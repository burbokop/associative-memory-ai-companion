//! Short-term memory (raw messages) and long-term memory (episodic summaries,
//! core facts), retention scoring and associative recall.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{config::Config, emotion::Emotions, timefmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    User,
    Companion,
    /// Something that happened outside the chat (absence, persona edits...).
    Event,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub speaker: Speaker,
    pub content: String,
    pub time: DateTime<Utc>,
    /// What the companion felt after writing this message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotions: Option<Emotions>,
    /// The companion's private thoughts behind this message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thoughts: Option<String>,
}

impl Message {
    pub fn new(speaker: Speaker, content: impl Into<String>, time: DateTime<Utc>) -> Self {
        Self {
            speaker,
            content: content.into(),
            time,
            emotions: None,
            thoughts: None,
        }
    }

    /// Rough token estimate (~4 characters per token), good enough for budgeting.
    pub fn tokens(&self) -> usize {
        let chars = self.content.len() + self.thoughts.as_ref().map_or(0, String::len);
        chars / 4 + 8
    }
}

/// How many of the oldest messages should be consolidated into long-term memory,
/// or `None` if short-term memory is still within budget.
pub fn overflow(messages: &[Message], max_tokens: usize, keep_tokens: usize) -> Option<usize> {
    let total: usize = messages.iter().map(Message::tokens).sum();
    if total <= max_tokens || messages.len() <= 2 {
        return None;
    }
    let mut kept = 0;
    let mut keep_from = messages.len();
    for (i, message) in messages.iter().enumerate().rev() {
        kept += message.tokens();
        if kept > keep_tokens && messages.len() - i > 2 {
            break;
        }
        keep_from = i;
    }
    (keep_from > 0).then_some(keep_from)
}

/// A conversation excerpt written out for memory operations.
pub fn transcript(messages: &[Message], companion: &str, user: &str) -> String {
    let mut out = String::new();
    for m in messages {
        let stamp = timefmt::stamp(m.time);
        match m.speaker {
            Speaker::User => out.push_str(&format!("[{stamp}] {user}: {}\n", m.content)),
            Speaker::Event => out.push_str(&format!("[{stamp}] (event) {}\n", m.content)),
            Speaker::Companion => {
                if let Some(thoughts) = &m.thoughts {
                    out.push_str(&format!("[{stamp}] ({companion} thinks: {thoughts})\n"));
                }
                out.push_str(&format!("[{stamp}] {companion}: {}", m.content));
                if let Some(emotions) = &m.emotions {
                    out.push_str(&format!("  [{companion} feels: {}]", emotions.describe()));
                }
                out.push('\n');
            }
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: u64,
    /// Written by the companion in first person.
    pub summary: String,
    pub emotions: Emotions,
    /// 0..1, how much this matters to the companion.
    pub importance: f32,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub last_recalled: DateTime<Utc>,
    pub recall_count: u32,
    /// How many original episodes were merged into this memory (1 = vivid, many = vague).
    pub merged_from: u32,
}

impl Memory {
    /// How strongly this memory holds on: emotional charge, importance and recency.
    pub fn retention(&self, now: DateTime<Utc>, config: &Config) -> f32 {
        let w = &config.retention_weights;
        let age_days = (now - self.last_recalled).num_seconds().max(0) as f64 / 86400.0;
        let recency = 0.5f64.powf(age_days / config.memory_half_life_days) as f32;
        w.emotion * self.emotions.peak() + w.importance * self.importance + w.recency * recency
    }

    /// One line for prompts: "(3 weeks ago; felt joy 70) summary".
    pub fn render(&self, now: DateTime<Utc>) -> String {
        let when = if self.merged_from > 1 {
            format!("around {}", timefmt::ago(self.ended_at, now))
        } else {
            timefmt::ago(self.ended_at, now)
        };
        let feelings = if self.emotions.is_empty() {
            String::new()
        } else {
            format!("; felt {}", self.emotions.describe())
        };
        format!("({when}{feelings}) {}", self.summary)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreFact {
    pub fact: String,
    pub added_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LongTerm {
    next_id: u64,
    pub core_facts: Vec<CoreFact>,
    pub memories: Vec<Memory>,
    /// Kept in a separate file so `long_term.json` stays readable.
    #[serde(skip)]
    pub embeddings: HashMap<u64, Vec<f32>>,
}

impl LongTerm {
    /// Stores a memory, assigning it a fresh id.
    pub fn add(&mut self, mut memory: Memory, embedding: Vec<f32>) -> u64 {
        self.next_id += 1;
        memory.id = self.next_id;
        self.embeddings.insert(memory.id, embedding);
        self.memories.push(memory);
        self.next_id
    }

    pub fn remove(&mut self, ids: &[u64]) {
        self.memories.retain(|m| !ids.contains(&m.id));
        for id in ids {
            self.embeddings.remove(id);
        }
    }

    pub fn get(&self, id: u64) -> Option<&Memory> {
        self.memories.iter().find(|m| m.id == id)
    }

    /// Adds a fact unless an equal one is already known.
    pub fn add_core_fact(&mut self, fact: &str, now: DateTime<Utc>) -> bool {
        let fact = fact.trim();
        let normalized = fact.to_lowercase();
        if fact.is_empty()
            || self
                .core_facts
                .iter()
                .any(|f| f.fact.to_lowercase() == normalized)
        {
            return false;
        }
        self.core_facts.push(CoreFact {
            fact: fact.to_string(),
            added_at: now,
        });
        true
    }

    /// Ids of the `n` most recent memories, newest first.
    pub fn recent(&self, n: usize) -> Vec<u64> {
        let mut list: Vec<&Memory> = self.memories.iter().collect();
        list.sort_by_key(|m| std::cmp::Reverse(m.ended_at));
        list.into_iter().take(n).map(|m| m.id).collect()
    }

    /// Ids of the `n` memories with the lowest retention.
    pub fn weakest(&self, n: usize, now: DateTime<Utc>, config: &Config) -> Vec<u64> {
        let mut list: Vec<(f32, u64)> = self
            .memories
            .iter()
            .map(|m| (m.retention(now, config), m.id))
            .collect();
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        list.into_iter().take(n).map(|(_, id)| id).collect()
    }

    /// Memories ranked by similarity to `query`, weighted by retention.
    pub fn search(&self, query: &[f32], now: DateTime<Utc>, config: &Config) -> Vec<(f32, u64)> {
        let mut ranked: Vec<(f32, u64)> = self
            .memories
            .iter()
            .filter_map(|m| {
                let similarity = cosine(query, self.embeddings.get(&m.id)?);
                (similarity >= config.recall_min_similarity)
                    .then(|| (similarity * (0.5 + 0.5 * m.retention(now, config)), m.id))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        ranked
    }

    /// Associative recall: the best matches come to mind and get refreshed.
    pub fn recall(
        &mut self,
        query: &[f32],
        exclude: &[u64],
        now: DateTime<Utc>,
        config: &Config,
    ) -> Vec<u64> {
        let ids: Vec<u64> = self
            .search(query, now, config)
            .into_iter()
            .map(|(_, id)| id)
            .filter(|id| !exclude.contains(id))
            .take(config.recall_top_k)
            .collect();
        for memory in self.memories.iter_mut().filter(|m| ids.contains(&m.id)) {
            memory.last_recalled = now;
            memory.recall_count += 1;
        }
        ids
    }

    pub fn missing_embeddings(&self) -> Vec<u64> {
        self.memories
            .iter()
            .filter(|m| !self.embeddings.contains_key(&m.id))
            .map(|m| m.id)
            .collect()
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{emotion::RawEmotion, llm::fake::bag_of_words};
    use chrono::TimeDelta;

    pub fn emotions(list: &[(&str, i64)]) -> Emotions {
        list.iter()
            .map(|(n, i)| RawEmotion {
                name: n.to_string(),
                intensity: *i,
            })
            .collect::<Vec<_>>()
            .into()
    }

    pub fn memory(
        summary: &str,
        feel: &[(&str, i64)],
        importance: f32,
        when: DateTime<Utc>,
    ) -> Memory {
        Memory {
            id: 0,
            summary: summary.into(),
            emotions: emotions(feel),
            importance,
            started_at: when,
            ended_at: when,
            last_recalled: when,
            recall_count: 0,
            merged_from: 1,
        }
    }

    #[test]
    fn emotional_and_recent_memories_retain_better() {
        let config = Config::default();
        let now = Utc::now();
        let calm = memory("weather talk", &[], 0.2, now);
        let intense = memory("big fight", &[("anger", 90)], 0.2, now);
        let old_intense = memory(
            "big fight",
            &[("anger", 90)],
            0.2,
            now - TimeDelta::days(120),
        );
        assert!(intense.retention(now, &config) > calm.retention(now, &config));
        assert!(intense.retention(now, &config) > old_intense.retention(now, &config));
    }

    #[test]
    fn recall_finds_related_memories_and_refreshes_them() {
        let config = Config::default();
        let now = Utc::now();
        let old = now - TimeDelta::days(60);
        let mut lt = LongTerm::default();
        let dog = memory(
            "Borys told me about his dog Rex chewing shoes",
            &[("amusement", 60)],
            0.5,
            old,
        );
        let job = memory("Borys got a new job at a bank", &[("pride", 50)], 0.6, old);
        let dog_id = lt.add(dog.clone(), bag_of_words(&dog.summary));
        lt.add(job.clone(), bag_of_words(&job.summary));

        let recalled = lt.recall(&bag_of_words("my dog Rex is sick"), &[], now, &config);
        assert_eq!(recalled, vec![dog_id]);
        let refreshed = lt.get(dog_id).unwrap();
        assert_eq!(refreshed.last_recalled, now);
        assert_eq!(refreshed.recall_count, 1);

        assert!(
            lt.recall(&bag_of_words("my dog Rex"), &[dog_id], now, &config)
                .is_empty()
        );
    }

    #[test]
    fn weakest_and_recent() {
        let config = Config::default();
        let now = Utc::now();
        let mut lt = LongTerm::default();
        let a = lt.add(memory("a", &[], 0.1, now - TimeDelta::days(90)), vec![]);
        let b = lt.add(
            memory("b", &[("joy", 90)], 0.9, now - TimeDelta::days(1)),
            vec![],
        );
        let c = lt.add(memory("c", &[], 0.3, now), vec![]);
        assert_eq!(lt.weakest(1, now, &config), vec![a]);
        assert_eq!(lt.recent(2), vec![c, b]);
        lt.remove(&[a]);
        assert_eq!(lt.memories.len(), 2);
        assert!(!lt.embeddings.contains_key(&a));
    }

    #[test]
    fn core_facts_are_deduplicated() {
        let mut lt = LongTerm::default();
        let now = Utc::now();
        assert!(lt.add_core_fact("Borys lives in Kyiv", now));
        assert!(!lt.add_core_fact("borys lives in kyiv ", now));
        assert!(!lt.add_core_fact("  ", now));
        assert_eq!(lt.core_facts.len(), 1);
    }

    #[test]
    fn overflow_keeps_recent_messages() {
        let now = Utc::now();
        let messages: Vec<Message> = (0..10)
            .map(|i| Message::new(Speaker::User, "x".repeat(368), now + TimeDelta::minutes(i)))
            .collect(); // 100 tokens each
        assert_eq!(overflow(&messages, 1000, 300), None);
        assert_eq!(overflow(&messages, 900, 300), Some(7));
        // Always keeps at least two messages, even if they exceed the budget.
        assert_eq!(overflow(&messages, 900, 50), Some(8));
    }
}
