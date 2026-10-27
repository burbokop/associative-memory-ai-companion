//! Memory operations that need the model: turning conversation into episodic
//! memories, and letting weak memories fade by merging them into vaguer ones.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::{
    config::Config,
    emotion::Emotions,
    llm::{InputMessage, JsonRequest, Llm, Role, schema},
    memory::{LongTerm, Memory, Message, transcript},
};

/// Who is remembering whom.
pub struct Names<'a> {
    pub companion: &'a str,
    pub user: &'a str,
}

#[derive(Debug, Deserialize)]
struct Episode {
    summary: String,
    emotions: Emotions,
    importance: i64,
}

#[derive(Debug, Deserialize)]
struct Consolidated {
    memories: Vec<Episode>,
    core_facts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Facts {
    facts: Vec<String>,
}

fn episode_schema(summary: &str) -> serde_json::Value {
    schema::object(&[
        ("summary", schema::string(summary)),
        ("emotions", Emotions::schema("What you felt about it")),
        (
            "importance",
            schema::integer("How much it matters to you, 0 to 100"),
        ),
    ])
}

fn to_unit(importance: i64) -> f32 {
    importance.clamp(0, 100) as f32 / 100.0
}

/// Turns a chunk of conversation into episodic memories and new core facts.
pub async fn consolidate(
    llm: &impl Llm,
    long_term: &mut LongTerm,
    messages: &[Message],
    names: &Names<'_>,
) -> Result<()> {
    let (Some(first), Some(last)) = (messages.first(), messages.last()) else {
        return Ok(());
    };
    let Names { companion, user } = names;
    let known_facts = long_term
        .core_facts
        .iter()
        .map(|f| format!("- {}", f.fact))
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = format!(
        "You are the memory of {companion}, a person who chats with {user}. \
         Turn the conversation excerpt into 1 to 5 episodic memories, written by {companion} \
         in first person (\"I\", \"{user}\"), each about one topic or moment. Keep concrete \
         details that a friend would remember; drop small talk. For every memory record what \
         {companion} felt and how important it is to them.\n\
         Also list NEW core facts: stable, important facts about {user}, about {companion}'s \
         own life, or about the relationship (names, family, where they live, big life events, \
         promises, how they relate to each other). Do not repeat already known facts. \
         Return an empty list if there is nothing new.\n\n\
         Already known core facts:\n{known_facts}"
    );
    let request = JsonRequest {
        instructions,
        input: vec![InputMessage::new(
            Role::User,
            transcript(messages, companion, user),
        )],
        schema_name: "consolidation",
        schema: schema::object(&[
            (
                "memories",
                schema::array(
                    episode_schema("What happened, 1-3 sentences"),
                    "Episodic memories",
                ),
            ),
            (
                "core_facts",
                schema::array(schema::string("One fact"), "New core facts"),
            ),
        ]),
    };
    let answer: Consolidated = llm.complete(request).await?;

    let summaries: Vec<String> = answer.memories.iter().map(|e| e.summary.clone()).collect();
    let embeddings = llm.embed(&summaries).await?;
    for (episode, embedding) in answer.memories.into_iter().zip(embeddings) {
        let memory = Memory {
            id: 0,
            summary: episode.summary,
            emotions: episode.emotions,
            importance: to_unit(episode.importance),
            started_at: first.time,
            ended_at: last.time,
            last_recalled: last.time,
            recall_count: 0,
            merged_from: 1,
        };
        long_term.add(memory, embedding);
    }
    for fact in &answer.core_facts {
        long_term.add_core_fact(fact, last.time);
    }
    Ok(())
}

/// Rewrites the core facts into a shorter list when there are too many.
/// Facts are condensed, never dropped.
pub async fn condense_core_facts(
    llm: &impl Llm,
    long_term: &mut LongTerm,
    names: &Names<'_>,
    config: &Config,
    now: DateTime<Utc>,
) -> Result<bool> {
    if long_term.core_facts.len() <= config.max_core_facts {
        return Ok(false);
    }
    let list = long_term
        .core_facts
        .iter()
        .map(|f| format!("- {}", f.fact))
        .collect::<Vec<_>>()
        .join("\n");
    let request = JsonRequest {
        instructions: format!(
            "These are the core facts {} knows about themselves, {} and their relationship. \
             Rewrite them as at most {} facts by combining related ones. Do not lose any \
             information; if two facts contradict, keep the newer one (later in the list).",
            names.companion, names.user, config.max_core_facts
        ),
        input: vec![InputMessage::new(Role::User, list)],
        schema_name: "core_facts",
        schema: schema::object(&[("facts", schema::array(schema::string("One fact"), ""))]),
    };
    let answer: Facts = llm.complete(request).await?;
    if answer.facts.is_empty() {
        return Ok(false);
    }
    long_term.core_facts.clear();
    for fact in &answer.facts {
        long_term.add_core_fact(fact, now);
    }
    Ok(true)
}

/// While there are too many memories, merges the weakest ones into a single,
/// vaguer memory — the way old, unremarkable days blur together.
/// Returns how many memories were merged away.
pub async fn fade_weak_memories(
    llm: &impl Llm,
    long_term: &mut LongTerm,
    names: &Names<'_>,
    config: &Config,
    now: DateTime<Utc>,
) -> Result<usize> {
    let batch = config.merge_batch.max(2);
    let mut merged = 0;
    while long_term.memories.len() > config.max_memories {
        let ids = long_term.weakest(batch, now, config);
        let mut group: Vec<Memory> = ids
            .iter()
            .filter_map(|id| long_term.get(*id).cloned())
            .collect();
        group.sort_by_key(|m| m.started_at);

        let list = group
            .iter()
            .map(|m| m.render(now))
            .collect::<Vec<_>>()
            .join("\n");
        let request = JsonRequest {
            instructions: format!(
                "You are the memory of {}. These memories have faded over time. Merge them \
                 into ONE vaguer, more general memory in first person, the way people remember \
                 old times: keep the gist, recurring themes and the strongest feelings, lose \
                 the small details.",
                names.companion
            ),
            input: vec![InputMessage::new(Role::User, list)],
            schema_name: "merged_memory",
            schema: episode_schema("The merged memory, 1-2 sentences"),
        };
        let episode: Episode = llm.complete(request).await?;
        let embedding = llm.embed_one(&episode.summary).await?;

        let memory = Memory {
            id: 0,
            summary: episode.summary,
            emotions: episode.emotions,
            importance: to_unit(episode.importance),
            started_at: group.iter().map(|m| m.started_at).min().unwrap_or(now),
            ended_at: group.iter().map(|m| m.ended_at).max().unwrap_or(now),
            last_recalled: group.iter().map(|m| m.last_recalled).max().unwrap_or(now),
            recall_count: group.iter().map(|m| m.recall_count).sum(),
            merged_from: group.iter().map(|m| m.merged_from).sum(),
        };
        long_term.remove(&ids);
        long_term.add(memory, embedding);
        merged += ids.len();
    }
    Ok(merged)
}

/// Embeds memories that have no vector yet (e.g. after changing the embedding model).
pub async fn ensure_embeddings(llm: &impl Llm, long_term: &mut LongTerm) -> Result<()> {
    let ids = long_term.missing_embeddings();
    if ids.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = ids
        .iter()
        .filter_map(|id| long_term.get(*id).map(|m| m.summary.clone()))
        .collect();
    let vectors = llm.embed(&texts).await?;
    long_term.embeddings.extend(ids.into_iter().zip(vectors));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        llm::fake::FakeLlm,
        memory::{Speaker, tests::memory},
    };
    use chrono::TimeDelta;
    use serde_json::json;

    const NAMES: Names = Names {
        companion: "Mira",
        user: "Borys",
    };

    #[tokio::test]
    async fn consolidates_conversation_into_memories_and_facts() {
        let now = Utc::now();
        let messages = vec![
            Message::new(Speaker::User, "My dog Rex ate my shoes", now),
            Message::new(
                Speaker::Companion,
                "Haha, poor shoes!",
                now + TimeDelta::minutes(1),
            ),
        ];
        let llm = FakeLlm::with_answers([json!({
            "memories": [{
                "summary": "Borys told me his dog Rex ate his shoes",
                "emotions": [{"name": "amusement", "intensity": 70}],
                "importance": 40
            }],
            "core_facts": ["Borys has a dog named Rex"]
        })]);
        let mut lt = LongTerm::default();
        consolidate(&llm, &mut lt, &messages, &NAMES).await.unwrap();

        assert_eq!(lt.memories.len(), 1);
        let m = &lt.memories[0];
        assert_eq!(m.importance, 0.4);
        assert_eq!(m.started_at, now);
        assert_eq!(m.ended_at, now + TimeDelta::minutes(1));
        assert!(lt.embeddings.contains_key(&m.id));
        assert_eq!(lt.core_facts[0].fact, "Borys has a dog named Rex");
        let prompt = &llm.requests.borrow()[0].input[0].content;
        assert!(prompt.contains("Borys: My dog Rex ate my shoes"));
    }

    #[tokio::test]
    async fn fades_the_weakest_memories_into_one() {
        let config = Config {
            max_memories: 3,
            merge_batch: 2,
            ..Config::default()
        };
        let now = Utc::now();
        let mut lt = LongTerm::default();
        let old = now - TimeDelta::days(200);
        let weak1 = lt.add(memory("talked about rain", &[], 0.1, old), vec![]);
        let weak2 = lt.add(memory("talked about traffic", &[], 0.1, old), vec![]);
        let strong = lt.add(
            memory("first 'I love you'", &[("love", 95)], 0.9, old),
            vec![],
        );
        lt.add(memory("lunch today", &[], 0.2, now), vec![]);

        let llm = FakeLlm::with_answers([json!({
            "summary": "We used to chat about everyday things like the weather",
            "emotions": [],
            "importance": 10
        })]);
        let merged = fade_weak_memories(&llm, &mut lt, &NAMES, &config, now)
            .await
            .unwrap();

        assert_eq!(merged, 2);
        assert_eq!(lt.memories.len(), 3);
        assert!(lt.get(weak1).is_none() && lt.get(weak2).is_none());
        assert!(lt.get(strong).is_some(), "emotional memory must survive");
        let blurred = lt.memories.iter().find(|m| m.merged_from == 2).unwrap();
        assert_eq!(blurred.started_at, old);
    }

    #[tokio::test]
    async fn condenses_core_facts_only_over_limit() {
        let config = Config {
            max_core_facts: 2,
            ..Config::default()
        };
        let now = Utc::now();
        let mut lt = LongTerm::default();
        lt.add_core_fact("Borys has a dog", now);
        lt.add_core_fact("The dog is named Rex", now);
        let llm = FakeLlm::default();
        assert!(
            !condense_core_facts(&llm, &mut lt, &NAMES, &config, now)
                .await
                .unwrap()
        );

        lt.add_core_fact("Borys lives in Kyiv", now);
        let llm = FakeLlm::with_answers([json!({
            "facts": ["Borys has a dog named Rex", "Borys lives in Kyiv"]
        })]);
        assert!(
            condense_core_facts(&llm, &mut lt, &NAMES, &config, now)
                .await
                .unwrap()
        );
        assert_eq!(lt.core_facts.len(), 2);
    }
}
