//! Conversation logic, independent of the console.

use anyhow::Result;
use chrono::{TimeDelta, Utc};

use crate::{
    config::Config,
    consolidation::{self, Names},
    emotion::Emotions,
    llm::{JsonRequest, Llm},
    memory::{self, Memory, Message, Speaker},
    mind::{self, Mind, MindRecord},
    persona,
    profile::Profile,
    prompt::{self, TurnContext, TurnOutput},
    timefmt,
};

/// How many recent messages are used to trigger associative recall.
const RECALL_CONTEXT_MESSAGES: usize = 3;

pub struct Reply {
    pub content: String,
    pub emotions: Emotions,
    pub thoughts: String,
    pub recalled: Vec<String>,
}

#[derive(Debug, Default)]
pub struct MaintenanceReport {
    pub consolidated_messages: usize,
    pub new_memories: usize,
    pub faded_memories: usize,
    pub core_facts_condensed: bool,
}

pub struct Session<L: Llm> {
    pub llm: L,
    pub config: Config,
    pub profile: Profile,
}

impl<L: Llm> Session<L> {
    pub fn new(llm: L, config: Config, profile: Profile) -> Self {
        Self {
            llm,
            config,
            profile,
        }
    }

    pub fn companion_name(&self) -> String {
        persona::name(&self.profile.persona)
    }

    pub fn user_name(&self) -> &str {
        &self.profile.state.user_name
    }

    /// Notices what happened since the last session (time passed, persona edits)
    /// and lets the companion speak first if there is a reason to.
    pub async fn start(&mut self) -> Result<Option<Reply>> {
        let now = Utc::now();
        let user = self.user_name().to_string();
        let mut speak_first = false;

        let changes = persona::detect_changes(&self.profile.persona, &self.profile.persona_meta);
        if !changes.is_empty() {
            persona::accept_user_changes(
                &mut self.profile.persona_meta,
                &self.profile.persona,
                &changes,
                now,
            );
            self.profile.save_persona_meta()?;
            let text = persona::describe_changes(&changes, &user);
            self.profile
                .short_term
                .push(Message::new(Speaker::Event, text, now));
            speak_first = true;
        }

        let current_mind = Mind::from_config(&self.config);
        match &self.profile.state.mind {
            Some(record) if record.mind == current_mind => {}
            previous => {
                if let Some(text) = previous.as_ref().and_then(|old| {
                    mind::describe_change(old, &current_mind, &self.config, &user, now)
                }) {
                    self.profile
                        .short_term
                        .push(Message::new(Speaker::Event, text, now));
                    speak_first = true;
                }
                self.profile.state.mind = Some(MindRecord {
                    mind: current_mind,
                    since: now,
                });
            }
        }

        match self.profile.state.last_interaction {
            None => {
                self.profile.short_term.push(Message::new(
                    Speaker::Event,
                    format!(
                        "This is the very first time you talk to {user}. Say hello and \
                         introduce yourself in your own way."
                    ),
                    now,
                ));
                speak_first = true;
            }
            Some(last) => {
                let gap = now - last;
                if gap >= TimeDelta::minutes(self.config.absence_greeting_after_minutes) {
                    self.profile.short_term.push(Message::new(
                        Speaker::Event,
                        format!(
                            "{user} has just opened the chat again after {} without talking \
                             to you (your last conversation was on {}). Notice how long it has \
                             been and react the way you genuinely feel about it.",
                            timefmt::duration(gap),
                            timefmt::stamp(last)
                        ),
                        now,
                    ));
                    speak_first = true;
                }
            }
        }

        let state = &mut self.profile.state;
        state.emotions = state
            .emotions
            .decayed(now - state.emotions_at, self.config.emotion_half_life_hours);
        state.emotions_at = now;

        consolidation::ensure_embeddings(&self.llm, &mut self.profile.long_term).await?;
        self.profile.save()?;

        if speak_first {
            Ok(Some(self.respond().await?))
        } else {
            Ok(None)
        }
    }

    pub async fn user_says(&mut self, text: &str) -> Result<Reply> {
        let now = Utc::now();
        let previous_interaction = self.profile.state.last_interaction;
        self.profile
            .short_term
            .push(Message::new(Speaker::User, text, now));
        self.profile.state.last_interaction = Some(now);

        match self.respond().await {
            Ok(reply) => Ok(reply),
            Err(e) => {
                // The message was never "heard"; let the user try again cleanly.
                self.profile.short_term.pop();
                self.profile.state.last_interaction = previous_interaction;
                Err(e)
            }
        }
    }

    async fn respond(&mut self) -> Result<Reply> {
        let now = Utc::now();
        let config = &self.config;
        let profile = &mut self.profile;

        let recent = profile.long_term.recent(config.recent_memories_in_prompt);
        let query: String = profile
            .short_term
            .iter()
            .rev()
            .take(RECALL_CONTEXT_MESSAGES)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let recalled = if profile.long_term.memories.is_empty() || query.is_empty() {
            Vec::new()
        } else {
            let vector = self.llm.embed_one(&query).await?;
            profile.long_term.recall(&vector, &recent, now, config)
        };

        let lookup = |ids: &[u64]| -> Vec<&Memory> {
            ids.iter()
                .filter_map(|id| profile.long_term.get(*id))
                .collect()
        };
        let context = TurnContext {
            persona: &profile.persona,
            persona_meta: &profile.persona_meta,
            user: &profile.state.user_name,
            met_at: profile.state.met_at,
            core_facts: &profile.long_term.core_facts,
            recent: lookup(&recent),
            recalled: lookup(&recalled),
            emotions: &profile.state.emotions,
            mind: mind::render_for_prompt(
                &profile.state.mind.clone().unwrap_or_else(|| MindRecord {
                    mind: Mind::from_config(config),
                    since: now,
                }),
                config,
                &profile.state.user_name,
                now,
            ),
            messages: &profile.short_term,
            now,
        };
        let request = JsonRequest {
            instructions: prompt::instructions(&context),
            input: prompt::input(&profile.short_term),
            schema_name: "turn",
            schema: prompt::turn_schema(),
        };
        let recalled_text: Vec<String> = context.recalled.iter().map(|m| m.render(now)).collect();
        let output: TurnOutput = self.llm.complete(request).await?;

        let now = Utc::now();
        let mut message = Message::new(Speaker::Companion, output.reply.clone(), now);
        message.emotions = Some(output.emotions.clone());
        message.thoughts = Some(output.inner_thoughts.clone());
        profile.short_term.push(message);
        profile.state.emotions = output.emotions.clone();
        profile.state.emotions_at = now;
        profile.state.last_interaction = Some(now);
        profile.save()?;

        Ok(Reply {
            content: output.reply,
            emotions: output.emotions,
            thoughts: output.inner_thoughts,
            recalled: recalled_text,
        })
    }

    /// Moves old messages into long-term memory when short-term memory overflows
    /// (or always, if `force`), then lets weak memories fade.
    pub async fn maintain_memory(&mut self, force: bool) -> Result<MaintenanceReport> {
        let now = Utc::now();
        let config = &self.config;
        let profile = &mut self.profile;
        let mut report = MaintenanceReport::default();

        let count = if force {
            Some(profile.short_term.len().saturating_sub(2)).filter(|n| *n > 0)
        } else {
            memory::overflow(
                &profile.short_term,
                config.short_term_max_tokens,
                config.short_term_keep_tokens,
            )
        };
        let Some(count) = count else {
            return Ok(report);
        };

        let companion = persona::name(&profile.persona);
        let names = Names {
            companion: &companion,
            user: &profile.state.user_name,
        };
        let before = profile.long_term.memories.len();
        consolidation::consolidate(
            &self.llm,
            &mut profile.long_term,
            &profile.short_term[..count],
            &names,
        )
        .await?;
        report.new_memories = profile.long_term.memories.len() - before;
        report.consolidated_messages = count;
        profile.short_term.drain(..count);
        profile.save()?;

        report.core_facts_condensed = consolidation::condense_core_facts(
            &self.llm,
            &mut profile.long_term,
            &names,
            config,
            now,
        )
        .await?;
        report.faded_memories = consolidation::fade_weak_memories(
            &self.llm,
            &mut profile.long_term,
            &names,
            config,
            now,
        )
        .await?;
        profile.save()?;
        Ok(report)
    }

    /// Memories most related to `query`, with their scores.
    pub async fn search_memories(&self, query: &str) -> Result<Vec<(f32, &Memory)>> {
        let vector = self.llm.embed_one(query).await?;
        let long_term = &self.profile.long_term;
        Ok(long_term
            .search(&vector, Utc::now(), &self.config)
            .into_iter()
            .filter_map(|(score, id)| Some((score, long_term.get(id)?)))
            .collect())
    }

    /// All memories, strongest first, with their retention scores.
    pub fn memories_by_retention(&self) -> Vec<(f32, &Memory)> {
        let now = Utc::now();
        let mut list: Vec<(f32, &Memory)> = self
            .profile
            .long_term
            .memories
            .iter()
            .map(|m| (m.retention(now, &self.config), m))
            .collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        llm::fake::FakeLlm,
        persona::{Author, PersonaMeta},
        profile::{Paths, State},
    };
    use serde_json::{Value, json};

    fn turn(reply: &str) -> Value {
        json!({
            "inner_thoughts": "thinking",
            "reply": reply,
            "emotions": [{"name": "joy", "intensity": 50}]
        })
    }

    fn session(test: &str, answers: Vec<Value>) -> (Session<FakeLlm>, std::path::PathBuf) {
        let data = std::env::temp_dir().join(format!("amc-session-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let now = Utc::now();
        let persona = json!({"name": "Mira", "personality": "calm"})
            .as_object()
            .unwrap()
            .clone();
        let meta = PersonaMeta::new(&persona, Author::Companion, now);
        let state = State {
            user_name: "Borys".into(),
            met_at: now,
            last_interaction: None,
            emotions: Emotions::default(),
            emotions_at: now,
            mind: None,
        };
        let profile =
            Profile::create("t", Paths::new(&data, "t"), persona, meta, state, "fake").unwrap();
        let config = Config {
            absence_greeting_after_minutes: 60,
            ..Config::default()
        };
        (
            Session::new(FakeLlm::with_answers(answers), config, profile),
            data,
        )
    }

    fn last_event(s: &Session<FakeLlm>) -> String {
        s.profile
            .short_term
            .iter()
            .rev()
            .find(|m| m.speaker == Speaker::Event)
            .unwrap()
            .content
            .clone()
    }

    #[tokio::test]
    async fn first_start_introduces_itself() {
        let (mut s, data) = session("first", vec![turn("Hi, I'm Mira!")]);
        let reply = s
            .start()
            .await
            .unwrap()
            .expect("companion should speak first");
        assert_eq!(reply.content, "Hi, I'm Mira!");
        assert!(last_event(&s).contains("very first time"));
        assert_eq!(s.profile.short_term.len(), 2);
        assert!(s.profile.state.last_interaction.is_some());
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn notices_long_absence_but_not_short_one() {
        let (mut s, data) = session("absence", vec![turn("Where were you?!")]);
        s.profile.state.last_interaction = Some(Utc::now() - TimeDelta::days(9));
        assert!(s.start().await.unwrap().is_some());
        assert!(last_event(&s).contains("after 9 days"));

        s.profile.state.last_interaction = Some(Utc::now() - TimeDelta::minutes(5));
        let events = s.profile.short_term.len();
        assert!(s.start().await.unwrap().is_none());
        assert_eq!(s.profile.short_term.len(), events);
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn reacts_to_model_changes() {
        let (mut s, data) = session("mind", vec![turn("Hey, what happened to my brain?")]);
        s.profile.state.last_interaction = Some(Utc::now() - TimeDelta::minutes(1));

        // First start only records the model, without making a fuss about it.
        assert!(s.start().await.unwrap().is_none());
        assert_eq!(
            s.profile.state.mind.as_ref().unwrap().mind.model,
            "gpt-6-luna"
        );

        s.config.model = "gpt-6-astra".into();
        assert!(s.start().await.unwrap().is_some());
        assert!(last_event(&s).contains("from gpt-6-luna (the cheapest"));
        assert_eq!(
            s.profile.state.mind.as_ref().unwrap().mind.model,
            "gpt-6-astra"
        );
        let requests = s.llm.requests.borrow();
        assert!(
            requests[0]
                .instructions
                .contains("runs on the AI model gpt-6-astra (OpenAI's most capable")
        );
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn reacts_to_persona_edits() {
        let (mut s, data) = session("persona", vec![turn("Why did you make me fiery?")]);
        s.profile.state.last_interaction = Some(Utc::now() - TimeDelta::minutes(1));
        s.profile.persona["personality"] = json!("fiery");
        assert!(s.start().await.unwrap().is_some());
        assert!(last_event(&s).contains("from \"calm\" (chosen by you) to \"fiery\""));
        assert_eq!(
            s.profile.persona_meta.author_of("personality"),
            Author::User
        );
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn failed_turn_leaves_no_trace() {
        let (mut s, data) = session("fail", vec![]);
        assert!(s.user_says("hello?").await.is_err());
        assert!(s.profile.short_term.is_empty());
        assert!(s.profile.state.last_interaction.is_none());
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn conversation_flows_into_long_term_memory_and_back() {
        let consolidation = json!({
            "memories": [{
                "summary": "Borys told me his dog Rex ate his shoes",
                "emotions": [{"name": "amusement", "intensity": 70}],
                "importance": 40
            }],
            "core_facts": ["Borys has a dog named Rex"]
        });
        let (mut s, data) = session(
            "flow",
            vec![turn("Oh no!"), consolidation, turn("Is Rex okay?")],
        );
        s.profile.state.last_interaction = Some(Utc::now());
        s.user_says("My dog Rex ate my shoes").await.unwrap();

        let report = s.maintain_memory(true).await.unwrap();
        assert_eq!(
            report.consolidated_messages, 0,
            "keeps the last two messages"
        );
        s.profile
            .short_term
            .insert(0, Message::new(Speaker::User, "earlier", Utc::now()));
        let report = s.maintain_memory(true).await.unwrap();
        assert_eq!(report.consolidated_messages, 1);
        assert_eq!(report.new_memories, 1);

        let reply = s.user_says("Rex is sick today").await.unwrap();
        assert_eq!(reply.content, "Is Rex okay?");
        let requests = s.llm.requests.borrow();
        let instructions = &requests.last().unwrap().instructions;
        assert!(instructions.contains("Borys has a dog named Rex"));
        assert!(instructions.contains("dog Rex ate his shoes"));
        std::fs::remove_dir_all(data).unwrap();
    }
}
