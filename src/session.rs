//! Conversation logic, independent of the console.

use anyhow::{Context, Result};
use chrono::{TimeDelta, Utc};

use crate::{
    config::Config,
    consolidation::{self, Names},
    emotion::Emotions,
    llm::{JsonRequest, Llm},
    memory::{self, Memory, Message, Speaker},
    mind::{self, Mind, MindRecord},
    persona::{self, Persona},
    profile::Profile,
    prompt::{self, TurnContext, TurnOutput},
    relationship::{self, DimensionChange, Relationship},
    storage, timefmt,
    tracking::Author,
};

/// How many recent messages are used to trigger associative recall.
const RECALL_CONTEXT_MESSAGES: usize = 3;

pub struct Reply {
    pub content: String,
    pub emotions: Emotions,
    pub thoughts: String,
    pub recalled: Vec<String>,
    pub relationship_changes: Vec<DimensionChange>,
}

#[derive(Debug, Default)]
pub struct MaintenanceReport {
    pub consolidated_messages: usize,
    pub new_memories: usize,
    pub faded_memories: usize,
    pub core_facts_condensed: bool,
    /// The companion now describes the relationship differently.
    pub new_relationship_label: Option<String>,
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

    /// Re-reads the user-editable files (persona, relationship) and tells the
    /// companion about edits. Returns whether anything was edited.
    fn sync_files(&mut self) -> Result<bool> {
        let now = Utc::now();
        let user = self.user_name().to_string();
        let profile = &mut self.profile;
        let mut edited = false;

        let persona_path = profile.paths.persona();
        let persona: Persona = storage::load(&persona_path)?
            .with_context(|| format!("{} is missing", persona_path.display()))?;
        let changes = profile
            .persona_meta
            .tracked
            .record(&persona, Author::User, now);
        if !changes.is_empty() {
            profile.persona = persona;
            profile.save_persona_meta()?;
            let text = persona::describe_changes(&changes, &user);
            profile
                .short_term
                .push(Message::new(Speaker::Event, text, now));
            edited = true;
        }

        match storage::load::<Relationship>(&profile.paths.relationship())? {
            // Deleted while running: put the current state back.
            None => profile.save_relationship()?,
            Some(relationship) => {
                let changes = profile
                    .relationship_meta
                    .record(&relationship, Author::User, now);
                if !changes.is_empty() {
                    profile.relationship = relationship;
                    profile.save_relationship_meta()?;
                    let text = relationship::describe_changes(&changes, &user);
                    profile
                        .short_term
                        .push(Message::new(Speaker::Event, text, now));
                    edited = true;
                }
            }
        }
        Ok(edited)
    }

    /// Notices what happened since the last session (time passed, edits of persona
    /// or relationship, model change) and lets the companion speak first if there is
    /// a reason to.
    pub async fn start(&mut self) -> Result<Option<Reply>> {
        let now = Utc::now();
        let user = self.user_name().to_string();
        let mut speak_first = self.sync_files()?;

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
        // Files may have been edited while the chat is open.
        self.sync_files()?;
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
            relationship: relationship::render_for_prompt(
                &profile.relationship,
                &profile.relationship_meta,
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

        let relationship_changes = relationship::apply_updates(
            &mut profile.relationship,
            &mut profile.relationship_meta,
            &output.relationship_changes,
            config.relationship_max_step,
            now,
        );
        if !relationship_changes.is_empty() {
            profile.save_relationship()?;
        }

        Ok(Reply {
            content: output.reply,
            emotions: output.emotions,
            thoughts: output.inner_thoughts,
            recalled: recalled_text,
            relationship_changes,
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

        let transcript =
            memory::transcript(&profile.short_term[..count], names.companion, names.user);
        report.new_relationship_label = relationship::reflect(
            &self.llm,
            &mut profile.relationship,
            &mut profile.relationship_meta,
            &transcript,
            &names,
            now,
        )
        .await?;
        profile.save_relationship()?;

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

    /// Lets the companion know the user looked at the relationship view.
    pub fn relationship_viewed(&mut self, saw_private: bool) -> Result<()> {
        let text = relationship::viewed_event(self.user_name(), saw_private);
        self.profile
            .short_term
            .push(Message::new(Speaker::Event, text, Utc::now()));
        self.profile.save()
    }

    /// The last `count` chat messages (without internal events), oldest first.
    /// Only messages still in short-term memory are available.
    pub fn recent_messages(&self, count: usize) -> Vec<&Message> {
        let mut list: Vec<&Message> = self
            .profile
            .short_term
            .iter()
            .rev()
            .filter(|m| m.speaker != Speaker::Event)
            .take(count)
            .collect();
        list.reverse();
        list
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
            "emotions": [{"name": "joy", "intensity": 50}],
            "relationship_changes": []
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

    #[test]
    fn recent_messages_skip_events() {
        let (mut s, data) = session("recent", vec![]);
        let now = Utc::now();
        for (speaker, text) in [
            (Speaker::User, "one"),
            (Speaker::Companion, "two"),
            (Speaker::Event, "event"),
            (Speaker::User, "three"),
        ] {
            s.profile.short_term.push(Message::new(speaker, text, now));
        }
        let texts = |n| -> Vec<String> {
            s.recent_messages(n)
                .iter()
                .map(|m| m.content.clone())
                .collect()
        };
        assert_eq!(texts(2), ["two", "three"]);
        assert_eq!(texts(10), ["one", "two", "three"]);
        assert!(texts(0).is_empty());
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
    async fn relationship_changes_with_each_message() {
        let mut answer = turn("You remembered! That means a lot.");
        answer["relationship_changes"] = json!([
            {"dimension": "trust", "value": 28},
            {"dimension": "closeness", "value": 90}
        ]);
        let (mut s, data) = session("rel-turn", vec![answer]);
        s.profile.state.last_interaction = Some(Utc::now());
        let reply = s.user_says("Happy birthday, Mira!").await.unwrap();

        let changes = relationship::render_changes(&reply.relationship_changes);
        assert_eq!(
            changes, "trust 20→28 ↑ · closeness 5→20 ↑",
            "big jumps are limited"
        );
        let on_disk: Relationship = storage::load(&s.profile.paths.relationship())
            .unwrap()
            .unwrap();
        assert_eq!(on_disk.dimensions["trust"], json!(28));
        let instructions = &s.llm.requests.borrow()[0].instructions;
        assert!(instructions.contains("# Your relationship with Borys"));
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn notices_relationship_edits_while_chatting() {
        let (mut s, data) = session(
            "rel-edit",
            vec![turn("Wait, did you just change how I feel?")],
        );
        s.profile.state.last_interaction = Some(Utc::now());
        let mut edited = s.profile.relationship.clone();
        edited.dimensions.insert("attraction".into(), json!(90));
        storage::save(&s.profile.paths.relationship(), &edited).unwrap();

        s.user_says("hi").await.unwrap();
        let event = last_event(&s);
        assert!(event.contains("edited the file that describes your relationship"));
        assert!(event.contains("\"dimensions.attraction\" from \"5\" (chosen by you) to \"90\""));
        // The event comes before the user's message, so the reply can react to it.
        let speakers: Vec<Speaker> = s.profile.short_term.iter().map(|m| m.speaker).collect();
        assert_eq!(
            speakers,
            [Speaker::Event, Speaker::User, Speaker::Companion]
        );
        assert_eq!(s.profile.relationship.dimensions["attraction"], json!(90));
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn knows_when_the_user_looked_at_the_relationship() {
        let (mut s, data) = session("rel-view", vec![]);
        s.relationship_viewed(false).unwrap();
        assert!(last_event(&s).contains("not at your private opinion"));
        s.relationship_viewed(true).unwrap();
        assert!(last_event(&s).contains("also your private opinion"));
        std::fs::remove_dir_all(data).unwrap();
    }

    #[tokio::test]
    async fn reacts_to_persona_edits() {
        let (mut s, data) = session("persona", vec![turn("Why did you make me fiery?")]);
        s.profile.state.last_interaction = Some(Utc::now() - TimeDelta::minutes(1));
        let mut persona = s.profile.persona.clone();
        persona["personality"] = json!("fiery");
        storage::save(&s.profile.paths.persona(), &persona).unwrap();
        assert!(s.start().await.unwrap().is_some());
        assert_eq!(s.profile.persona["personality"], json!("fiery"));
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
            vec![
                turn("Oh no!"),
                consolidation,
                json!({"label": "new friends", "opinion": "Funny guy.", "wishes": "More dog stories."}),
                turn("Is Rex okay?"),
            ],
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
        assert_eq!(
            report.new_relationship_label.as_deref(),
            Some("new friends")
        );
        assert_eq!(s.profile.relationship.opinion, "Funny guy.");
        assert_eq!(s.profile.relationship_meta.history.len(), 2);

        let reply = s.user_says("Rex is sick today").await.unwrap();
        assert_eq!(reply.content, "Is Rex okay?");
        let requests = s.llm.requests.borrow();
        let instructions = &requests.last().unwrap().instructions;
        assert!(instructions.contains("Borys has a dog named Rex"));
        assert!(instructions.contains("dog Rex ate his shoes"));
        std::fs::remove_dir_all(data).unwrap();
    }
}
