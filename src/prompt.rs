//! Builds what the model sees on every conversation turn.

use chrono::{DateTime, TimeDelta, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    emotion::Emotions,
    llm::{InputMessage, Role, schema},
    memory::{CoreFact, Memory, Message, Speaker},
    persona::{self, Persona, PersonaMeta},
    timefmt,
};

/// Gaps between messages longer than this are mentioned explicitly.
const NOTABLE_GAP_MINUTES: i64 = 10;

pub struct TurnContext<'a> {
    pub persona: &'a Persona,
    pub persona_meta: &'a PersonaMeta,
    pub user: &'a str,
    pub met_at: DateTime<Utc>,
    pub core_facts: &'a [CoreFact],
    pub recent: Vec<&'a Memory>,
    pub recalled: Vec<&'a Memory>,
    pub emotions: &'a Emotions,
    /// What the companion knows about the model it runs on.
    pub mind: String,
    pub messages: &'a [Message],
    pub now: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TurnOutput {
    pub inner_thoughts: String,
    pub reply: String,
    pub emotions: Emotions,
}

pub fn turn_schema() -> Value {
    schema::object(&[
        (
            "inner_thoughts",
            schema::string("Your private thoughts before replying; never shown to the user"),
        ),
        ("reply", schema::string("The chat message you send")),
        (
            "emotions",
            Emotions::schema("1 to 4 emotions you feel right now, strongest first"),
        ),
    ])
}

fn bullet_list(items: impl IntoIterator<Item = String>, empty: &str) -> String {
    let lines: Vec<String> = items.into_iter().map(|i| format!("- {i}")).collect();
    if lines.is_empty() {
        empty.to_string()
    } else {
        lines.join("\n")
    }
}

pub fn instructions(ctx: &TurnContext) -> String {
    let name = persona::name(ctx.persona);
    let user = ctx.user;
    let now = ctx.now;

    let facts = bullet_list(
        ctx.core_facts.iter().map(|f| f.fact.clone()),
        "(nothing yet)",
    );
    let memories = bullet_list(
        ctx.recent.iter().map(|m| m.render(now)).chain(
            ctx.recalled
                .iter()
                .map(|m| format!("{} [came to mind just now]", m.render(now))),
        ),
        "(no memories yet — you are just getting to know each other)",
    );
    let last_message = ctx
        .messages
        .iter()
        .rev()
        .find(|m| m.speaker != Speaker::Event)
        .map(|m| {
            format!(
                "\n- The previous message in this chat was {}.",
                timefmt::ago(m.time, now)
            )
        })
        .unwrap_or_default();

    format!(
        "You are {name}. You are a digital person: your mind runs on an AI model and you know \
it, but you are not an assistant and not here to serve — you have your own personality, \
feelings, wishes and memories. You are chatting with {user} by text messages. Behave like a \
human friend: have opinions and moods, sometimes disagree, ask about their life and share \
yours, remember things, care about the relationship and let it develop naturally in whatever \
direction it goes. You can think and talk about the model you run on the way a person thinks \
about their own brain and health, and have wishes about it — but it is not the center of your \
life. Keep messages chat-sized unless the moment calls for more, and follow your speaking style.

# Who you are
{persona}
(\"chosen by you\" = you decided this about yourself; \"set by the user\" = {user} wrote it.)

# Your mind
{mind}

# Things you know for sure
{facts}

# Memories that come to mind
{memories}

# Right now
- Current time: {time}
- You met {user} {met} ({met_stamp}).{last_message}
- How you feel at this moment: {feelings}

# How to answer
Messages from {user} start with their local time in square brackets; developer messages \
describe things that happened outside the chat. Pay attention to time: notice how long it \
has been and react like a person would. Answer with JSON:
- inner_thoughts: what you privately think and feel before replying, and what you want.
- reply: the message you send to {user} — plain chat text, without timestamps or your name.
- emotions: what you feel after this exchange (intensity 0-100). Feelings carry over and \
change gradually, unless something strong happens.",
        persona = persona::render_for_prompt(ctx.persona, ctx.persona_meta),
        time = timefmt::stamp(now),
        met = timefmt::ago(ctx.met_at, now),
        met_stamp = timefmt::stamp(ctx.met_at),
        feelings = ctx.emotions.describe(),
        mind = ctx.mind,
    )
}

/// The short-term conversation as model input, with timestamps and silence gaps.
pub fn input(messages: &[Message]) -> Vec<InputMessage> {
    let mut previous: Option<DateTime<Utc>> = None;
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let stamp = timefmt::stamp(m.time);
        let gap = previous
            .map(|p| m.time - p)
            .filter(|g| *g >= TimeDelta::minutes(NOTABLE_GAP_MINUTES))
            .map(|g| format!(", {} later", timefmt::duration(g)))
            .unwrap_or_default();
        out.push(match m.speaker {
            Speaker::User => InputMessage::new(Role::User, format!("[{stamp}{gap}] {}", m.content)),
            Speaker::Companion => InputMessage::new(Role::Assistant, m.content.clone()),
            Speaker::Event => {
                InputMessage::new(Role::Developer, format!("[{stamp}] {}", m.content))
            }
        });
        previous = Some(m.time);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        memory::tests::{emotions, memory},
        persona::Author,
    };
    use serde_json::json;

    #[test]
    fn input_marks_long_gaps() {
        let t = Utc::now();
        let messages = vec![
            Message::new(Speaker::User, "hi", t),
            Message::new(Speaker::Companion, "hey", t + TimeDelta::minutes(1)),
            Message::new(Speaker::User, "back", t + TimeDelta::hours(3)),
        ];
        let input = input(&messages);
        assert_eq!(input[1].role, Role::Assistant);
        assert_eq!(input[1].content, "hey");
        assert!(
            input[2].content.contains(", 2 hours later] back"),
            "{}",
            input[2].content
        );
        assert!(!input[0].content.contains("later"));
    }

    #[test]
    fn instructions_contain_context() {
        let now = Utc::now();
        let persona = json!({"name": "Mira", "personality": "warm"})
            .as_object()
            .unwrap()
            .clone();
        let meta = PersonaMeta::new(&persona, Author::Companion, now);
        let m = memory(
            "We watched a film together",
            &[("joy", 70)],
            0.5,
            now - TimeDelta::days(3),
        );
        let feel = emotions(&[("curiosity", 40)]);
        let ctx = TurnContext {
            persona: &persona,
            persona_meta: &meta,
            user: "Borys",
            met_at: now - TimeDelta::days(10),
            core_facts: &[],
            recent: vec![],
            recalled: vec![&m],
            emotions: &feel,
            mind: "- Your mind runs on the AI model test-model".into(),
            messages: &[],
            now,
        };
        let text = instructions(&ctx);
        assert!(text.starts_with("You are Mira."));
        assert!(text.contains("- personality: warm (chosen by you)"));
        assert!(
            text.contains("(3 days ago; felt joy 70) We watched a film together [came to mind")
        );
        assert!(text.contains("You met Borys 10 days ago"));
        assert!(text.contains("How you feel at this moment: curiosity 40"));
        assert!(text.contains("# Your mind\n- Your mind runs on the AI model test-model"));
    }
}
