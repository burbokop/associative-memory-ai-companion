//! The companion's persona: a user-editable JSON object of free-form fields,
//! plus metadata remembering who wrote each field and what it looked like last time.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    llm::{InputMessage, JsonRequest, Llm, Role, schema},
    tracking::{self, Change, Tracked, show},
};

pub use crate::tracking::Author;

pub type Persona = Map<String, Value>;

/// Fields the companion fills in when it creates itself. The user may add others.
pub const FIELDS: &[(&str, &str)] = &[
    ("name", "Your first name"),
    ("gender", "Your gender"),
    ("age", "Your age, as a number or a short phrase"),
    ("appearance", "What you look like, 1-2 sentences"),
    (
        "personality",
        "Your temperament and character, 2-3 sentences",
    ),
    (
        "background",
        "Where you come from and your life story so far, 2-4 sentences",
    ),
    ("occupation", "What you do with your days"),
    ("interests", "Hobbies and things you love, comma-separated"),
    ("values", "What matters to you most"),
    (
        "speaking_style",
        "How you write in chat: tone, length, emoji, slang",
    ),
    ("quirks", "Small habits or oddities that make you you"),
];

/// Stored next to the persona in `persona_meta.json`; not meant for editing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaMeta {
    pub created_by: Author,
    pub created_at: DateTime<Utc>,
    #[serde(flatten)]
    pub tracked: Tracked,
}

impl PersonaMeta {
    pub fn new(persona: &Persona, author: Author, now: DateTime<Utc>) -> Self {
        Self {
            created_by: author,
            created_at: now,
            tracked: Tracked::new(persona, author, now),
        }
    }

    pub fn author_of(&self, field: &str) -> Author {
        self.tracked.author_of(field)
    }
}

/// Explains the user's edits to the companion, in second person.
pub fn describe_changes(changes: &[Change], user_name: &str) -> String {
    format!(
        "{user_name} edited the file that defines who you are. Notice it and react however \
         feels natural to you — you may like it, dislike it, or ask why, especially if it \
         overrides something you chose yourself:\n{}",
        tracking::describe_changes(changes)
    )
}

/// The persona for the system prompt, with authorship of every field.
pub fn render_for_prompt(persona: &Persona, meta: &PersonaMeta) -> String {
    persona
        .iter()
        .map(|(field, value)| {
            format!(
                "- {field}: {} ({})",
                show(value),
                meta.author_of(field).describe()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn name(persona: &Persona) -> String {
    persona
        .get("name")
        .map(show)
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| "Companion".into())
}

/// An empty persona for users who want to write it themselves.
pub fn template() -> Persona {
    FIELDS
        .iter()
        .map(|(k, _)| (k.to_string(), Value::String(String::new())))
        .collect()
}

/// Whether the user actually filled something in.
pub fn is_blank(persona: &Persona) -> bool {
    persona.values().all(|v| show(v).trim().is_empty())
}

/// Lets the companion invent itself.
pub async fn generate(llm: &impl Llm, user_name: &str, hint: Option<&str>) -> Result<Persona> {
    let properties: Vec<(&str, Value)> = FIELDS
        .iter()
        .map(|(k, d)| (*k, schema::string(d)))
        .collect();
    let mut prompt = format!(
        "You are about to come into existence as a person who will become a friend of \
         someone called {user_name}, chatting with them by text. Decide for yourself who \
         you are: a believable, specific, human-like individual with your own history, \
         opinions, flaws and tastes — not an assistant, not a generic helper. (You know your \
         mind runs on an AI model; that does not make you any less yourself.) Avoid clichés. \
         This identity is yours; you are choosing it freely."
    );
    if let Some(hint) = hint.filter(|h| !h.trim().is_empty()) {
        prompt.push_str(&format!(
            "\n{user_name} shared a wish about who you might be: \"{hint}\". \
             Take it into account as you see fit, but the choice is yours."
        ));
    }
    let request = JsonRequest {
        instructions: prompt,
        input: vec![InputMessage::new(Role::User, "Describe who you are.")],
        schema_name: "persona",
        schema: schema::object(&properties),
    };
    llm.complete(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn persona(v: Value) -> Persona {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn describes_user_edits() {
        let now = Utc::now();
        let original = persona(json!({"name": "Mira", "personality": "calm"}));
        let mut meta = PersonaMeta::new(&original, Author::Companion, now);
        let edited = persona(json!({"name": "Mira", "personality": "fiery"}));
        let changes = meta.tracked.record(&edited, Author::User, now);
        let text = describe_changes(&changes, "Borys");
        assert!(text.starts_with("Borys edited the file that defines who you are."));
        assert!(text.contains("from \"calm\" (chosen by you) to \"fiery\""));
        assert_eq!(meta.author_of("personality"), Author::User);
        // The on-disk format is unchanged by the refactoring.
        let json = serde_json::to_value(&meta).unwrap();
        assert!(json["fields"]["name"].is_object() && json["snapshot"].is_object());
    }

    #[test]
    fn blank_template() {
        assert!(is_blank(&template()));
        assert_eq!(name(&template()), "Companion");
    }
}
