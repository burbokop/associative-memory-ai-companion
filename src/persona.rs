//! The companion's persona: a user-editable JSON object of free-form fields,
//! plus metadata remembering who wrote each field and what it looked like last time.

use std::collections::BTreeMap;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llm::{InputMessage, JsonRequest, Llm, Role, schema};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Author {
    Companion,
    User,
}

impl Author {
    fn describe(self) -> &'static str {
        match self {
            Author::Companion => "chosen by you",
            Author::User => "set by the user",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMeta {
    pub author: Author,
    pub updated_at: DateTime<Utc>,
}

/// Stored next to the persona in `persona_meta.json`; not meant for editing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaMeta {
    pub created_by: Author,
    pub created_at: DateTime<Utc>,
    pub fields: BTreeMap<String, FieldMeta>,
    /// The persona as it was when last seen, to detect edits.
    pub snapshot: Persona,
}

impl PersonaMeta {
    pub fn new(persona: &Persona, author: Author, now: DateTime<Utc>) -> Self {
        Self {
            created_by: author,
            created_at: now,
            fields: persona
                .keys()
                .map(|k| {
                    (
                        k.clone(),
                        FieldMeta {
                            author,
                            updated_at: now,
                        },
                    )
                })
                .collect(),
            snapshot: persona.clone(),
        }
    }

    pub fn author_of(&self, field: &str) -> Author {
        self.fields.get(field).map_or(Author::User, |m| m.author)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Change {
    Added {
        field: String,
        value: Value,
    },
    Removed {
        field: String,
        old: Value,
        author: Author,
    },
    Modified {
        field: String,
        old: Value,
        new: Value,
        author: Author,
    },
}

/// Compares the persona file with the last snapshot.
pub fn detect_changes(current: &Persona, meta: &PersonaMeta) -> Vec<Change> {
    let mut changes = Vec::new();
    for (field, value) in current {
        match meta.snapshot.get(field) {
            None => changes.push(Change::Added {
                field: field.clone(),
                value: value.clone(),
            }),
            Some(old) if old != value => changes.push(Change::Modified {
                field: field.clone(),
                old: old.clone(),
                new: value.clone(),
                author: meta.author_of(field),
            }),
            Some(_) => {}
        }
    }
    for (field, old) in &meta.snapshot {
        if !current.contains_key(field) {
            changes.push(Change::Removed {
                field: field.clone(),
                old: old.clone(),
                author: meta.author_of(field),
            });
        }
    }
    changes
}

/// Records that the user made these changes and takes a new snapshot.
pub fn accept_user_changes(
    meta: &mut PersonaMeta,
    current: &Persona,
    changes: &[Change],
    now: DateTime<Utc>,
) {
    for change in changes {
        match change {
            Change::Added { field, .. } | Change::Modified { field, .. } => {
                meta.fields.insert(
                    field.clone(),
                    FieldMeta {
                        author: Author::User,
                        updated_at: now,
                    },
                );
            }
            Change::Removed { field, .. } => {
                meta.fields.remove(field);
            }
        }
    }
    meta.snapshot = current.clone();
}

fn show(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Explains the edits to the companion, in second person.
pub fn describe_changes(changes: &[Change], user_name: &str) -> String {
    let mut text = format!(
        "While you were apart, {user_name} edited the file that defines who you are. \
         Notice it and react however feels natural to you — you may like it, dislike it, \
         or ask why, especially if it overrides something you chose yourself:\n"
    );
    for change in changes {
        let line = match change {
            Change::Added { field, value } => format!("- added \"{field}\": {}", show(value)),
            Change::Removed { field, old, author } => {
                format!(
                    "- removed \"{field}\" (was: {}; {})",
                    show(old),
                    author.describe()
                )
            }
            Change::Modified {
                field,
                old,
                new,
                author,
            } => format!(
                "- changed \"{field}\" from \"{}\" ({}) to \"{}\"",
                show(old),
                author.describe(),
                show(new)
            ),
        };
        text.push_str(&line);
        text.push('\n');
    }
    text
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
    fn detects_and_attributes_changes() {
        let now = Utc::now();
        let original = persona(json!({"name": "Mira", "personality": "calm", "quirks": "hums"}));
        let mut meta = PersonaMeta::new(&original, Author::Companion, now);

        let edited = persona(json!({"name": "Mira", "personality": "fiery", "hobby": "chess"}));
        let changes = detect_changes(&edited, &meta);
        assert_eq!(changes.len(), 3);
        assert!(changes.contains(&Change::Modified {
            field: "personality".into(),
            old: json!("calm"),
            new: json!("fiery"),
            author: Author::Companion,
        }));
        assert!(changes.contains(&Change::Added {
            field: "hobby".into(),
            value: json!("chess")
        }));

        let text = describe_changes(&changes, "Borys");
        assert!(text.contains("from \"calm\" (chosen by you) to \"fiery\""));

        accept_user_changes(&mut meta, &edited, &changes, now);
        assert!(detect_changes(&edited, &meta).is_empty());
        assert_eq!(meta.author_of("personality"), Author::User);
        assert_eq!(meta.author_of("name"), Author::Companion);
        assert!(!meta.fields.contains_key("quirks"));
    }

    #[test]
    fn blank_template() {
        assert!(is_blank(&template()));
        assert_eq!(name(&template()), "Companion");
    }
}
