//! Who wrote what: per-field authorship and edit detection for files that both
//! the companion and the user may change (persona, relationship).

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Author {
    Companion,
    User,
}

impl Author {
    pub fn describe(self) -> &'static str {
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

/// Authorship of every field plus a snapshot of the last known contents.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tracked {
    pub fields: BTreeMap<String, FieldMeta>,
    /// The contents as they were when last seen, to detect edits.
    pub snapshot: Map<String, Value>,
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

impl Tracked {
    pub fn new(contents: &Map<String, Value>, author: Author, now: DateTime<Utc>) -> Self {
        let mut tracked = Self::default();
        tracked.record(contents, author, now);
        tracked
    }

    pub fn author_of(&self, field: &str) -> Author {
        self.fields.get(field).map_or(Author::User, |m| m.author)
    }

    /// Differences between `current` and the snapshot.
    pub fn detect(&self, current: &Map<String, Value>) -> Vec<Change> {
        let mut changes = Vec::new();
        for (field, value) in current {
            match self.snapshot.get(field) {
                None => changes.push(Change::Added {
                    field: field.clone(),
                    value: value.clone(),
                }),
                Some(old) if old != value => changes.push(Change::Modified {
                    field: field.clone(),
                    old: old.clone(),
                    new: value.clone(),
                    author: self.author_of(field),
                }),
                Some(_) => {}
            }
        }
        for (field, old) in &self.snapshot {
            if !current.contains_key(field) {
                changes.push(Change::Removed {
                    field: field.clone(),
                    old: old.clone(),
                    author: self.author_of(field),
                });
            }
        }
        changes
    }

    /// Attributes every field that differs from the snapshot to `author`
    /// and takes a new snapshot. Returns the changes.
    pub fn record(
        &mut self,
        current: &Map<String, Value>,
        author: Author,
        now: DateTime<Utc>,
    ) -> Vec<Change> {
        let changes = self.detect(current);
        for change in &changes {
            match change {
                Change::Added { field, .. } | Change::Modified { field, .. } => {
                    self.fields.insert(
                        field.clone(),
                        FieldMeta {
                            author,
                            updated_at: now,
                        },
                    );
                }
                Change::Removed { field, .. } => {
                    self.fields.remove(field);
                }
            }
        }
        self.snapshot = current.clone();
        changes
    }
}

pub fn show(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One line per change, for telling the companion what the user edited.
pub fn describe_changes(changes: &[Change]) -> String {
    changes
        .iter()
        .map(|change| match change {
            Change::Added { field, value } => format!("- added \"{field}\": {}", show(value)),
            Change::Removed { field, old, author } => format!(
                "- removed \"{field}\" (was: {}; {})",
                show(old),
                author.describe()
            ),
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
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn detects_and_attributes_changes() {
        let now = Utc::now();
        let original = map(json!({"name": "Mira", "personality": "calm", "quirks": "hums"}));
        let mut tracked = Tracked::new(&original, Author::Companion, now);
        assert!(tracked.detect(&original).is_empty());

        let edited = map(json!({"name": "Mira", "personality": "fiery", "hobby": "chess"}));
        let changes = tracked.detect(&edited);
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
        assert!(describe_changes(&changes).contains("from \"calm\" (chosen by you) to \"fiery\""));

        assert_eq!(tracked.record(&edited, Author::User, now), changes);
        assert!(tracked.detect(&edited).is_empty());
        assert_eq!(tracked.author_of("personality"), Author::User);
        assert_eq!(tracked.author_of("name"), Author::Companion);
        assert!(!tracked.fields.contains_key("quirks"));
    }
}
