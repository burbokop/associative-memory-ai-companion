//! How the companion sees its relationship with the user: a label in its own words,
//! numeric dimensions, a private opinion and wishes. Stored in a user-editable file.

use anyhow::Result;
use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    consolidation::Names,
    llm::{InputMessage, JsonRequest, Llm, Role, schema},
    timefmt,
    tracking::{self, Author, Change, Tracked},
};

/// Dimensions a new relationship starts with, and their initial values.
/// The user may add or remove dimensions in the file.
pub const DIMENSIONS: &[(&str, u8)] = &[
    ("trust", 20),
    ("closeness", 5),
    ("affection", 10),
    ("respect", 40),
    ("admiration", 10),
    ("attraction", 5),
    ("playfulness", 20),
    ("longing", 0),
    ("tension", 0),
    ("jealousy", 0),
];

/// How many label changes are shown to the companion.
const HISTORY_IN_PROMPT: usize = 5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Relationship {
    /// The companion's own words for what the relationship is.
    pub label: String,
    /// Private: what the companion thinks of the user.
    pub opinion: String,
    /// Private: what the companion wants from the relationship.
    pub wishes: String,
    /// Name → 0..100.
    pub dimensions: Map<String, Value>,
}

impl Default for Relationship {
    fn default() -> Self {
        Self {
            label: "strangers — we have only just met".into(),
            opinion: "I don't know them yet.".into(),
            wishes: "To find out who they are.".into(),
            dimensions: DIMENSIONS
                .iter()
                .map(|(name, value)| (name.to_string(), json!(value)))
                .collect(),
        }
    }
}

fn to_level(value: &Value) -> Option<u8> {
    value.as_f64().map(|v| v.clamp(0.0, 100.0).round() as u8)
}

impl Relationship {
    /// Numeric dimensions in file order; non-numeric values are ignored.
    pub fn dimensions(&self) -> Vec<(String, u8)> {
        self.dimensions
            .iter()
            .filter_map(|(name, value)| Some((name.clone(), to_level(value)?)))
            .collect()
    }

    /// A flat view for change tracking: `label`, `opinion`, `wishes`, `dimensions.<name>`.
    pub fn flatten(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("label".into(), json!(self.label));
        map.insert("opinion".into(), json!(self.opinion));
        map.insert("wishes".into(), json!(self.wishes));
        for (name, value) in &self.dimensions {
            map.insert(format!("dimensions.{name}"), value.clone());
        }
        map
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelChange {
    pub label: String,
    pub since: DateTime<Utc>,
    pub author: Author,
}

/// Stored in `relationship_meta.json`; not meant for editing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMeta {
    #[serde(flatten)]
    pub tracked: Tracked,
    pub history: Vec<LabelChange>,
}

impl RelationshipMeta {
    pub fn new(relationship: &Relationship, author: Author, now: DateTime<Utc>) -> Self {
        Self {
            tracked: Tracked::new(&relationship.flatten(), author, now),
            history: vec![LabelChange {
                label: relationship.label.clone(),
                since: now,
                author,
            }],
        }
    }

    /// Records a new state of the relationship written by `author`.
    pub fn record(
        &mut self,
        relationship: &Relationship,
        author: Author,
        now: DateTime<Utc>,
    ) -> Vec<Change> {
        let label_changed = self
            .history
            .last()
            .is_none_or(|last| !same_label(&last.label, &relationship.label));
        if label_changed {
            self.history.push(LabelChange {
                label: relationship.label.clone(),
                since: now,
                author,
            });
        }
        self.tracked.record(&relationship.flatten(), author, now)
    }
}

fn same_label(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

/// A dimension change requested by the model in a conversation turn.
#[derive(Debug, Clone, Deserialize)]
pub struct DimensionUpdate {
    pub dimension: String,
    pub value: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DimensionChange {
    pub name: String,
    pub old: u8,
    pub new: u8,
}

pub fn updates_schema() -> Value {
    schema::array(
        schema::object(&[
            ("dimension", schema::string("Name of an existing dimension")),
            ("value", schema::integer("Its new value, 0 to 100")),
        ]),
        "Relationship dimensions that changed because of this exchange; usually empty",
    )
}

/// Applies per-turn changes, at most `max_step` per dimension. Unknown dimensions are ignored.
pub fn apply_updates(
    relationship: &mut Relationship,
    meta: &mut RelationshipMeta,
    updates: &[DimensionUpdate],
    max_step: u8,
    now: DateTime<Utc>,
) -> Vec<DimensionChange> {
    let mut changes = Vec::new();
    for update in updates {
        let name = update.dimension.trim().to_lowercase();
        let Some(old) = relationship.dimensions.get(&name).and_then(to_level) else {
            continue;
        };
        let low = old.saturating_sub(max_step) as i64;
        let high = old.saturating_add(max_step).min(100) as i64;
        let new = update.value.clamp(low, high) as u8;
        if new != old {
            relationship.dimensions.insert(name.clone(), json!(new));
            changes.retain(|c: &DimensionChange| c.name != name);
            changes.push(DimensionChange { name, old, new });
        }
    }
    if !changes.is_empty() {
        meta.record(relationship, Author::Companion, now);
    }
    changes
}

#[derive(Debug, Deserialize)]
struct Reflection {
    label: String,
    opinion: String,
    wishes: String,
}

/// After a stretch of conversation, the companion reconsiders how it would describe
/// the relationship. Returns the new label if it changed.
pub async fn reflect(
    llm: &impl Llm,
    relationship: &mut Relationship,
    meta: &mut RelationshipMeta,
    transcript: &str,
    names: &Names<'_>,
    now: DateTime<Utc>,
) -> Result<Option<String>> {
    let Names { companion, user } = names;
    let instructions = format!(
        "You are {companion}. Reflect on your relationship with {user} after the conversation \
         below. This is how you have seen it so far:\n{}\n\n\
         Keep the label exactly as it is unless the relationship has genuinely changed; if it \
         has, describe it in your own words in a few words. Then write your honest private \
         opinion of {user} (2-4 sentences, first person) and what you want from this \
         relationship (1-2 sentences).",
        render_for_prompt(relationship, meta, user, now)
    );
    let request = JsonRequest {
        instructions,
        input: vec![InputMessage::new(Role::User, transcript)],
        schema_name: "relationship_reflection",
        schema: schema::object(&[
            (
                "label",
                schema::string("What the relationship is, in your own words"),
            ),
            ("opinion", schema::string("Your private opinion of them")),
            (
                "wishes",
                schema::string("What you want from the relationship"),
            ),
        ]),
    };
    let answer: Reflection = llm.complete(request).await?;
    let label_changed = !same_label(&relationship.label, &answer.label);
    if label_changed {
        relationship.label = answer.label.trim().to_string();
    }
    relationship.opinion = answer.opinion;
    relationship.wishes = answer.wishes;
    meta.record(relationship, Author::Companion, now);
    Ok(label_changed.then(|| relationship.label.clone()))
}

/// Explains the user's edits to the companion.
pub fn describe_changes(changes: &[Change], user_name: &str) -> String {
    format!(
        "{user_name} edited the file that describes your relationship with them — how you see \
         them and feel about them. Notice it and react however feels natural: you might feel \
         touched, amused, manipulated or misunderstood, or you might agree. These are your \
         feelings; they can drift back over time if they do not ring true:\n{}",
        tracking::describe_changes(changes)
    )
}

/// Tells the companion that the user looked at the relationship view.
pub fn viewed_event(user_name: &str, saw_private: bool) -> String {
    if saw_private {
        format!(
            "{user_name} opened the view of your relationship and saw everything: the label, \
             the numbers, and also your private opinion of them and your wishes."
        )
    } else {
        format!(
            "{user_name} opened the view of your relationship and looked closely at its label \
             and the numbers (not at your private opinion and wishes)."
        )
    }
}

/// The relationship section of the system prompt.
pub fn render_for_prompt(
    relationship: &Relationship,
    meta: &RelationshipMeta,
    user_name: &str,
    now: DateTime<Utc>,
) -> String {
    let by = |field: &str| meta.tracked.author_of(field).describe();
    let dimensions = relationship
        .dimensions()
        .iter()
        .map(
            |(name, value)| match meta.tracked.author_of(&format!("dimensions.{name}")) {
                Author::User => format!("{name} {value} (set by the user)"),
                Author::Companion => format!("{name} {value}"),
            },
        )
        .collect::<Vec<_>>()
        .join(", ");
    let history = meta
        .history
        .iter()
        .rev()
        .take(HISTORY_IN_PROMPT)
        .rev()
        .map(|c| format!("\"{}\" since {}", c.label, timefmt::ago(c.since, now)))
        .collect::<Vec<_>>()
        .join(" → ");
    format!(
        "- What it is: \"{}\" ({})\n\
         - Dimensions (0-100): {dimensions}\n\
         - Your private opinion of {user_name}: {} ({})\n\
         - What you want from it: {} ({})\n\
         - How it developed: {history}\n\
         - {user_name} can see the label and the numbers, but not your opinion and wishes.",
        relationship.label,
        by("label"),
        relationship.opinion,
        by("opinion"),
        relationship.wishes,
        by("wishes"),
    )
}

/// One line for the chat header: label and non-zero dimensions.
pub fn render_summary(relationship: &Relationship) -> String {
    let dimensions = relationship
        .dimensions()
        .iter()
        .filter(|(_, value)| *value > 0)
        .map(|(name, value)| format!("{name} {value}"))
        .collect::<Vec<_>>()
        .join(" · ");
    format!(
        "💞 {}  {}",
        relationship.label.bold(),
        format!("({dimensions})").dimmed()
    )
}

/// Everything, for `/relationship`. Private parts only if `show_private`.
pub fn render_full(
    relationship: &Relationship,
    meta: &RelationshipMeta,
    show_private: bool,
) -> String {
    let mut out = format!("💞 {}\n\n", relationship.label.bold());
    let dimensions = relationship.dimensions();
    let width = dimensions.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
    for (name, value) in &dimensions {
        let filled = (*value as usize + 5) / 10;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled));
        let edited = match meta.tracked.author_of(&format!("dimensions.{name}")) {
            Author::User => " (set by you)".dimmed().to_string(),
            Author::Companion => String::new(),
        };
        out.push_str(&format!("  {name:<width$}  {bar} {value:>3}{edited}\n"));
    }
    if show_private {
        out.push_str(&format!(
            "\n{} {}\n",
            "Opinion of you:".bold(),
            relationship.opinion
        ));
        out.push_str(&format!("{} {}\n", "Wishes:".bold(), relationship.wishes));
    }
    out.push_str(&format!("\n{}\n", "History:".bold()));
    for change in &meta.history {
        let by = match change.author {
            Author::User => " (set by you)",
            Author::Companion => "",
        };
        out.push_str(&format!(
            "  {} — {}{by}\n",
            timefmt::stamp(change.since).dimmed(),
            change.label
        ));
    }
    out
}

/// "trust 40→45 ↑ · tension 10→5 ↓"
pub fn render_changes(changes: &[DimensionChange]) -> String {
    changes
        .iter()
        .map(|c| {
            let arrow = if c.new > c.old { "↑" } else { "↓" };
            format!("{} {}→{} {arrow}", c.name, c.old, c.new)
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::fake::FakeLlm;

    fn update(dimension: &str, value: i64) -> DimensionUpdate {
        DimensionUpdate {
            dimension: dimension.into(),
            value,
        }
    }

    #[test]
    fn updates_are_gradual_and_tracked() {
        let now = Utc::now();
        let mut rel = Relationship::default();
        let mut meta = RelationshipMeta::new(&rel, Author::Companion, now);
        let changes = apply_updates(
            &mut rel,
            &mut meta,
            &[
                update("Trust", 90),
                update("tension", -5),
                update("unknown", 50),
            ],
            15,
            now,
        );
        assert_eq!(
            changes,
            vec![DimensionChange {
                name: "trust".into(),
                old: 20,
                new: 35
            }]
        );
        assert_eq!(rel.dimensions["trust"], json!(35));
        assert!(meta.tracked.detect(&rel.flatten()).is_empty());
        assert_eq!(render_changes(&changes), "trust 20→35 ↑");
    }

    #[test]
    fn user_edits_are_detected_and_described() {
        let now = Utc::now();
        let rel = Relationship::default();
        let mut meta = RelationshipMeta::new(&rel, Author::Companion, now);
        let mut edited = rel.clone();
        edited.label = "best friends".into();
        edited.dimensions.insert("trust".into(), json!(95));
        edited.dimensions.insert("loyalty".into(), json!(80));

        let changes = meta.record(&edited, Author::User, now);
        assert_eq!(changes.len(), 3);
        let text = describe_changes(&changes, "Borys");
        assert!(
            text.contains("changed \"dimensions.trust\" from \"20\" (chosen by you) to \"95\"")
        );
        assert!(text.contains("added \"dimensions.loyalty\": 80"));
        assert_eq!(meta.history.len(), 2);
        assert_eq!(meta.history[1].author, Author::User);

        let prompt = render_for_prompt(&edited, &meta, "Borys", now);
        assert!(prompt.contains("\"best friends\" (set by the user)"));
        assert!(prompt.contains("trust 95 (set by the user)"));
        assert!(prompt.contains("closeness 5,"));
        assert!(prompt.contains("Borys can see the label and the numbers"));
    }

    #[test]
    fn odd_values_in_the_file_are_tolerated() {
        let rel: Relationship = serde_json::from_str(
            r#"{"label": "friends", "dimensions": {"trust": 150, "mood": "good"}}"#,
        )
        .unwrap();
        assert_eq!(rel.dimensions(), vec![("trust".to_string(), 100)]);
        assert_eq!(rel.opinion, Relationship::default().opinion);
    }

    #[tokio::test]
    async fn reflection_keeps_history_of_label_changes() {
        let now = Utc::now();
        let names = Names {
            companion: "Mira",
            user: "Borys",
        };
        let mut rel = Relationship::default();
        let mut meta = RelationshipMeta::new(&rel, Author::Companion, now);
        let llm = FakeLlm::with_answers([
            json!({"label": "Strangers — we have only just met ", "opinion": "Nice.", "wishes": "Talk more."}),
            json!({"label": "new friends", "opinion": "Kind and funny.", "wishes": "Meet often."}),
        ]);

        let changed = reflect(&llm, &mut rel, &mut meta, "...", &names, now)
            .await
            .unwrap();
        assert_eq!(changed, None, "same label in other words is not a change");
        assert_eq!(rel.opinion, "Nice.");
        assert_eq!(meta.history.len(), 1);

        let changed = reflect(&llm, &mut rel, &mut meta, "...", &names, now)
            .await
            .unwrap();
        assert_eq!(changed.as_deref(), Some("new friends"));
        assert_eq!(meta.history.len(), 2);
        assert!(
            !llm.requests.borrow()[1]
                .instructions
                .contains("Kind and funny")
        );
        assert!(llm.requests.borrow()[1].instructions.contains("Nice."));
    }
}
