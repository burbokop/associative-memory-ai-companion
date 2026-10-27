//! A mix of emotions felt at once, each with an intensity 0–100.

use chrono::TimeDelta;
use colored::{ColoredString, Colorize};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm::schema;

/// Emotions weaker than this are considered gone.
const FADED: u8 = 5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Emotion {
    pub name: String,
    pub intensity: u8,
}

/// Sorted from strongest to weakest, names lowercase and unique.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "Vec<RawEmotion>", into = "Vec<Emotion>")]
pub struct Emotions(Vec<Emotion>);

/// Emotion as produced by the model, before clamping and normalisation.
#[derive(Debug, Clone, Deserialize)]
pub struct RawEmotion {
    pub name: String,
    pub intensity: i64,
}

impl From<Vec<RawEmotion>> for Emotions {
    fn from(raw: Vec<RawEmotion>) -> Self {
        let mut list: Vec<Emotion> = Vec::new();
        for e in raw {
            let name = e.name.trim().to_lowercase();
            let intensity = e.intensity.clamp(0, 100) as u8;
            if name.is_empty() || intensity < FADED {
                continue;
            }
            match list.iter_mut().find(|x| x.name == name) {
                Some(existing) => existing.intensity = existing.intensity.max(intensity),
                None => list.push(Emotion { name, intensity }),
            }
        }
        list.sort_by_key(|e| std::cmp::Reverse(e.intensity));
        Self(list)
    }
}

impl From<Emotions> for Vec<Emotion> {
    fn from(e: Emotions) -> Self {
        e.0
    }
}

impl Emotions {
    pub fn iter(&self) -> impl Iterator<Item = &Emotion> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The strongest intensity as 0..1 — how emotionally charged this is.
    pub fn peak(&self) -> f32 {
        self.0.first().map_or(0.0, |e| e.intensity as f32 / 100.0)
    }

    /// Emotions fade exponentially while time passes.
    pub fn decayed(&self, elapsed: TimeDelta, half_life_hours: f64) -> Self {
        let hours = elapsed.num_seconds().max(0) as f64 / 3600.0;
        let factor = 0.5f64.powf(hours / half_life_hours);
        Self(
            self.0
                .iter()
                .map(|e| Emotion {
                    name: e.name.clone(),
                    intensity: (e.intensity as f64 * factor).round() as u8,
                })
                .filter(|e| e.intensity >= FADED)
                .collect(),
        )
    }

    /// "joy 60, anxiety 20" for prompts.
    pub fn describe(&self) -> String {
        if self.0.is_empty() {
            return "calm, nothing in particular".into();
        }
        self.0
            .iter()
            .map(|e| format!("{} {}", e.name, e.intensity))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Coloured, emoji-decorated version for the console.
    pub fn render(&self) -> String {
        if self.0.is_empty() {
            return "😌 calm".dimmed().to_string();
        }
        self.0
            .iter()
            .map(|e| render_one(e).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// JSON schema for a list of emotions in model answers.
    pub fn schema(description: &str) -> Value {
        schema::array(
            schema::object(&[
                (
                    "name",
                    schema::string("A single lowercase word, e.g. joy, sadness, curiosity"),
                ),
                ("intensity", schema::integer("0 to 100")),
            ]),
            description,
        )
    }
}

#[derive(Clone, Copy)]
enum Tone {
    Positive,
    Negative,
    Agitated,
    Neutral,
}

fn classify(name: &str) -> (&'static str, Tone) {
    const TABLE: &[(&[&str], &str, Tone)] = &[
        (
            &["joy", "happ", "delight", "cheer", "glad"],
            "😊",
            Tone::Positive,
        ),
        (
            &["excite", "enthusias", "thrill", "eager"],
            "🤩",
            Tone::Positive,
        ),
        (
            &["love", "affection", "tender", "fond", "adore"],
            "🥰",
            Tone::Positive,
        ),
        (&["amuse", "playful", "fun"], "😄", Tone::Positive),
        (&["grat", "thank", "apprecia"], "🙏", Tone::Positive),
        (
            &["content", "relax", "peace", "calm", "seren", "relief"],
            "😌",
            Tone::Positive,
        ),
        (&["hope", "optimis"], "🌱", Tone::Positive),
        (&["pride", "proud", "confiden"], "😎", Tone::Positive),
        (
            &["curio", "interest", "intrigu", "wonder"],
            "🤔",
            Tone::Neutral,
        ),
        (&["surpris", "astonish", "amaze"], "😮", Tone::Neutral),
        (&["nostalg", "wistful"], "🥲", Tone::Neutral),
        (
            &["sad", "sorrow", "grief", "melanchol", "disappoint"],
            "😢",
            Tone::Negative,
        ),
        (&["lonel", "miss", "abandon"], "🥺", Tone::Negative),
        (&["hurt", "betray", "jealous"], "💔", Tone::Negative),
        (
            &["fear", "anxi", "worr", "nervous", "scare", "afraid"],
            "😟",
            Tone::Agitated,
        ),
        (
            &[
                "anger", "angry", "annoy", "irritat", "frustrat", "furious", "resent",
            ],
            "😠",
            Tone::Agitated,
        ),
        (
            &["embarrass", "shy", "awkward", "shame"],
            "😳",
            Tone::Agitated,
        ),
        (&["confus", "puzzl", "uncertain"], "😕", Tone::Neutral),
        (&["bored", "tired", "indifferen"], "😐", Tone::Neutral),
    ];
    TABLE
        .iter()
        .find(|(keys, _, _)| keys.iter().any(|k| name.contains(k)))
        .map_or(("💭", Tone::Neutral), |(_, emoji, tone)| (*emoji, *tone))
}

fn render_one(e: &Emotion) -> ColoredString {
    let (emoji, tone) = classify(&e.name);
    let text = format!("{emoji} {} {}%", e.name, e.intensity);
    match tone {
        Tone::Positive => text.green(),
        Tone::Negative => text.blue(),
        Tone::Agitated => text.red(),
        Tone::Neutral => text.yellow(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(list: &[(&str, i64)]) -> Emotions {
        list.iter()
            .map(|(n, i)| RawEmotion {
                name: n.to_string(),
                intensity: *i,
            })
            .collect::<Vec<_>>()
            .into()
    }

    #[test]
    fn normalises_model_output() {
        let e = raw(&[
            ("Joy", 40),
            ("anxiety", 150),
            (" joy ", 60),
            ("boredom", 2),
            ("", 50),
        ]);
        assert_eq!(e.describe(), "anxiety 100, joy 60");
        assert_eq!(e.peak(), 1.0);
    }

    #[test]
    fn decays_over_time() {
        let e = raw(&[("joy", 80), ("worry", 8)]);
        let later = e.decayed(TimeDelta::hours(12), 12.0);
        assert_eq!(later.describe(), "joy 40");
        assert!(e.decayed(TimeDelta::days(10), 12.0).is_empty());
    }

    #[test]
    fn serde_roundtrip_normalises() {
        let e: Emotions = serde_json::from_str(r#"[{"name":"Sad","intensity":30}]"#).unwrap();
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"[{"name":"sad","intensity":30}]"#
        );
    }
}
