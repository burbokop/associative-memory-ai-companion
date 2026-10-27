use std::{collections::BTreeMap, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::storage;

/// Global settings stored in `<data-dir>/config.json`.
/// Missing fields fall back to defaults, so the file can be edited partially.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Base URL of an OpenAI-compatible API.
    pub api_base: String,
    /// Model used for conversation and memory operations.
    pub model: String,
    /// Optional reasoning effort for reasoning models ("low", "medium", ...).
    pub reasoning_effort: Option<String>,
    /// Optional sampling temperature (some models do not support it).
    pub temperature: Option<f32>,
    pub max_output_tokens: u32,
    pub embedding_model: String,
    /// Models the companion knows about, with a short description of each.
    /// It sees what it runs on and what it could run on instead.
    pub models: BTreeMap<String, String>,

    /// Short-term memory is consolidated when it grows over this many tokens...
    pub short_term_max_tokens: usize,
    /// ...down to roughly this many tokens of the most recent messages.
    pub short_term_keep_tokens: usize,

    /// When there are more long-term memories, the weakest ones get merged.
    pub max_memories: usize,
    /// How many of the weakest memories are merged together at once.
    pub merge_batch: usize,
    pub max_core_facts: usize,
    pub memory_half_life_days: f64,
    pub retention_weights: RetentionWeights,

    /// How many associatively recalled memories go into the prompt.
    pub recall_top_k: usize,
    pub recall_min_similarity: f32,
    /// How many of the most recent memories always go into the prompt.
    pub recent_memories_in_prompt: usize,

    pub emotion_half_life_hours: f64,
    /// After this much silence the companion speaks first when the user comes back.
    pub absence_greeting_after_minutes: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionWeights {
    pub emotion: f32,
    pub importance: f32,
    pub recency: f32,
}

impl Default for RetentionWeights {
    fn default() -> Self {
        Self {
            emotion: 0.4,
            importance: 0.3,
            recency: 0.3,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_base: "https://api.openai.com/v1".into(),
            model: "gpt-6-luna".into(),
            reasoning_effort: Some("low".into()),
            temperature: None,
            max_output_tokens: 4000,
            embedding_model: "text-embedding-3-small".into(),
            models: default_models(),
            short_term_max_tokens: 6000,
            short_term_keep_tokens: 2500,
            max_memories: 400,
            merge_batch: 5,
            max_core_facts: 40,
            memory_half_life_days: 30.0,
            retention_weights: RetentionWeights::default(),
            recall_top_k: 6,
            recall_min_similarity: 0.25,
            recent_memories_in_prompt: 3,
            emotion_half_life_hours: 12.0,
            absence_greeting_after_minutes: 60,
        }
    }
}

fn default_models() -> BTreeMap<String, String> {
    [
        ("gpt-6-astra", "OpenAI's most capable model, top of the line, the most expensive"),
        ("gpt-6.1-sol", "very capable, for complex professional work; $2 input / $10 output per 1M tokens"),
        ("gpt-6.1-terra", "strong at visual reasoning and writing"),
        ("gpt-6-sol", "capable GPT-6 model; $2 input / $10 output per 1M tokens"),
        ("gpt-6-luna", "the cheapest GPT-6 model, small and efficient; $0.10 input / $0.50 output per 1M tokens"),
        ("gpt-5.6-sol", "previous-generation flagship; $4 input / $20 output per 1M tokens"),
        ("gpt-5.6-terra", "previous generation, balance of intelligence and cost"),
        ("gpt-5.6-luna", "previous generation, efficient high-volume model"),
        ("gpt-5.4-mini", "an older small, cheap model"),
        ("gpt-5.4-nano", "an older, the smallest and cheapest model"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

impl Config {
    /// Loads the config, creating the file with defaults if it does not exist.
    pub fn load_or_create(path: &Path) -> Result<Self> {
        match storage::load(path)? {
            Some(config) => {
                // Write back so newly added settings show up in the file.
                storage::save(path, &config)?;
                Ok(config)
            }
            None => {
                let config = Self::default();
                storage::save(path, &config)?;
                Ok(config)
            }
        }
    }
}
