use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Emotion {
    pub emotion: String,
    pub level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UserMessage {
    pub content: String,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssistantMessage {
    pub content: String,
    pub emotion: Emotion,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SystemMessage {
    pub content: String,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LongTermMemory {
    pub summary: String,
    pub emotion: Emotion,
    pub time: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub(crate) enum MemoryQuant {
    User(UserMessage),
    Assistant(AssistantMessage),
    System(SystemMessage),
    LongTermMemory(LongTermMemory),
}
