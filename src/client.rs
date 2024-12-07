use std::{fmt::Display, str::FromStr as _};

use chrono::Utc;
use openai_api_rust::{
    chat::{ChatApi as _, ChatBody},
    OpenAI, Role,
};

use crate::types::{AssistantMessage, Emotion, MemoryQuant};

static MODEL: &str = "gpt-3.5-turbo";

/// Warning: value is converted with loss of data
fn memory_quant_to_openai_message(m: MemoryQuant) -> openai_api_rust::Message {
    match m {
        MemoryQuant::User(message) => openai_api_rust::Message {
            role: Role::User,
            content: message.content,
        },
        MemoryQuant::Assistant(message) => openai_api_rust::Message {
            role: Role::Assistant,
            content: message.content,
        },
        MemoryQuant::System(message) => openai_api_rust::Message {
            role: Role::System,
            content: message.content,
        },
        MemoryQuant::LongTermMemory(mem) => openai_api_rust::Message {
            role: Role::System,
            content: mem.summary,
        },
    }
}

fn get_emotion(openai: &OpenAI, messages: Vec<MemoryQuant>, max_tokens: i32) -> Emotion {
    let mut messages: Vec<openai_api_rust::Message> = messages
        .into_iter()
        .map(memory_quant_to_openai_message)
        .collect();
    messages.push(openai_api_rust::Message {
        role: openai_api_rust::Role::System,
        content: "Tell in one word your emotion and level from 0 to 100 (Example: \"Neutral, 4\")"
            .to_string(),
    });

    let body = ChatBody {
        model: MODEL.to_string(),
        max_tokens: Some(max_tokens),
        temperature: Some(0_f32),
        top_p: Some(0_f32),
        n: Some(1),
        stream: Some(false),
        stop: None,
        presence_penalty: Some(2.),
        frequency_penalty: Some(2.),
        logit_bias: None,
        user: None,
        messages,
    };
    let completion = openai.chat_completion_create(&body).unwrap();
    let choise = &completion.choices[0];
    let answer = choise.message.as_ref().unwrap().clone();

    let mut parts = answer.content.split(",").map(str::trim);
    let emotion = String::from_str(parts.next().unwrap()).unwrap();
    let level = parts.next().unwrap().parse::<f32>().unwrap() / 100.;
    Emotion { emotion, level }
}

#[derive(Debug)]
pub(crate) struct TokensUsage(u32);

impl Display for TokensUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.to_string())
    }
}

pub(crate) fn get_answer(
    openai: &OpenAI,
    messages: Vec<MemoryQuant>,
    max_tokens: i32,
) -> (AssistantMessage, TokensUsage) {
    let body = ChatBody {
        model: MODEL.to_string(),
        max_tokens: Some(max_tokens),
        temperature: Some(0_f32),
        top_p: Some(0_f32),
        n: Some(1),
        stream: Some(false),
        stop: None,
        presence_penalty: Some(2.),
        frequency_penalty: Some(2.),
        logit_bias: None,
        user: None,
        messages: messages
            .clone()
            .into_iter()
            .map(memory_quant_to_openai_message)
            .collect(),
    };
    let completion = openai.chat_completion_create(&body).unwrap();
    let choise = &completion.choices[0];
    let answer = choise.message.as_ref().unwrap().clone();
    let emotion = get_emotion(&openai, messages, 100);

    if choise.finish_reason != Some("stop".to_string()) {
        println!(
            "world: warning: finish reason is {:?}",
            choise.finish_reason
        );
    }

    (
        AssistantMessage {
            content: answer.content,
            emotion,
            time: Utc::now(),
        },
        TokensUsage(completion.usage.total_tokens.unwrap()),
    )
}
