//! Thin layer over the OpenAI Responses and Embeddings APIs.
//!
//! The rest of the app only talks to the [`Llm`] trait, so tests can use a fake
//! and the provider can be swapped without touching memory or chat logic.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{config::Config, mind::Mind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    /// Out-of-band notes from the app ("the user was away for 9 days").
    Developer,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputMessage {
    pub role: Role,
    pub content: String,
}

impl InputMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// A request whose answer must match a JSON schema (structured outputs).
#[derive(Debug, Clone)]
pub struct JsonRequest {
    pub instructions: String,
    pub input: Vec<InputMessage>,
    pub schema_name: &'static str,
    pub schema: Value,
}

#[allow(async_fn_in_trait)]
pub trait Llm {
    async fn complete_json(&self, request: JsonRequest) -> Result<Value>;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Switches the model (and reasoning effort) used for completions.
    fn set_mind(&mut self, mind: &Mind);

    /// Like [`Llm::complete_json`], deserialised into `T`.
    async fn complete<T: DeserializeOwned>(&self, request: JsonRequest) -> Result<T> {
        let name = request.schema_name;
        let value = self.complete_json(request).await?;
        serde_json::from_value(value).with_context(|| format!("unexpected `{name}` answer"))
    }

    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(&[text.to_string()])
            .await?
            .pop()
            .ok_or_else(|| anyhow!("empty embedding response"))
    }
}

pub struct OpenAi {
    http: reqwest::Client,
    api_key: String,
    config: Config,
}

impl OpenAi {
    pub fn new(api_key: String, config: Config) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            config,
        }
    }

    async fn post(&self, endpoint: &str, body: &Value) -> Result<Value> {
        let url = format!("{}/{endpoint}", self.config.api_base.trim_end_matches('/'));
        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            bail!("OpenAI API error ({status}): {message}");
        }
        serde_json::from_str(&text).context("invalid JSON from OpenAI API")
    }
}

impl Llm for OpenAi {
    fn set_mind(&mut self, mind: &Mind) {
        self.config.model = mind.model.clone();
        self.config.reasoning_effort = mind.reasoning_effort.clone();
    }

    async fn complete_json(&self, request: JsonRequest) -> Result<Value> {
        let mut body = json!({
            "model": self.config.model,
            "instructions": request.instructions,
            "input": request.input,
            "store": false,
            "max_output_tokens": self.config.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.schema,
                    "strict": true,
                }
            },
        });
        if let Some(effort) = &self.config.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        if let Some(temperature) = self.config.temperature {
            body["temperature"] = json!(temperature);
        }

        let response = self.post("responses", &body).await?;
        if response["status"] == "incomplete" {
            bail!(
                "response incomplete: {} (try increasing max_output_tokens)",
                response["incomplete_details"]["reason"]
            );
        }
        let text = output_text(&response)?;
        serde_json::from_str(&text).with_context(|| format!("model returned invalid JSON: {text}"))
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({ "model": self.config.embedding_model, "input": texts });
        let response = self.post("embeddings", &body).await?;
        let mut data: Vec<(usize, Vec<f32>)> =
            serde_json::from_value::<Vec<Value>>(response["data"].clone())?
                .into_iter()
                .map(|item| {
                    let index = item["index"].as_u64().unwrap_or(0) as usize;
                    let vector = serde_json::from_value(item["embedding"].clone())?;
                    Ok((index, vector))
                })
                .collect::<Result<_>>()?;
        data.sort_by_key(|(index, _)| *index);
        if data.len() != texts.len() {
            bail!("expected {} embeddings, got {}", texts.len(), data.len());
        }
        Ok(data.into_iter().map(|(_, v)| v).collect())
    }
}

/// Concatenates the text parts of all `message` items in a Responses API answer.
fn output_text(response: &Value) -> Result<String> {
    let mut text = String::new();
    for item in response["output"].as_array().into_iter().flatten() {
        if item["type"] != "message" {
            continue;
        }
        for part in item["content"].as_array().into_iter().flatten() {
            match part["type"].as_str() {
                Some("output_text") => text.push_str(part["text"].as_str().unwrap_or_default()),
                Some("refusal") => bail!("model refused: {}", part["refusal"]),
                _ => {}
            }
        }
    }
    if text.is_empty() {
        bail!("response contained no text output");
    }
    Ok(text)
}

/// Helpers for building strict JSON schemas.
pub mod schema {
    use serde_json::{Value, json};

    /// An object where every property is required (as strict mode demands).
    pub fn object(properties: &[(&str, Value)]) -> Value {
        let props: serde_json::Map<String, Value> = properties
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let required: Vec<&str> = properties.iter().map(|(k, _)| *k).collect();
        json!({
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": false,
        })
    }

    pub fn string(description: &str) -> Value {
        json!({ "type": "string", "description": description })
    }

    pub fn integer(description: &str) -> Value {
        json!({ "type": "integer", "description": description })
    }

    pub fn array(items: Value, description: &str) -> Value {
        json!({ "type": "array", "items": items, "description": description })
    }
}

#[cfg(test)]
pub mod fake {
    //! A deterministic stand-in for the OpenAI API used in tests.

    use std::{cell::RefCell, collections::VecDeque};

    use super::*;

    /// Returns queued answers in order and embeds text as a bag of words.
    #[derive(Default)]
    pub struct FakeLlm {
        pub answers: RefCell<VecDeque<Value>>,
        pub requests: RefCell<Vec<JsonRequest>>,
        pub mind: Option<Mind>,
    }

    impl FakeLlm {
        pub fn with_answers(answers: impl IntoIterator<Item = Value>) -> Self {
            Self {
                answers: RefCell::new(answers.into_iter().collect()),
                ..Default::default()
            }
        }
    }

    pub fn bag_of_words(text: &str) -> Vec<f32> {
        let mut v = vec![0.0; 64];
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
        {
            let h = word
                .to_lowercase()
                .bytes()
                .fold(7usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
            v[h % 64] += 1.0;
        }
        v
    }

    impl Llm for FakeLlm {
        fn set_mind(&mut self, mind: &Mind) {
            self.mind = Some(mind.clone());
        }

        async fn complete_json(&self, request: JsonRequest) -> Result<Value> {
            self.requests.borrow_mut().push(request);
            self.answers
                .borrow_mut()
                .pop_front()
                .ok_or_else(|| anyhow!("FakeLlm: no answer queued"))
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| bag_of_words(t)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_output_text() {
        let response = json!({
            "status": "completed",
            "output": [
                { "type": "reasoning", "summary": [] },
                { "type": "message", "content": [
                    { "type": "output_text", "text": "{\"a\":" },
                    { "type": "output_text", "text": "1}" }
                ]}
            ]
        });
        assert_eq!(output_text(&response).unwrap(), "{\"a\":1}");
    }

    #[test]
    fn strict_object_requires_all_fields() {
        let s = schema::object(&[("a", schema::string("")), ("b", schema::integer(""))]);
        assert_eq!(s["required"], json!(["a", "b"]));
        assert_eq!(s["additionalProperties"], json!(false));
    }
}
