//! OpenAI Chat Completions interface.

use serde::{Deserialize, Serialize};

pub const SYSTEM: &str = include_str!("data/SYSTEM.md");

/// Valid `role` entries for a [`Message`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Assistant,
    System,
    Developer,
    User,
}

/// One unit of data passed from the user to the model or vice versa.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The `Role` of the actor sending the message.
    pub role: Role,
    /// The text associated with this message.
    pub content: String,
}

impl Message {
    /// Initialize a `Role::System` message with the tart system prompt.
    #[inline]
    pub fn system() -> Self {
        Self {
            role: Role::System,
            content: SYSTEM.to_string(),
        }
    }
}

/// Why the model stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    FunctionCall,
}

/// A client for an OpenAI-compatible Chat Completions endpoint.
pub struct ChatCompletionsClient {
    /// A Chat-Completions URL.
    completions_url: String,
    api_key: String,
    model: String,
}

impl ChatCompletionsClient {
    /// Configure a client for an OpenAI-compatible endpoint.
    #[inline]
    pub fn new(
        completions_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            completions_url: completions_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Request a completion for a conversation, returning the sole assistant
    /// message and why the model stopped generating.
    ///
    /// We always want a single choice: the wire default for `n` is 1, and this
    /// errors if the endpoint answers with anything else.
    pub fn create(&self, messages: &[Message]) -> anyhow::Result<(Message, FinishReason)> {
        let request = CompletionRequest {
            model: &self.model,
            messages,
        };

        let response = match ureq::post(&self.completions_url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(&request)
        {
            Ok(response) => response,
            // Surface the provider's error body, which explains what went wrong.
            Err(ureq::Error::Status(code, response)) => anyhow::bail!(
                "chat completions endpoint returned {code}: {}",
                response
                    .into_string()
                    .unwrap_or_else(|_| "<no body>".to_string())
            ),
            Err(error) => return Err(error.into()),
        };

        let completion: CompletionResponse = response.into_json()?;
        let [choice] = &completion.choices[..] else {
            anyhow::bail!(
                "expected exactly one choice, but the endpoint returned {}",
                completion.choices.len()
            )
        };

        Ok((choice.message.clone(), choice.finish_reason))
    }
}

/// The JSON body of a chat completion request.
#[derive(Serialize)]
struct CompletionRequest<'a> {
    /// A model name the provider recognizes.
    model: &'a str,
    /// The conversation so far.
    messages: &'a [Message],
}

/// The JSON body of a chat completion response, limited to the fields we use.
#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
}

/// One entry of `choices` in a [`CompletionResponse`].
#[derive(Deserialize)]
struct CompletionChoice {
    /// Text and tool data for the completion.
    message: Message,
    /// Reason why the completion exited.
    finish_reason: FinishReason,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_serialize_as_expected() {
        let wire = serde_json::to_value(Message::system()).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({ "role": "system", "content": SYSTEM })
        );
    }

    #[test]
    fn finish_reasons_deserialize_as_expected() {
        let reason: FinishReason = serde_json::from_value(serde_json::json!("tool_calls")).unwrap();
        assert_eq!(reason, FinishReason::ToolCalls);
    }
}
