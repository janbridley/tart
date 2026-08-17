//! OpenAI Chat Completions interface.

use std::io::{BufRead, BufReader, Read};

use crate::{ContextHistory, ReasoningEffort};
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
    /// A value this client does not recognize.
    #[serde(other)]
    Unknown,
}

/// One streaming delta of a completion.
///
/// Thinking-capable models emit reasoning deltas first, then answer deltas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delta {
    /// A fragment of the model's chain-of-thought reasoning.
    Thinking(String),
    /// A fragment of the final answer.
    Answer(String),
}

/// A client for an OpenAI-compatible Chat Completions endpoint.
pub struct ChatCompletionsClient {
    /// A Chat-Completions URL.
    completions_url: String,
    api_key: String,
    model: String,
    /// How hard the model reasons. `None` uses the provider default.
    reasoning_effort: Option<ReasoningEffort>,
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
            reasoning_effort: None,
        }
    }

    /// Set how hard the model reasons before answering.
    #[inline]
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Begin a streaming completion for the transcript held in `history`.
    ///
    /// - Connection failures and provider error bodies come through as `Err`
    /// - Content deltas arrive as a stream, which can be iterated to completion
    ///   (or consumed entirely by [`CompletionStream::complete`])
    /// - Dropping the stream early closes the connection
    pub fn create(&self, history: &ContextHistory) -> anyhow::Result<CompletionStream> {
        let request = CompletionRequest {
            model: &self.model,
            messages: history.as_slice(),
            reasoning_effort: self.reasoning_effort,
            stream: true,
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

        Ok(CompletionStream::from_reader(response.into_reader()))
    }
}

/// An in-flight streaming completion.
///
/// Yields assistant content deltas as they arrive over SSE. The stream ends
/// once it yields `None`; if it ended cleanly, [`CompletionStream::message`]
/// and [`CompletionStream::finish_reason`] turn `Some`.
pub struct CompletionStream {
    /// The SSE body, read line by line.
    reader: BufReader<Box<dyn Read + Send + Sync>>,
    /// Reasoning deltas seen so far.
    thinking: String,
    /// Answer deltas seen so far.
    content: String,
    /// The reason generation stopped, set once the terminal chunk arrives.
    finish_reason: Option<FinishReason>,
    /// The stream reached `[DONE]` or the end of the body.
    done: bool,
    /// An error was yielded; no further items will be produced.
    errored: bool,
}

impl CompletionStream {
    fn from_reader(reader: impl Read + Send + Sync + 'static) -> Self {
        Self {
            reader: BufReader::new(Box::new(reader)),
            thinking: String::new(),
            content: String::new(),
            finish_reason: None,
            done: false,
            errored: false,
        }
    }

    /// The chain-of-thought reasoning seen so far.
    pub fn thinking(&self) -> &str {
        &self.thinking
    }

    /// The reason generation stopped; `Some` once the terminal chunk has been
    /// seen.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// The assembled assistant message; `Some` only after the stream ran to
    /// completion.
    pub fn message(&self) -> Option<Message> {
        (self.done && !self.errored && self.finish_reason.is_some()).then(|| Message {
            role: Role::Assistant,
            content: self.content.clone(),
        })
    }

    /// Consume the stream, forwarding each delta to `on_delta` as it arrives,
    /// then return the assembled message and its finish reason. The assembled
    /// message contains only the answer; reasoning never enters it.
    pub fn complete(
        mut self,
        mut on_delta: impl FnMut(Delta),
    ) -> anyhow::Result<(Message, FinishReason)> {
        for delta in self.by_ref() {
            on_delta(delta?);
        }

        let Some(finish_reason) = self.finish_reason else {
            anyhow::bail!("stream ended without a finish reason");
        };

        Ok((
            Message {
                role: Role::Assistant,
                content: self.content,
            },
            finish_reason,
        ))
    }
}

impl Iterator for CompletionStream {
    type Item = anyhow::Result<Delta>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.errored {
            return None;
        }

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                // End of body without `[DONE]`: the stream was truncated.
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {}
                Err(error) => {
                    self.errored = true;
                    return Some(Err(error.into()));
                }
            }

            // Event separators, keep-alive comments, `event:` lines, and so on.
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.strip_prefix(' ').unwrap_or(payload).trim_end();

            // The chat-completions terminator; not part of SSE itself.
            if payload == "[DONE]" {
                self.done = true;
                return None;
            }

            let chunk: CompletionChunk = match serde_json::from_str(payload) {
                Ok(chunk) => chunk,
                Err(error) => {
                    self.errored = true;
                    return Some(Err(error.into()));
                }
            };
            let Some(choice) = chunk.choices.first() else {
                // Chunks may legitimately carry an empty `choices` array.
                continue;
            };
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            if let Some(reasoning) = choice
                .delta
                .reasoning_content
                .as_deref()
                .filter(|r| !r.is_empty())
            {
                self.thinking.push_str(reasoning);
                return Some(Ok(Delta::Thinking(reasoning.to_string())));
            }
            let Some(delta) = choice.delta.content.as_deref().filter(|d| !d.is_empty()) else {
                continue;
            };
            self.content.push_str(delta);
            return Some(Ok(Delta::Answer(delta.to_string())));
        }
    }
}

/// The JSON body of a chat completion request.
#[derive(Serialize)]
struct CompletionRequest<'a> {
    /// A model name the provider recognizes.
    model: &'a str,
    /// The conversation so far.
    messages: &'a [Message],
    /// How hard the model reasons; omitted to use the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    /// Ask for an SSE stream of deltas rather than a single JSON body.
    stream: bool,
}

/// One SSE `data:` payload from a streaming response.
#[derive(Deserialize)]
struct CompletionChunk {
    choices: Vec<ChunkChoice>,
}

/// One entry of `choices` in a [`CompletionChunk`].
#[derive(Deserialize)]
struct ChunkChoice {
    /// The partial message; fields arrive as they fill in. Some servers omit
    /// `delta` on the terminal chunk.
    #[serde(default)]
    delta: ChunkDelta,
    /// Reason why the completion exited, on the terminal chunk.
    finish_reason: Option<FinishReason>,
}

/// The `delta` field of a [`ChunkChoice`].
#[derive(Default, Deserialize)]
struct ChunkDelta {
    /// Chain-of-thought reasoning, on thinking-capable models.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// The answer text.
    #[serde(default)]
    content: Option<String>,
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

    #[test]
    fn stream_yields_deltas_then_completes() {
        let sse = concat!(
            r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "\n\n",
            "data: [DONE]\n\n",
        );
        let mut stream = CompletionStream::from_reader(sse.as_bytes());

        assert_eq!(
            stream.next().unwrap().unwrap(),
            Delta::Thinking("hmm".into())
        );
        assert_eq!(stream.next().unwrap().unwrap(), Delta::Answer("Hel".into()));
        assert_eq!(stream.next().unwrap().unwrap(), Delta::Answer("lo".into()));
        assert!(stream.next().is_none());
        assert_eq!(stream.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(stream.thinking(), "hmm");
        assert_eq!(stream.message().unwrap().content, "Hello");
    }

    #[test]
    fn complete_assembles_the_message() {
        let sse = concat!(
            r#"data: {"choices":[{"delta":{"reasoning_content":"Let me "},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            "\n\n",
            "data: [DONE]\n",
        );
        let stream = CompletionStream::from_reader(sse.as_bytes());

        let (message, finish_reason) = stream.complete(|_| {}).unwrap();
        // Reasoning never leaks into the assembled message.
        assert_eq!(message.content, "Hi");
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(finish_reason, FinishReason::Length);
    }

    #[test]
    fn complete_forwards_thinking_and_answer() {
        let sse = concat!(
            r#"data: {"choices":[{"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            "\n\n",
            "data: [DONE]\n",
        );
        let stream = CompletionStream::from_reader(sse.as_bytes());

        let mut thinking = String::new();
        let mut answer = String::new();
        stream
            .complete(|delta| match delta {
                Delta::Thinking(text) => thinking.push_str(&text),
                Delta::Answer(text) => answer.push_str(&text),
            })
            .unwrap();
        assert_eq!(thinking, "Let me think");
        assert_eq!(answer, "Hi");
    }

    #[test]
    fn truncated_stream_has_no_finish_reason() {
        let sse = r#"data: {"choices":[{"delta":{"content":"par"}}]}"#;
        let mut stream = CompletionStream::from_reader(sse.as_bytes());

        assert_eq!(stream.next().unwrap().unwrap(), Delta::Answer("par".into()));
        assert!(stream.next().is_none());
        assert_eq!(stream.finish_reason(), None);
        assert!(stream.message().is_none());
    }

    #[test]
    fn malformed_chunk_errors_and_fuses() {
        let sse = "data: not json\n";
        let mut stream = CompletionStream::from_reader(sse.as_bytes());

        assert!(stream.next().unwrap().is_err());
        assert!(stream.next().is_none());
        assert!(stream.message().is_none());
    }
}
