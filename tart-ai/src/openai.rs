//! OpenAI Chat Completions interface.

use std::io::{BufRead, BufReader, Read};

use crate::{ContextHistory, ReasoningEffort, Usage};
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
    #[inline]
    pub fn user(content: String) -> Self {
        Self {
            role: Role::System,
            content,
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
    /// Token usage, set once a chunk carries it.
    usage: Option<Usage>,
    /// Content deferred from a chunk that also carried reasoning.
    pending: Option<String>,
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
            usage: None,
            pending: None,
            done: false,
            errored: false,
        }
    }

    /// The chain-of-thought reasoning seen so far.
    pub fn thinking(&self) -> &str {
        &self.thinking
    }

    /// The reason generation stopped; `Some` once the terminal chunk has been seen.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Token usage for the request; `Some` once it has been received.
    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// The assembled assistant message; `Some` only after the stream ran to
    /// completion.
    pub fn message(&self) -> Option<Message> {
        (self.done && !self.errored && self.finish_reason.is_some()).then(|| Message {
            role: Role::Assistant,
            content: self.content.clone(),
        })
    }

    /// Exhaust the stream, forwarding each delta and returning the assembled message.
    ///
    /// ```rust
    /// # fn main() -> anyhow::Result<()> {
    /// use tart_ai::openai::{ChatCompletionsClient, Delta, Message, Role};
    /// use tart_ai::ContextHistory;
    ///
    /// let client = ChatCompletionsClient::new(
    ///     "https://api.deepseek.com/chat/completions",
    ///     std::env::var("DEEPSEEK_API_KEY")?,
    ///     "deepseek-v4-flash",
    /// );
    /// let mut history = ContextHistory::from(Message::system());
    /// history.append_message(Message {
    ///     role: Role::User,
    ///     content: "Who are you?".to_string(),
    /// });
    ///
    /// let mut stream = client.create(&history)?;
    /// let (message, finish_reason) = stream.complete(|delta| match delta {
    ///     Delta::Thinking(text) => print!("\x1b[2m{text}\x1b[0m"),
    ///     Delta::Answer(text) => print!("{text}"),
    /// })?;
    ///
    /// // The stream is spent but still readable, so we can get our usage data out.
    /// if let Some(usage) = stream.usage() {
    ///     history.record_usage(usage);
    /// }
    /// history.append_message(message);
    /// # Ok(())
    /// # }
    /// ```
    pub fn complete(
        &mut self,
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
                content: self.content.clone(),
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

        // Accumulate content from chunks that carry both reasoning and answers.
        if let Some(pending) = self.pending.take() {
            return Some(Ok(Delta::Answer(pending)));
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
                    // Surface the provider error to the caller verbatim.
                    return Some(Err(anyhow::anyhow!(
                        "chat completions stream sent an unparseable chunk: {payload} ({error})"
                    )));
                }
            };
            // Usage arrives on the final chunk, empty `choices` or not.
            if let Some(usage) = chunk.usage {
                self.usage = Some(usage.into());
            }
            let Some(choice) = chunk.choices.first() else {
                // Chunks may legitimately carry an empty `choices` array.
                continue;
            };
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            // A chunk may carry reasoning and content together. If so, defer content
            // until we empty all reasoning data.
            let content = choice.delta.content.as_deref().filter(|c| !c.is_empty());
            let reasoning = choice
                .delta
                .reasoning_content
                .as_deref()
                .filter(|r| !r.is_empty());

            if let Some(c) = content {
                self.content.push_str(c);
            }

            if let Some(r) = reasoning {
                self.thinking.push_str(r);
                self.pending = content.map(String::from);
                return Some(Ok(Delta::Thinking(r.to_string())));
            }

            let Some(c) = content else {
                continue;
            };
            return Some(Ok(Delta::Answer(c.to_string())));
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
    /// Token usage, carried by the final chunk (`null` elsewhere).
    #[serde(default)]
    usage: Option<WireUsage>,
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

/// The `usage` object, tolerating every provider's spelling.
#[derive(Clone, Copy, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: PromptTokensDetails,
    #[serde(default)]
    completion_tokens_details: CompletionTokensDetails,
    /// DeepSeek's flat spelling of the cache hit count.
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
    /// Tokens newly written to the cache.
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

impl From<WireUsage> for Usage {
    fn from(wire: WireUsage) -> Self {
        Self {
            prompt_tokens: wire.prompt_tokens,
            completion_tokens: wire.completion_tokens,
            reasoning_tokens: wire.completion_tokens_details.reasoning_tokens,
            cached_tokens: if wire.prompt_tokens_details.cached_tokens > 0 {
                wire.prompt_tokens_details.cached_tokens
            } else {
                wire.prompt_cache_hit_tokens
            },
            cache_write_tokens: wire.prompt_tokens_details.cache_write_tokens,
        }
    }
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
        let mut stream = CompletionStream::from_reader(sse.as_bytes());

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
        let mut stream = CompletionStream::from_reader(sse.as_bytes());

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
