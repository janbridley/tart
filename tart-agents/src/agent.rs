use std::panic::AssertUnwindSafe;

use async_compat::Compat;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponseArgs, FunctionToolCall, InputParam, OutputItem, Reasoning, ReasoningEffort,
        ReasoningItem, ResponseStreamEvent,
    },
};
use futures::{StreamExt, executor::block_on};

use crate::{MAX_TOOL_ROUNDS, Progress, Transcript, debug, sandbox::Policy, tools};

/// A Responses-API model configured to run the tart tool loop.
#[derive(Clone)]
pub struct Agent {
    /// HTTP client for an OpenAI-compatible endpoint; shares its connection pool.
    client: Client<OpenAIConfig>,
    /// Model name sent with every request.
    model: String,
    /// How hard the model reasons; `None` uses the provider default.
    effort: Option<ReasoningEffort>,
    /// Most tool rounds one generation may take.
    max_rounds: usize,
    /// The seatbelt policy every bash tool call runs under.
    policy: Policy,
}

impl Agent {
    /// Configure a client for an OpenAI-compatible Responses endpoint, running
    /// tool calls under `policy`.
    #[inline]
    pub fn new<U: Into<String>, K: Into<String>, M: Into<String>>(
        base_url: U,
        api_key: K,
        model: M,
        policy: Policy,
    ) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(base_url.into())
            .with_api_key(api_key.into());
        Self {
            client: Client::with_config(config),
            model: model.into(),
            effort: None,
            max_rounds: MAX_TOOL_ROUNDS,
            policy,
        }
    }

    /// Set how hard the model reasons before answering.
    #[must_use]
    #[inline]
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Run one generation on its own thread, reporting progress to `on_progress`.
    ///
    /// The worker records its turns (reasoning, tool exchanges, final answer) into the
    /// shared transcript as it goes. Exactly one terminal event ([`Progress::Done`] or
    /// [`Progress::Failed`]) is delivered, even if the worker panics.
    #[inline]
    pub fn spawn<F: Fn(Progress) + Send + 'static>(&self, transcript: &Transcript, on_progress: F) {
        let agent = self.clone();
        let transcript = transcript.clone();
        std::thread::spawn(move || {
            // A panicking worker must still deliver the terminal event to the caller
            let outcome =
                std::panic::catch_unwind(AssertUnwindSafe(|| agent.run(&transcript, &on_progress)));
            if outcome.is_err() {
                terminate_and_log(
                    &on_progress,
                    Progress::Failed("generation panicked".to_string()),
                );
            }
        });
    }

    /// Drive one generation to completion on the current thread.
    ///
    /// Each round streams model output; when the model calls a tool (which
    /// could happen more than once per round) the round's reasoning, any text
    /// it streamed alongside the calls, the calls, and their outputs are
    /// recorded in the shared transcript and the next round continues from
    /// there. The final answer is recorded on completion, so later turns replay
    /// the whole exchange. The generation ends after at
    /// most `max_rounds` rounds, always with exactly one terminal event:
    /// [`Progress::Done`] with the final answer (`None` if nothing arrived), or
    /// [`Progress::Failed`] on a request error, an explicit error or incomplete event,
    /// or a truncated stream (one that ended mid tool call or delivered nothing).
    /// A stream that closes without a terminal event still ends its round with whatever
    /// the model produced.
    ///
    /// Blocks the current thread until the generation finishes.
    #[allow(
        clippy::too_many_lines,
        reason = "the round loop reads best as one straight-line function"
    )]
    fn run<F: Fn(Progress)>(&self, transcript: &Transcript, on_progress: &F) {
        for _ in 0..self.max_rounds {
            let request = match CreateResponseArgs::default()
                .model(self.model.as_str())
                .stream(true)
                .reasoning(Reasoning {
                    effort: self.effort.clone(),
                    summary: None,
                })
                .input(InputParam::Items(transcript.request_items()))
                .tools(vec![tools::bash(), tools::read(), tools::edit()])
                .build()
            {
                Ok(request) => request,
                Err(error) => {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
            };

            debug::log_json("round request", || serde_json::to_string(&request));

            // `Compat` enters the global tokio runtime and exposes `futures` blocking control.
            let mut stream =
                match block_on(Compat::new(self.client.responses().create_stream(request))) {
                    Ok(stream) => stream,
                    Err(error) => {
                        return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                    }
                };

            let mut answer = String::new();
            // Completed calls, captured from finished output items, in stream order.
            let mut calls: Vec<FunctionToolCall> = Vec::new();
            // The reasoning that preceded them; thinking mode requires its replay.
            let mut reasoning: Option<ReasoningItem> = None;
            // A function-call item started but never reported done.
            let mut call_in_flight = false;
            // Some output arrived, so the stream was delivering.
            let mut saw_output = false;
            // The last transport error, skipped so a trailing one cannot discard an
            // otherwise complete answer.
            let mut last_error: Option<String> = None;

            while let Some(item) = block_on(stream.next()) {
                match item {
                    Ok(ResponseStreamEvent::ResponseOutputTextDelta(delta)) => {
                        answer.push_str(&delta.delta);
                        saw_output = true;
                        on_progress(Progress::Answer(delta.delta));
                    }
                    Ok(ResponseStreamEvent::ResponseReasoningTextDelta(delta)) => {
                        saw_output = true;
                        on_progress(Progress::Thinking(delta.delta));
                    }
                    Ok(ResponseStreamEvent::ResponseOutputItemAdded(added)) => {
                        if matches!(added.item, OutputItem::FunctionCall(_)) {
                            call_in_flight = true;
                        }
                    }
                    Ok(ResponseStreamEvent::ResponseOutputItemDone(done)) => match &done.item {
                        OutputItem::Reasoning(item) => {
                            debug::log_json("captured reasoning item", || {
                                serde_json::to_string(item)
                            });
                            reasoning = Some(item.clone());
                            saw_output = true;
                        }
                        OutputItem::FunctionCall(call) => {
                            calls.push(call.clone());
                            call_in_flight = false;
                            saw_output = true;
                        }
                        _ => {}
                    },
                    Ok(ResponseStreamEvent::ResponseCompleted(completed)) => {
                        // A completed response is output even when it holds no items.
                        saw_output = true;
                        if let Some(usage) = completed.response.usage {
                            on_progress(Progress::Usage {
                                input: u64::from(usage.input_tokens),
                                cached: u64::from(usage.input_tokens_details.cached_tokens),
                                output: u64::from(usage.output_tokens),
                            });
                        }
                    }
                    Ok(ResponseStreamEvent::ResponseFailed(failed)) => {
                        debug::log_json("response failed event", || serde_json::to_string(&failed));
                        return terminate_and_log(
                            on_progress,
                            Progress::Failed(failed.response.error.map_or_else(
                                || "response failed".to_string(),
                                |error| error.message,
                            )),
                        );
                    }
                    // The provider said the stream broke; not recoverable here.
                    Ok(ResponseStreamEvent::ResponseError(error)) => {
                        debug::log_json("response error event", || serde_json::to_string(&error));
                        return terminate_and_log(
                            on_progress,
                            Progress::Failed(format!(
                                "{}: {}",
                                error.code.unwrap_or_else(|| "error".to_string()),
                                error.message
                            )),
                        );
                    }
                    // Truncated (max output tokens, content filter): report it
                    Ok(ResponseStreamEvent::ResponseIncomplete(incomplete)) => {
                        debug::log_json("response incomplete event", || {
                            serde_json::to_string(&incomplete)
                        });
                        let reason = incomplete
                            .response
                            .incomplete_details
                            .map_or_else(|| "unknown reason".to_string(), |details| details.reason);
                        return terminate_and_log(
                            on_progress,
                            Progress::Failed(format!("response incomplete: {reason}")),
                        );
                    }
                    Ok(_) => {}
                    // Transport errors are skipped: the stream ends on its own,
                    // and the checks after the loop decide what the round got.
                    Err(error) => {
                        let error = error.to_string();
                        debug::log("stream error", || error.clone());
                        last_error = Some(error);
                    }
                }
            }

            if call_in_flight {
                return terminate_and_log(
                    on_progress,
                    Progress::Failed(with_last_error(
                        "stream ended mid tool call",
                        last_error, // The stream ended mid tool call
                    )),
                );
            }

            if !saw_output {
                return terminate_and_log(
                    on_progress,
                    Progress::Failed(with_last_error(
                        "stream ended without output",
                        last_error, // Nothing arrived at all
                    )),
                );
            }

            // No calls pending: this round's answer is the turn's message.
            if calls.is_empty() {
                if !answer.is_empty()
                    && let Err(error) = transcript.push_assistant(answer.clone())
                {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
                return terminate_and_log(
                    on_progress,
                    Progress::Done {
                        message: (!answer.is_empty()).then_some(answer),
                    },
                );
            }

            // Run the round's calls in order, then record the round as one group.
            let mut exchanges = Vec::with_capacity(calls.len());
            // A call the harness cannot run ends the round, but is recorded with its
            // error so the round's earlier effects stay in the transcript.
            let mut failure: Option<String> = None;
            for call in calls {
                match tools::execute(&call, &self.policy, on_progress) {
                    Ok(output) => exchanges.push((call, output)),
                    Err(error) => {
                        exchanges.push((call, format!("error: {error}")));
                        failure = Some(error.to_string());
                        break;
                    }
                }
            }
            if let Some(item) = reasoning {
                transcript.push_reasoning(item);
            }
            if !answer.is_empty()
                && let Err(error) = transcript.push_assistant(std::mem::take(&mut answer))
            {
                return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
            }
            transcript.push_tool_round(exchanges);
            if let Some(reason) = failure {
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }
        }
        let rounds = self.max_rounds;
        terminate_and_log(
            on_progress,
            Progress::Failed(format!("gave up after {rounds} tool rounds")),
        );
    }
}

/// A failure message with the last skipped transport error, if any, appended.
fn with_last_error(message: &str, last_error: Option<String>) -> String {
    match last_error {
        Some(error) => format!("{message}: {error}"),
        None => message.to_string(),
    }
}

/// Deliver the generation's terminal event, mirroring its outcome to the debug lob.
fn terminate_and_log<F: Fn(Progress)>(on_progress: &F, event: Progress) {
    debug::log("generation outcome", || match &event {
        Progress::Done { message } => {
            format!("done ({} answer chars)", message.as_deref().map_or(0, str::len))
        }
        Progress::Failed(reason) => format!("failed: {reason}"),
        _ => "not a terminal event".to_string(),
    });
    on_progress(event);
}
