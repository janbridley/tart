use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::time::Duration;

use async_compat::Compat;
use async_openai::{
    Client,
    config::OpenAIConfig,
    error::OpenAIError,
    types::responses::{
        CreateResponse, CreateResponseArgs, FunctionToolCall, InputParam, OutputItem, Reasoning,
        ReasoningEffort, ReasoningItem, ResponseStreamEvent, Tool,
    },
};
use futures::FutureExt;
use futures::channel::mpsc;
use futures::future::Either;
use futures::stream::{Stream, StreamExt};

use crate::{MAX_TOOL_ROUNDS, Progress, Transcript, debug, sandbox::Policy, tools};

/// How long one HTTP connection attempt may take before the request fails.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// How long one socket read may stall, receiving nothing, before the stream fails into `Progress::Failed`.
const READ_TIMEOUT: Duration = Duration::from_secs(120);

/// The session's collaboration mode, mirroring Codex's `ModeKind`.
///
/// Plan mode is a property of the *session*, not of the composer: it selects which
/// policy tool calls run under and which tools are offered. The front end's own
/// input modes (`!` manual commands) never reach the model and are not represented here
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatMode {
    /// Ordinary chat: the granted roots are writable and `edit` is offered.
    #[default]
    Default,
    /// Plan mode: research and plan, blocking writes to the working directory.
    Plan,
}

/// One item of a response stream, as the provider yields it.
type StreamItem = Result<ResponseStreamEvent, OpenAIError>;

/// One response stream, as the provider yields it.
type ResponseStream = Pin<Box<dyn Stream<Item = StreamItem> + Send>>;

/// The future that opens one round's response stream.
type OpenStreamFuture = Pin<Box<dyn Future<Output = Result<ResponseStream, OpenAIError>> + Send>>;

/// One generation task, boxed for whichever executor drives it.
pub type TurnTask = Pin<Box<dyn Future<Output = ()> + Send>>;

/// A Responses-API model configured to run the tart tool loop.
#[derive(Clone)]
pub struct Agent {
    /// HTTP client for an OpenAI-compatible endpoint with a shared connection pool.
    client: Client<OpenAIConfig>,
    /// Model name sent with every request.
    model: String,
    /// How hard the model reasons; `None` uses the provider default.
    effort: Option<ReasoningEffort>,
    /// Most tool rounds one generation may take.
    max_rounds: usize,
    /// The session's mode; picks the active policy and tool list.
    mode: ChatMode,
    /// The Default-mode policy: the granted roots stay writable.
    writable: Policy,
    /// The Plan-mode policy: the same grants, but most paths are read-only.
    planning: Policy,
}

/// One interrupt the front end can send a running turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poke {
    /// Esc: save the partial answer, if any, and stop.
    Cancel,
    /// A steering message: save the partial, record this text, restart the round.
    Steer(String),
}

/// The front end's lever on one running turn.
pub struct TurnHandle {
    /// The turn's poke sender, where interrupts reach the round loop.
    sender: mpsc::Sender<Poke>,
}

impl TurnHandle {
    /// Cancel the turn, keeping any partial answer and every exchange that already ran
    ///
    /// Mid-stream the stream is dropped (aborting the request); mid-tool the
    /// call's process group is killed within one watchdog slice and the call
    /// is awaited to its `[cancelled]` framing; between rounds no further
    /// request is spent.
    #[inline]
    pub fn cancel(&mut self) {
        let _ = self.sender.try_send(Poke::Cancel);
    }

    /// Deliver `text` as a steering poke. [`Steering`] ensures we go one at a time
    #[inline]
    fn steer(&mut self, text: String) {
        let _ = self.sender.try_send(Poke::Steer(text));
    }
}

/// The session's steering surface.
#[derive(Default)]
pub struct Steering {
    /// The running turn's lever; absent while the model is idle.
    handle: Option<TurnHandle>,
    /// The queued steering message, or None otherwise.
    slot: Option<String>,
}

impl Steering {
    /// Install a newly spawned turn's lever, dropping any slot a finished turn left.
    #[inline]
    pub fn begin(&mut self, handle: TurnHandle) {
        self.slot = None;
        self.handle = Some(handle);
    }

    /// The turn ended; retire its lever.
    #[inline]
    pub fn end(&mut self) {
        self.handle = None;
    }

    /// Esc's lever on the running turn: a no-op while the model is idle.
    #[inline]
    pub fn cancel(&mut self) {
        if let Some(handle) = &mut self.handle {
            handle.cancel();
        }
    }

    /// Queue `text` to interrupt and redirect the running turn.
    ///
    /// Returns `false` when a message is already queued.
    #[must_use]
    #[inline]
    pub fn steer(&mut self, text: String) -> bool {
        if self.slot.is_some() {
            return false;
        }
        if let Some(handle) = &mut self.handle {
            handle.steer(text.clone());
        }
        self.slot = Some(text);
        true
    }

    /// The waiting steering message, if any.
    #[inline]
    pub fn steering(&self) -> Option<&str> {
        self.slot.as_deref()
    }

    /// Take the waiting steering message, if any.
    #[inline]
    pub fn take(&mut self) -> Option<String> {
        self.slot.take()
    }

    /// Clear the slot when `text` is the message the worker just consumed —
    /// matched, so a different steer queued in the same breath survives.
    #[inline]
    pub fn clear_if(&mut self, text: &str) {
        if self.slot.as_deref() == Some(text) {
            self.slot = None;
        }
    }
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
        // Register a cryptography backend, otherwise reqwest rustls-no-provider panics
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = OpenAIConfig::new()
            .with_api_base(base_url.into())
            .with_api_key(api_key.into());
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()
            .unwrap_or_default();
        let planning = policy.clone().read_only();
        Self {
            client: Client::build(http, config),
            model: model.into(),
            effort: None,
            max_rounds: MAX_TOOL_ROUNDS,
            mode: ChatMode::Default,
            writable: policy,
            planning,
        }
    }

    /// Switch the session's collaboration mode for subsequent turns.
    ///
    /// A turn already in flight keeps the policy it started with, since
    /// [`Agent::spawn`] copied the agent before it began, so this cannot change the
    /// sandbox under a running command.
    #[inline]
    pub fn set_mode(&mut self, mode: ChatMode) {
        self.mode = mode;
    }

    /// The policy the current mode runs tool calls under.
    fn policy(&self) -> &Policy {
        match self.mode {
            ChatMode::Default => &self.writable,
            ChatMode::Plan => &self.planning,
        }
    }

    /// The tools `mode` offers to the model. Plan mode withholds the `edit` tool.
    fn tools_for(mode: ChatMode) -> Vec<Tool> {
        let mut definitions = vec![tools::bash(), tools::read()];
        if mode == ChatMode::Default {
            definitions.push(tools::edit());
        }
        definitions.extend(tools::search());
        definitions.extend(tools::fetch());
        definitions
    }

    /// Record one steering message, reporting it so the front end can echo
    fn record_steer<F: Fn(Progress)>(
        transcript: &Transcript,
        text: &str,
        on_progress: &F,
    ) -> anyhow::Result<()> {
        transcript.push_user(text.to_string())?;
        on_progress(Progress::Steered(text.to_string()));
        Ok(())
    }

    /// Set how hard the model reasons before answering.
    #[must_use]
    #[inline]
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Change the reasoning effort for subsequent turns.
    #[inline]
    pub fn set_reasoning_effort(&mut self, effort: ReasoningEffort) {
        self.effort = Some(effort);
    }

    /// Run one generation as a task on the injected executor, reporting progress.
    ///
    /// The task records its turns (reasoning, tool exchanges, final answer) into the
    /// shared transcript as it goes. Exactly one terminal event ([`Progress::Done`],
    /// [`Progress::Failed`], or [`Progress::Cancelled`]) is delivered, even if the
    /// task panics. The handle needs no retirement: when the turn ends its
    /// receiver is gone, and later pokes simply go unread.
    #[must_use]
    #[inline]
    pub fn spawn<F, D>(&self, transcript: &Transcript, on_progress: F, drive: D) -> TurnHandle
    where
        F: Fn(Progress) + Send + Sync + 'static,
        D: FnOnce(TurnTask),
    {
        let agent = self.clone();
        let transcript = transcript.clone();
        // This turn's poke channel: the sender waits where Esc and steering
        // reach it, and the receiver races the stream inside the task.
        // Four slots, so a Cancel can queue behind a Steer.
        let (sender, pokes) = mpsc::channel(4);
        // One provider call per round, from the agent's connection pool.
        let open_client = agent.client.clone();
        drive(
            async move {
                let open = move |request: CreateResponse| -> OpenStreamFuture {
                    let client = open_client.clone();
                    Box::pin(async move { client.responses().create_stream(request).await })
                };
                // A panicking task must still deliver the terminal event.
                let outcome = AssertUnwindSafe(agent.run(&transcript, pokes, open, &on_progress))
                    .catch_unwind()
                    .await;
                if outcome.is_err() {
                    terminate_and_log(
                        &on_progress,
                        Progress::Failed("generation panicked".to_string()),
                    );
                }
            }
            .boxed(),
        );
        TurnHandle { sender }
    }

    /// Drive one generation to completion as one task, on whatever executor
    /// the caller injected.
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
    /// or a truncated stream (one that ended mid tool call or delivered nothing),
    /// or [`Progress::Cancelled`] when the front end cancels the turn. *Keeps*
    /// *everything recorded so far, including a partial answer*.
    ///
    /// A steering message the front end queues mid-turn interrupts the stream
    /// the same way, is recorded as the next user message, and the round
    /// continues from it.
    ///
    /// A stream that closes without a terminal event still ends its round with whatever
    /// the model produced.
    ///
    /// Resolves with the generation's terminal `Progress`.
    #[allow(
        clippy::too_many_lines,
        reason = "the round loop reads best as one straight-line function"
    )]
    async fn run<O, F>(
        &self,
        transcript: &Transcript,
        mut pokes: mpsc::Receiver<Poke>,
        open: O,
        on_progress: &F,
    ) -> Progress
    where
        O: Fn(CreateResponse) -> OpenStreamFuture + Send,
        F: Fn(Progress) + Sync,
    {
        // Set once a cancel is consumed, so the rest of the turn's calls and rounds
        // skip without re-reading the channel.
        let mut cancelled = false;
        for _ in 0..self.max_rounds {
            // A cancelled generation stops before spending another request.
            if cancelled {
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            // Pokes that arrived between rounds ride on this request; Esc > steer.
            let (cancel, steer) = drain_pokes(&mut pokes);
            if cancel {
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            // Steering left over from the last round rides on this request.
            if let Some(text) = steer
                && let Err(error) = Self::record_steer(transcript, &text, on_progress)
            {
                return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
            }
            // The sandboxed trio less `edit` in plan mode, plus web tools if available
            let definitions = Self::tools_for(self.mode);
            let request = match CreateResponseArgs::default()
                .model(self.model.as_str())
                .stream(true)
                .reasoning(Reasoning {
                    effort: self.effort.clone(),
                    summary: None,
                })
                .input(InputParam::Items(transcript.request_items()))
                .tools(definitions)
                .build()
            {
                Ok(request) => request,
                Err(error) => {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
            };

            debug::log_json("round request", || serde_json::to_string(&request));

            // `Compat` enters the global tokio runtime so the provider's future polls
            // correctly from any executor. Its stream is a channel the pump feeds
            let mut stream = match Compat::new(open(request)).await {
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
            // The stream was dropped mid-round for a steering message.
            let mut aborted = false;

            // Race the stream against Esc and steering pokes; a won cancel
            // returns and drops the stream, a won steer restarts the round
            let mut pokes_closed = false;
            'events: loop {
                // The poke future borrows the receiver; scoping it here frees the drain
                let woke = {
                    let mut poke_fut = if pokes_closed {
                        futures::future::pending::<Option<Poke>>().boxed().fuse()
                    } else {
                        pokes.next().boxed().fuse()
                    };
                    let mut closed = false;
                    let woke = futures::select! {
                     item = stream.next().fuse() => {
                         let Some(item) = item else { break 'events };
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
                         None
                     }
                     // The poke arm hands back what it woke on
                    poke = poke_fut => {
                         if poke.is_none() {
                             closed = true;
                         }
                         poke
                     },
                     };
                    pokes_closed = closed;
                    woke
                };
                let Some(poke) = woke else {
                    continue;
                };
                // The poke that woke us has been consumed; drain its company.
                // An Esc queued behind a steer still wins.
                let (cancel, steer) = match poke {
                    Poke::Cancel => (true, None),
                    Poke::Steer(text) => {
                        let (cancel, more) = drain_pokes(&mut pokes);
                        (cancel, more.or(Some(text)))
                    }
                };
                if cancel {
                    // Esc won: dropping the stream keeps what it streamed.
                    if !answer.is_empty()
                        && let Err(error) = transcript.push_assistant(answer.clone())
                    {
                        return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                    }
                    return terminate_and_log(on_progress, Progress::Cancelled);
                }
                let Some(text) = steer else {
                    // Nothing actionable woke us: keep streaming.
                    continue;
                };
                // A steer won: record the partial, then the steered input, then restart
                if !answer.is_empty()
                    && let Err(error) = transcript.push_assistant(std::mem::take(&mut answer))
                {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
                if let Err(error) = Self::record_steer(transcript, &text, on_progress) {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
                aborted = true;
                break;
            }

            // The stream was dropped for a steer -> rebuild from truncated output
            if aborted {
                continue;
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
            // The tool loop broke early to consume a steering message.
            let mut steered_text: Option<String> = None;
            for call in calls {
                // Pokes that arrived between calls stop the loop the way
                // they always have.
                let (cancel, steer) = drain_pokes(&mut pokes);
                if cancel || cancelled {
                    cancelled = true;
                    break;
                }
                if let Some(text) = steer {
                    steered_text = Some(text);
                    break;
                }
                // Interrupt control per call
                let recorded = call.clone();
                let kill = tools::CancelToken::new();
                let raced = futures::future::select(
                    tools::execute(&call, self.policy(), &kill, on_progress).boxed(),
                    pokes.next(),
                )
                .await;
                let outcome = match raced {
                    Either::Left((outcome, _)) => outcome,
                    Either::Right((poke, in_flight)) => {
                        // Drain the company, as every other wake point does:
                        // an Esc queued behind a steer still wins, and the
                        // steer it retracts is discarded instead of recorded.
                        let (cancel, steer) = match poke {
                            Some(Poke::Cancel) => (true, None),
                            Some(Poke::Steer(text)) => {
                                let (cancel, more) = drain_pokes(&mut pokes);
                                (cancel, more.or(Some(text)))
                            }
                            None => (false, None),
                        };
                        if cancel {
                            kill.cancel();
                            cancelled = true;
                        } else if let Some(text) = steer {
                            steered_text = Some(text);
                        }
                        in_flight.await
                    }
                };
                match outcome {
                    Ok(output) => exchanges.push((recorded, output)),
                    Err(error) => {
                        exchanges.push((recorded, format!("error: {error}")));
                        failure = Some(error.to_string());
                        break;
                    }
                }
                if cancelled || steered_text.is_some() {
                    break;
                }
            }
            // A round cut short before its first executed call records no
            // reasoning: without its calls the item would dangle in the replay.
            if !exchanges.is_empty()
                && let Some(item) = reasoning
            {
                transcript.push_reasoning(item);
            }
            if !answer.is_empty()
                && let Err(error) = transcript.push_assistant(std::mem::take(&mut answer))
            {
                return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
            }
            transcript.push_tool_round(exchanges);
            if cancelled {
                // The turn stops here, keeping the rounds recorded so far.
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            if let Some(text) = steered_text {
                // The steered input rides on the next round's request.
                if let Err(error) = Self::record_steer(transcript, &text, on_progress) {
                    return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
                }
                continue;
            }
            if let Some(reason) = failure {
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }
        }
        let rounds = self.max_rounds;
        terminate_and_log(
            on_progress,
            Progress::Failed(format!("gave up after {rounds} tool rounds")),
        )
    }
}

/// Drain every pending poke, reporting whether an Esc is among them.
fn drain_pokes(pokes: &mut mpsc::Receiver<Poke>) -> (bool, Option<String>) {
    let mut cancel = false;
    let mut steer = None;
    while let Ok(poke) = pokes.try_recv() {
        match poke {
            Poke::Cancel => cancel = true,
            Poke::Steer(text) => steer = Some(text),
        }
    }
    (cancel, steer)
}

/// A failure message with the last skipped transport error, if any, appended.
fn with_last_error(message: &str, last_error: Option<String>) -> String {
    match last_error {
        Some(error) => format!("{message}: {error}"),
        None => message.to_string(),
    }
}

/// Deliver the generation's terminal event, mirroring its outcome to the debug
/// log, and hand it back so a driving wrapper can observe the outcome.
fn terminate_and_log<F: Fn(Progress)>(on_progress: &F, event: Progress) -> Progress {
    debug::log("generation outcome", || match &event {
        Progress::Done { message } => {
            format!("done ({} answer chars)", message.as_deref().map_or(0, str::len))
        }
        Progress::Failed(reason) => format!("failed: {reason}"),
        Progress::Cancelled => "cancelled".to_string(),
        _ => "not a terminal event".to_string(),
    });
    on_progress(event.clone());
    event
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use futures::executor::block_on;
    use serde_json::json;
    use std::sync::Mutex;

    /// `Agent::new` must install a TLS crypto provider before building its
    /// reqwest client; the `rustls-no-provider` build panics otherwise.
    #[test]
    fn new_installs_tls_provider() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let agent = Agent::new("http://localhost:9", "key", "model", policy);
        // Reaching here means the client constructed without the provider panic.
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
        let _ = agent.model;
    }

    /// Plan mode runs under the read-only twin of the Default policy and witholds
    /// the `edit` tool.
    #[test]
    fn plan_mode_is_read_only_and_drops_edit() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let mut agent = Agent::new("http://localhost:9", "key", "model", policy);

        // Default: the granted roots are writable, and `edit` is offered.
        assert!(!agent.policy().writable_roots().is_empty());
        assert!(names(&Agent::tools_for(ChatMode::Default)).contains(&"edit"));

        // Plan: the workspace is not writable, the temp scratch alone is, and not edit
        agent.set_mode(ChatMode::Plan);
        let writable = agent.policy().writable_roots();
        assert_eq!(writable.len(), 1, "exactly the scratch root: {writable:?}");
        let scratch =
            std::fs::canonicalize(std::env::temp_dir()).expect("the temp dir canonicalizes");
        assert_eq!(writable[0], scratch);
        let plan_tools = Agent::tools_for(ChatMode::Plan);
        let offered = names(&plan_tools);
        assert!(!offered.contains(&"edit"));
        assert!(offered.contains(&"read") && offered.contains(&"bash"));
    }

    /// The names of the function tools among `definitions`.
    fn names(definitions: &[Tool]) -> Vec<&str> {
        definitions
            .iter()
            .filter_map(|tool| match tool {
                Tool::Function(function) => Some(function.name.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn steering_gates_matches_and_survives_its_turn() {
        let mut steering = Steering::default();

        // One message waits at a time: a second submission is refused.
        assert!(steering.steer("first".to_string()));
        assert!(!steering.steer("second".to_string()));
        assert_eq!(steering.steering(), Some("first"));
        assert_eq!(steering.take(), Some("first".to_string()));

        // A steer with no turn running still waits, for the next one.
        assert!(steering.steer("queued while idle".to_string()));
        assert_eq!(steering.steering(), Some("queued while idle"));

        // Begin drops what the finished turn left behind, and installs the lever.
        let (sender, mut pokes) = mpsc::channel(4);
        steering.begin(TurnHandle { sender });
        assert_eq!(steering.steering(), None);
        assert_eq!(drain_pokes(&mut pokes), (false, None));

        // Cancel and steer deliver typed pokes the worker drains; an Esc
        // queued behind a steer still wins.
        assert!(steering.steer("redirect".to_string()));
        steering.cancel();
        assert_eq!(drain_pokes(&mut pokes), (true, Some("redirect".to_string())));

        // The matched clear frees the slot the worker just consumed; a
        // different steer queued in the same breath survives it.
        steering.clear_if("redirect");
        assert_eq!(steering.steering(), None);
        assert!(steering.steer("next".to_string()));
        steering.clear_if("redirect");
        assert_eq!(steering.steering(), Some("next"));

        // A retired lever's pokes go into the void harmlessly.
        let (sender, pokes) = mpsc::channel(4);
        steering.begin(TurnHandle { sender });
        steering.end();
        steering.cancel();
        drop(pokes);
    }

    #[test]
    fn a_poked_steer_records_and_restarts_the_round() {
        let agent = test_agent();
        let open = scripted(vec![
            // Round one streams a partial answer, then a poke redirects it.
            stays_open(vec![delta("partial ")]),
            // Round two answers from the steered input.
            ends(vec![delta("redirected")]),
        ]);
        let (sender, pokes) = mpsc::channel::<Poke>(4);
        let transcript = Transcript::new().unwrap();
        transcript.push_user("original".to_string()).unwrap();
        // The poke rides in mid-stream: the first delta queues the redirect,
        // the way a user's message lands during a live generation.
        let mid_stream = Mutex::new(Some(sender));
        let events = Mutex::new(Vec::new());
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::Answer(delta) if delta == "partial ")
                && let Some(mut sender) = mid_stream.lock().unwrap().take()
            {
                sender
                    .try_send(Poke::Steer("redirect".to_string()))
                    .expect("four slots");
            }
            events.lock().unwrap().push(progress);
        }));

        assert!(matches!(outcome, Progress::Done { message: Some(text) } if text == "redirected"));
        // The record reads: user, assistant partial, user redirect, assistant.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"], "original");
        assert_eq!(items[2]["role"], "assistant");
        assert_eq!(items[2]["content"], "partial ");
        assert_eq!(items[3]["role"], "user");
        assert_eq!(items[3]["content"], "redirect");
        assert_eq!(items[4]["role"], "assistant");
        assert_eq!(items[4]["content"], "redirected");
        // The steer was reported so the front end can echo it.
        assert!(events.lock().unwrap().iter().any(|event| matches!(
            event,
            Progress::Steered(text) if text == "redirect"
        )));
    }

    #[test]
    fn a_steer_mid_tool_waits_for_the_call_and_rides_the_next_round() {
        let agent = test_agent();
        let open = scripted(vec![
            ends(vec![item(json!({
                "type": "response.output_item.done",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "function_call", "id": "fc_1", "call_id": "c1", "name": "bash",
                    "arguments": "{\"command\":\"echo tool ran\"}", "status": "completed"
                }
            }))]),
            ends(vec![delta("after the steer")]),
        ]);
        let (sender, pokes) = mpsc::channel::<Poke>(4);
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        let mid_tool = Mutex::new(Some(sender));
        let events = Mutex::new(Vec::new());
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::ToolStart { name: "Bash", .. })
                && let Some(mut sender) = mid_tool.lock().unwrap().take()
            {
                sender
                    .try_send(Poke::Steer("redirect".to_string()))
                    .expect("four slots");
            }
            events.lock().unwrap().push(progress);
        }));

        assert!(
            matches!(outcome, Progress::Done { message: Some(text) } if text == "after the steer")
        );
        // The record reads: user, the tool round, the steered input, the final
        // answer (whichever select arm won).
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().unwrap();
        let position = |probe: &dyn Fn(&serde_json::Value) -> bool| {
            items.iter().position(probe).expect("the item recorded")
        };
        let tool = position(&|item| item["type"] == "function_call");
        let steer = position(&|item| item["role"] == "user" && item["content"] == "redirect");
        let answer = position(&|item| item["role"] == "assistant");
        assert!(tool < steer && steer < answer);
        assert!(events.lock().unwrap().iter().any(|event| matches!(
            event,
            Progress::Steered(text) if text == "redirect"
        )));
    }

    #[test]
    fn an_esc_queued_behind_a_mid_tool_steer_still_wins() {
        let agent = test_agent();
        let open = scripted(vec![
            ends(vec![item(json!({
                "type": "response.output_item.done",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "function_call", "id": "fc_1", "call_id": "c1", "name": "bash",
                    "arguments": "{\"command\":\"echo tool ran\"}", "status": "completed"
                }
            }))]),
            // Never reached: the cancel wins before another round opens.
            ends(vec![delta("unreachable")]),
        ]);
        let (sender, pokes) = mpsc::channel::<Poke>(4);
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        let mid_tool = Mutex::new(Some(sender));
        let events = Mutex::new(Vec::new());
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::ToolStart { name: "Bash", .. })
                && let Some(mut sender) = mid_tool.lock().unwrap().take()
            {
                sender
                    .try_send(Poke::Steer("redirect".to_string()))
                    .expect("four slots");
                sender.try_send(Poke::Cancel).expect("four slots");
            }
            events.lock().unwrap().push(progress);
        }));

        assert!(matches!(outcome, Progress::Cancelled));
        // The retracted steer was neither recorded nor echoed; the killed
        // call is kept, as a cancel always keeps what ran.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().unwrap();
        assert!(
            !items
                .iter()
                .any(|item| item["role"] == "user" && item["content"] == "redirect")
        );
        assert!(items.iter().any(|item| item["type"] == "function_call"));
        assert!(!events.lock().unwrap().iter().any(|event| matches!(
            event,
            Progress::Steered(text) if text == "redirect"
        )));
    }

    /// A poked cancel mid-stream saves the partial answer and stops.
    #[test]
    fn a_poked_cancel_saves_the_partial_answer() {
        let agent = test_agent();
        let open = scripted(vec![stays_open(vec![delta("partial answer")])]);
        let (sender, pokes) = mpsc::channel::<Poke>(4);
        let transcript = Transcript::new().unwrap();
        // Esc lands after the first delta: the partial answer is kept.
        let mid_stream = Mutex::new(Some(sender));
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::Answer(delta) if delta == "partial answer")
                && let Some(mut sender) = mid_stream.lock().unwrap().take()
            {
                sender.try_send(Poke::Cancel).expect("four slots");
            }
        }));

        assert!(matches!(outcome, Progress::Cancelled));
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items[1]["role"], "assistant");
        assert_eq!(items[1]["content"], "partial answer");
    }

    /// The truncation rules port verbatim: a stream that ends mid tool call
    /// or delivers nothing fails the turn.
    #[test]
    fn truncated_streams_fail_the_turn() {
        let agent = test_agent();
        let open = scripted(vec![ends(vec![item(json!({
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "function_call", "id": "fc_1", "call_id": "c1", "name": "bash",
                "arguments": "", "status": "in_progress"
            }
        }))])]);
        let (_sender, pokes) = mpsc::channel::<Poke>(4);
        let outcome = block_on(agent.run(&Transcript::new().unwrap(), pokes, open, &|_| {}));
        assert!(matches!(outcome, Progress::Failed(reason) if reason.contains("mid tool call")));

        let agent = test_agent();
        let open = scripted(vec![ends(Vec::new())]);
        let (_sender, pokes) = mpsc::channel::<Poke>(4);
        let outcome = block_on(agent.run(&Transcript::new().unwrap(), pokes, open, &|_| {}));
        assert!(matches!(outcome, Progress::Failed(reason) if reason.contains("without output")));
    }

    /// The reasoning item is captured and replayed before its round's call
    /// exchanges: thinking-mode providers 400 without it.
    #[test]
    fn reasoning_replays_before_its_rounds_calls() {
        let agent = test_agent();
        let open = scripted(vec![
            ends(vec![
                item(json!({
                    "type": "response.output_item.done",
                    "sequence_number": 2,
                    "output_index": 0,
                    "item": {"type": "reasoning", "id": "rs_1", "summary": []}
                })),
                item(json!({
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 1,
                    "item": {
                        "type": "function_call", "id": "fc_1", "call_id": "c1", "name": "bash",
                        "arguments": "{\"command\":\"true\"}", "status": "completed"
                    }
                })),
            ]),
            ends(vec![delta("done now")]),
        ]);
        let (_sender, pokes) = mpsc::channel::<Poke>(4);
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        let outcome = block_on(agent.run(&transcript, pokes, open, &|_| {}));

        assert!(matches!(outcome, Progress::Done { .. }));
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let kinds: Vec<&str> = items
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                item["type"]
                    .as_str()
                    .unwrap_or(item["role"].as_str().unwrap_or(""))
            })
            .collect();
        let reasoning_at = kinds.iter().position(|k| *k == "reasoning").unwrap();
        let call_at = kinds.iter().position(|k| *k == "function_call").unwrap();
        assert!(
            reasoning_at < call_at,
            "reasoning must precede its calls: {kinds:?}"
        );
    }

    /// One canned stream item from JSON: the wire shape the provider sends.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the stream's items are Results; canned ones are always Ok"
    )]
    fn item(json: serde_json::Value) -> StreamItem {
        Ok(serde_json::from_value::<ResponseStreamEvent>(json).unwrap())
    }

    /// A text delta event carrying `text`.
    fn delta(text: &str) -> StreamItem {
        item(json!({
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "item_0",
            "output_index": 0,
            "content_index": 0,
            "delta": text
        }))
    }

    /// One scripted round: its events, and whether the stream stays open
    /// after them (a mid-stream poke test needs the poke as its only wake).
    struct Script {
        items: Vec<StreamItem>,
        stay_open: bool,
    }

    /// A round that ends with its items: the stream's natural end.
    fn ends(items: Vec<StreamItem>) -> Script {
        Script { items, stay_open: false }
    }

    /// A round whose stream stays open after its items.
    fn stays_open(items: Vec<StreamItem>) -> Script {
        Script { items, stay_open: true }
    }

    /// An agent pointed at nothing, for driving `run` with a scripted opener.
    fn test_agent() -> Agent {
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        Agent::new("http://localhost:9", "key", "model", policy)
    }

    /// An opener that replays canned rounds, one stream per round.
    fn scripted(rounds: Vec<Script>) -> impl Fn(CreateResponse) -> OpenStreamFuture {
        use std::sync::Mutex;

        let rounds = Mutex::new(rounds.into_iter().rev().collect::<Vec<_>>());
        move |request: CreateResponse| -> OpenStreamFuture {
            let _ = request;
            let script = rounds.lock().unwrap().pop().unwrap_or_else(|| ends(Vec::new()));
            Box::pin(async move {
                let items = futures::stream::iter(script.items).boxed();
                let stream: ResponseStream = if script.stay_open {
                    items.chain(futures::stream::pending().boxed()).boxed()
                } else {
                    items
                };
                Ok::<ResponseStream, OpenAIError>(stream)
            })
        }
    }
}
