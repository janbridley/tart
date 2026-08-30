use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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
use futures::executor::block_on;
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
    /// The front end's lever on the running turn (cancel + steer).
    control: TurnControl,
}

/// One interrupt the front end can send a running turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poke {
    /// Esc: save the partial answer, if any, and stop.
    Cancel,
    /// A steering message: save the partial, record this text, restart the round.
    Steer(String),
}

/// The front end's control plane for running turns.
#[derive(Clone, Default)]
pub struct TurnControl {
    state: Arc<Mutex<TurnState>>,
}

/// The turn control state, under one lock.
#[derive(Default)]
struct TurnState {
    /// Which turn owns the lever; a finishing worker retires only its own.
    generation: u64,
    /// The running turn's poke sender, where interrupts reach it. `None` at idle
    sender: Option<mpsc::Sender<Poke>>,
    /// Message queue and effective label for the frontend to observe what's waiting.
    steer: Option<String>,
}

impl TurnControl {
    /// The state under its lock.
    fn state(&self) -> MutexGuard<'_, TurnState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Cancel the turn by dropping the stream, keeping any partial answer.
    ///
    /// NOTE: this must wait until tool calls complete, and the provider will likely
    /// take the cancelled turn to completion.
    #[inline]
    pub fn cancel(&self) {
        let mut state = self.state();
        if let Some(sender) = state.sender.as_mut() {
            // A failed send means a poke is already queued; a pending Cancel
            // says everything a second one would.
            let _ = sender.try_send(Poke::Cancel);
        }
    }

    /// Queue `text` to interrupt and redirect the turn.
    ///
    /// One message waits at a time: `false` when one already does, so the
    /// caller keeps its draft (Option+Up edits the queued message instead).
    #[must_use = "the caller keeps its draft when the slot is taken"]
    #[inline]
    pub fn steer(&self, text: String) -> bool {
        let mut state = self.state();
        if state.steer.is_some() {
            return false;
        }
        if let Some(sender) = state.sender.as_mut() {
            // A failed send means a poke is already queued behind this text.
            let _ = sender.try_send(Poke::Steer(text.clone()));
        }
        state.steer = Some(text);
        true
    }

    /// A copy of the waiting steering message, if any.
    #[inline]
    pub fn steering(&self) -> Option<String> {
        self.state().steer.clone()
    }

    /// Take the waiting steering message, if any.
    #[inline]
    pub fn take_steer(&self) -> Option<String> {
        self.state().steer.take()
    }

    /// Install `sender` as the next turn's lever, forgetting the last turn's
    /// intents, and report the turn's id.
    fn claim(&self, sender: mpsc::Sender<Poke>) -> u64 {
        let mut state = self.state();
        state.generation += 1;
        state.steer = None;
        state.sender = Some(sender);
        state.generation
    }

    /// Retire the lever, unless a newer turn already claimed it. A steering
    /// message survives: the front end reads it when the terminal event lands.
    fn release(&self, generation: u64) {
        let mut state = self.state();
        if state.generation == generation {
            state.sender = None;
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
            control: TurnControl::default(),
        }
    }

    /// The session's collaboration mode.
    #[inline]
    pub fn mode(&self) -> ChatMode {
        self.mode
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

    /// The front end's lever on the running turn, for the pane to hold.
    #[inline]
    pub fn control(&self) -> TurnControl {
        self.control.clone()
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

    /// Run one generation on its own thread, reporting progress to `on_progress`.
    ///
    /// The worker records its turns (reasoning, tool exchanges, final answer) into the
    /// shared transcript as it goes. Exactly one terminal event ([`Progress::Done`],
    /// [`Progress::Failed`], or [`Progress::Cancelled`]) is delivered, even if the
    /// worker panics.
    #[inline]
    pub fn spawn<F: Fn(Progress) + Send + 'static>(&self, transcript: &Transcript, on_progress: F) {
        let agent = self.clone();
        let transcript = transcript.clone();
        // This turn's poke channel: the sender waits where Esc and steering
        // reach it, and the receiver races the stream inside the worker.
        // Four slots, so a Cancel can queue behind a Steer.
        let (sender, pokes) = mpsc::channel(4);
        let generation = self.control.claim(sender);
        std::thread::spawn(move || {
            // One provider call per round, from the agent's collection pool.
            let open_client = agent.client.clone();
            let open = move |request: CreateResponse| -> OpenStreamFuture {
                let client = open_client.clone();
                Box::pin(async move { client.responses().create_stream(request).await })
            };
            // A panicking worker must still deliver the terminal event to the caller
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                block_on(agent.run(&transcript, pokes, open, &on_progress))
            }));
            // The turn is over: retire the lever unless a newer turn claimed it.
            agent.control.release(generation);
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
    /// Blocks the current thread until the generation finishes.
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
        O: Fn(CreateResponse) -> OpenStreamFuture,
        F: Fn(Progress),
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
                        futures::future::pending::<Option<Poke>>().boxed_local().fuse()
                    } else {
                        pokes.next().boxed_local().fuse()
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
                // Pokes that arrived mid-tools stop the loop the way Esc
                // between calls always has; the one in flight finishes first.
                let (cancel, steer) = drain_pokes(&mut pokes);
                if cancel || cancelled {
                    cancelled = true;
                    break;
                }
                if let Some(text) = steer {
                    steered_text = Some(text);
                    break;
                }
                match tools::execute(&call, self.policy(), on_progress) {
                    Ok(output) => exchanges.push((call, output)),
                    Err(error) => {
                        exchanges.push((call, format!("error: {error}")));
                        failure = Some(error.to_string());
                        break;
                    }
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
    use serde_json::json;
    use std::cell::RefCell;

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
        assert_eq!(agent.mode(), ChatMode::Default);
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
    fn turn_control_intents_are_generation_owned() {
        let control = TurnControl::default();

        // One message waits at a time: a second submission is refused.
        assert!(control.steer("first".to_string()));
        assert!(!control.steer("second".to_string()));
        assert_eq!(control.steering(), Some("first".to_string()));
        assert_eq!(control.take_steer(), Some("first".to_string()));

        // Claiming resets the slot and installs the poke sender.
        let (sender, mut pokes) = mpsc::channel(4);
        let first = control.claim(sender);
        assert_eq!(control.steering(), None);
        assert_eq!(drain_pokes(&mut pokes), (false, None));

        // Cancel sends a typed poke the worker drains.
        control.cancel();
        assert_eq!(drain_pokes(&mut pokes), (true, None));

        // A steer delivers its text in the message; the slot mirrors it.
        assert!(control.steer("redirect".to_string()));
        assert_eq!(control.steering(), Some("redirect".to_string()));
        assert_eq!(drain_pokes(&mut pokes), (false, Some("redirect".to_string())));
        // The pane clears the mirror when the steer is echoed.
        assert_eq!(control.take_steer().as_deref(), Some("redirect"));

        // An Esc queued behind a steer still wins the drain.
        assert!(control.steer("second thought".to_string()));
        control.cancel();
        assert_eq!(
            drain_pokes(&mut pokes),
            (true, Some("second thought".to_string()))
        );

        // A stale release must not retire the newer turn's lever.
        let (sender, _) = mpsc::channel(4);
        let second = control.claim(sender);
        control.release(first);

        // Retiring the right turn clears its intents but not a later steer.
        control.release(second);
        assert!(control.steer("survives".to_string()));
        assert_eq!(control.take_steer(), Some("survives".to_string()));
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
        let mid_stream = RefCell::new(Some(sender));
        let events = RefCell::new(Vec::new());
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::Answer(delta) if delta == "partial ")
                && let Some(mut sender) = mid_stream.borrow_mut().take()
            {
                sender
                    .try_send(Poke::Steer("redirect".to_string()))
                    .expect("four slots");
            }
            events.borrow_mut().push(progress);
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
        assert!(events.borrow().iter().any(|event| matches!(
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
        let mid_stream = RefCell::new(Some(sender));
        let outcome = block_on(agent.run(&transcript, pokes, open, &|progress| {
            if matches!(&progress, Progress::Answer(delta) if delta == "partial answer")
                && let Some(mut sender) = mid_stream.borrow_mut().take()
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
