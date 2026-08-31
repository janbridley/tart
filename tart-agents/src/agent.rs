use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponseArgs, FunctionToolCall, InputParam, OutputItem, Reasoning, ReasoningEffort,
        ReasoningItem, ResponseStreamEvent, Tool,
    },
};
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::{Either, select};
use tokio::runtime::Runtime;

use crate::{
    Agents, CancelToken, MAX_TOOL_ROUNDS, Progress, Transcript, debug, sandbox::Policy, tools,
};

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
    /// The session's mode; picks the active policy and tool list.
    mode: ChatMode,
    /// The Default-mode policy: the granted roots stay writable.
    writable: Policy,
    /// The shared runtime every agent and turn drives its futures on.
    runtime: Arc<Runtime>,
    /// The subagent registry, which also arms the subagent tools; `None` for
    /// subagents themselves.
    subagents: Option<Arc<Agents>>,
    /// The front end's lever on the running turn (cancel).
    control: TurnHandle,
}

/// The front end's control plane for running turns.
#[derive(Clone, Default)]
pub struct TurnHandle {
    state: Arc<Mutex<TurnState>>,
}

/// The turn control state, under one lock.
#[derive(Default)]
struct TurnState {
    /// Which turn owns the lever; a finishing worker retires only its own.
    generation: u64,
    /// The running turn's wake sender, where pokes reach it.
    sender: Option<mpsc::Sender<()>>,
    /// Esc was pressed and we should attempt to cancel.
    cancelled: bool,
    /// The running turn's command lever: cancelling kills a bash in flight.
    token: CancelToken,
}

impl TurnHandle {
    /// The state under its lock.
    fn state(&self) -> MutexGuard<'_, TurnState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Cancel the turn: kill any command it is running, drop the stream, and
    /// keep any partial answer.
    ///
    /// The provider will likely still take its side of the cancelled stream
    /// to completion; the record here ends at the drop.
    #[inline]
    pub fn cancel(&self) {
        let mut state = self.state();
        state.cancelled = true;
        state.token.cancel();
        if let Some(sender) = &mut state.sender {
            // A failed poke means one is already pending; the flag decides.
            let _ = sender.try_send(());
        }
    }

    /// Install `sender` as the next turn's lever, forgetting the last turn's
    /// cancel, and report the turn's id with the turn's fresh command lever.
    fn claim(&self, sender: mpsc::Sender<()>) -> (u64, CancelToken) {
        let mut state = self.state();
        state.generation += 1;
        state.cancelled = false;
        state.sender = Some(sender);
        state.token = CancelToken::new();
        (state.generation, state.token.clone())
    }

    /// Retire the lever, unless a newer turn already claimed it.
    fn release(&self, generation: u64) {
        let mut state = self.state();
        if state.generation == generation {
            state.sender = None;
            state.cancelled = false;
        }
    }

    /// Whether the front end cancelled the turn: pokes only wake a parked wait
    fn cancelled(&self, cancel_rx: &mut mpsc::Receiver<()>) -> bool {
        while cancel_rx.try_recv().is_ok() {}
        self.state().cancelled
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
        Self {
            client: Client::with_config(config),
            model: model.into(),
            effort: None,
            max_rounds: MAX_TOOL_ROUNDS,
            mode: ChatMode::Default,
            writable: policy,
            runtime: Arc::new(Runtime::new().expect("tokio runtime did not start")),
            subagents: None,
            control: TurnHandle::default(),
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
    fn policy(&self) -> Policy {
        match self.mode {
            ChatMode::Default => self.writable.clone(),
            ChatMode::Plan => self.writable.clone().read_only(),
        }
    }

    /// The tools this agent offers the model. Plan mode withholds the `edit` tool.
    fn tools_for(&self) -> Vec<Tool> {
        [tools::bash(), tools::read()]
            .into_iter()
            .chain((self.mode == ChatMode::Default).then_some(tools::edit()))
            .chain([tools::search(), tools::fetch()].into_iter().flatten())
            .chain(
                self.subagents
                    .iter()
                    .flat_map(|_| [tools::spawn(), tools::wait()]),
            )
            .collect()
    }

    /// A child agent: this agent's model and policy, with no subagent recursion.
    pub(crate) fn child(&self) -> Agent {
        let mut child = self.clone();
        child.control = TurnHandle::default();
        child.subagents = None;
        child
    }

    /// Arm the subagent tools with a registry, for the front end's agent.
    #[inline]
    pub fn set_subagents(&mut self, agents: Arc<Agents>) {
        self.subagents = Some(agents);
    }

    /// The front end's lever on the running turn, for the pane to hold.
    #[inline]
    pub fn handle(&self) -> TurnHandle {
        self.control.clone()
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
        // This turn's wake channel: the sender waits where Esc can reach it, and
        // the receiver races the stream inside the worker.
        let (esc_sender, receiver) = mpsc::channel(1);
        let (generation, token) = self.control.claim(esc_sender);
        std::thread::spawn(move || {
            // A panicking worker must still deliver the terminal event to the caller
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                agent.run(&transcript, receiver, &token, &on_progress);
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
    /// A message the front end queues mid-turn is not the worker's concern:
    /// queueing cancels the turn, and the front end starts the next turn on
    /// the queued message when this one ends. The record keeps its shape
    /// either way: the partial answer and the calls that ran, then the
    /// queued message as the next user item.
    ///
    /// A stream that closes without a terminal event still ends its round with whatever
    /// the model produced.
    ///
    /// Blocks the current thread until the generation finishes.
    #[allow(
        clippy::too_many_lines,
        reason = "the round loop reads best as one straight-line function"
    )]
    fn run<F: Fn(Progress)>(
        &self,
        transcript: &Transcript,
        mut cancel_rx: mpsc::Receiver<()>,
        token: &CancelToken,
        on_progress: &F,
    ) {
        // Completed rounds so far: a dropped stream retries its round without
        // counting it, so a flaky connection cannot burn the budget.
        let mut round = 0;
        // Consecutive dropped streams; any completed round resets it.
        let mut retries = 0;
        while round < self.max_rounds {
            // A cancelled generation stops before spending another request.
            if self.control.cancelled(&mut cancel_rx) {
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            // The sandboxed trio less `edit` in plan mode, plus web tools if
            // available, plus the subagent pair for a spawning agent.
            let definitions = self.tools_for();
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

            // The owned runtime drives the future
            let mut stream = match self
                .runtime
                .block_on(self.client.responses().create_stream(request))
            {
                Ok(stream) => stream,
                Err(error) => {
                    // A request that never opened dropped the round before recording,
                    // so we can retry it
                    let reason = error.to_string();
                    if retry_dropped(on_progress, &reason, &mut retries) {
                        continue;
                    }
                    return terminate_and_log(on_progress, Progress::Failed(reason));
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

            // Race the stream against Esc pokes; a won cancel returns and
            // drops the stream, keeping whatever it streamed
            loop {
                let item = match self.runtime.block_on(select(stream.next(), cancel_rx.next())) {
                    Either::Right(_) => {
                        // A stale poke only wakes the wait: the flag decides.
                        if !self.control.cancelled(&mut cancel_rx) {
                            continue;
                        }
                        // Esc won: dropping the stream keeps what it streamed.
                        if let Err(error) = record_answer(transcript, &answer) {
                            return terminate_and_log(
                                on_progress,
                                Progress::Failed(error.to_string()),
                            );
                        }
                        return terminate_and_log(on_progress, Progress::Cancelled);
                    }
                    Either::Left((item, _)) => item,
                };
                let Some(item) = item else { break };
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

            // A stream that broke before completing its round dropped it -> retry.
            if call_in_flight || !saw_output {
                let reason = with_last_error(
                    if call_in_flight {
                        "stream ended mid tool call"
                    } else {
                        "stream ended without output"
                    },
                    last_error,
                );
                if retry_dropped(on_progress, &reason, &mut retries) {
                    continue;
                }
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }

            // No calls pending: this round's answer is the turn's message.
            if calls.is_empty() {
                if let Err(error) = record_answer(transcript, &answer) {
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
            let policy = self.policy();
            let tooling = tools::Tooling {
                policy: &policy,
                cancel: token,
                agents: self.subagents.as_deref(),
                template: self,
            };
            let mut exchanges = Vec::with_capacity(calls.len());
            // A call the harness cannot run ends the round, but is recorded with its
            // error so the round's earlier effects stay in the transcript.
            let mut failure: Option<String> = None;
            for call in calls {
                // A cancelled turn skips its remaining calls; the one in
                // flight finishes first.
                if self.control.cancelled(&mut cancel_rx) {
                    break;
                }
                match tools::execute(&call, &tooling, on_progress) {
                    Ok(output) => {
                        let bounded = bounded_for_history(&call.name, &output, on_progress);
                        exchanges.push((call, bounded));
                    }
                    Err(error) => {
                        let bounded = bounded_for_history(
                            &call.name,
                            &format!("error: {error}"),
                            on_progress,
                        );
                        exchanges.push((call, bounded));
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
            if let Err(error) = record_answer(transcript, &answer) {
                return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
            }
            transcript.push_tool_round(exchanges);
            if self.control.cancelled(&mut cancel_rx) {
                // The turn stops here, keeping the rounds recorded so far.
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            if let Some(reason) = failure {
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }
            // The round completed and is recorded; the next one starts fresh.
            round += 1;
            retries = 0;
        }
        terminate_and_log(
            on_progress,
            Progress::Failed(format!("gave up after {round} tool rounds")),
        );
    }
}

fn with_last_error(message: &str, last_error: Option<String>) -> String {
    last_error.map_or_else(|| message.to_string(), |error| format!("{message}: {error}"))
}

/// The most consecutive rounds a dropped stream retries before failing.
const MAX_STREAM_RETRIES: usize = 4;

/// The pause before the first retry; it doubles on each further one
/// (2s, 4s, 8s, 16s).
#[cfg(not(test))]
const RETRY_PAUSE: Duration = Duration::from_secs(2);
/// A test's pause is nothing, so budget tests run in milliseconds.
#[cfg(test)]
const RETRY_PAUSE: Duration = Duration::from_millis(1);

/// Count one dropped round, reporting the retry and pausing while one remains.
///
/// Returns whether the round should re-run. `false` means the caller fails the
/// turn with the reason. A completed round resets the count, so four *consecutive*
/// drops end the turn.
fn retry_dropped<F: Fn(Progress)>(on_progress: &F, reason: &str, retries: &mut usize) -> bool {
    *retries += 1;
    if *retries > MAX_STREAM_RETRIES {
        return false;
    }
    on_progress(Progress::Note(format!(
        "{reason}; retry {retries}/{MAX_STREAM_RETRIES}"
    )));
    std::thread::sleep(RETRY_PAUSE * (1u32 << (*retries - 1)));
    true
}

/// Bound one tool result before it enters history.
fn bounded_for_history<F: Fn(Progress)>(name: &str, output: &str, on_progress: &F) -> String {
    let capped = tools::bounded(output, tools::CONTENT_CAP);
    if output.len() > tools::CONTENT_CAP {
        on_progress(Progress::Note(format!(
            "{name} output truncated to {} KiB",
            tools::CONTENT_CAP / 1024
        )));
    }
    capped
}

/// Record the answer a round streamed, when it streamed one at all.
fn record_answer(transcript: &Transcript, answer: &str) -> anyhow::Result<()> {
    if answer.is_empty() {
        return Ok(());
    }
    transcript.push_assistant(answer.to_string())
}

/// Deliver the generation's terminal event, mirroring its outcome to the debug lob.
fn terminate_and_log<F: Fn(Progress)>(on_progress: &F, event: Progress) {
    debug::log("generation outcome", || match &event {
        Progress::Done { message } => {
            format!("done ({} answer chars)", message.as_deref().map_or(0, str::len))
        }
        Progress::Failed(reason) => format!("failed: {reason}"),
        Progress::Cancelled => "cancelled".to_string(),
        _ => "not a terminal event".to_string(),
    });
    on_progress(event);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use async_openai::types::responses::ResponseTextDeltaEvent;
    use std::io::{Read, Write};

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

    /// Every turn worker blocks onto the agent's runtime from its own thread:
    /// eight concurrent `block_on`s on the one runtime must all come back.
    #[test]
    fn the_runtime_blocks_eight_concurrent_threads() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let agent = Agent::new("http://localhost:9", "key", "model", policy);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let runtime = Arc::clone(&agent.runtime);
                std::thread::spawn(move || {
                    // The `async` block defers the timer's construction to
                    // inside the runtime context `block_on` provides.
                    runtime.block_on(async {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    });
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("a blocked thread comes back");
        }
    }

    /// Plan mode runs under the read-only twin of the Default policy and witholds
    /// the `edit` tool.
    #[test]
    fn plan_mode_is_read_only_and_drops_edit() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let mut agent = Agent::new("http://localhost:9", "key", "model", policy);
        agent.set_subagents(Arc::new(Agents::new(|_, _| ())));

        // Default: the granted roots are writable, and `edit` is offered alongside the
        // subagent pair only a spawning agent gets.
        assert_eq!(agent.mode(), ChatMode::Default);
        assert!(!agent.policy().writable_roots().is_empty());
        assert!(names(&agent.tools_for()).contains(&"edit"));
        assert!(names(&agent.tools_for()).contains(&"spawn"));
        assert!(names(&agent.tools_for()).contains(&"wait"));

        // Plan: the workspace is not writable, the temp scratch alone is, and not edit
        agent.set_mode(ChatMode::Plan);
        let plan = agent.policy();
        let writable = plan.writable_roots();
        assert_eq!(writable.len(), 1, "exactly the scratch root: {writable:?}");
        let scratch =
            std::fs::canonicalize(std::env::temp_dir()).expect("the temp dir canonicalizes");
        assert_eq!(writable[0], scratch);
        let plan_tools = agent.tools_for();
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

    /// Claiming installs the wake sender and retires the last turn's cancel;
    /// a stale release must not retire a newer turn's lever.
    #[test]
    fn turn_control_cancel_is_generation_owned() {
        let control = TurnHandle::default();

        // Claiming resets the cancel and installs the wake sender.
        let (sender, mut rx) = mpsc::channel(1);
        let (first, _) = control.claim(sender);
        assert!(!control.cancelled(&mut rx));

        // Cancel pokes the claimed sender; the drained flag decides.
        control.cancel();
        assert!(control.cancelled(&mut rx));

        // A stale release must not retire the newer turn's lever.
        let (sender, _) = mpsc::channel(1);
        let (second, _) = control.claim(sender);
        control.release(first);
        control.cancel();
        assert!(control.cancelled(&mut rx));

        // Retiring the right turn clears its cancel for the next one.
        control.release(second);
        assert!(!control.cancelled(&mut rx));
    }

    /// Cancelling a turn kills the commands of the turn that claimed the lever,
    /// and only those: the next claim mints a fresh command lever.
    #[test]
    fn cancel_kills_the_claims_commands_and_no_one_elses() {
        let control = TurnHandle::default();

        let (sender, _) = mpsc::channel(1);
        let (_, first) = control.claim(sender);
        control.cancel();
        assert!(first.cancelled());

        // A fresh claim is not born cancelled: the prior cancel died with its turn.
        let (sender, _) = mpsc::channel(1);
        let (_, second) = control.claim(sender);
        assert!(!second.cancelled());
    }

    /// A cancel mid-stream keeps the partial answer and ends the turn
    /// [`Progress::Cancelled`], with nothing recorded after it: what an
    /// interrupted turn leaves behind for the front end to requeue onto.
    #[test]
    fn cancel_mid_stream_keeps_the_partial_answer() {
        let Some(listener) = loopback() else { return };
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let address = listener.local_addr().expect("a bound address");
        let agent = Agent::new(format!("http://{address}"), "key", "model", policy);
        let transcript = Transcript::new().expect("a transcript opens");
        transcript
            .push_user("tell me a story".to_string())
            .expect("the user turn records");

        // One round streams a partial answer, then parks: the stream never
        // ends on its own, so only a cancel can end the turn.
        let server = serve(listener, vec![delta("Once upon"), delta(" a time")]);

        // Esc lands once both fragments have streamed, so the cancel wakes
        // the parked select with the partial already in hand.
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let interrupt = {
            let log = Arc::clone(&log);
            let handle = agent.handle();
            std::thread::spawn(move || {
                while !log.lock().unwrap().iter().any(|entry| entry.contains(" a time")) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                handle.cancel();
            })
        };

        // Drive the generation as `spawn` would: claim, run, retire.
        let (sender, receiver) = mpsc::channel(1);
        let handle = agent.handle();
        let (generation, token) = handle.claim(sender);
        agent.run(&transcript, receiver, &token, &|progress| {
            log.lock().unwrap().push(format!("{progress:?}"));
        });
        handle.release(generation);
        interrupt.join().expect("the interrupter exits");

        // The turn ends with exactly one terminal event: the cancel.
        let log = log.lock().unwrap();
        assert_eq!(log.last().map(String::as_str), Some("Cancelled"));
        assert_eq!(
            log.iter().filter(|entry| entry.as_str() == "Cancelled").count(),
            1
        );

        // The partial answer is recorded: the user turn, then it, then nothing.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().map(Vec::len), Some(3));
        assert_eq!(items[2]["content"], "Once upon a time");

        server.join().expect("the server exits");
    }

    /// The retry budget: each drop reports its retry, and the fifth
    /// consecutive one ends the turn instead.
    #[test]
    fn four_consecutive_drops_then_give_up() {
        let notes = Mutex::new(Vec::new());
        let mut retries = 0;
        // Keep dropping until `retry_dropped` stops offering a retry.
        while retry_dropped(
            &|progress| {
                if let Progress::Note(text) = progress {
                    notes.lock().unwrap().push(text);
                }
            },
            "stream ended without output",
            &mut retries,
        ) {}

        let notes = notes.lock().unwrap();
        assert_eq!(
            *notes,
            vec![
                "stream ended without output; retry 1/4",
                "stream ended without output; retry 2/4",
                "stream ended without output; retry 3/4",
                "stream ended without output; retry 4/4",
            ],
        );
        // The count a completed round resets.
        assert_eq!(retries, 5);
    }

    /// A stream that delivers nothing and closes drops its round; the retry
    /// re-sends the request verbatim and the turn completes on the second one.
    #[test]
    fn a_dropped_stream_retries_its_round() {
        let Some(listener) = loopback() else { return };
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let address = listener.local_addr().expect("a bound address");
        let agent = Agent::new(format!("http://{address}"), "key", "model", policy);
        let transcript = Transcript::new().expect("a transcript opens");
        transcript
            .push_user("tell me a story".to_string())
            .expect("the user turn records");

        // The first request answers with an empty stream that closes at once;
        // the retried request gets the complete one, so only the retry ends
        // the turn.
        let server = std::thread::spawn(move || {
            let head = "HTTP/1.1 200 OK\r\n\
                        content-type: text/event-stream\r\n\
                        connection: close\r\n\r\n";
            let (mut stream, _) = listener.accept().expect("the first client arrives");
            read_request(&mut stream);
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.flush();
            drop(stream);
            let (mut stream, _) = listener.accept().expect("the retried client arrives");
            read_request(&mut stream);
            let _ = stream.write_all(head.as_bytes());
            let event = serde_json::to_string(&delta("Once upon a time")).expect("serializes");
            let _ = stream.write_all(format!("data: {event}\n\n").as_bytes());
            let _ = stream.flush();
        });

        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let (sender, receiver) = mpsc::channel(1);
        let handle = agent.handle();
        let (generation, token) = handle.claim(sender);
        agent.run(&transcript, receiver, &token, &|progress| {
            log.lock().unwrap().push(format!("{progress:?}"));
        });
        handle.release(generation);
        server.join().expect("the server exits");

        let log = log.lock().unwrap();
        // The stall is visible as a note, and the retry completes the turn.
        assert!(
            log.iter().any(|entry| entry.contains("retry 1/4")),
            "the retry is noted: {log:?}"
        );
        assert_eq!(log.last().map(|entry| entry.starts_with("Done")), Some(true));

        // The dropped round recorded nothing: the answer is the only
        // assistant item, exactly as a clean single-round turn would leave.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        assert_eq!(items.as_array().map(Vec::len), Some(3));
        assert_eq!(items[2]["content"], "Once upon a time");
    }

    #[test]
    fn spawned_subagents_run_and_report() {
        let Some(listener) = loopback() else { return };
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let address = listener.local_addr().expect("a bound address");
        let agent = Agent::new(format!("http://{address}"), "key", "model", policy);
        let events = Arc::new(Mutex::new(Vec::<(crate::AgentId, Progress)>::new()));
        let log = Arc::clone(&events);
        let agents = crate::Agents::new(move |id, progress| {
            log.lock().unwrap().push((id, progress));
        });

        // The child's one round streams an answer, then the stream closes:
        // the turn ends on its own, no cancel needed.
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("the child connects");
            read_request(&mut stream);
            let head = "HTTP/1.1 200 OK\r\n\
                        content-type: text/event-stream\r\n\
                        connection: close\r\n\r\n";
            let _ = stream.write_all(head.as_bytes());
            let event = serde_json::to_string(&delta("found it")).expect("serializes");
            let _ = stream.write_all(format!("data: {event}\n\n").as_bytes());
            let _ = stream.flush();
        });

        let id = agents
            .spawn(&agent, "find the flaky test")
            .expect("the registry has room");
        let outcome = agents
            .wait(id, Duration::from_secs(5), &CancelToken::new())
            .expect("the child is registered");
        assert_eq!(outcome, Some(crate::Outcome::Done(Some("found it".to_string()))));
        server.join().expect("the server exits");

        // The box opener precedes every child event.
        let events = events.lock().unwrap();
        assert!(
            matches!(events.first(), Some((first, Progress::ToolStart { name, .. })) if *first == id && name == "agent"),
            "{events:?}"
        );
        assert_eq!(events.len(), 3, "opener, one answer delta, terminal: {events:?}");

        // The delivered report is gone: one delivery, and the slot is freed.
        assert_eq!(agents.take_outcome(id), None);
        assert!(
            agents
                .wait(id, Duration::from_secs(1), &CancelToken::new())
                .is_err()
        );
    }

    #[test]
    fn reports_deliver_once_and_free_their_slots() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let agent = Agent::new("http://localhost:9", "key", "model", policy);
        let agents = crate::Agents::new(|_, _| {});

        let mut ids = Vec::new();
        for _ in 0..crate::MAX_SUBAGENTS {
            ids.push(agents.spawn(&agent, "task").expect("a slot is free"));
        }
        assert!(agents.spawn(&agent, "task").is_err(), "the slots are full");

        // The unreachable endpoint fails each child fast (a test's retry
        // pause is 1ms); waiting delivers the report and prunes the child.
        let token = CancelToken::new();
        for id in ids {
            let outcome = agents
                .wait(id, Duration::from_secs(30), &token)
                .expect("the child ends");
            assert!(outcome.is_some(), "a failed child still reports: {outcome:?}");
            assert_eq!(agents.take_outcome(id), None, "one delivery only");
        }
        assert!(agents.running().is_empty());
        agents
            .spawn(&agent, "task")
            .expect("the slots freed with the reports");
    }

    /// The loopback listener the worker test streams from, or `None` where denied
    fn loopback() -> Option<std::net::TcpListener> {
        match std::net::TcpListener::bind(("127.0.0.1", 0)) {
            Ok(listener) => Some(listener),
            Err(error) => {
                let current = std::thread::current();
                let name = current.name().unwrap_or("<unnamed>");
                let mut stderr = std::io::stderr().lock();
                let _ = writeln!(
                    stderr,
                    "note: skipping live test {name} (needs loopback the sandbox denies): {error}"
                );
                None
            }
        }
    }

    /// An output-text delta event carrying `text`.
    fn delta(text: &str) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
            sequence_number: 0,
            item_id: "item_0".to_string(),
            output_index: 0,
            content_index: 0,
            delta: text.to_string(),
            logprobs: None,
        })
    }

    /// Answer one `/responses` request with `events` as `text/event-stream`,
    /// then park until the client drops the connection: the stream never ends
    /// on its own, so a test decides when the turn ends.
    fn serve(
        listener: std::net::TcpListener,
        events: Vec<ResponseStreamEvent>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            read_request(&mut stream);
            let head = "HTTP/1.1 200 OK\r\n\
                        content-type: text/event-stream\r\n\
                        connection: close\r\n\r\n";
            let _ = stream.write_all(head.as_bytes());
            for event in events {
                let event = serde_json::to_string(&event).expect("a stream event serializes");
                let _ = stream.write_all(format!("data: {event}\n\n").as_bytes());
            }
            let _ = stream.flush();
            // Park until the client drops the stream: a cancel does.
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
            let mut drain = [0; 1024];
            while !matches!(stream.read(&mut drain), Ok(0) | Err(_)) {}
        })
    }

    /// Read one request off `stream`: the head, then the body its
    /// `content-length` names. Nothing beyond that length is parsed.
    fn read_request(stream: &mut std::net::TcpStream) {
        let mut seen = Vec::new();
        let mut chunk = [0; 1024];
        let head = loop {
            let read = stream.read(&mut chunk).unwrap_or(0);
            if read == 0 {
                return; // the client gave up
            }
            seen.extend_from_slice(&chunk[..read]);
            if let Some(at) = seen.windows(4).position(|window| window == b"\r\n\r\n") {
                break at + 4;
            }
        };
        let length = String::from_utf8_lossy(&seen[..head])
            .to_uppercase()
            .lines()
            .find_map(|line| line.strip_prefix("CONTENT-LENGTH:"))
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        while seen.len() < head + length {
            let read = stream.read(&mut chunk).unwrap_or(0);
            if read == 0 {
                return;
            }
            seen.extend_from_slice(&chunk[..read]);
        }
    }
}
