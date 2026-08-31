use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

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

use crate::{MAX_TOOL_ROUNDS, Progress, Transcript, debug, sandbox::Policy, tools};

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
    /// The Plan-mode policy: the same grants, but most paths are read-only.
    planning: Policy,
    /// The shared runtime every agent and turn drives its futures on.
    runtime: Arc<Runtime>,
    /// The front end's lever on the running turn (cancel).
    control: TurnControl,
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
    /// The running turn's wake sender, where pokes reach it.
    sender: Option<mpsc::Sender<()>>,
    /// Esc was pressed and we should attempt to cancel.
    cancelled: bool,
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
        state.cancelled = true;
        if let Some(sender) = &mut state.sender {
            // A failed poke means one is already pending; the flag decides.
            let _ = sender.try_send(());
        }
    }

    /// Install `sender` as the next turn's lever, forgetting the last turn's
    /// cancel, and report the turn's id.
    fn claim(&self, sender: mpsc::Sender<()>) -> u64 {
        let mut state = self.state();
        state.generation += 1;
        state.cancelled = false;
        state.sender = Some(sender);
        state.generation
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
        let planning = policy.clone().read_only();
        Self {
            client: Client::with_config(config),
            model: model.into(),
            effort: None,
            max_rounds: MAX_TOOL_ROUNDS,
            mode: ChatMode::Default,
            writable: policy,
            planning,
            runtime: Arc::new(Runtime::new().expect("tokio runtime did not start")),
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
        let generation = self.control.claim(esc_sender);
        std::thread::spawn(move || {
            // A panicking worker must still deliver the terminal event to the caller
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                agent.run(&transcript, receiver, &on_progress);
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
        on_progress: &F,
    ) {
        // Set once a cancel is consumed, so the rest of the turn's calls and rounds
        // skip without re-reading the channel.
        let mut cancelled = false;
        for _ in 0..self.max_rounds {
            // A cancelled generation stops before spending another request.
            if cancelled || self.control.cancelled(&mut cancel_rx) {
                return terminate_and_log(on_progress, Progress::Cancelled);
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

            // The owned runtime drives the future
            let mut stream = match self
                .runtime
                .block_on(self.client.responses().create_stream(request))
            {
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
                        if !answer.is_empty()
                            && let Err(error) = transcript.push_assistant(answer.clone())
                        {
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
                // A cancelled turn skips its remaining calls; the one in
                // flight finishes first.
                if cancelled || self.control.cancelled(&mut cancel_rx) {
                    cancelled = true;
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
                let runtime = agent.runtime.clone();
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

    /// Claiming installs the wake sender and retires the last turn's cancel;
    /// a stale release must not retire a newer turn's lever.
    #[test]
    fn turn_control_cancel_is_generation_owned() {
        let control = TurnControl::default();

        // Claiming resets the cancel and installs the wake sender.
        let (sender, mut rx) = mpsc::channel(1);
        let first = control.claim(sender);
        assert!(!control.cancelled(&mut rx));

        // Cancel pokes the claimed sender; the drained flag decides.
        control.cancel();
        assert!(control.cancelled(&mut rx));

        // A stale release must not retire the newer turn's lever.
        let (sender, _) = mpsc::channel(1);
        let second = control.claim(sender);
        control.release(first);
        control.cancel();
        assert!(control.cancelled(&mut rx));

        // Retiring the right turn clears its cancel for the next one.
        control.release(second);
        assert!(!control.cancelled(&mut rx));
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
            let log = log.clone();
            let control = agent.control();
            std::thread::spawn(move || {
                while !log.lock().unwrap().iter().any(|entry| entry.contains(" a time")) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                control.cancel();
            })
        };

        // Drive the generation as `spawn` would: claim, run, retire.
        let (sender, receiver) = mpsc::channel(1);
        let control = agent.control();
        let generation = control.claim(sender);
        agent.run(&transcript, receiver, &|progress| {
            log.lock().unwrap().push(format!("{progress:?}"));
        });
        control.release(generation);
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

    /// The loopback listener the worker test streams from, or `None` where the
    /// enclosing sandbox denies binding one: the test then skips, exactly like
    /// the live sandbox tests do.
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
