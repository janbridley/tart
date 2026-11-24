use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard};

use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::{Either, select};
use tokio::runtime::Runtime;

use crate::backends::openai_responses::Responses;
use crate::backends::{Event, FunctionToolCall, InputItem, Item, ReasoningEffort, Tool};
use crate::usage::MAIN_AGENT;
use crate::{
    AgentId, Agents, CancelToken, MAIN, Progress, Provider, Transcript, debug, errors,
    max_tool_rounds, sandbox::Policy, tools,
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
    /// Chat mode: no shell and no filesystem access, web tools only. Sessions
    /// run under [`Policy::no_access`] and the sandboxed tools are denied at
    /// execution, not merely withheld from the offered list.
    Chat,
}

/// A Responses-API model configured to run the tart tool loop.
#[derive(Clone)]
pub struct Agent {
    backend: Responses,
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
    /// This agent's conversation: MAIN, or the subagent's registry id.
    id: AgentId,
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
        crate::locked(&self.state)
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
        Self::with_provider(Provider::api_key(base_url, api_key), model, policy)
    }

    /// Create an agent for a provider and model, running tools under `policy`.
    #[inline]
    pub fn with_provider<M: Into<String>>(provider: Provider, model: M, policy: Policy) -> Self {
        Self {
            backend: Responses::new(provider),
            model: model.into(),
            effort: None,
            max_rounds: max_tool_rounds(),
            mode: ChatMode::Default,
            writable: policy,
            runtime: Arc::new(Runtime::new().expect("tokio runtime did not start")),
            subagents: None,
            control: TurnHandle::default(),
            id: MAIN,
        }
    }

    /// Switch providers or models, clearing the previous model's effort setting.
    #[inline]
    pub fn set_provider<M: Into<String>>(&mut self, provider: Provider, model: M) {
        self.backend = Responses::new(provider);
        self.model = model.into();
        self.effort = None;
    }

    /// The session's collaboration mode.
    #[inline]
    pub const fn mode(&self) -> ChatMode {
        self.mode
    }

    /// Switch the session's collaboration mode for subsequent turns.
    ///
    /// A turn already in flight keeps the policy it started with, since
    /// [`Agent::spawn`] copied the agent before it began, so this cannot change the
    /// sandbox under a running command.
    #[inline]
    pub const fn set_mode(&mut self, mode: ChatMode) {
        self.mode = mode;
    }

    /// Toggle the writable policy's GPU grants mid-session, returning the new state.
    #[inline]
    pub const fn toggle_gpu(&mut self) -> bool {
        self.writable.toggle_gpu()
    }

    /// The policy the current mode runs tool calls under.
    fn policy(&self) -> Policy {
        match self.mode {
            ChatMode::Default => self.writable.clone(),
            ChatMode::Plan => self.writable.clone().read_only(),
            ChatMode::Chat => Policy::no_access(),
        }
    }

    /// The tools this agent offers the model.
    ///
    /// Plan mode withholds the `edit` tool, chat mode removes all but web tools.
    fn tools_for(&self) -> Vec<Tool> {
        if self.mode == ChatMode::Chat {
            return tools::search().into_iter().chain([tools::fetch()]).collect();
        }
        [tools::bash(), tools::read()]
            .into_iter()
            .chain((self.mode == ChatMode::Default).then_some(tools::edit()))
            .chain(tools::search().into_iter().chain([tools::fetch()]))
            .chain(
                self.subagents
                    .iter()
                    .flat_map(|_| [tools::spawn_agent(), tools::check_agent()]),
            )
            .collect()
    }

    /// Name this agent's conversation, as a spawned subagent's registry id.
    #[inline]
    pub(crate) const fn identify(mut self, id: AgentId) -> Self {
        self.id = id;
        self
    }

    /// Clone this agent as a spawned subagent of itself: a fresh lever, no
    /// registry of its own, and its registry id still to name.
    pub(crate) fn child(&self) -> Self {
        let mut child = self.clone();
        child.control = TurnHandle::default();
        child.subagents = None;
        child
    }

    /// The ledger's tag for this agent: `"main"`, or `"child-N"` for the
    /// Nth subagent the session spawned.
    #[inline]
    #[must_use]
    fn agent_tag(&self) -> String {
        if self.id == MAIN {
            MAIN_AGENT.to_string()
        } else {
            format!("child-{}", self.id)
        }
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
    pub const fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Change the reasoning effort for subsequent turns.
    #[inline]
    pub const fn set_reasoning_effort(&mut self, effort: ReasoningEffort) {
        self.effort = Some(effort);
    }

    /// The configured effort, so a caller can layer a one-turn boost on it.
    #[inline]
    #[must_use]
    pub fn configured_effort(&self) -> Option<ReasoningEffort> {
        self.effort.clone()
    }

    /// Run one generation on its own thread, reporting progress to `on_progress`.
    ///
    /// The worker records its turns (reasoning, tool exchanges, final answer) into the
    /// shared transcript as it goes. Exactly one terminal event ([`Progress::Done`],
    /// [`Progress::Failed`], or [`Progress::Cancelled`]) is delivered, even if the
    /// worker panics.
    #[inline]
    pub fn spawn<F: Fn(Progress) + Send + 'static>(&self, transcript: &Transcript, on_progress: F) {
        self.spawn_with(self.effort.clone(), transcript, on_progress);
    }

    /// [`Self::spawn`] at a caller-chosen effort: the `ultrathink` boost
    /// passes max without touching the agent, so children keep their effort.
    #[inline]
    pub fn spawn_with<F: Fn(Progress) + Send + 'static>(
        &self,
        effort: Option<ReasoningEffort>,
        transcript: &Transcript,
        on_progress: F,
    ) {
        let agent = self.clone();
        let transcript = transcript.clone();
        // This turn's wake channel: the sender waits where Esc can reach it, and
        // the receiver races the stream inside the worker.
        let (esc_sender, receiver) = mpsc::channel(1);
        let (generation, token) = self.control.claim(esc_sender);
        std::thread::spawn(move || {
            // A panicking worker must still deliver the terminal event to the caller
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                agent.run(effort.as_ref(), &transcript, receiver, &token, &on_progress);
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
    /// there. Failed tool calls record an error as its output, which the next round
    /// answers like any other. The final answer is recorded on completion, so later
    /// turns replay the whole exchange. The generation ends after at
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
        effort: Option<&ReasoningEffort>,
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
        'round: while round < self.max_rounds {
            // A cancelled generation stops before spending another request.
            if self.control.cancelled(&mut cancel_rx) {
                return terminate_and_log(on_progress, Progress::Cancelled);
            }
            // The owned runtime drives the future: the model, the effort, the
            // record, and the tools the model has this round.
            let mut stream = match self.runtime.block_on(self.backend.round(
                &self.model,
                effort.cloned(),
                transcript.request_items(),
                self.tools_for(),
            )) {
                Ok(stream) => stream,
                Err(error) => {
                    // The error's class decides the round's next move.
                    let reason = error.to_string();
                    if let Some(terminal) =
                        errors::absorb_provider_error(on_progress, &reason, &mut retries)
                    {
                        return terminate_and_log(on_progress, terminal);
                    }
                    continue;
                }
            };

            let mut answer = String::new();
            // The round's finished output items, exactly as delivered and in
            // stream order: reasoning, calls, and assistant messages alike.
            let mut items: Vec<InputItem> = Vec::new();
            // The typed view of its calls, for execution and their outputs.
            let mut calls: Vec<FunctionToolCall> = Vec::new();
            // A function-call item started but never reported done.
            let mut call_in_flight = false;
            // Some output arrived, so the stream was delivering.
            let mut saw_output = false;
            let mut completed = false;
            let requires_completion = self.backend.requires_completion();
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
                        transcript.push_answer(&answer);
                        return terminate_and_log(on_progress, Progress::Cancelled);
                    }
                    Either::Left((item, _)) => item,
                };
                let Some(item) = item else { break };
                match item {
                    Ok(Event::Answer(delta)) => {
                        answer.push_str(&delta);
                        saw_output = true;
                        on_progress(Progress::Answer(delta));
                    }
                    Ok(Event::Thinking(delta)) => {
                        saw_output = true;
                        on_progress(Progress::Thinking(delta));
                    }
                    Ok(Event::CallStarted) => {
                        call_in_flight = true;
                    }
                    Ok(Event::ItemDone(item)) => {
                        saw_output = true;
                        if let InputItem::Item(Item::FunctionCall(call)) = &item {
                            calls.push(call.clone());
                            call_in_flight = false;
                        }
                        items.push(item);
                    }
                    Ok(Event::Completed(usage)) => {
                        completed = true;
                        // A completed response is output even when it holds no items.
                        saw_output = true;
                        if let Some(spend) = usage {
                            // The bill lands whatever happens to the round
                            // after this.
                            spend.bill(&self.agent_tag(), &self.model);
                            let (input, cached, output) = spend.gauge();
                            on_progress(Progress::Usage { input, cached, output });
                        }
                        if requires_completion {
                            break;
                        }
                    }
                    Ok(Event::Failed(reason)) => {
                        if let Some(terminal) =
                            errors::absorb_provider_error(on_progress, &reason, &mut retries)
                        {
                            return terminate_and_log(on_progress, terminal);
                        }
                        // Retry: the partial round is discarded unrecorded.
                        continue 'round;
                    }
                    // Truncated (max output tokens, content filter): report it
                    Ok(Event::Incomplete(reason)) => {
                        return terminate_and_log(
                            on_progress,
                            Progress::Failed(format!("response incomplete: {reason}")),
                        );
                    }
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
            if requires_completion && !completed && saw_output {
                let reason = with_last_error("stream ended before response.completed", last_error);
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }
            if call_in_flight || !saw_output {
                let reason = with_last_error(
                    if call_in_flight {
                        "stream ended mid tool call"
                    } else {
                        "stream ended without output"
                    },
                    last_error,
                );
                if errors::retry_dropped(on_progress, &reason, &mut retries) {
                    continue;
                }
                return terminate_and_log(on_progress, Progress::Failed(reason));
            }

            // No calls pending: this round's answer is the turn's message.
            if calls.is_empty() {
                transcript.push_round(items, &answer, Vec::new());
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
            for call in calls {
                // A cancelled turn skips its remaining calls; the one in
                // flight finishes first.
                if self.control.cancelled(&mut cancel_rx) {
                    break;
                }
                // Tool calls exit with errors/denial text/parse failures if they fail
                let output = tools::execute(&call, &tooling, on_progress);
                // History keeps no more than the cap: an oversized output is
                // spilled whole to a file the model can read back, noted with
                // where it landed. A spill that cannot be written falls back
                // to the head-and-tail excerpt rather than promising a file
                // that does not exist.
                let kept = if output.len() > tools::CONTENT_CAP
                    && let Some((pointer, path)) = tools::spill_to_file(&output)
                {
                    on_progress(Progress::Note(format!(
                        "{} output spilled to {} (over {} KiB)",
                        call.name,
                        path.display(),
                        tools::CONTENT_CAP / 1024
                    )));
                    pointer
                } else if output.len() > tools::CONTENT_CAP {
                    // The spill could not be written: say truncated, and
                    // deliver the head-and-tail excerpt that implies.
                    on_progress(Progress::Note(format!(
                        "{} output truncated to {} KiB",
                        call.name,
                        tools::CONTENT_CAP / 1024
                    )));
                    tools::bounded(&output, tools::CONTENT_CAP)
                } else {
                    output
                };
                exchanges.push((call, kept));
            }
            // A round cut short before its first executed call records only the
            // partial answer: its calls would have no outputs to pair with, and
            // the retried round delivers its items again. Calls skipped after
            // the first one ran are left out of the record by `push_round`.
            if exchanges.is_empty() {
                transcript.push_answer(&answer);
            } else {
                transcript.push_round(items, &answer, exchanges);
            }
            if self.control.cancelled(&mut cancel_rx) {
                // The turn stops here, keeping the rounds recorded so far.
                return terminate_and_log(on_progress, Progress::Cancelled);
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
    #![allow(clippy::unwrap_used, clippy::panic, reason = "test assertions")]

    use super::*;
    use crate::backends::openai_responses::{
        OutputItem, ResponseOutputItemDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
        Responses,
    };
    use crate::usage::TokenUsage;
    use crate::usage::tests::sample_usage;

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

    /// `/model` swaps the endpoint and model through `set_provider`, dropping the
    /// previous effort while the mode survives it; a child inherits the effort.
    #[test]
    fn swapping_the_provider_resets_effort_and_children_inherit_it() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let mut agent = Agent::new("http://localhost:9", "key", "model", policy);
        agent.set_reasoning_effort(ReasoningEffort::High);
        assert_eq!(agent.child().configured_effort(), Some(ReasoningEffort::High));

        agent.set_provider(
            Provider::api_key("https://api.deepseek.com", "key2"),
            "glm-4-flash",
        );
        assert_eq!(agent.model, "glm-4-flash");
        assert_eq!(agent.effort, None, "the previous agent's effort is left behind");
        agent.set_mode(ChatMode::Chat);
        agent.set_provider(
            Provider::api_key("https://api.deepseek.com", "key2"),
            "glm-4-flash",
        );
        assert_eq!(agent.mode(), ChatMode::Chat, "a /model swap keeps chat mode");
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
        assert_ne!(agent.policy().writable_roots(), [] as [std::path::PathBuf; 0]);
        assert!(names(&agent.tools_for()).contains(&"edit"));
        assert!(names(&agent.tools_for()).contains(&"spawn_agent"));
        assert!(names(&agent.tools_for()).contains(&"check_agent"));

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

    #[test]
    fn chat_mode_offers_only_the_web_tools_under_an_empty_policy() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let mut agent = Agent::new("http://localhost:9", "key", "model", policy);
        agent.set_subagents(Arc::new(Agents::new(|_, _| ())));

        agent.set_mode(ChatMode::Chat);
        let chat_tools = agent.tools_for();
        let offered = names(&chat_tools);
        for withheld in ["bash", "read", "edit", "spawn_agent", "check_agent"] {
            assert!(!offered.contains(&withheld), "chat offers no {withheld}");
        }
        assert!(
            offered.iter().all(|name| ["search", "fetch"].contains(name)),
            "nothing beyond the web pair: {offered:?}"
        );
        assert!(
            agent.policy().writable_roots().is_empty(),
            "chat grants no roots at all"
        );
    }

    /// The names of the function tools among `definitions`.
    fn names(definitions: &[Tool]) -> Vec<&str> {
        definitions.iter().map(|tool| tool.name.as_str()).collect()
    }

    /// An agent whose backend serves `rounds` in turn, each stream held open
    /// after its events when `hold` is set.
    fn scripted(rounds: Vec<Vec<ResponseStreamEvent>>, hold: bool) -> Agent {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let mut agent = Agent::new("http://unused", "key", "model", policy);
        agent.backend = Responses::scripted(rounds, hold);
        agent
    }

    /// A transcript holding the user's `prompt`.
    fn primed(prompt: &str) -> Transcript {
        let transcript = Transcript::new().expect("a transcript opens");
        transcript
            .push_user(prompt.to_string())
            .expect("the user turn records");
        transcript
    }

    fn drive(agent: &Agent, transcript: &Transcript) -> Vec<String> {
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let (sender, receiver) = mpsc::channel(1);
        let handle = agent.handle();
        let (generation, token) = handle.claim(sender);
        agent.run(None, transcript, receiver, &token, &|progress| {
            log.lock().unwrap().push(format!("{progress:?}"));
        });
        handle.release(generation);
        Arc::try_unwrap(log)
            .expect("the sole owner")
            .into_inner()
            .unwrap()
    }

    /// Claiming installs the wake sender and retires the last turn's cancel: a
    /// stale release must not retire a newer turn's lever, cancelling kills the
    /// commands of the turn that claimed it and only those, and the next claim
    /// mints a fresh command lever.
    #[test]
    fn turn_control_is_generation_owned() {
        let control = TurnHandle::default();

        // Claiming resets the cancel and installs the wake sender.
        let (sender, mut rx) = mpsc::channel(1);
        let (first, commands) = control.claim(sender);
        assert!(!control.cancelled(&mut rx));

        // Cancel pokes the claimed sender and kills its commands; the drained
        // flag decides.
        control.cancel();
        assert!(control.cancelled(&mut rx));
        assert!(commands.cancelled());

        // A stale release must not retire the newer turn's lever.
        let (sender, _) = mpsc::channel(1);
        let (second, fresh) = control.claim(sender);
        assert!(!fresh.cancelled(), "a fresh claim is not born cancelled");
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
        let transcript = primed("tell me a story");

        // One round streams a partial answer, then parks: the stream never
        // ends on its own, so only a cancel can end the turn.
        let agent = scripted(vec![vec![delta("Once upon"), delta(" a time")]], true);

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
        agent.run(None, &transcript, receiver, &token, &|progress| {
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
    }

    /// A stream that delivers nothing and closes drops its round; the retry
    /// re-sends the request verbatim and the turn completes on the second one.
    /// A round that streams its text and then the finished message item
    /// records the provider's item alone: the streamed text is the fallback,
    /// not a second copy beside it, while `Done` still carries it.
    #[test]
    fn a_message_item_replaces_the_streamed_answer_in_the_record() {
        let transcript = primed("tell me a story");
        let agent = scripted(
            vec![vec![delta("All "), delta("done."), message_done("All done.")]],
            false,
        );

        let log = drive(&agent, &transcript);

        assert_eq!(
            log.last().map(String::as_str),
            Some(r#"Done { message: Some("All done.") }"#)
        );
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().unwrap();
        assert_eq!(items.len(), 3, "system, user, the provider's message: {items:?}");
        assert_eq!(items[2]["id"], "msg_0", "the provider's item, not a stand-in");
        assert_eq!(items[2]["phase"], "final_answer");
        assert_eq!(items[2]["content"][0]["text"], "All done.");
    }

    #[test]
    fn a_dropped_stream_retries_its_round() {
        let transcript = primed("tell me a story");

        // The first request answers with an empty stream that closes at once;
        // the retried request gets the complete one, so only the retry ends
        // the turn.
        let agent = scripted(vec![vec![], vec![delta("Once upon a time")]], false);

        let log = drive(&agent, &transcript);

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
    fn explicit_completion_is_required_before_dispatching_a_tool() {
        let mut agent = scripted(vec![vec![call_done(0)]], false);
        agent.backend.use_chatgpt_wire_format_for_test();
        let transcript = primed("Read a file");
        let before = transcript.request_items().len();
        let log = drive(&agent, &transcript);
        assert_eq!(log, [r#"Failed("stream ended before response.completed")"#]);
        assert_eq!(transcript.request_items().len(), before);
    }

    /// One round whose `read` output passes the content cap: history records
    /// a pointer to a private scratch file holding the whole output, and the
    /// turn's notes say where it landed. The bash tool's own 30 KiB spill
    /// never reaches this branch, so `read` is the tool that exercises it.
    #[test]
    fn an_oversized_tool_output_spills_into_history() {
        use std::os::unix::fs::PermissionsExt as _;

        // The round runs a real `read`, under the harness's own sandbox.
        if let Err(why) = crate::sandbox::live::probe() {
            use std::io::Write as _;
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "note: skipping live test an_oversized_tool_output_spills_into_history (reaches sandbox-exec): {why}"
            );
            return;
        }

        let transcript = primed("read it all");

        // A file whose numbered rendering passes the content cap.
        let big = std::env::temp_dir().join("tart-spill-test-input.txt");
        std::fs::write(&big, "x".repeat(tools::CONTENT_CAP + 4_096)).unwrap();
        let arguments = serde_json::json!({ "file_path": &big }).to_string();

        // The first round calls read on the big file; the second answers.
        let agent = scripted(
            vec![
                vec![call_done_with(
                    0,
                    FunctionToolCall {
                        namespace: None,
                        name: "read".to_string(),
                        arguments,
                        call_id: "call_0".to_string(),
                        id: Some("item_0".to_string()),
                        status: None,
                        caller: None,
                        r#async: None,
                    },
                )],
                vec![delta("done")],
            ],
            false,
        );

        let log = drive(&agent, &transcript);
        assert!(
            log.last().is_some_and(|entry| entry.starts_with("Done")),
            "the turn must finish: {log:?}"
        );

        // The recorded output is the pointer, and the file it names is the
        // whole output, private to its owner.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().expect("the record is a list of items");
        let output = items[3]["output"].as_str().expect("the call's output");
        assert!(output.starts_with("<persisted-output>"), "{output}");
        // The pointer carries its path mid-line: "Output too large (68.0KB).
        // Full output saved to: {path}".
        let saved = output
            .split_once("Full output saved to: ")
            .and_then(|(_, rest)| rest.lines().next())
            .unwrap_or_else(|| panic!("the pointer names its file: {output}"));
        let spilled = std::fs::read_to_string(saved).expect("the spill file exists");
        assert!(
            spilled.len() > tools::CONTENT_CAP,
            "the whole output: {}",
            spilled.len()
        );
        let mode = std::fs::metadata(saved).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "{saved}");

        // The note told the user where it went, after the fact.
        assert!(
            log.iter().any(|entry| entry.contains("output spilled to ")),
            "the spill is noted with its path: {log:?}"
        );

        let _ = std::fs::remove_file(saved);
        let _ = std::fs::remove_file(&big);
    }

    /// A call the harness cannot run records its misuse as the output and ends
    /// neither the round nor its siblings: the old loop broke the round on a
    /// call it could not run, and this pins that a misuse output skips no
    /// sibling call.
    #[test]
    fn the_rounds_remaining_calls_run_after_a_malformed_one() {
        let transcript = primed("read something");

        // One round carries two malformed read calls; the next round answers.
        let agent = scripted(
            vec![vec![call_done(0), call_done(1)], vec![delta("both answered")]],
            false,
        );

        let log = drive(&agent, &transcript);

        assert!(
            log.iter().all(|entry| !entry.starts_with("Failed")),
            "neither malformed call fails the turn: {log:?}"
        );
        assert_eq!(log.last().map(|entry| entry.starts_with("Done")), Some(true));

        // Both calls ran and recorded, each with its error as its output.
        let items = serde_json::to_value(transcript.request_items()).unwrap();
        let items = items.as_array().expect("the record is a list of items");
        assert_eq!(
            items.len(),
            7,
            "system, user, two calls, two outputs, answer: {items:?}"
        );
        assert_eq!(items[2]["name"], "read");
        assert_eq!(items[2]["call_id"], "call_0");
        assert_eq!(items[3]["call_id"], "call_1");
        for output in [4, 5] {
            assert!(
                items[output]["output"]
                    .as_str()
                    .is_some_and(|text| text.contains("`file_path` is missing")),
                "the call's error is its output: {}",
                items[output]["output"]
            );
        }
        assert_eq!(items[6]["content"], "both answered");
    }

    /// A claim on a running child answers "still running" at once, as often
    /// as asked: nothing blocks, so nothing has to recover from blocking.
    #[test]
    fn a_claim_on_a_running_child_is_instant_and_repeatable() {
        // A stream that never ends holds the child mid-request, so it stays
        // running through every claim.
        let agent = scripted(Vec::new(), true);
        let agents = Agents::new(|_, _| {});
        let id = agents.spawn(&agent, "outlive the claims").unwrap();
        assert!(agents.claim(id).unwrap().is_none());
        assert!(agents.claim(id).unwrap().is_none());
    }

    #[test]
    fn spawned_subagents_run_and_report() {
        let events = Arc::new(Mutex::new(Vec::<(crate::AgentId, Progress)>::new()));
        let log = Arc::clone(&events);
        let agents = crate::Agents::new(move |id, progress| {
            log.lock().unwrap().push((id, progress));
        });

        // The child's one round streams an answer, then the stream closes:
        // the turn ends on its own, no cancel needed.
        let agent = scripted(vec![vec![delta("found it")]], false);

        let id = agents
            .spawn(&agent, "find the flaky test")
            .expect("the registry has room");
        settled(&agents, id);
        let outcome = agents.claim(id).expect("the child is registered");
        assert_eq!(outcome, Some(crate::Outcome::Done(Some("found it".to_string()))));

        // The box opener precedes every child event.
        let events = events.lock().unwrap();
        assert!(
            matches!(events.first(), Some((first, Progress::ToolStart { name, .. })) if *first == id && name == "agent"),
            "{events:?}"
        );
        assert_eq!(events.len(), 3, "opener, one answer delta, terminal: {events:?}");

        // The delivered report is gone: one delivery, and the slot is freed.
        assert_eq!(agents.take_outcome(id), None);
        // The second claim says where the report went, so the model never
        // reads a delivered report as a lost one.
        let delivered = agents
            .claim(id)
            .expect_err("the report left with the first delivery")
            .to_string();
        assert!(delivered.contains("already delivered"), "{delivered}");
    }

    /// A subagent's completed round bills the ledger as a child row: the hard
    /// requirement, pinned end to end through the real spawn path.
    #[test]
    fn a_subagents_round_bills_as_a_child() {
        let _held = crate::usage::tests::held_for_test();
        let _ledger = crate::usage::tests::ledger_root();
        crate::usage::set_session("child-ledger".to_string());
        let agents = crate::Agents::new(|_, _| ());

        // The child's one round answers, bills, and the stream ends.
        let agent = scripted(vec![vec![delta("found it"), completed(&sample_usage())]], false);

        let id = agents
            .spawn(&agent, "find the flaky test")
            .expect("room for a child");
        settled(&agents, id);

        let row = crate::usage::Ledger::last_entry().expect("the child's round billed");
        assert_eq!(row.agent, format!("child-{}", id.id()), "the row names the child");
        assert_eq!(row.session.as_deref(), Some("child-ledger"));
        assert_eq!(row.usage, sample_usage());
    }

    /// A report delivers once and frees its slot: a second claim on the id
    /// says the report was delivered, a claim on an id that never ran says so,
    /// and the freed slots take new children.
    #[test]
    fn reports_deliver_once_and_free_their_slots() {
        let policy = Policy::new(std::env::temp_dir()).expect("temp dir is a valid root");
        let agent = Agent::new("http://localhost:9", "key", "model", policy);
        let agents = crate::Agents::new(|_, _| {});

        let never = agents
            .claim(crate::AgentId::from(99))
            .expect_err("no such subagent")
            .to_string();
        assert!(
            never.contains("never ran") && !never.contains("already delivered"),
            "{never}"
        );

        let mut ids = Vec::new();
        for _ in 0..crate::MAX_SUBAGENTS {
            ids.push(agents.spawn(&agent, "task").expect("a slot is free"));
        }
        assert!(agents.spawn(&agent, "task").is_err(), "the slots are full");

        // The unreachable endpoint fails each child fast (a test's retry
        // pause is 1ms); the claim delivers the report and prunes the child.
        for id in ids {
            settled(&agents, id);
            let outcome = agents.claim(id).expect("the child ended");
            assert!(outcome.is_some(), "a failed child still reports: {outcome:?}");
            assert_eq!(agents.take_outcome(id), None, "one delivery only");
            let delivered = agents.claim(id).expect_err("one delivery only").to_string();
            assert!(
                delivered.contains("already delivered") && !delivered.contains("never ran"),
                "{delivered}"
            );
        }
        assert_eq!(agents.running(), []);
        agents
            .spawn(&agent, "task")
            .expect("the slots freed with the reports");
    }

    /// Settle a child the way the front end does: poll its stored outcome
    /// until the worker reports one, bounded so a stuck child fails the test.
    fn settled(agents: &crate::Agents, id: crate::AgentId) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !agents.done(id) {
            assert!(std::time::Instant::now() < deadline, "the child ends");
            std::thread::sleep(std::time::Duration::from_millis(5));
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

    /// An output-item-done event carrying one finished function call: the
    /// shape a provider streams when the model calls a tool. This one calls
    /// `read` without the required `file_path`; `index` keeps concurrent calls'
    /// ids distinct.
    fn call_done(index: usize) -> ResponseStreamEvent {
        call_done_with(
            index,
            FunctionToolCall {
                namespace: None,
                name: "read".to_string(),
                arguments: r#"{"start_line": 1}"#.to_string(),
                call_id: format!("call_{index}"),
                id: Some(format!("item_{index}")),
                status: None,
                caller: None,
                r#async: None,
            },
        )
    }

    /// An output-item-done event carrying any finished function call, for
    /// tests whose calls must be well formed.
    fn call_done_with(index: usize, call: FunctionToolCall) -> ResponseStreamEvent {
        ResponseStreamEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
            sequence_number: index as u64,
            output_index: index as u32,
            item: OutputItem::FunctionCall(call),
        })
    }

    /// An output-item-done event carrying the finished assistant message, as
    /// a provider streams it after the text deltas: the item the record keeps.
    fn message_done(text: &str) -> ResponseStreamEvent {
        let event = serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_0",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        });
        serde_json::from_value(event).expect("a message item done event parses")
    }

    /// A `response.completed` event whose response burned `spend` tokens.
    fn completed(spend: &TokenUsage) -> ResponseStreamEvent {
        let event = serde_json::json!({
            "sequence_number": 1,
            "response": {
                "created_at": 0,
                "id": "resp_0",
                "model": "model",
                "object": "response",
                "output": [],
                "status": "completed",
                "usage": {
                    "input_tokens": spend.input,
                    "input_tokens_details": { "cached_tokens": spend.cached },
                    "output_tokens": spend.output,
                    "output_tokens_details": { "reasoning_tokens": spend.reasoning },
                    "total_tokens": spend.total,
                }
            }
        });
        ResponseStreamEvent::ResponseCompleted(
            serde_json::from_value(event).expect("a completed event parses"),
        )
    }

    #[test]
    fn a_completed_round_bills_the_ledger() {
        let transcript = primed("hello");
        let _held = crate::usage::tests::held_for_test();
        let _ledger = crate::usage::tests::ledger_root();
        crate::usage::set_session("ledgered".to_string());

        // One round answers, bills, and the stream ends: the turn completes.
        let agent = scripted(vec![vec![delta("hi there"), completed(&sample_usage())]], false);

        let log = drive(&agent, &transcript);

        // The turn completes, and the gauge saw the round's bill.
        assert_eq!(log.last().map(|entry| entry.starts_with("Done")), Some(true));
        assert_eq!(
            log.iter().filter(|entry| entry.starts_with("Usage")).count(),
            1,
            "the completed round reports its usage once: {log:?}"
        );

        // The answer is recorded, and the round's bill landed in the ledger:
        // one row, tagged for this agent, session, and model.
        let lines = transcript.lines();
        assert_eq!(lines.len(), 3);
        let row = crate::usage::Ledger::last_entry().expect("the round billed");
        assert_eq!(row.agent, "main");
        assert_eq!(row.model, "model");
        assert_eq!(row.session.as_deref(), Some("ledgered"));
        assert_eq!(row.usage, crate::usage::tests::sample_usage());
    }
}
