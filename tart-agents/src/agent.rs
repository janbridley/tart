use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_compat::Compat;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::responses::{
        CreateResponseArgs, FunctionToolCall, InputParam, OutputItem, Reasoning, ReasoningEffort,
        ReasoningItem, ResponseStreamEvent,
    },
};
use futures::channel::mpsc;
use futures::future::{Either, select};
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
    /// The front end's lever on the running turn (cancel + steer).
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
    /// The steering message waiting to interrupt the turn, if any.
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
        state.cancelled = true;
        if let Some(sender) = &mut state.sender {
            // A failed poke means one is already pending; the flag decides.
            let _ = sender.try_send(());
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
        state.steer = Some(text);
        if let Some(sender) = &mut state.sender {
            let _ = sender.try_send(());
        }
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
    fn claim(&self, sender: mpsc::Sender<()>) -> u64 {
        let mut state = self.state();
        state.generation += 1;
        state.cancelled = false;
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
            state.cancelled = false;
        }
    }

    /// Whether the front end cancelled the turn: pokes only wake a parked wait
    fn cancelled(&self, cancel_rx: &mut mpsc::Receiver<()>) -> bool {
        while cancel_rx.try_recv().is_ok() {}
        self.state().cancelled
    }

    /// The cancel flag, without draining pokes — the select already woke.
    fn is_cancelled(&self) -> bool {
        self.state().cancelled
    }

    /// Whether a steering message waits.
    fn has_steer(&self) -> bool {
        self.state().steer.is_some()
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
            policy,
            control: TurnControl::default(),
        }
    }

    /// The front end's lever on the running turn, for the pane to hold.
    #[inline]
    pub fn control(&self) -> TurnControl {
        self.control.clone()
    }

    /// Record the queued steering message, reporting it so the front end can echo
    fn record_steer<F: Fn(Progress)>(
        &self,
        transcript: &Transcript,
        on_progress: &F,
    ) -> anyhow::Result<()> {
        if let Some(text) = self.control.take_steer() {
            transcript.push_user(text.clone())?;
            on_progress(Progress::Steered(text));
        }
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
            // Steering left over from the last round rides on this request.
            if let Err(error) = self.record_steer(transcript, on_progress) {
                return terminate_and_log(on_progress, Progress::Failed(error.to_string()));
            }
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
            // The stream was dropped mid-round for a steering message.
            let mut aborted = false;

            // Race the stream against Esc and steering pokes; a won cancel
            // returns and drops the stream, a won steer restarts the round
            loop {
                let item = match block_on(Compat::new(select(stream.next(), cancel_rx.next()))) {
                    Either::Right(_) => {
                        if self.control.is_cancelled() {
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
                        if !self.control.has_steer() {
                            // A stale poke: nothing to do but keep streaming.
                            continue;
                        }
                        // A steer won: record the partial, then the steered input,
                        // and restart the round from it.
                        if !answer.is_empty()
                            && let Err(error) =
                                transcript.push_assistant(std::mem::take(&mut answer))
                        {
                            return terminate_and_log(
                                on_progress,
                                Progress::Failed(error.to_string()),
                            );
                        }
                        if let Err(error) = self.record_steer(transcript, on_progress) {
                            return terminate_and_log(
                                on_progress,
                                Progress::Failed(error.to_string()),
                            );
                        }
                        aborted = true;
                        break;
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
            // (A cancelled turn's recording is unwound by the front end)
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
            let mut steered = false;
            for call in calls {
                // A cancelled turn skips its remaining calls; the one in
                // flight finishes first.
                if cancelled || self.control.cancelled(&mut cancel_rx) {
                    cancelled = true;
                    break;
                }
                if self.control.has_steer() {
                    steered = true;
                    break;
                }
                match tools::execute(&call, &self.policy, on_progress) {
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
            if steered {
                // The steered input rides on the next round's request.
                if let Err(error) = self.record_steer(transcript, on_progress) {
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
    use super::*;

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

    #[test]
    fn turn_control_intents_are_generation_owned() {
        let control = TurnControl::default();

        // One message waits at a time: a second submission is refused.
        assert!(control.steer("first".to_string()));
        assert!(!control.steer("second".to_string()));
        assert_eq!(control.steering(), Some("first".to_string()));
        assert_eq!(control.take_steer(), Some("first".to_string()));

        // Claiming resets both intents and installs the wake sender.
        let (sender, mut rx) = mpsc::channel(1);
        let first = control.claim(sender);
        assert_eq!(control.steering(), None);
        assert!(!control.cancelled(&mut rx));

        // Cancel pokes the claimed sender; the drained flag decides.
        control.cancel();
        assert!(control.cancelled(&mut rx));

        // A stale release must not retire the newer turn's lever.
        let (sender, _) = mpsc::channel(1);
        let second = control.claim(sender);
        control.release(first);
        control.cancel();
        assert!(control.is_cancelled());

        // Retiring the right turn clears its intents but not a later steer —
        // the front end reads that when the terminal event lands.
        control.release(second);
        assert!(control.steer("survives".to_string()));
        assert_eq!(control.take_steer(), Some("survives".to_string()));
    }
}
