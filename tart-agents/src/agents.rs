//! The process's registry of conversations: the main turn's lever and its subagents.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use crate::{Agent, CancelToken, Progress, Transcript, TurnHandle};

/// The subagent prompt, appended after the system like the front end's instructions
const AGENT_PROMPT: &str = include_str!("data/AGENT.md");

/// How often a bounded `wait` wakes to check its cancel token.
const WAIT_POLL: Duration = Duration::from_millis(100);

/// One conversation in the process: the main one, or a subagent it spawned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AgentId(u64);

/// The main conversation the front end drives.
pub const MAIN: AgentId = AgentId(0);

impl AgentId {
    /// The id's number, as `spawn` reports it and `wait` takes it.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.0
    }

    /// The id as its agent box names it: `agent-N`.
    #[inline]
    #[must_use]
    pub fn tag(&self) -> String {
        format!("agent-{self}")
    }
}

impl From<u64> for AgentId {
    #[inline]
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl Outcome {
    /// The outcome as text for the model and its box: what ended the child,
    /// and what it had to say.
    #[inline]
    #[must_use]
    pub fn report(&self) -> String {
        match self {
            Outcome::Done(report) => report.clone().unwrap_or_else(|| "(no report)".to_string()),
            Outcome::Failed(error) => format!("failed: {error}"),
            Outcome::Cancelled => "cancelled".to_string(),
        }
    }
}

impl fmt::Display for AgentId {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A subagent's terminal result.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Outcome {
    /// The subagent finished; its final message is its report.
    Done(Option<String>),
    /// The subagent's turn failed.
    Failed(String),
    /// The subagent was cancelled.
    Cancelled,
}

/// The most subagents that may run (or await delivery) at once.
pub const MAX_SUBAGENTS: usize = 8;

/// The wire name of the synthetic tool call that carries a subagent's box.
pub const AGENT_TOOL: &str = "agent";

/// One registered subagent.
struct Child {
    /// The task it was given, for its box and its injection message.
    task: String,
    /// Its lever: cancelling reaches its stream and its commands.
    handle: TurnHandle,
    /// Its terminal result, once its worker reported one and until it is taken.
    outcome: Option<Outcome>,
    /// Where `wait` blocks for the outcome; taken by the first waiter.
    terminal: Option<Receiver<Outcome>>,
    /// The terminal channel's sending end, through which the worker reports.
    sender: Sender<Outcome>,
}

/// Every conversation's lever in one place, so one key cancels them all.
#[derive(Clone)]
pub struct Agents {
    inner: Arc<Inner>,
}

/// The registry's shared state.
struct Inner {
    /// MAIN's lever, adopted by the front end.
    main: Mutex<Option<TurnHandle>>,
    /// Every spawned child, whose outcomes are waiting for injection or `wait`
    children: Mutex<Vec<(AgentId, Child)>>,
    /// The next child's id.
    next: AtomicU64,
    /// Where every child's progress goes, tagged with its id.
    events: Box<dyn Fn(AgentId, Progress) + Send + Sync>,
}

impl Agents {
    /// A registry that forwards every child's progress to `events`, tagged
    /// with the child's id.
    #[inline]
    pub fn new<F: Fn(AgentId, Progress) + Send + Sync + 'static>(events: F) -> Self {
        Self {
            inner: Arc::new(Inner {
                main: Mutex::new(None),
                children: Mutex::new(Vec::new()),
                next: AtomicU64::new(1),
                events: Box::new(events),
            }),
        }
    }

    /// Register MAIN's lever, so [`Agents::cancel_all`] reaches the main turn.
    #[inline]
    pub fn adopt(&self, handle: TurnHandle) {
        *self.lock_main() = Some(handle);
    }

    /// Fork a subagent on `task`, cloned from `template`, and return its id
    /// at once: the child runs on its own thread and never blocks the caller.
    #[inline]
    pub fn spawn(&self, template: &Agent, task: &str) -> anyhow::Result<AgentId> {
        if self.inner.lock_children().len() >= MAX_SUBAGENTS {
            anyhow::bail!("at most {MAX_SUBAGENTS} subagents may run at once");
        }
        let id = AgentId(self.inner.next.fetch_add(1, Ordering::Relaxed));
        // The child's own lever, with fresh state: MAIN's generation guard
        // must never retire the child's wake sender, nor the child's retire
        // MAIN's.
        let agent = template.child();
        let transcript = Transcript::with_instructions(AGENT_PROMPT.to_string())?;
        transcript.push_user(task.to_string())?;
        let (sender, terminal) = mpsc::channel::<Outcome>();
        self.inner.lock_children().push((
            id,
            Child {
                task: task.to_string(),
                handle: agent.handle(),
                outcome: None,
                terminal: Some(terminal),
                sender,
            },
        ));
        // The box opens before any child event can arrive, so the front end
        // never sees a call for a box it lacks.
        (self.inner.events)(
            id,
            Progress::ToolStart {
                id: id.tag(),
                name: AGENT_TOOL.to_string(),
                arguments: serde_json::json!({ "task": task }).to_string(),
            },
        );

        let inner = Arc::clone(&self.inner);
        agent.spawn(&transcript, move |progress| {
            // A terminal outcome is stored and its waiter woken before
            // the event forwards, so whoever acts on the event always finds
            // the outcome already registered.
            if let Some(outcome) = progress.outcome() {
                inner.with_child(id, |child| {
                    child.outcome = Some(outcome.clone());
                    let _ = child.sender.send(outcome);
                });
            }
            (inner.events)(id, progress);
        });
        Ok(id)
    }

    /// The subagent's terminal result, when it has ended: a peek that claims
    /// nothing, for drawing its box. Delivery is [`Agents::take_outcome`].
    #[inline]
    pub fn outcome(&self, id: AgentId) -> Option<Outcome> {
        self.inner.with_child(id, |child| child.outcome.clone()).flatten()
    }

    /// Claim the subagent's terminal result, with the task it ran: the one
    /// delivery of that report, removing the child from the registry. `None`
    /// when no such child exists, it is still running, or its report was
    /// already delivered: one answer for all three.
    #[inline]
    pub fn take_outcome(&self, id: AgentId) -> Option<(String, Outcome)> {
        let mut children = self.inner.lock_children();
        let index = children.iter().position(|(sid, _)| *sid == id)?;
        // Taking the outcome delivers it: the child leaves the registry,
        // freeing its slot.
        let outcome = children[index].1.outcome.take();
        outcome.map(|outcome| (children.remove(index).1.task, outcome))
    }

    /// The running subagents, as (id, task) for listing.
    #[inline]
    pub fn running(&self) -> Vec<(AgentId, String)> {
        self.inner
            .lock_children()
            .iter()
            .filter(|(_, child)| child.outcome.is_none())
            .map(|(id, child)| (*id, child.task.clone()))
            .collect()
    }

    /// Block until the subagent ends, `timeout` passes, or `cancel` fires.
    ///
    /// The registry's lock is never held while blocked: Esc's
    /// [`Agents::cancel_all`] must reach the child whose end is the wakeup,
    /// and it takes the lock to find the child.
    #[inline]
    pub fn wait(
        &self,
        id: AgentId,
        timeout: Duration,
        cancel: &CancelToken,
    ) -> anyhow::Result<Option<Outcome>> {
        // An ended, undelivered child answers at once.
        if let Some((_, outcome)) = self.take_outcome(id) {
            return Ok(Some(outcome));
        }
        let receiver = self
            .inner
            .with_child(id, |child| child.terminal.take())
            .ok_or_else(|| {
                anyhow::anyhow!("no subagent {id}, or its report was already delivered")
            })?
            .ok_or_else(|| anyhow::anyhow!("subagent {id} is already being waited on"))?;
        // Off the lock: the child's terminal event is the wakeup.
        let deadline = Instant::now() + timeout;
        loop {
            if cancel.cancelled() {
                return Ok(None);
            }
            match receiver.recv_timeout(WAIT_POLL) {
                Ok(_) => {
                    // The worker stored the outcome before sending, so the
                    // take succeeds unless the front end delivered first.
                    return match self.take_outcome(id) {
                        Some((_, outcome)) => Ok(Some(outcome)),
                        None => anyhow::bail!("subagent {id}'s report was already delivered"),
                    };
                }
                Err(RecvTimeoutError::Timeout) if Instant::now() >= deadline => {
                    return Ok(None);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    anyhow::bail!("subagent {id} ended without a report");
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }

    /// Cancel one subagent.
    #[inline]
    pub fn cancel(&self, id: AgentId) {
        // The lever is cloned out from under the lock; cancelling takes the
        // handle's own lock.
        if let Some(handle) = self.inner.with_child(id, |child| child.handle.clone()) {
            handle.cancel();
        }
    }

    /// Cancel every child and forget them: the conversation they belonged to
    /// is gone, and no report may follow it into the next one.
    #[inline]
    pub fn clear(&self) {
        for handle in self.inner.handles() {
            handle.cancel();
        }
        self.inner.lock_children().clear();
    }

    /// Cancel every registered conversation: the main turn, and any children.
    #[inline]
    pub fn cancel_all(&self) {
        // Every lever is cloned out from under the locks; cancelling takes
        // each handle's own lock.
        let mut handles = self.inner.handles();
        handles.extend(self.lock_main().clone());
        for handle in handles {
            handle.cancel();
        }
    }

    /// The registry's lock on MAIN's lever.
    fn lock_main(&self) -> MutexGuard<'_, Option<TurnHandle>> {
        self.inner.main.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Inner {
    /// The registry's lock on its children.
    fn lock_children(&self) -> MutexGuard<'_, Vec<(AgentId, Child)>> {
        self.children.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Run `f` on the child `id`, under the registry's lock.
    fn with_child<R>(&self, id: AgentId, f: impl FnOnce(&mut Child) -> R) -> Option<R> {
        self.lock_children()
            .iter_mut()
            .find(|(sid, _)| *sid == id)
            .map(|(_, child)| f(child))
    }

    /// Every child's lever, cloned out from under the lock.
    fn handles(&self) -> Vec<TurnHandle> {
        self.lock_children()
            .iter()
            .map(|(_, child)| child.handle.clone())
            .collect()
    }
}
