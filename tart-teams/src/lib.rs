//! One lead, its children, and the peer layer between them: the team tart
//! actually runs, with the graph's rules carried by the types instead of
//! checked at send time.
//!
//! Each handle's method set is exactly its edge set: the [`Lead`] steers,
//! cancels, and broadcasts down; a [`Child`] reports up, and `tell`s or
//! steers its peers (the one flat sibling layer) sideways. There is no
//! `to` parameter on the up path, no way to address MAIN sideways (peer
//! methods take [`ChildId`], which MAIN is not), and no peer-cancellation
//! method at all: it is unrepresentable, not merely forbidden. Ids are
//! minted opaque by [`Team::spawn_child`], so a child that was never
//! spawned cannot be named; dropping a handle retires its slot, and later
//! sends answer [`SendError::Gone`]. What stays runtime is data: policy
//! refusals, self-addressed peer sends, and finished children.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crossbeam_channel::{Receiver, Sender, unbounded};

/// An agent's id inside a [`Team`].
pub type AgentId = u64;

/// The lead's id, matching `tart_agents::MAIN`.
pub const MAIN: AgentId = 0;

/// The kind of a [`Message`]; policies are defined over kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    /// Mid-flight course correction.
    Steer,
    /// Request to stop working.
    Cancel,
    /// Ordinary payload.
    Data,
}

/// What one agent can say to another.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Message {
    /// Adjust the receiving agent's directive while it keeps working.
    Steer(String),
    /// Ask the receiving agent to stop.
    Cancel(String),
    /// Any other payload.
    Data(String),
}

impl Message {
    /// The [`Kind`] of this message.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Self::Steer(_) => Kind::Steer,
            Self::Cancel(_) => Kind::Cancel,
            Self::Data(_) => Kind::Data,
        }
    }
}

/// A message stamped with its sender and recipient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Envelope {
    /// The sending agent.
    pub from: AgentId,
    /// The receiving agent.
    pub to: AgentId,
    /// What was said.
    pub message: Message,
}

/// What may be said on a link, independent of the wiring.
pub trait Policy: Send + Sync {
    /// Whether `from` may send `kind` to `to`.
    fn permits(&self, from: AgentId, to: AgentId, kind: Kind) -> bool;
}

/// Open by default, with cancellations that can be switched off per sender
/// or per link while steering and data keep flowing.
#[derive(Clone, Debug, Default)]
pub struct Permissions {
    no_cancel_from: HashSet<AgentId>,
    no_cancel_on: HashSet<(AgentId, AgentId)>,
}

impl Permissions {
    /// Everything is permitted.
    #[inline]
    #[must_use]
    pub fn open() -> Self {
        Self::default()
    }

    /// Bar every cancellation `agent` sends.
    #[inline]
    #[must_use]
    pub fn without_cancel_from(mut self, agent: AgentId) -> Self {
        self.no_cancel_from.insert(agent);
        self
    }

    /// Bar cancellation from `from` to `to` specifically.
    #[inline]
    #[must_use]
    pub fn without_cancel_between(mut self, from: AgentId, to: AgentId) -> Self {
        self.no_cancel_on.insert((from, to));
        self
    }
}

impl Policy for Permissions {
    #[inline]
    fn permits(&self, from: AgentId, to: AgentId, kind: Kind) -> bool {
        kind != Kind::Cancel
            || (!self.no_cancel_from.contains(&from) && !self.no_cancel_on.contains(&(from, to)))
    }
}

/// Why a send was refused. Adjacency cannot fail here, since in one flat
/// layer the live children are the whole address space, so what remains is
/// data: policy, self-addressed peer sends, and finished children.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SendError {
    /// The policy forbids this kind of message on this link, including a
    /// peer sending to itself.
    Forbidden {
        /// The sending agent.
        from: AgentId,
        /// The intended recipient.
        to: AgentId,
        /// The refused message kind.
        kind: Kind,
    },
    /// The child has finished; its id is stale.
    Gone(AgentId),
}

impl fmt::Display for SendError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forbidden { from, to, kind } => {
                write!(f, "agent {from} may not send {kind:?} to agent {to}")
            }
            Self::Gone(id) => write!(f, "child {id} has finished"),
        }
    }
}

impl Error for SendError {}

/// A child's id: an index into the team's routing table, minted by
/// [`Team::spawn_child`] alone, with no public constructor and no `From`,
/// so a child that was never spawned cannot be named. MAIN is not one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChildId(AgentId);

impl ChildId {
    /// The id's number, for registries and ledgers.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> AgentId {
        self.0
    }

    /// The routing-table slot this id indexes; ids are minted one past
    /// their slot, so the mapping is exact.
    fn slot(self) -> usize {
        (self.0 - 1) as usize
    }
}

impl fmt::Display for ChildId {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The lead: steers, cancels, and broadcasts down to the children, and
/// receives every child's reports.
pub struct Lead {
    inbox: Receiver<Envelope>,
    inner: Arc<Inner>,
}

/// A child agent: reports up, tells and steers its peers sideways, and is
/// steered and cancelled from above. Peer cancellation has no method: it
/// is unrepresentable, not merely forbidden.
pub struct Child {
    id: ChildId,
    inbox: Receiver<Envelope>,
    inner: Arc<Inner>,
}

impl Drop for Child {
    #[inline]
    fn drop(&mut self) {
        // Retire the slot, so later sends answer Gone instead of
        // vanishing into a dead inbox.
        let mut slots = locked(&self.inner.children);
        if let Some(slot) = slots.get_mut(self.id.slot()) {
            *slot = None;
        }
    }
}

/// The lead, the children, and the peer layer between them: one channel
/// pair per agent, policy on every send.
#[derive(Clone)]
pub struct Team {
    inner: Arc<Inner>,
}

/// The team's shared state, behind the `Arc` every handle holds.
struct Inner {
    policy: Box<dyn Policy>,
    lead: Sender<Envelope>,
    /// The routing table: one slot per child in mint order, holding its
    /// sender until the handle is dropped. Slot `i` is `ChildId(i + 1)`,
    /// so membership and address are the same thing.
    children: Mutex<Vec<Option<Sender<Envelope>>>>,
}

/// Recover a mutex's content, poisoned or otherwise: no path panics while
/// holding one of these locks.
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Team {
    /// A team under `policy`, returned with its one lead handle.
    #[inline]
    pub fn new<P: Policy + 'static>(policy: P) -> (Self, Lead) {
        let (lead_tx, lead_rx) = unbounded();
        let inner = Arc::new(Inner {
            policy: Box::new(policy),
            lead: lead_tx,
            children: Mutex::new(Vec::new()),
        });
        (Self { inner: Arc::clone(&inner) }, Lead { inbox: lead_rx, inner })
    }

    /// Spawn a child, minting its id, its channel, and its handle.
    #[inline]
    pub fn spawn_child(&self) -> Child {
        let (tx, rx) = unbounded();
        let mut slots = locked(&self.inner.children);
        slots.push(Some(tx));
        Child {
            id: ChildId(slots.len() as u64),
            inbox: rx,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Lead {
    /// The lead's inbox: every report from below.
    #[inline]
    #[must_use]
    pub fn inbox(&self) -> &Receiver<Envelope> {
        &self.inbox
    }

    /// Steer `child` mid-flight.
    #[inline]
    pub fn steer<S: Into<String>>(&self, child: ChildId, text: S) -> Result<(), SendError> {
        self.inner.to_child(MAIN, child, Message::Steer(text.into()))
    }

    /// Ask `child` to stop.
    #[inline]
    pub fn cancel<S: Into<String>>(&self, child: ChildId, reason: S) -> Result<(), SendError> {
        self.inner.to_child(MAIN, child, Message::Cancel(reason.into()))
    }

    /// Send to every live child the policy permits, one result each.
    #[inline]
    pub fn broadcast(&self, message: &Message) -> Vec<(ChildId, Result<(), SendError>)> {
        let inner = &self.inner;
        let slots = locked(&inner.children);
        let mut results = Vec::new();
        for (index, slot) in slots.iter().enumerate() {
            if let Some(sender) = slot.as_ref() {
                let child = ChildId((index + 1) as u64);
                results.push((child, inner.gate(MAIN, child.0, message.clone(), sender)));
            }
        }
        results
    }
}

impl Child {
    /// This child's id.
    #[inline]
    #[must_use]
    pub fn id(&self) -> ChildId {
        self.id
    }

    /// The child's inbox: steering and cancellation from above, and tells
    /// and steering from peers, each distinguishable by `from`.
    #[inline]
    #[must_use]
    pub fn inbox(&self) -> &Receiver<Envelope> {
        &self.inbox
    }

    /// Report `text` up to the lead: the only send that leaves the layer.
    #[inline]
    pub fn report<S: Into<String>>(&self, text: S) -> Result<(), SendError> {
        self.inner
            .gate(self.id.0, MAIN, Message::Data(text.into()), &self.inner.lead)
    }

    /// The siblings still running in the one flat peer layer, without
    /// self, as a snapshot in mint order.
    #[inline]
    pub fn peers(&self) -> impl Iterator<Item = ChildId> {
        let mut ids = live_ids(&locked(&self.inner.children));
        ids.retain(|id| *id != self.id);
        ids.into_iter()
    }

    /// Tell a sibling something; it arrives in the sibling's inbox.
    #[inline]
    pub fn tell<S: Into<String>>(&self, peer: ChildId, text: S) -> Result<(), SendError> {
        self.inner
            .route_lateral(self.id, peer, Message::Data(text.into()))
    }

    /// Steer a sibling mid-flight, on the same round-boundary path as the
    /// lead's steering.
    #[inline]
    pub fn steer_peer<S: Into<String>>(&self, peer: ChildId, text: S) -> Result<(), SendError> {
        self.inner
            .route_lateral(self.id, peer, Message::Steer(text.into()))
    }
}

/// The live ids in mint order, under the table's lock.
fn live_ids(slots: &[Option<Sender<Envelope>>]) -> Vec<ChildId> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.as_ref().map(|_| ChildId((index + 1) as u64)))
        .collect()
}

impl Inner {
    /// To one child, from MAIN above or a sibling beside: its live slot's
    /// sender, then the policy gate.
    fn to_child(&self, from: AgentId, child: ChildId, message: Message) -> Result<(), SendError> {
        let slots = locked(&self.children);
        let sender = slots.get(child.slot()).and_then(Option::as_ref);
        match sender {
            Some(sender) => self.gate(from, child.0, message, sender),
            None => Err(SendError::Gone(child.0)),
        }
    }

    /// Sideways to one sibling: self-sends are refused, then the same
    /// gated route every child send takes.
    fn route_lateral(&self, from: ChildId, to: ChildId, message: Message) -> Result<(), SendError> {
        if from == to {
            return Err(SendError::Forbidden {
                from: from.0,
                to: to.0,
                kind: message.kind(),
            });
        }
        self.to_child(from.0, to, message)
    }

    /// The one policy gate every send passes, then delivery.
    fn gate(
        &self,
        from: AgentId,
        to: AgentId,
        message: Message,
        sender: &Sender<Envelope>,
    ) -> Result<(), SendError> {
        if !self.policy.permits(from, to, message.kind()) {
            return Err(SendError::Forbidden { from, to, kind: message.kind() });
        }
        let envelope = Envelope { from, to, message };
        sender.send(envelope.clone()).map_err(|_| SendError::Gone(to))?;
        Ok(())
    }
}
