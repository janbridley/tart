//! The turn control plane: the front end's lever and the worker's half.

use std::sync::{Arc, Mutex, MutexGuard};

use futures::channel::mpsc;

use crate::CancelToken;

/// The worker's half of a running turn's control plane
pub(super) struct Turn {
    /// The claimed lever; `retire` releases only its own generation.
    control: TurnHandle,
    /// This turn's generation of the lever.
    generation: u64,
    /// Drains pokes so the cancel flag decides: a poke only wakes a wait.
    pokes: mpsc::Receiver<()>,
    /// The running turn's command lever.
    token: CancelToken,
}

impl Turn {
    /// Claim `control` for a new turn, arming the wake channel Esc pokes land on.
    pub(super) fn claim(control: &TurnHandle) -> Self {
        let (sender, pokes) = mpsc::channel(1);
        let control = control.clone();
        let (generation, token) = control.claim(sender);
        Self { control, generation, pokes, token }
    }

    /// Whether the front end cancelled the turn, draining any pokes.
    pub(super) fn cancelled(&mut self) -> bool {
        self.control.cancelled(&mut self.pokes)
    }

    /// Get the reciever that allow pokes to un-park a waiting worker.
    pub(super) fn wakes(&mut self) -> &mut mpsc::Receiver<()> {
        &mut self.pokes
    }

    /// The command lever for this turn's tool calls.
    pub(super) fn token(&self) -> &CancelToken {
        &self.token
    }

    /// Retire the lever, unless a newer turn claimed it.
    pub(super) fn retire(self) {
        self.control.release(self.generation);
    }
}

/// The front end's control plane for running turns.
#[derive(Clone, Default)]
pub struct TurnHandle {
    state: Arc<Mutex<TurnState>>,
}

/// The turn control state, under one lock.
#[derive(Debug, Default)]
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

    /// Cancel the turn: kill running commands, drop the stream, and keep partial answers.
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

    /// Install `sender` as the next turn's lever, forgetting the last turn's cancel
    pub(super) fn claim(&self, sender: mpsc::Sender<()>) -> (u64, CancelToken) {
        let mut state = self.state();
        state.generation += 1;
        state.cancelled = false;
        state.sender = Some(sender);
        state.token = CancelToken::new();
        (state.generation, state.token.clone())
    }

    /// Retire the lever, unless a newer turn already claimed it.
    pub(super) fn release(&self, generation: u64) {
        let mut state = self.state();
        if state.generation == generation {
            state.sender = None;
            state.cancelled = false;
        }
    }

    /// Whether the front end cancelled the turn: pokes only wake a parked wait
    pub(super) fn cancelled(&self, cancel_rx: &mut mpsc::Receiver<()>) -> bool {
        while cancel_rx.try_recv().is_ok() {}
        self.state().cancelled
    }
}
