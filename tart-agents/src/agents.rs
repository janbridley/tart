//! The process's registry of subagent conversations.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::TurnHandle;

/// One conversation in the process: the main one, or a subagent it spawned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentId(u64);

/// The main conversation the front end drives.
pub const MAIN: AgentId = AgentId(0);

/// Every conversation's lever in one place, so one key cancels them all.
#[derive(Default)]
pub struct Agents {
    /// MAIN's lever, adopted by the front end.
    main: Mutex<Option<TurnHandle>>,
}

impl Agents {
    /// An empty registry; the front end adopts the main lever into it.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register MAIN's lever, so [`Agents::cancel_all`] reaches the main turn.
    #[inline]
    pub fn adopt(&self, handle: TurnHandle) {
        *self.lock_main() = Some(handle);
    }

    /// Cancel every registered conversation: the main turn, and any children.
    #[inline]
    pub fn cancel_all(&self) {
        // Clone out before cancelling: the handle takes its own lock.
        if let Some(handle) = self.lock_main().clone() {
            handle.cancel();
        }
    }

    /// MAIN's lever slot under its lock.
    fn lock_main(&self) -> MutexGuard<'_, Option<TurnHandle>> {
        self.main.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
