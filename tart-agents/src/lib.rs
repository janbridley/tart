//! The *tart* agent harness.

mod agent;
mod agents;
mod debug;
mod errors;
mod history;
mod progress;
pub mod prompts;
pub mod session;
mod tools;
pub mod usage;

pub mod backends;

#[cfg(target_os = "macos")]
pub mod sandbox;

pub use history::Transcript;
pub use progress::Progress;
pub use session::{CHAT_PROJECT, SESSIONS_ROOT, Session};

pub use agent::{Agent, ChatMode, TurnHandle};
pub use agents::{AGENT_TOOL, AgentId, Agents, MAIN, MAX_SUBAGENTS, Outcome};

pub use tools::{CONTENT_CAP, CancelToken, head_cap, manual_command};

/// Most model rounds one generation may take before giving up.
///
/// Read from the `TART_MAX_TOOL_ROUNDS` environment variable when it holds a positive
/// integer, [`DEFAULT_MAX_TOOL_ROUNDS`] otherwise. Smaller values help prevent looping
/// but may prematurely end a turn in extended agentic work.
#[inline]
pub fn max_tool_rounds() -> usize {
    std::env::var("TART_MAX_TOOL_ROUNDS")
        .ok()
        .and_then(|rounds| rounds.parse().ok())
        .filter(|&rounds| rounds > 0)
        .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS)
}

/// Recover the content of a mutex, poisoned or otherwise.
#[inline]
pub(crate) fn locked<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The round cap when `TART_MAX_TOOL_ROUNDS` is unset or invalid.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 4096;

/// The reasoning efforts a backend accepts, so callers can pick one without
/// depending on the backend's own vocabulary.
pub use backends::ReasoningEffort;
