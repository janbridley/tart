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

#[cfg(target_os = "macos")]
pub mod sandbox;

pub use history::Transcript;
pub use progress::Progress;
pub use session::{SESSIONS_ROOT, Session};

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

/// The round cap when `TART_MAX_TOOL_ROUNDS` is unset or invalid.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 4096;

/// Re-exported so callers can pick a reasoning effort without depending on
/// `async-openai`.
pub use async_openai::types::responses::ReasoningEffort;
