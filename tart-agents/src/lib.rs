//! The *tart* agent harness.

mod agent;
mod debug;
mod history;
mod progress;
pub mod session;
mod tools;

#[cfg(target_os = "macos")]
pub mod sandbox;

pub use history::Transcript;
pub use progress::Progress;
pub use session::{SESSIONS_ROOT, Session};

pub use agent::{Agent, TurnControl};

pub use tools::{CancelToken, manual_command, merge_digests};

/// Most model rounds one generation may take before giving up.
pub const MAX_TOOL_ROUNDS: usize = 128;

/// Re-exported so callers can pick a reasoning effort without depending on
/// `async-openai`.
pub use async_openai::types::responses::ReasoningEffort;
