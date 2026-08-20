//! The *tart* agent harness.

mod agent;
mod history;
mod progress;
mod tools;

#[cfg(target_os = "macos")]
pub mod sandbox;

pub use history::Transcript;
pub use progress::Progress;

pub use agent::Agent;

/// Most model rounds one generation may take before giving up.
pub const MAX_TOOL_ROUNDS: usize = 10;

/// Re-exported so callers can pick a reasoning effort without depending on
/// `async-openai`.
pub use async_openai::types::responses::ReasoningEffort;
