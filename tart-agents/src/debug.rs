//! Debug logging for the agent loop.
//!
//! ```bash
//! TART_DEBUG=1 cargo run -p tart-tui 2>dump.txt
//! ```
//!
//! ```text
//! [tart] round request: {"model":"deepseek-reasoner",...}
//! ```

use std::ffi::OsStr;
use std::sync::OnceLock;

/// The cached `TART_DEBUG` verdict, decided once per process.
static ENABLED: OnceLock<bool> = OnceLock::new();

/// Log one debug line with a preformatted body, if `TART_DEBUG` is on.
#[inline]
pub(crate) fn log<F: FnOnce() -> String>(label: &str, render: F) {
    if enabled() {
        emit(&line(label, &render()));
    }
}

/// Log one debug line with a JSON body, if `TART_DEBUG` is on.
#[inline]
pub(crate) fn log_json<F: FnOnce() -> Result<String, serde_json::Error>>(label: &str, render: F) {
    if enabled() {
        emit(&line(label, &rendered(render())));
    }
}

/// The prefixed line for one event.
fn line(label: &str, body: &str) -> String {
    format!("[tart] {label}: {body}")
}

/// A serialization result for a debug line, degrading instead of panicking.
fn rendered(result: Result<String, serde_json::Error>) -> String {
    result.unwrap_or_else(|error| format!("<unserializable: {error}>"))
}

/// Whether debug logging is on for this process.
fn enabled() -> bool {
    *ENABLED.get_or_init(|| flag(std::env::var_os("TART_DEBUG").as_deref()))
}

/// Interpret one `TART_DEBUG` value: on for `1` or any other non-`0` text.
fn flag(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty() && value != OsStr::new("0"))
}

/// Emit one finished line to stderr.
#[allow(
    clippy::print_stderr,
    reason = "stderr is this module's entire sink; the TUI owns stdout"
)]
fn emit(text: &str) {
    eprintln!("{text}");
}
