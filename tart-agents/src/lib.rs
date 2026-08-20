//! Run commands under a macOS Seatbelt sandbox.
#[cfg(target_os = "macos")]
pub mod sandbox;

/// Run `command` under `bash -c`, returning its combined stdout and stderr.
///
/// A failure to launch comes back as an error string rather than a `Result`,
/// so the output can be handed straight back to the model.
#[must_use]
#[inline]
pub fn run_bash(command: &str) -> String {
    std::process::Command::new("bash")
        .arg("-c")
        .arg(command)
        .output()
        .map_or_else(
            |error| format!("error: {error}"),
            |output| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            },
        )
}
