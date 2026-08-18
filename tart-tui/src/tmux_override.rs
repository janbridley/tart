//! Override tmux defaults so our pseudo-copy mode takes precedence over tmux.

use std::env;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

/// With the alternate screen on, forward the key; otherwise the default.
const REBIND: &str = "if -F '#{alternate_on}' 'send-keys S-Up' 'copy-mode'";
const RESTORE: &str = "copy-mode";
static REBOUND: AtomicBool = AtomicBool::new(false);

/// A `tmux` client for the server this pane belongs to, or `None` outside tmux.
fn tmux() -> Option<Command> {
    let tmux = env::var_os("TMUX")?;
    // $TMUX is "<socket>,<pid>,<start time>"; -S targets that exact server.
    let socket = tmux.to_str().and_then(|s| s.split(',').next()).unwrap_or("");
    let mut cmd = Command::new("tmux");
    if !socket.is_empty() {
        cmd.arg("-S").arg(socket);
    }
    Some(cmd)
}

fn tmux_bind(binding: &str, mark: bool) -> bool {
    let Some(mut cmd) = tmux() else { return false };
    let ok = cmd
        .args(["bind-key", "-n", "S-Up"])
        .arg(binding) // one argv element; tmux re-parses it
        .output()
        .is_ok_and(|out| out.status.success());
    if ok && mark {
        REBOUND.store(true, Ordering::SeqCst);
    }
    ok
}

/// Undo the rebind exactly once.
pub fn restore_tmux() {
    if REBOUND.swap(false, Ordering::SeqCst) {
        tmux_bind(RESTORE, false);
    }
}

/// Take the override once the alternate screen is live; dropping the guard
/// restores the binding, covering `?` early returns in `main`.
pub fn override_shift_up() -> Option<TmuxGuard> {
    tmux_bind(REBIND, true).then_some(TmuxGuard)
}

pub struct TmuxGuard;

impl Drop for TmuxGuard {
    fn drop(&mut self) {
        restore_tmux();
    }
}
