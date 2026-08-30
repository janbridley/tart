use std::future::Future;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;

use crate::{Progress, sandbox::Policy};

mod render_tool;
mod web;

pub use render_tool::merge_digests;
pub(crate) use web::{fetch, search};

/// Perform a raw string find-and-replace operation, holding a lock for thread safety..
///
/// This string contains perl source code to perform the required work, dispatching a
/// platform independent flock to ensure concurrent agents cannot collide.
/// Exits 1 with a warning, file untouched, or when the match count is wrong.
const EDIT_PROGRAM: &str = include_str!("data/edit.pl");

/// Emulate `cat -n … | sed -n …` , using a shared `flock` with the edit tool.
const READ_PROGRAM: &str = include_str!("data/read.pl");

/// The timeout a bash call runs under when none is requested.
const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(120);

/// The longest timeout a bash call may ask for.
const MAX_BASH_TIMEOUT: Duration = Duration::from_secs(600);

/// The most of any one blob the model is handed, in bytes.
pub const CONTENT_CAP: usize = 64 * 1024;

/// How often the watchdog wakes to check its deadline and kill lever.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// The front end's kill lever for a running process group: one tool call or command
#[derive(Clone, Default)]
pub struct CancelToken {
    /// Set by `cancel`, watched by the runner's watchdog.
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    /// A token for a command nobody has cancelled yet.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel the run this token fronts, if it is still going.
    #[inline]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Whether [`CancelToken::cancel`] was called.
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Cat a (subregion of a) file, numbering the lines in the output.
fn numbered_read(start: Option<u64>, end: Option<u64>) -> String {
    format!(
        "$start = {}; $end = {};\n{READ_PROGRAM}", // Bake the bounds into the script
        start.unwrap_or(0),
        end.unwrap_or(0)
    )
}

/// A function tool with the given name, description, and JSON-schema parameters.
#[must_use]
fn tool(name: &str, description: &str, parameters: serde_json::Value) -> Tool {
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: name.to_string(),
        description: Some(description.to_string()),
        parameters: Some(parameters),
        strict: None,
    })
}

/// The bash tool; commands execute under the caller's [`Policy`].
#[must_use]
pub(crate) fn bash() -> Tool {
    tool(
        "bash",
        "Run a bash command in a sandbox (writes restricted to granted roots, no network) \
        and return its combined stdout/stderr",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds the command may run before it is killed (1-600); default 120"
                }
            },
            "required": ["command"]
        }),
    )
}

/// The read tool; files are read under the caller's [`Policy`].
#[must_use]
pub(crate) fn read() -> Tool {
    tool(
        "read",
        "Read a file with line numbers (cat -n style) in a sandbox (reads restricted to \
        granted roots); optionally pass start_line/end_line (1-based, inclusive) to read \
        a range",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to read"},
                "start_line": {"type": "integer", "description": "First line to read (1-based); omit to start at the top"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive); omit to read to the end"}
            },
            "required": ["path"]
        }),
    )
}

/// The edit tool; replacements execute under the caller's [`Policy`].
#[must_use]
pub(crate) fn edit() -> Tool {
    tool(
        "edit",
        "Replace an exact string in an existing file. old_string must match the file exactly, including whitespace and \
        newlines, and occur exactly once unless replace_all is true: include surrounding \
        lines to make it unique. An empty new_string deletes old_string. The file must \
        already exist and be valid UTF-8, so use bash to create files. Prefer this tool \
        over bash for changing existing files",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the existing file to edit"},
                "old_string": {"type": "string", "description": "Text to replace; must match exactly and be unique unless replace_all"},
                "new_string": {"type": "string", "description": "Replacement text; empty deletes old_string"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of one unique match"}
            },
            "required": ["path", "old_string", "new_string"]
        }),
    )
}

/// Parse a tool call's arguments as JSON.
fn parse_arguments(arguments: &str) -> anyhow::Result<serde_json::Value> {
    serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))
}

/// A required string field from parsed tool arguments.
fn string_field(args: &serde_json::Value, name: &str) -> anyhow::Result<String> {
    args[name]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("tool call missing '{name}'"))
}

/// One parsed bash tool call.
#[derive(Debug)]
struct Bash {
    /// The command to run.
    command: String,
    /// How long the command may run, clamped into the allowed range.
    timeout: Duration,
}

/// Extract the fields from a bash tool call's JSON arguments.
fn parse_bash(arguments: &str) -> anyhow::Result<Bash> {
    let args = parse_arguments(arguments)?;
    // `as_i64` so negatives join the clamp instead of falling to the default.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the clamp bounds the value to 1-600 seconds before either cast"
    )]
    let seconds = args["timeout"]
        .as_i64()
        .unwrap_or(DEFAULT_BASH_TIMEOUT.as_secs() as i64)
        .clamp(1, MAX_BASH_TIMEOUT.as_secs() as i64) as u64;
    Ok(Bash {
        command: string_field(&args, "command")?,
        timeout: Duration::from_secs(seconds),
    })
}

/// One parsed read tool call.
#[derive(Debug)]
struct Read {
    /// The file to read.
    path: String,
    /// First line to read, 1-based; `None` starts at the top.
    start_line: Option<u64>,
    /// Last line to read, inclusive; `None` reads to the end.
    end_line: Option<u64>,
}

/// Extract the fields from a read tool call's JSON arguments.
///
/// The line bounds are optional; wrong-typed bounds are ignored.
fn parse_read(arguments: &str) -> anyhow::Result<Read> {
    let args = parse_arguments(arguments)?;
    Ok(Read {
        path: string_field(&args, "path")?,
        start_line: args["start_line"].as_u64(),
        end_line: args["end_line"].as_u64(),
    })
}

/// One parsed edit tool call.
#[derive(Debug)]
struct Edit {
    /// The file to edit.
    path: String,
    /// The exact text to replace.
    old_string: String,
    /// What replaces it; empty deletes `old_string`.
    new_string: String,
    /// Replace every occurrence instead of requiring a unique match.
    replace_all: bool,
}

/// Extract the fields from an edit tool call's JSON arguments.
///
/// `replace_all` is optional and defaults to false.
fn parse_edit(arguments: &str) -> anyhow::Result<Edit> {
    let args = parse_arguments(arguments)?;
    Ok(Edit {
        path: string_field(&args, "path")?,
        old_string: string_field(&args, "old_string")?,
        new_string: string_field(&args, "new_string")?,
        replace_all: args["replace_all"].as_bool().unwrap_or(false),
    })
}

/// Run one tool call under `policy`, report each step to `on_progress`, and return
/// output to the model
///
/// Each tool announces itself with [`Progress::ToolStart`] once its arguments
/// parse and always follows with a [`Progress::ToolOutput`] so the front end knows the
/// task concluded.
///
/// Tool *failures* (a non-zero exit, an edit that did not apply, or a command the
/// sandbox denies) are not errors here: their output is content that the model
/// should see.
pub(crate) async fn execute<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    kill: &CancelToken,
    on_progress: &F,
) -> anyhow::Result<String> {
    match call.name.as_str() {
        "bash" => run_bash(call, policy, kill, on_progress).await,
        "read" => run_read(call, policy, kill, on_progress).await,
        "edit" => run_edit(call, policy, kill, on_progress).await,
        // The unsandboxed pair: they need the network the sandbox denies.
        "search" => web::run_search(call, kill, on_progress).await,
        "fetch" => web::run_fetch(call, kill, on_progress).await,
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

/// One finished process's combined streams: stdout then stderr, lossily decoded.
fn combined_output(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// The model-facing framing of a finished command's text: prefixed with
/// `[exit N]` when it failed, or `done` when a success printed nothing.
fn command_text(text: &str, status: ExitStatus) -> String {
    if status.success() {
        return if text.is_empty() {
            "done".to_string()
        } else {
            text.to_string()
        };
    }

    let status = status.code().map_or("signal".into(), |c| c.to_string());

    let separator = if text.is_empty() { "" } else { "\n" };
    format!("[exit {status}]{separator}{text}")
}

/// How a watched run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stopped {
    /// The watchdog killed the group at its deadline.
    Deadline,
    /// Esc killed the group mid-run.
    Esc,
}

/// One finished process run under the watchdog.
pub(crate) struct WatchedRun {
    /// The command's streams and exit status, as `output()` returns them.
    output: Output,
    /// What stopped it: `None` when it finished on its own.
    stopped: Option<Stopped>,
}

/// Frame one finished (or stopped) run for the model and the pane.
///
/// `[timed out after Ns]` for a deadline, `[cancelled]` for Esc, and the
/// status-prefixed text otherwise.
fn frame_run(run: WatchedRun, timeout: Duration) -> (String, String, Option<i32>) {
    let WatchedRun { output, stopped } = run;
    let text = combined_output(&output);
    let exit = output.status.code();
    match stopped {
        Some(Stopped::Deadline) => {
            let marked = timeout_text(&text, timeout);
            (marked.clone(), marked, exit)
        }
        // Esc leaves no exit code for the header, so the body is marked up.
        Some(Stopped::Esc) => {
            let marked = cancel_text(&text);
            (marked.clone(), marked, exit)
        }
        None => (command_text(&text, output.status), text, exit),
    }
}

/// Model-facing explanation for a command the timeout killed.
fn timeout_text(text: &str, timeout: Duration) -> String {
    let separator = if text.is_empty() { "" } else { "\n" };
    format!("[timed out after {}s]{separator}{text}", timeout.as_secs())
}

/// Model-facing explanation for a command Esc killed.
pub(crate) fn cancel_text(text: &str) -> String {
    let separator = if text.is_empty() { "" } else { "\n" };
    format!("[cancelled]{separator}{text}")
}

/// Keep the first `cap` bytes of `text`, suffixing a marker when it cut: an
/// attached file reads from the top, where its interesting part usually is.
#[inline]
pub fn head_cap(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    // Slicing must land on a char boundary; lossy decoding made `text` valid.
    let mut end = cap;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated; first {} KB shown]", &text[..end], cap / 1024)
}

/// Keep the last `cap` bytes of `text`, prefixing a marker when it cut.
fn tail_cap(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    // Slicing must land on a char boundary; lossy decoding made `text` valid.
    let mut start = text.len() - cap;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("[truncated; last {} KB shown]\n{}", cap / 1024, &text[start..])
}

/// The display name and digest for a recorded call, to be replayed as a tool header.
///
/// Recorded arguments are re-parsed so the header matches what the front end showed
/// live. Unparseable arguments and unknown tools degrade to the raw JSON.
pub(crate) fn describe(call: &FunctionToolCall) -> (&'static str, String) {
    let (name, digest) = match call.name.as_str() {
        "bash" => ("Bash", parse_bash(&call.arguments).ok().map(|bash| bash.command)),
        "read" => (
            "Read",
            parse_read(&call.arguments).ok().map(|read| read_digest(&read)),
        ),
        "edit" => ("Edit", parse_edit(&call.arguments).ok().map(|edit| edit.path)),
        "fetch" => (
            "Fetch",
            web::parse_fetch(&call.arguments).ok().map(|fetch| fetch.url),
        ),
        "search" => (
            "Search",
            web::parse_search(&call.arguments).ok().map(|search| {
                if search.news {
                    format!("{} [news]", search.query)
                } else {
                    search.query
                }
            }),
        ),
        _ => ("Tool", None),
    };
    (name, digest.unwrap_or_else(|| call.arguments.clone()))
}

/// Spawn `command` in its own process group with piped output and no input,
/// returning the child and its group id.
fn spawn_grouped(command: &mut Command) -> io::Result<(std::process::Child, Pid)> {
    command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = command.spawn()?;
    // A pid always fits an i32, and the child leads its own group.
    let group = Pid::from_raw(child.id().cast_signed());
    Ok((child, group))
}

/// Run `command` to completion under the one watchdog, killing its process
/// group at `deadline` (if any) or the moment `kill` sets.
fn run_watched_sync(
    command: &mut Command,
    deadline: Option<Duration>,
    kill: &CancelToken,
) -> io::Result<WatchedRun> {
    let (child, group) = spawn_grouped(command)?;
    let started = Instant::now();

    // The sender drops once the command is reaped, waking the watchdog at once.
    let (finished, slept) = mpsc::channel::<()>();
    // The stop reason, sent only when the watchdog stopped the run.
    let (reason, verdict) = mpsc::channel::<Stopped>();
    let token = kill.clone();
    let watchdog = std::thread::spawn(move || {
        loop {
            if token.cancelled() {
                // Flag by sending before killing, so the caller cannot mistake
                // the death for a natural signal once the wait returns.
                let _ = reason.send(Stopped::Esc);
                break;
            }
            if let Some(deadline) = deadline
                && started.elapsed() >= deadline
            {
                let _ = reason.send(Stopped::Deadline);
                break;
            }
            if matches!(
                slept.recv_timeout(CANCEL_POLL),
                Err(RecvTimeoutError::Disconnected)
            ) {
                return;
            }
        }
        let _ = killpg(group, Signal::SIGKILL);
    });

    let output = child.wait_with_output();
    drop(finished);
    // Double check we've killed everything even if the wait errors.
    if output.is_err() {
        let _ = killpg(group, Signal::SIGKILL);
    }
    let _ = watchdog.join();
    let stopped = verdict.try_recv().ok();

    Ok(WatchedRun { output: output?, stopped })
}

/// [`run_watched_sync`] off the calling task, so a caller can send pokes.
async fn run_watched(
    command: Command,
    deadline: Option<Duration>,
    kill: CancelToken,
) -> io::Result<WatchedRun> {
    let (done, done_rx) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let mut command = command;
        let _ = done.send(run_watched_sync(&mut command, deadline, &kill));
    });
    done_rx
        .await
        .unwrap_or_else(|_| Err(io::Error::other("the watched run's thread died")))
}

/// Announce a tool call, run it, and report its conclusion.
///
/// The run arrives as a future: it is inert until polled, so the
/// [`Progress::ToolStart`] pair always opens before any work and closes after
/// it, whatever the body awaits.
async fn traced<F: Fn(Progress), Run>(
    call: &FunctionToolCall,
    name: &'static str,
    digest: String,
    on_progress: &F,
    run: Run,
) -> String
where
    Run: Future<Output = (String, String, Option<i32>)>,
{
    on_progress(Progress::ToolStart {
        id: call.call_id.clone(),
        name,
        digest,
    });
    let (result, output, exit) = run.await;
    on_progress(Progress::ToolOutput {
        id: call.call_id.clone(),
        output,
        exit,
    });
    result
}

/// Run `command` to a `timeout` deadline and frame its streams for the model & pane
async fn run_framed(
    command: Command,
    timeout: Duration,
    kill: CancelToken,
) -> (String, String, Option<i32>) {
    // All tools run through here, so they can't get stuck forever.
    match run_watched(command, Some(timeout), kill).await {
        Ok(run) => frame_run(run, timeout),
        Err(error) => {
            let text = format!("error: {error}");
            (text.clone(), text, None)
        }
    }
}

/// Run one bash tool call under `policy`, reporting its steps to `on_progress`.
///
/// A command that outlives its timeout is killed with everything it started.
/// The model sees `[timed out after Ns]` and any partial output.
async fn run_bash<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    kill: &CancelToken,
    on_progress: &F,
) -> anyhow::Result<String> {
    let bash = parse_bash(&call.arguments)?;
    // A failure to launch comes back as an error string rather than a `Result`,
    // so the output can be handed straight back to the model.
    let mut sandboxed = policy.command("/bin/bash");
    sandboxed.arg("-c").arg(&bash.command);
    Ok(traced(
        call,
        "Bash",
        bash.command.clone(),
        on_progress,
        run_framed(sandboxed, bash.timeout, kill.clone()),
    )
    .await)
}

/// Run one command the user typed, with their privileges and *no deadline*.
#[inline]
pub fn manual_command(command: &str, cancel: &CancelToken) -> String {
    let mut shell = Command::new("/bin/bash");
    shell.arg("-c").arg(command);
    match run_watched_sync(&mut shell, None, cancel) {
        Ok(WatchedRun { output, stopped }) => {
            let text = tail_cap(&combined_output(&output), CONTENT_CAP);
            if stopped.is_some() {
                cancel_text(&text)
            } else {
                command_text(&text, output.status)
            }
        }
        Err(error) => format!("error: {error}"),
    }
}

/// The digest for a read call: the path, with any bounds as `path:start-end`.
fn read_digest(read: &Read) -> String {
    match (read.start_line, read.end_line) {
        (None, None) => read.path.clone(),
        (Some(start), Some(end)) => format!("{}:{start}-{end}", read.path),
        (Some(start), None) => format!("{}:{start}-", read.path),
        (None, Some(end)) => format!("{}:-{end}", read.path),
    }
}

/// Run one read tool call under `policy`, reporting its steps to `on_progress`.
async fn run_read<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    kill: &CancelToken,
    on_progress: &F,
) -> anyhow::Result<String> {
    let read = parse_read(&call.arguments)?;
    let mut command = policy.command("/usr/bin/perl");
    command
        .arg("-e")
        .arg(numbered_read(read.start_line, read.end_line))
        .arg("--")
        .arg(&read.path);
    Ok(traced(
        call,
        "Read",
        read_digest(&read),
        on_progress,
        // The watchdog bounds the read like bash: a FIFO with no writer dies
        // at the deadline — or on Esc — instead of wedging the generation.
        run_framed(command, DEFAULT_BASH_TIMEOUT, kill.clone()),
    )
    .await)
}

/// Run one edit tool call: report the target, apply it, and report the outcome.
///
/// As with bash, edit *failures* (an unreadable file, no or ambiguous match, a
/// sandbox denial) are not errors: their message is content the model can act
/// on and retry.
async fn run_edit<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    kill: &CancelToken,
    on_progress: &F,
) -> anyhow::Result<String> {
    let edit = parse_edit(&call.arguments)?;
    Ok(traced(call, "Edit", edit.path.clone(), on_progress, async {
        let (result, exit) = apply_edit(&edit, policy, kill).await;
        (result.clone(), result, exit)
    })
    .await)
}

/// Apply one parsed edit under `policy`, returning outcome message and the exit code.
///
/// We pre-check that the edit is valid in rust for performance, though the perl script
/// verifies to ensure we don't run into TOCTOU issues between here and the lock.
async fn apply_edit(edit: &Edit, policy: &Policy, kill: &CancelToken) -> (String, Option<i32>) {
    let path = Path::new(&edit.path);
    if edit.old_string.is_empty() {
        return (
            format!("edit: old_string must not be empty: {}", path.display()),
            None,
        );
    }
    if edit.old_string == edit.new_string {
        return (
            format!(
                "edit: old_string and new_string are identical: {}",
                path.display()
            ),
            None,
        );
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => return (format!("edit: cannot read {}: {error}", path.display()), None),
    };
    let count = content.matches(&edit.old_string).count();
    if count == 0 {
        return (
            format!(
                "edit: old_string not found in {}; the match must be exact, including whitespace",
                path.display()
            ),
            None,
        );
    }
    if count > 1 && !edit.replace_all {
        return (
            format!(
                "edit: old_string matches {count} times in {}; pass replace_all or include more \
                surrounding lines to make it unique",
                path.display()
            ),
            None,
        );
    }
    let mut cmd = policy.command("/usr/bin/perl");
    arm_perl(edit, &mut cmd);
    frame_perl(
        run_watched(cmd, Some(DEFAULT_BASH_TIMEOUT), kill.clone()).await,
        Path::new(&edit.path),
    )
}

/// Set [`EDIT_PROGRAM`] and its arguments and environment on `cmd`.
fn arm_perl(edit: &Edit, cmd: &mut std::process::Command) {
    cmd.arg("-e")
        .arg(EDIT_PROGRAM)
        .arg("--")
        .arg(&edit.path)
        .env("TART_OLD", &edit.old_string)
        .env("TART_NEW", &edit.new_string)
        .envs(edit.replace_all.then_some(("TART_ALL", "1")));
}

/// Map one watched perl run to the message the model sees: perl's report on
/// success, its warning as retryable content otherwise.
fn frame_perl(run: io::Result<WatchedRun>, path: &Path) -> (String, Option<i32>) {
    match run {
        Ok(WatchedRun { output, .. }) if output.status.success() => (
            String::from_utf8_lossy(&output.stdout).trim_end().to_string(),
            Some(0),
        ),
        Ok(WatchedRun { output, stopped }) if stopped.is_some() => (
            frame_run(WatchedRun { output, stopped }, DEFAULT_BASH_TIMEOUT).0,
            None,
        ),
        Ok(WatchedRun { output, .. }) => (
            format!(
                "edit failed on {}: {}{}",
                path.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            Some(1),
        ),
        Err(error) => (
            format!("edit failed on {}: failed to run perl: {error}", path.display()),
            None,
        ),
    }
}

/// Run [`EDIT_PROGRAM`] through an already-configured `perl` command.
#[cfg(test)]
fn spawn_perl(edit: &Edit, cmd: &mut std::process::Command) -> (String, Option<i32>) {
    arm_perl(edit, cmd);
    frame_perl(
        run_watched_sync(cmd, Some(DEFAULT_BASH_TIMEOUT), &CancelToken::new()),
        Path::new(&edit.path),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use std::fmt::Write as _;

    use crate::sandbox::live::skip_unless_live;
    use macro_rules_attribute::apply;
    use std::time::Instant;

    /// A `bash` tool call requesting `command`.
    fn bash_call(arguments: &str) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: "bash".to_string(),
            arguments: arguments.to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
        }
    }

    #[test]
    fn bash_definition_has_one_required_command_parameter() {
        let tool = serde_json::to_value(bash()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "bash");
        assert_eq!(tool["parameters"]["required"][0], "command");
    }

    #[test]
    fn bash_definition_offers_an_optional_timeout_parameter() {
        let tool = serde_json::to_value(bash()).unwrap();

        assert_eq!(tool["parameters"]["properties"]["timeout"]["type"], "integer");
        // Only the command is required; the timeout keeps its default.
        assert_eq!(tool["parameters"]["required"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn parse_bash_reads_the_command_field() {
        assert_eq!(parse_bash(r#"{"command":"ls -la"}"#).unwrap().command, "ls -la");
    }

    #[test]
    fn parse_bash_defaults_the_timeout_when_absent() {
        let bash = parse_bash(r#"{"command":"ls"}"#).unwrap();

        assert_eq!(bash.timeout, DEFAULT_BASH_TIMEOUT);
    }

    #[test]
    fn parse_bash_clamps_out_of_range_timeouts() {
        let bash = parse_bash(r#"{"command":"sleep 5","timeout":9000}"#).unwrap();
        assert_eq!(bash.timeout, MAX_BASH_TIMEOUT);

        // Both ends: a negative joins the clamp rather than falling to the default
        let bash = parse_bash(r#"{"command":"ls","timeout":0}"#).unwrap();
        assert_eq!(bash.timeout, Duration::from_secs(1));
        let bash = parse_bash(r#"{"command":"ls","timeout":-5}"#).unwrap();
        assert_eq!(bash.timeout, Duration::from_secs(1));
    }

    #[test]
    fn parse_bash_ignores_a_wrong_typed_timeout() {
        let bash = parse_bash(r#"{"command":"ls","timeout":"300"}"#).unwrap();

        assert_eq!(bash.timeout, DEFAULT_BASH_TIMEOUT);
    }

    #[test]
    fn parse_bash_rejects_non_json() {
        let error = parse_bash("not json").unwrap_err().to_string();

        assert!(error.contains("weren't JSON"), "{error}");
    }

    #[test]
    fn parse_bash_rejects_a_missing_command() {
        let error = parse_bash(r#"{"other":1}"#).unwrap_err().to_string();

        assert!(error.contains("missing 'command'"), "{error}");
    }

    /// The tool result for a finished command with a Unix wait status: 0 is a success,
    /// `code << 8` an exit code, a small number a signal, framed as in the tui.
    fn framed(raw: i32, stdout: &str, stderr: &str) -> String {
        use std::os::unix::process::ExitStatusExt;

        let output = Output {
            status: std::process::ExitStatus::from_raw(raw),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        };
        command_text(&combined_output(&output), output.status)
    }

    #[test]
    fn command_result_passes_successful_output_through() {
        assert_eq!(framed(0, "hi\n", ""), "hi\n");
    }

    #[test]
    fn command_result_reports_a_silent_success_as_done() {
        assert_eq!(framed(0, "", ""), "done");
    }

    #[test]
    fn command_result_prefixes_a_failure_with_its_exit_code() {
        assert_eq!(framed(1 << 8, "hi\n", "boom\n"), "[exit 1]\nhi\nboom\n");
    }

    #[test]
    fn command_result_marks_a_silent_failure_with_just_the_exit_code() {
        assert_eq!(framed(1 << 8, "", ""), "[exit 1]");
    }

    #[test]
    fn command_result_marks_a_signal_death() {
        assert_eq!(framed(9, "", ""), "[exit signal]");
    }

    #[test]
    fn timeout_text_prefixes_the_partial_output() {
        assert_eq!(
            timeout_text("partial\ntail\n", DEFAULT_BASH_TIMEOUT),
            "[timed out after 120s]\npartial\ntail\n"
        );
        assert_eq!(
            timeout_text("still going\n", MAX_BASH_TIMEOUT),
            "[timed out after 600s]\nstill going\n"
        );
    }

    /// A short text passes through untouched; a long one keeps its head under a
    /// trailing marker, cut on a char boundary.
    #[test]
    fn head_cap_keeps_the_head_and_marks_the_cut() {
        assert_eq!(head_cap("hi\n", 10), "hi\n");

        let text = "abcdef".repeat(1024); // 6 KB
        let capped = head_cap(&text, 1024);
        let (kept, marker) = capped.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first 1 KB shown]");
        assert_eq!(kept, &text[..1024]);

        // A multi-byte head must not split a character at the cut.
        let wide = "語".repeat(4_000); // 12 KB of 3-byte characters
        let capped = head_cap(&wide, 1024);
        let (kept, marker) = capped.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first 1 KB shown]");
        assert!(kept.chars().all(|c| c == '語'), "cut inside a character");
    }

    #[test]
    fn timeout_text_without_output_is_just_the_marker() {
        assert_eq!(timeout_text("", DEFAULT_BASH_TIMEOUT), "[timed out after 120s]");
    }

    #[test]
    fn tail_cap_keeps_the_tail_and_marks_the_cut() {
        assert_eq!(tail_cap("hi\n", 10), "hi\n");

        let text = "abcdef".repeat(1024); // 6 KB
        let capped = tail_cap(&text, 1024);
        let (marker, tail) = capped.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; last 1 KB shown]");
        assert_eq!(tail, &text[text.len() - 1024..]);

        // A multi-byte tail must not split a character at the cut.
        let wide = "語".repeat(4_000); // 12 KB of 3-byte characters
        let capped = tail_cap(&wide, 1024);
        let (marker, tail) = capped.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; last 1 KB shown]");
        assert!(tail.chars().all(|c| c == '語'), "cut inside a character");
    }

    /// Manual runs reuse the bash tool's framing; these run unsandboxed
    #[test]
    fn manual_command_frames_success_and_failure() {
        assert_eq!(manual_command("echo hi", &CancelToken::new()), "hi\n");
        assert_eq!(manual_command("true", &CancelToken::new()), "done");
        assert_eq!(manual_command("false", &CancelToken::new()), "[exit 1]");
        assert_eq!(
            manual_command("echo boom >&2; exit 3", &CancelToken::new()),
            "[exit 3]\nboom\n"
        );
    }

    /// A token cancelled before the run kills the command at once; one cancelled
    /// mid-run keeps the output the command had already written.
    #[test]
    fn a_cancelled_manual_command_dies_with_its_group() {
        let ahead = CancelToken::new();
        ahead.cancel();
        let started = Instant::now();
        assert_eq!(
            manual_command("sleep 9871 & sleep 9871 & wait", &ahead),
            "[cancelled]"
        );
        assert!(started.elapsed() < Duration::from_secs(5));

        let live = CancelToken::new();
        let token = live.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            token.cancel();
        });
        let started = Instant::now();
        // The sleeps hold the output pipe, so returning promptly proves the
        // whole group died and not just bash.
        let framed = manual_command("echo started; sleep 9871 & sleep 9871 & wait", &live);
        assert_eq!(framed, "[cancelled]\nstarted\n");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Esc's kill lever stops a run immediately.
    #[test]
    fn the_kill_lever_beats_the_deadline() {
        let mut command = Command::new("/bin/bash");
        command
            .arg("-c")
            .arg("echo started; sleep 9871 & sleep 9871 & wait");
        let kill = CancelToken::new();
        let signer = kill.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            signer.cancel();
        });
        let started = Instant::now();

        let run =
            futures::executor::block_on(run_watched(command, Some(Duration::from_secs(600)), kill))
                .unwrap();

        assert_eq!(run.stopped, Some(Stopped::Esc));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn the_kill_lever_frees_a_fifo_wedge() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("wedge");
        let made = Command::new("/usr/bin/mkfifo").arg(&fifo).output().unwrap();
        assert!(made.status.success());
        let mut command = Command::new("/usr/bin/perl");
        command.arg("-e").arg(format!(
            "open(my $f, '<', '{}') or exit 1; print <$f>;",
            fifo.display()
        ));
        let kill = CancelToken::new();
        let signer = kill.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            signer.cancel();
        });
        let started = Instant::now();

        let run = futures::executor::block_on(run_watched(command, None, kill)).unwrap();

        assert_eq!(run.stopped, Some(Stopped::Esc));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// An untouched token leaves the command to finish on its own.
    #[test]
    fn manual_command_runs_to_completion_without_a_cancel() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");

        let run =
            futures::executor::block_on(run_watched(command, None, CancelToken::new())).unwrap();

        assert_eq!(run.stopped, None);
        assert_eq!(combined_output(&run.output), "hi\n");
    }

    /// The user's shell context is inherited: same working directory, same
    /// environment, unlike the web tools' cleared environment. `pwd -P`, not
    /// `$PWD`, since bash re-derives that from its own cwd at startup.
    #[test]
    fn manual_command_inherits_the_parents_directory_and_environment() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            manual_command("pwd -P", &CancelToken::new()),
            format!("{}\n", cwd.display())
        );
        if let Ok(home) = std::env::var("HOME") {
            assert_eq!(
                manual_command("printenv HOME", &CancelToken::new()),
                format!("{home}\n")
            );
        }
    }

    /// These drive `run_watched` with plain commands, so they run without
    /// the sandbox; their deadlines are milliseconds to stay fast.
    #[test]
    fn run_watched_returns_a_fast_command_normally() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");

        let run = futures::executor::block_on(run_watched(
            command,
            Some(Duration::from_secs(10)),
            CancelToken::new(),
        ))
        .unwrap();

        assert_eq!(run.stopped, None);
        assert_eq!(combined_output(&run.output), "hi\n");
        assert!(run.output.status.success());
    }

    #[test]
    fn run_watched_kills_a_command_that_outruns_the_deadline() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        let started = Instant::now();

        let run = futures::executor::block_on(run_watched(
            command,
            Some(Duration::from_millis(300)),
            CancelToken::new(),
        ))
        .unwrap();

        assert_eq!(run.stopped, Some(Stopped::Deadline));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!run.output.status.success());
    }

    #[test]
    fn run_watched_kills_the_whole_process_group() {
        // The backgrounded sleeps outlive bash and hold the output pipe; only a
        // group kill frees the capture, so returning promptly proves they died.
        let mut command = Command::new("/bin/bash");
        command.arg("-c").arg("sleep 9871 & sleep 9871 & wait");
        let started = Instant::now();

        let run = futures::executor::block_on(run_watched(
            command,
            Some(Duration::from_millis(300)),
            CancelToken::new(),
        ))
        .unwrap();

        assert_eq!(run.stopped, Some(Stopped::Deadline));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn run_watched_wakes_the_watchdog_when_the_command_finishes_early() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");
        let started = Instant::now();

        let run = futures::executor::block_on(run_watched(
            command,
            Some(DEFAULT_BASH_TIMEOUT),
            CancelToken::new(),
        ))
        .unwrap();

        assert_eq!(run.stopped, None);
        // The sender's drop joins the watchdog at once rather than letting it
        // sleep out the full timeout.
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// A killed run is framed for the model with the timeout marker, keeping
    /// whatever partial output the child had already written.
    #[test]
    fn run_framed_marks_a_killed_run_with_the_deadline() {
        let mut command = Command::new("/bin/bash");
        command
            .arg("-c")
            .arg("echo partial; sleep 9871 & sleep 9871 & wait");
        let started = Instant::now();

        let (result, display, exit) = futures::executor::block_on(run_framed(
            command,
            Duration::from_millis(300),
            CancelToken::new(),
        ));

        assert!(result.starts_with("[timed out after 0s]\npartial"), "{result}");
        // The pane carries the marker too, exactly as bash always has.
        assert_eq!(display, result);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(exit, None);
    }

    /// A successful run keeps bash's framing: output for the model, plain text
    /// for the pane, and the child's exit code.
    #[test]
    fn run_framed_passes_a_finished_run_through() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");

        let (result, display, exit) = futures::executor::block_on(run_framed(
            command,
            DEFAULT_BASH_TIMEOUT,
            CancelToken::new(),
        ));

        assert_eq!(result, "hi\n");
        assert_eq!(display, "hi\n");
        assert_eq!(exit, Some(0));
    }

    /// Live: reaches `sandbox-exec`, so it only passes outside a nested sandbox.
    #[apply(skip_unless_live!)]
    #[test]
    fn execute_reports_command_then_output() {
        let policy = Policy::new(std::env::current_dir().unwrap()).unwrap();
        let events = std::cell::RefCell::new(Vec::new());
        let output = futures::executor::block_on(execute(
            &bash_call(r#"{"command":"echo hi"}"#),
            &policy,
            &CancelToken::new(),
            &|progress| {
                events.borrow_mut().push(progress);
            },
        ))
        .unwrap();

        assert_eq!(output, "hi\n");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart {
                    id,
                    name: "Bash",
                    digest
                },
                Progress::ToolOutput {
                    output,
                    exit: Some(0),
                    ..
                }
            ] if id == "call_0" && digest == "echo hi" && output == "hi\n"
        ));

        // The exit status reaches the model verbatim.
        assert_eq!(
            futures::executor::block_on(execute(
                &bash_call(r#"{"command":"false"}"#),
                &policy,
                &CancelToken::new(),
                &|_| {},
            ))
            .unwrap(),
            "[exit 1]"
        );
        assert_eq!(
            futures::executor::block_on(execute(
                &bash_call(r#"{"command":"true"}"#),
                &policy,
                &CancelToken::new(),
                &|_| {},
            ))
            .unwrap(),
            "done"
        );
    }

    #[test]
    fn execute_rejects_unknown_tool_names() {
        let policy = Policy::new(std::env::current_dir().unwrap()).unwrap();
        let mut call = bash_call(r#"{"command":"ls"}"#);
        call.name = "rm".to_string();

        let error =
            futures::executor::block_on(execute(&call, &policy, &CancelToken::new(), &|_| {}))
                .unwrap_err()
                .to_string();

        assert!(error.contains("unknown tool"), "{error}");
    }

    /// A temporary file holding `contents`, removed when the guard drops.
    fn scratch(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    /// An `edit` tool call replacing `old` with `new` in `path`.
    fn edit_call(path: &Path, old: &str, new: &str) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: "edit".to_string(),
            arguments: serde_json::json!({"path": path, "old_string": old, "new_string": new})
                .to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
        }
    }

    /// Drive [`EDIT_PROGRAM`] with a plain, unsandboxed perl command.
    fn perl_edit(path: &Path, old: &str, new: &str, replace_all: bool) -> String {
        spawn_perl(
            &Edit {
                path: path.display().to_string(),
                old_string: old.to_string(),
                new_string: new.to_string(),
                replace_all,
            },
            &mut std::process::Command::new("/usr/bin/perl"),
        )
        .0
    }

    #[test]
    fn perl_edit_replaces_a_unique_multiline_string_literally() {
        let file = scratch("line1: cost $5.00 (a)\nline2: b.*x [y]\nline3\n");
        let output = perl_edit(file.path(), "b.*x [y]\nline3", r"REPL($1)$&\E", false);

        assert_eq!(
            output,
            format!("edited {}: 1 replacement(s)", file.path().display())
        );
        assert_eq!(
            std::fs::read_to_string(file.path()).unwrap(),
            "line1: cost $5.00 (a)\nline2: REPL($1)$&\\E\n"
        );
    }

    #[test]
    fn perl_edit_replaces_every_occurrence_with_replace_all() {
        let file = scratch("a a a\n");
        let output = perl_edit(file.path(), "a", "b", true);

        assert_eq!(
            output,
            format!("edited {}: 3 replacement(s)", file.path().display())
        );
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "b b b\n");
    }

    #[test]
    fn perl_edit_deletes_via_an_empty_new_string() {
        let file = scratch("keep\ndrop me\nkeep\n");
        let output = perl_edit(file.path(), "drop me\n", "", false);

        assert!(output.contains("1 replacement(s)"), "{output}");
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "keep\nkeep\n");
    }

    #[test]
    fn perl_edit_reports_a_missing_match_and_leaves_the_file_untouched() {
        let file = scratch("alpha beta\n");
        let output = perl_edit(file.path(), "gamma", "delta", false);

        assert!(output.contains("not found"), "{output}");
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "alpha beta\n");
    }

    #[test]
    fn perl_edit_reports_an_ambiguous_match_without_replace_all() {
        let file = scratch("x x x\n");
        let output = perl_edit(file.path(), "x", "y", false);

        assert!(output.contains("matches 3 times"), "{output}");
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "x x x\n");
    }

    #[test]
    fn perl_edit_reports_an_unopenable_file() {
        let missing = std::env::temp_dir().join("tart-edit-does-not-exist");
        let output = perl_edit(&missing, "a", "b", false);

        assert!(output.contains("cannot open"), "{output}");
    }

    #[test]
    fn concurrent_perl_edits_to_one_file_both_apply() {
        let file = scratch("AA eleven\nmid\nBB twelve\n");
        let spawn_edit = |old: &'static str, new: &'static str| {
            let path = file.path().to_path_buf();
            std::thread::spawn(move || perl_edit(&path, old, new, false))
        };
        let first = spawn_edit("AA", "aa");
        let second = spawn_edit("BB", "bb");

        assert!(first.join().unwrap().contains("1 replacement(s)"));
        assert!(second.join().unwrap().contains("1 replacement(s)"));
        assert_eq!(
            std::fs::read_to_string(file.path()).unwrap(),
            "aa eleven\nmid\nbb twelve\n"
        );
    }

    /// Live: reaches `sandbox-exec`, so it only passes outside a nested sandbox.
    #[apply(skip_unless_live!)]
    #[test]
    fn concurrent_edits_to_one_file_both_apply() {
        let file = scratch("one UNO alpha\ntwo DOS beta\n");
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let spawn_edit = |old: &str, new: &str| {
            let call = edit_call(file.path(), old, new);
            let policy = policy.clone();
            std::thread::spawn(move || {
                futures::executor::block_on(execute(&call, &policy, &CancelToken::new(), &|_| {}))
                    .unwrap()
            })
        };
        let first = spawn_edit("UNO", "uno");
        let second = spawn_edit("DOS", "dos");

        assert!(first.join().unwrap().contains("1 replacement(s)"));
        assert!(second.join().unwrap().contains("1 replacement(s)"));
        assert_eq!(
            std::fs::read_to_string(file.path()).unwrap(),
            "one uno alpha\ntwo dos beta\n"
        );
    }

    #[test]
    fn read_definition_requires_path() {
        let tool = serde_json::to_value(read()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "read");
        assert_eq!(tool["parameters"]["required"][0], "path");
        assert!(tool["parameters"]["properties"]["start_line"].is_object());
        assert!(tool["parameters"]["properties"]["end_line"].is_object());
    }

    #[test]
    fn parse_read_reads_the_path_and_bounds() {
        let read = parse_read(r#"{"path":"src/main.rs","start_line":10,"end_line":50}"#).unwrap();

        assert_eq!(read.path, "src/main.rs");
        assert_eq!((read.start_line, read.end_line), (Some(10), Some(50)));

        let whole = parse_read(r#"{"path":"src/main.rs"}"#).unwrap();
        assert_eq!((whole.start_line, whole.end_line), (None, None));
    }

    #[test]
    fn parse_read_rejects_a_missing_path() {
        let error = parse_read(r#"{"start_line":1}"#).unwrap_err().to_string();

        assert!(error.contains("missing 'path'"), "{error}");
    }

    /// A `read` tool call for `path`, optionally bounded to a line range.
    fn read_call(path: &Path, start_line: Option<u64>, end_line: Option<u64>) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: "read".to_string(),
            arguments: serde_json::json!({
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            })
            .to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
        }
    }

    #[test]
    fn reads_wait_for_the_edit_lock() {
        let file = scratch("one\ntwo\n");
        // Stand in for an in-flight edit: an exclusive flock, as edit.pl takes.
        let editor = std::fs::OpenOptions::new()
            .append(true)
            .open(file.path())
            .unwrap();
        editor.lock().unwrap();
        let path = file.path().to_path_buf();
        let reader = std::thread::spawn(move || {
            let output = std::process::Command::new("/usr/bin/perl")
                .arg("-e")
                .arg(numbered_read(None, None))
                .arg("--")
                .arg(&path)
                .output()
                .unwrap();
            String::from_utf8_lossy(&output.stdout).to_string()
        });

        // The read blocks on the shared lock until the "edit" finishes.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!reader.is_finished());
        drop(editor);
        assert_eq!(reader.join().unwrap(), "     1\tone\n     2\ttwo\n");
    }

    /// Live: reaches `sandbox-exec`, so it only passes outside a nested sandbox.
    #[apply(skip_unless_live!)]
    #[test]
    fn read_returns_numbered_contents_and_ranges() {
        let mut contents = String::new();
        for line in 1..=30 {
            let _ = writeln!(contents, "line {line}");
        }
        let file = scratch(&contents);
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let events = std::cell::RefCell::new(Vec::new());

        let whole = futures::executor::block_on(execute(
            &read_call(file.path(), None, None),
            &policy,
            &CancelToken::new(),
            &|progress| {
                events.borrow_mut().push(progress);
            },
        ))
        .unwrap();

        assert!(whole.starts_with("     1\tline 1\n"), "{whole}");
        assert!(whole.ends_with("    30\tline 30\n"), "{whole}");
        assert_eq!(whole.lines().count(), 30);
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart {
                    name: "Read",
                    digest,
                    ..
                },
                Progress::ToolOutput { .. }
            ] if digest == &file.path().display().to_string()
        ));

        // A bounded read digests its range into the box header.
        events.borrow_mut().clear();
        let range = futures::executor::block_on(execute(
            &read_call(file.path(), Some(10), Some(12)),
            &policy,
            &CancelToken::new(),
            &|progress| {
                events.borrow_mut().push(progress);
            },
        ))
        .unwrap();
        assert_eq!(range, "    10\tline 10\n    11\tline 11\n    12\tline 12\n");
        assert!(matches!(
            events.borrow().as_slice(),
            [Progress::ToolStart { digest, .. }, _]
                if digest == &format!("{}:10-12", file.path().display())
        ));

        let tail = futures::executor::block_on(execute(
            &read_call(file.path(), Some(28), None),
            &policy,
            &CancelToken::new(),
            &|_| {},
        ))
        .unwrap();
        assert!(tail.starts_with("    28\tline 28\n"), "{tail}");
        assert_eq!(tail.lines().count(), 3);

        let missing = std::env::temp_dir().join("tart-read-does-not-exist");
        let absent = futures::executor::block_on(execute(
            &read_call(&missing, None, None),
            &policy,
            &CancelToken::new(),
            &|_| {},
        ))
        .unwrap();
        assert!(absent.contains("No such file or directory"), "{absent}");
    }
}
