use std::io;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime};

use crate::backends::{FunctionToolCall, Tool, tool};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;

use crate::backends::Backend;
use crate::backends::openai_responses::Responses;
use crate::{Agent, AgentId, Agents, ChatMode, Progress, sandbox::Policy};

mod web;

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

/// read.pl's exit code for a missing file.
const READ_EXIT_NOT_FOUND: i32 = 3;

/// The most of any one blob the model is handed, in bytes.
pub const CONTENT_CAP: usize = 64 * 1024;

/// How often a manual command's watchdog wakes to check its cancel token.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// The front end's control for a running manual command.
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
    pub(crate) fn cancelled(&self) -> bool {
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

/// The bash tool; commands execute under the caller's [`Policy`].
#[must_use]
pub(crate) fn bash() -> Tool {
    tool(
        "bash",
        "Run a bash command in a sandbox (writes restricted to granted roots, no network) \
        and return its stdout, with stderr as a separate paragraph below",
        // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/bash.ts#L565>
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in milliseconds (max 600000)"
                }
            },
            "required": ["command"]
        }),
    )
}

/// The read tool; files are read under the caller's [`Policy`].
///
/// The parameters match Claude Code's Read tool (and zcode's): `file_path`, with a
/// 1-based `offset` and a line-count `limit`, rather than absolute end bounds.
#[must_use]
pub(crate) fn read() -> Tool {
    tool(
        "read",
        "Read a file with line numbers (cat -n style) in a sandbox (reads restricted to \
        granted roots); optionally pass an offset (1-based line to start from) and a \
        limit (number of lines) to read a range",
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from, 1-based; omit to start at the top"},
                "limit": {"type": "integer", "description": "Number of lines to read; omit to read to the end"}
            },
            "required": ["file_path"]
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
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "Text to replace; must match exactly and be unique unless replace_all"},
                "new_string": {"type": "string", "description": "Replacement text; empty deletes old_string"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of one unique match"}
            },
            "required": ["file_path", "old_string", "new_string"]
        }),
    )
}

/// The `spawn_agent` tool; only the main agent is offered it.
#[must_use]
pub(crate) fn spawn_agent() -> Tool {
    tool(
        "spawn_agent",
        "Spawn a subagent for a well-scoped task. Returns an id immediately; the subagent \
        runs independently with your tools (minus `spawn_agent` and `check_agent`) and its \
        final message becomes its report, delivered to you as a message when it finishes. \
        `check_agent` can check for it without blocking, but waiting is never required. \
        Only call this tool for a concrete, bounded subtask that can run independently \
        alongside useful local work; otherwise continue locally. Do not spawn subagents \
        unless the user explicitly asks for subagents, delegation, or parallel agent work. \
        At most 8 subagents run or await delivery at once",
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The complete task for the subagent: it sees nothing else of this conversation"}
            },
            "required": ["task"]
        }),
    )
}

/// The `check_agent` tool; only the main agent is offered it.
#[must_use]
pub(crate) fn check_agent() -> Tool {
    tool(
        "check_agent",
        "Check one subagent's status without blocking. Returns its report when it has \
        finished (claiming it, so it will not also arrive as a message), or says it is \
        still running: in which case end the turn and let the report arrive on its own \
        instead of polling. Check only when the very next step is blocked on the result and you are unsure whether the agent is making progress.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The subagent's id, as `spawn_agent` reported it"}
            },
            "required": ["id"]
        }),
    )
}

/// Parse a tool call's arguments as JSON.
fn parse_arguments(arguments: &str) -> anyhow::Result<serde_json::Value> {
    serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))
}

/// A required string field from parsed tool arguments, absent ones reported.
fn string_field(args: &serde_json::Value, tool: &str, name: &str) -> anyhow::Result<String> {
    args[name].as_str().map(str::to_string).ok_or_else(|| {
        // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/input-validation-model-content.ts#L18>
        anyhow::anyhow!(
            "InputValidationError: {tool} failed due to the following issue:\nThe required \
             parameter `{name}` is missing"
        )
    })
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
///
/// The timeout arrives in milliseconds as a JSON number, clamped to the 10-minute
/// ceiling with fractions rounded to the nearest millisecond. A request under a second
/// is refused rather than clamped.
fn parse_bash(arguments: &str) -> anyhow::Result<Bash> {
    let args = parse_arguments(arguments)?;
    let requested = args["timeout"].as_f64();
    if requested.is_some_and(|milliseconds| milliseconds < 1000.0) {
        anyhow::bail!("timeout is measured in milliseconds; pass at least 1000");
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the clamp bounds the value to 1-600 seconds before the cast"
    )]
    let milliseconds = requested
        .unwrap_or(DEFAULT_BASH_TIMEOUT.as_millis() as f64)
        .clamp(1000.0, MAX_BASH_TIMEOUT.as_millis() as f64)
        .round() as u64;
    Ok(Bash {
        command: string_field(&args, "bash", "command")?,
        timeout: Duration::from_millis(milliseconds),
    })
}

/// One parsed read tool call.
#[derive(Debug)]
struct Read {
    /// The file to read.
    file_path: String,
    /// Line number to start from, 1-based; `None` starts at the top.
    offset: Option<u64>,
    /// How many lines to read; `None` reads to the end.
    limit: Option<u64>,
}

impl Read {
    /// The inclusive line bounds the perl reader takes: an offset becomes the
    /// first line, a limit counts from it (or from the top).
    fn bounds(&self) -> (Option<u64>, Option<u64>) {
        match (self.offset, self.limit) {
            (Some(offset), Some(limit)) => {
                (Some(offset), Some(offset.saturating_add(limit).saturating_sub(1)))
            }
            (Some(offset), None) => (Some(offset), None),
            // A limit without an offset reads the first `limit` lines.
            (None, Some(limit)) => (None, Some(limit)),
            (None, None) => (None, None),
        }
    }
}

/// Extract the fields from a read tool call's JSON arguments.
fn parse_read(arguments: &str) -> anyhow::Result<Read> {
    let args = parse_arguments(arguments)?;
    Ok(Read {
        file_path: string_field(&args, "read", "file_path")?,
        offset: args["offset"].as_u64().filter(|&offset| offset > 0),
        limit: args["limit"].as_u64().filter(|&limit| limit > 0),
    })
}

/// One parsed edit tool call.
#[derive(Debug)]
struct Edit {
    /// The file to edit.
    file_path: String,
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
        file_path: string_field(&args, "edit", "file_path")?,
        old_string: string_field(&args, "edit", "old_string")?,
        new_string: string_field(&args, "edit", "new_string")?,
        replace_all: args["replace_all"].as_bool().unwrap_or(false),
    })
}

/// One parsed spawn tool call.
#[derive(Debug)]
struct Spawn {
    /// The complete, self-contained task the subagent runs on.
    task: String,
}

/// Extract the fields from a spawn tool call's JSON arguments.
fn parse_spawn(arguments: &str) -> anyhow::Result<Spawn> {
    let args = parse_arguments(arguments)?;
    Ok(Spawn {
        task: string_field(&args, "spawn_agent", "task")?,
    })
}

/// One parsed `check_agent` tool call.
#[derive(Debug)]
struct Check {
    /// The subagent to check on, as `spawn_agent` reported it.
    id: u64,
}

/// Extract the fields from a `check_agent` tool call's JSON arguments.
fn parse_check(arguments: &str) -> anyhow::Result<Check> {
    let args = parse_arguments(arguments)?;
    Ok(Check {
        id: args["id"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("check_agent needs an integer 'id'"))?,
    })
}

/// Run one tool call under `policy`, report each step to `on_progress`, and
/// return its output to the model.
///
/// Each tool announces itself with [`Progress::ToolStart`] and always follows with
/// a [`Progress::ToolOutput`] so the front end knows the task concluded. Malformed tool
/// calls return their error as output.
///
/// Tool *failures* (a non-zero exit, an edit that did not apply, or a command the
/// sandbox denies) are content that the model should see, and so is a *malformed call*.
pub(crate) fn execute<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    // Force-disable tools, lest hallucinated calls attempt execution regardless.
    if tools.template.mode() == ChatMode::Chat
        && matches!(
            call.name.as_str(),
            "bash" | "read" | "edit" | "spawn_agent" | "check_agent"
        )
    {
        return misuse(
            call,
            on_progress,
            &anyhow::anyhow!("the {} tool is not available in chat mode", call.name),
        );
    }
    match call.name.as_str() {
        "bash" => run_bash(call, tools, on_progress),
        "read" => run_read(call, tools, on_progress),
        "edit" => run_edit(call, tools, on_progress),
        // The unsandboxed pair: they need the network the sandbox denies.
        "search" => web::run_search(call, on_progress),
        "fetch" => web::run_fetch(call, on_progress),
        // The subagent pair, offered to spawning agents only.
        "spawn_agent" => run_spawn_agent(call, tools, on_progress),
        "check_agent" => run_check_agent(call, tools, on_progress),
        // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/executor/call-runner.ts#L128>
        other => misuse(call, on_progress, &anyhow::anyhow!("Tool not found: {other}")),
    }
}

/// Fork a subagent on the task and return at once.
fn run_spawn_agent<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    let spawn = match parse_spawn(&call.arguments) {
        Ok(spawn) => spawn,
        Err(error) => return misuse(call, on_progress, &error),
    };
    // A subagent can still call a tool it was never offered, so this too is passed
    // content for the model, not a harness failure.
    let Some(agents) = tools.agents else {
        return misuse(
            call,
            on_progress,
            &anyhow::anyhow!("subagents cannot spawn their own"),
        );
    };
    traced(call, on_progress, || {
        match agents.spawn(tools.template, &spawn.task) {
            Ok(id) => {
                let text = format!("started subagent {id}: {}", spawn.task);
                (text.clone(), text, Some(0))
            }
            Err(error) => {
                let text = error.to_string();
                (text.clone(), text, None)
            }
        }
    })
}

/// Run one `check_agent` tool call: an instant check, never a block.
fn run_check_agent<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    let check = match parse_check(&call.arguments) {
        Ok(check) => check,
        Err(error) => return misuse(call, on_progress, &error),
    };
    // As with spawn: a call a subagent was never offered is content, not failure.
    let Some(agents) = tools.agents else {
        return misuse(
            call,
            on_progress,
            &anyhow::anyhow!("subagents have no subagents to check on"),
        );
    };
    let id = check.id;
    traced(call, on_progress, || {
        match agents.claim(AgentId::from(id)) {
            // Attribute errors from children as the wait's own error, as we can't
            // distinguish the two (TODO: fix?).
            Ok(Some(outcome)) => {
                let text = format!("subagent {id} finished: {}", outcome.report());
                (text.clone(), text, Some(0))
            }
            Ok(None) => {
                let text = format!(
                    "subagent {id} is still running; its report arrives as a \
                     message when it finishes, so end the turn rather than \
                     waiting for a result"
                );
                (text.clone(), text, None)
            }
            Err(error) => {
                let text = error.to_string();
                (text.clone(), text, None)
            }
        }
    })
}

/// One finished process's combined streams: stdout, then stderr as its own paragraph.
fn combined_output(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (true, false) => stderr.into_owned(),
        (false, true) => stdout.into_owned(),
        (false, false) => format!("{stdout}\n{stderr}"),
    }
}

/// Whether an exit code of 1 from `command` counts as success.
fn exit_one_is_success(command: &str) -> bool {
    // The exempt list is transcribed from Claude Code's documented Bash behavior.
    let mut words = command.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    // A leading path belongs to the binary's name.
    let first = first.rsplit('/').next().unwrap_or(first);
    match first {
        "grep" | "rg" | "egrep" | "fgrep" | "find" | "diff" | "test" | "[" => true,
        "git" => matches!(words.next(), Some("diff" | "grep")),
        _ => false,
    }
}

/// Maximum output sizes in bytes before the data is spilled to a file.
const BASH_SUCCESS_CAP: usize = 30_000;
const BASH_FAILURE_CAP: usize = 10_000;

/// How much of a spilled output previews inline, in characters.
const SPILL_PREVIEW: usize = 2_000;

/// The first `max_chars` characters of `text`, snapped back to a newline in the
/// back half so a cut lands on a line boundary; the flag says it was cut.
// <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/result-persistence-format.ts#L50>
fn preview_first_chars(text: &str, max_chars: usize) -> (&str, bool) {
    // `None` means at most `max_chars` characters: there is no cut to make.
    let Some(end) = text.char_indices().nth(max_chars).map(|(index, _)| index) else {
        return (text, false);
    };
    let head = &text[..end];
    let snapped = head
        .rfind('\n')
        .filter(|&newline| 2 * head[..newline].chars().count() > max_chars)
        .unwrap_or(end);
    (&head[..snapped], true)
}

/// A size in binary units with one decimal, a trailing `.0` trimmed: `2KB`, `1.5MB`.
// <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/bash-model-content.ts#L177>
fn bash_byte_size(bytes: usize) -> String {
    let unit = |value: f64| {
        let fixed = format!("{value:.1}");
        fixed.strip_suffix(".0").unwrap_or(&fixed).to_owned()
    };
    let kilobytes = bytes as f64 / 1024.0;
    if bytes == 0 {
        "0 bytes".to_string()
    } else if kilobytes < 1.0 {
        format!("{bytes} bytes")
    } else if kilobytes < 1024.0 {
        format!("{}KB", unit(kilobytes))
    } else if kilobytes < 1024.0 * 1024.0 {
        format!("{}MB", unit(kilobytes / 1024.0))
    } else {
        format!("{}GB", unit(kilobytes / 1024.0 / 1024.0))
    }
}

/// The model-facing framing of a finished command's text; the bash tool's, which
/// spills an oversized success to a scratch file.
fn command_text(text: &str, status: ExitStatus, one_is_success: bool) -> String {
    frame(text, status, one_is_success, true)
}

/// The framing for callers whose text is returned inline however large: the web
/// tools and the user's own manual commands.
fn command_text_inline(text: &str, status: ExitStatus, one_is_success: bool) -> String {
    frame(text, status, one_is_success, false)
}

/// The shared framing: the exit-one exemption, a trailing exit code on failure,
/// and (when `spill` is set) an oversized success written to a scratch file.
fn frame(text: &str, status: ExitStatus, one_is_success: bool, spill: bool) -> String {
    let success = status.success() || (one_is_success && status.code() == Some(1));
    if success {
        if spill && text.len() > BASH_SUCCESS_CAP {
            return spill_to_file(text);
        }
        return if text.is_empty() {
            "(Bash completed with no output)".to_string()
        } else {
            text.to_string()
        };
    }

    if text.len() > BASH_FAILURE_CAP {
        return append_line(&excerpt(text), &exit_line(status));
    }
    append_line(text, &exit_line(status))
}

/// `text` with `line` appended as its own final line, without doubling a trailing `\n`
fn append_line(text: &str, line: &str) -> String {
    if text.is_empty() || text.ends_with('\n') {
        format!("{text}{line}")
    } else {
        format!("{text}\n{line}")
    }
}

/// The trailing failure line: the exit code, or the shell's conventional
/// 128+signal for a death by signal.
fn exit_line(status: ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;

    let code = status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal));
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/bash-model-content.ts#L52>
    format!("Exit code {}", code.unwrap_or(1))
}

/// A unique-enough scratch suffix: nanoseconds since the epoch.
fn nanos() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos())
}

/// Write `text` to a private scratch file and point the model at it, previewing head.
fn spill_to_file(text: &str) -> String {
    let dir = std::env::temp_dir().join("tart");
    let name = format!("tart-bash-{}-{}.output", std::process::id(), nanos());
    let path = dir.join(name);
    if write_private(&dir, &path, text).is_err() {
        return excerpt(text);
    }
    let (preview, more) = preview_first_chars(text, SPILL_PREVIEW);
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/result-persistence-format.ts#L29>
    let tail = if more { "...\n" } else { "" };
    format!(
        "<persisted-output>\nOutput too large ({}). Full output saved to: {}\n\nPreview (first \
         {}):\n{preview}\n{tail}</persisted-output>",
        bash_byte_size(text.len()),
        path.display(),
        bash_byte_size(SPILL_PREVIEW),
    )
}

/// Create `dir` and write `text` to `path` owner-only, so a command's output
/// spilled out of the transcript cannot be read by other local users.
fn write_private(dir: &Path, path: &Path, text: &str) -> io::Result<()> {
    use std::io::Write as _;

    // A pre-planted symlink would otherwise redirect the write (and the chmod).
    if dir.symlink_metadata().is_ok_and(|meta| meta.is_symlink()) {
        return Err(io::Error::other("scratch directory is a symlink"));
    }
    // Private from the moment of creation, and tightened if it predates us.
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(text.as_bytes())
}

/// A head-and-tail excerpt of `text` at the failure cap: [`bounded`], whose
/// marker keeps the seam from reading as contiguous output.
fn excerpt(text: &str) -> String {
    bounded(text.to_string(), BASH_FAILURE_CAP)
}

/// Information required for a tool call, including sandbox and cancellation info.
pub(crate) struct Tooling<'a, B: Backend = Responses> {
    /// The policy the call's commands run sandboxed under.
    pub(crate) policy: &'a Policy,
    /// The turn's cancel lever: Esc kills a command in flight.
    pub(crate) cancel: &'a CancelToken,
    /// The subagent registry, when this agent can spawn.
    pub(crate) agents: Option<&'a Agents>,
    /// The agent whose turn this is: the template a `spawn_agent` clones.
    pub(crate) template: &'a Agent<B>,
}

/// One finished command run under a watchdog: a deadline or a cancel.
struct WatchedRun {
    /// The command's streams and exit status, as `output()` returns them.
    output: Output,
    /// Why the watchdog killed the process group, when it did.
    killed: Option<KillReason>,
}

/// Which lever a watchdog pulled to kill a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KillReason {
    /// The deadline passed.
    Timeout,
    /// The cancel token fired.
    Cancelled,
}

/// Model-facing explanation for a command the timeout killed.
// <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/bash-semantics.ts#L105>
fn timeout_text(text: &str, timeout: Duration) -> String {
    format!(
        "{}\nCommand timed out after {}",
        append_line(text, "Exit code 137"),
        duration_text(timeout)
    )
}

/// Styled duration: `1s`, `2m 0s`, `6m 40s`.
fn duration_text(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {}s", seconds / 60, seconds % 60)
    }
}

/// Model-facing explanation for a command the user cancelled with Esc.
fn cancel_text(text: &str) -> String {
    let separator = if text.is_empty() { "" } else { "\n" };
    format!("[cancelled]{separator}{text}")
}

/// The first `head` and last `tail` bytes of `text`, both snapped to character
/// boundaries so neither slice can split one; the two never overlap.
fn ends(text: &str, head: usize, tail: usize) -> (&str, &str) {
    // Slicing must land on a char boundary; lossy decoding made `text` valid.
    let head_end = text.floor_char_boundary(head.min(text.len()));
    let tail_start = text.ceil_char_boundary(text.len().saturating_sub(tail));
    let head_end = head_end.min(tail_start);
    (&text[..head_end], &text[tail_start..])
}

/// Keep the first `cap` bytes of `text`, suffixing a marker when it cut: an
/// attached file reads from the top, where its interesting part usually is.
///
/// ```
/// use tart_agents::head_cap;
///
/// assert_eq!(head_cap("hi\n", 10), "hi\n");
///
/// let text = "abcdef".repeat(1024); // 6 KB
/// let capped = head_cap(&text, 1024);
/// let (kept, marker) = capped.split_once('\n').unwrap();
/// assert_eq!(kept, &text[..1024]);
/// assert_eq!(marker, "[truncated; first 1 KB shown]");
/// ```
#[inline]
pub fn head_cap(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    let (head, _) = ends(text, cap, 0);
    format!("{head}\n[truncated; first {} KB shown]", cap / 1024)
}

/// Keep the last `cap` bytes of `text`, prefixing a marker when it cut.
fn tail_cap(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    let (_, tail) = ends(text, 0, cap);
    format!("[truncated; last {} KB shown]\n{tail}", cap / 1024)
}

/// Keep the first and last `cap / 2` bytes of `text`, marking the omitted
/// middle; under-cap text moves through untouched, so a caller handing over
/// dead ownership pays nothing.
pub fn bounded(text: String, cap: usize) -> String {
    if text.len() <= cap {
        return text;
    }
    let half = cap / 2;
    let (head, tail) = ends(&text, half, half);
    format!(
        "{head}\n[truncated; first and last {} KB shown]\n{tail}",
        half / 1024
    )
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

/// Run `command` to completion, killing its process group the moment `cancel`
/// fires or, with a `deadline`, once it outlives it.
///
/// The watchdog polls both levers, so a kill lands within one [`CANCEL_POLL`]
/// of its cause. Four rules hold arm-for-arm: the reason is reported before
/// the kill lands, the reaped child's sender drop wakes the watchdog, the
/// group dies even when the wait errors, and the watchdog joins before
/// returning.
fn run_watched(
    command: &mut Command,
    deadline: Option<Duration>,
    cancel: &CancelToken,
) -> io::Result<WatchedRun> {
    let (child, group) = spawn_grouped(command)?;

    // The watchdog's kill reason, when it killed: sent before the kill lands
    // and read after the watchdog joins, so a killed group is never mistaken
    // for a natural exit.
    let (killer, killed_by) = mpsc::channel::<KillReason>();
    // The sender drops once the command is reaped, waking the watchdog at once.
    let (finished, slept) = mpsc::channel::<()>();
    let token = cancel.clone();
    let end = deadline.map(|timeout| Instant::now() + timeout);
    let watchdog = std::thread::spawn(move || {
        loop {
            if token.cancelled() {
                let _ = killer.send(KillReason::Cancelled);
                let _ = killpg(group, Signal::SIGKILL);
                return;
            }
            // Sleep only until the nearer of the poll and the deadline, so a
            // deadline kills on the right cycle
            let wait = CANCEL_POLL
                .min(end.map_or(CANCEL_POLL, |end| end.saturating_duration_since(Instant::now())));
            if matches!(slept.recv_timeout(wait), Err(RecvTimeoutError::Disconnected)) {
                return;
            }
            if end.is_some_and(|end| Instant::now() >= end) {
                let _ = killer.send(KillReason::Timeout);
                let _ = killpg(group, Signal::SIGKILL);
                return;
            }
        }
    });

    let output = child.wait_with_output();
    drop(finished);
    // Double check we've killed everything even if the wait errors.
    if output.is_err() {
        let _ = killpg(group, Signal::SIGKILL);
    }
    let _ = watchdog.join();

    Ok(WatchedRun {
        output: output?,
        killed: killed_by.try_recv().ok(),
    })
}

/// Announce a tool call, run it, and report its conclusion.
///
/// The announcement carries the call's identity as the provider sent it; how
/// that reads on screen is the front end's business.
fn traced<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
    run: impl FnOnce() -> (String, String, Option<i32>),
) -> String {
    on_progress(Progress::ToolStart {
        id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    });
    let (result, output, exit) = run();
    // Display-only copy of the text is bounded to match the string we pass to the model
    let output = bounded(output, CONTENT_CAP);
    on_progress(Progress::ToolOutput {
        id: call.call_id.clone(),
        output,
        exit,
    });
    result
}

/// A malformed call the model can fix returns its error as the call's output.
///
/// The parse helpers would rather error than invent defaults, and routing those errors
/// through here keeps them content the model acts on and retries.
fn misuse<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
    error: &anyhow::Error,
) -> String {
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/executor/errors.ts#L22>
    let text = format!("<tool_use_error>{error}</tool_use_error>");
    traced(call, on_progress, || (text.clone(), text, None))
}

/// Run one bash tool call under `tools`, reporting its steps to `on_progress`.
///
/// A command that outlives its timeout is killed with everything it started.
fn run_bash<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    let bash = match parse_bash(&call.arguments) {
        Ok(bash) => bash,
        Err(error) => return misuse(call, on_progress, &error),
    };
    // A failure to launch comes back as an error string, so the output can be
    // handed straight back to the model.
    let mut sandboxed = tools.policy.command("/bin/bash");
    sandboxed.arg("-c").arg(&bash.command);
    traced(call, on_progress, || {
        // Decode the stream into output for the front and backends.
        match run_watched(&mut sandboxed, Some(bash.timeout), tools.cancel) {
            Ok(run) => {
                let WatchedRun { output, killed } = run;
                let text = combined_output(&output);
                let exit = output.status.code();
                let one_is_success = exit == Some(1) && exit_one_is_success(&bash.command);
                match killed {
                    // A kill has no exit status of its own, so we mark up the body.
                    Some(KillReason::Timeout) => {
                        let marked = timeout_text(&text, bash.timeout);
                        (marked.clone(), marked, exit)
                    }
                    Some(KillReason::Cancelled) => {
                        let marked = cancel_text(&text);
                        (marked.clone(), marked, exit)
                    }
                    None => {
                        // An exempt "no match" exit is a success everywhere: the
                        // box paints green instead of showing `exit 1`.
                        let reported = if one_is_success { Some(0) } else { exit };
                        (command_text(&text, output.status, one_is_success), text, reported)
                    }
                }
            }
            Err(error) => {
                let text = format!("error: {error}");
                (text.clone(), text, None)
            }
        }
    })
}

/// Run one command the user typed, with their privileges.
#[inline]
pub fn manual_command(command: &str, cancel: &CancelToken) -> String {
    let mut shell = Command::new("/bin/bash");
    shell.arg("-c").arg(command);
    match run_watched(&mut shell, None, cancel) {
        Ok(WatchedRun { output, killed }) => {
            let text = tail_cap(&combined_output(&output), CONTENT_CAP);
            match killed {
                Some(KillReason::Cancelled) => cancel_text(&text),
                // No deadline exists to outlive, so nothing else kills it.
                _ => command_text_inline(&text, output.status, false),
            }
        }
        Err(error) => format!("error: {error}"),
    }
}

/// Run one read tool call under `tools`, reporting its steps to `on_progress`.
fn run_read<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    let read = match parse_read(&call.arguments) {
        Ok(read) => read,
        Err(error) => return misuse(call, on_progress, &error),
    };
    let (start_line, end_line) = read.bounds();
    let mut command = tools.policy.command("/usr/bin/perl");
    command
        .arg("-e")
        .arg(numbered_read(start_line, end_line))
        .arg("--")
        .arg(&read.file_path);
    traced(call, on_progress, || {
        // A failure to launch comes back as an error string for the model to deal with.
        match &command.output() {
            Ok(spawned) => {
                let text = combined_output(spawned);
                // A missing file reports its typed exit code, or perl warning if unknown
                if spawned.status.code() == Some(READ_EXIT_NOT_FOUND) {
                    let missing = missing_file_message();
                    (missing.clone(), missing, None)
                } else {
                    (text.clone(), text, spawned.status.code())
                }
            }
            Err(error) => {
                let text = format!("error: {error}");
                (text.clone(), text, None)
            }
        }
    })
}

/// Run one edit tool call: report the target, apply it, and report the outcome.
///
/// As with bash, edit *failures* (an unreadable file, no or ambiguous match, a
/// sandbox denial) are not errors: their message is content the model can act
/// on and retry.
fn run_edit<B: Backend, F: Fn(Progress)>(
    call: &FunctionToolCall,
    tools: &Tooling<'_, B>,
    on_progress: &F,
) -> String {
    let edit = match parse_edit(&call.arguments) {
        Ok(edit) => edit,
        Err(error) => return misuse(call, on_progress, &error),
    };
    traced(call, on_progress, || {
        let (result, exit_code) = apply_edit(&edit, tools.policy);
        (result.clone(), result, exit_code)
    })
}

/// Apply one parsed edit under `policy`, returning outcome message and the exit code.
///
/// We pre-check that the edit is valid in rust for performance, though the perl script
/// verifies to ensure we don't run into TOCTOU issues between here and the lock.
fn apply_edit(edit: &Edit, policy: &Policy) -> (String, Option<i32>) {
    let path = Path::new(&edit.file_path);
    if edit.old_string.is_empty() {
        return edit_failure(&format!("edit: old_string must not be empty: {}", path.display()));
    }
    if edit.old_string == edit.new_string {
        // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/edit.ts#L107>
        return edit_failure("No changes to make: old_string and new_string are exactly the same.");
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return edit_failure(&missing_file_message());
        }
        Err(error) => return (format!("edit: cannot read {}: {error}", path.display()), None),
    };
    let count = content.matches(&edit.old_string).count();
    if count == 0 {
        return edit_failure(&no_match_message(edit));
    }
    if count > 1 && !edit.replace_all {
        return edit_failure(&ambiguous_match_message(count, edit));
    }
    spawn_perl(edit, &mut policy.command("/usr/bin/perl"))
}

/// An edit-tool failure as the model sees it.
// <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/executor/errors.ts#L22>
fn edit_failure(message: &str) -> (String, Option<i32>) {
    (format!("<tool_use_error>{message}</tool_use_error>"), None)
}

/// Missing-file message; the cwd note keeps relative paths debuggable.
fn missing_file_message() -> String {
    let cwd =
        std::env::current_dir().map_or_else(|_| ".".to_string(), |dir| dir.display().to_string());
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/read.ts#L401>
    format!("File does not exist. Note: your current working directory is {cwd}.")
}

/// No-match error, echoing the string so quoting slips are visible.
fn no_match_message(edit: &Edit) -> String {
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/edit.ts#L213>
    format!(
        "String to replace not found in file.\nString: {}",
        edit.old_string
    )
}

/// Multi-match error for a call without `replace_all`.
fn ambiguous_match_message(count: usize, edit: &Edit) -> String {
    // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/edit.ts#L639>
    format!(
        "Found {count} matches of the string to replace, but replace_all is false. To replace \
         all occurrences, set replace_all to true. To replace only one occurrence, please \
         provide more context to uniquely identify the instance.\nString: {}",
        edit.old_string
    )
}

/// Run [`EDIT_PROGRAM`] through an already-configured `perl` command and map
/// its exit status to the message the model sees.
///
/// Split out so tests can drive the program with a plain command, exercising
/// its locking and matching semantics without the sandbox.
fn spawn_perl(edit: &Edit, cmd: &mut std::process::Command) -> (String, Option<i32>) {
    cmd.arg("-e")
        .arg(EDIT_PROGRAM)
        .arg("--")
        .arg(&edit.file_path)
        .env("TART_OLD", &edit.old_string)
        .env("TART_NEW", &edit.new_string)
        .envs(edit.replace_all.then_some(("TART_ALL", "1")));
    match cmd.output() {
        Ok(output) if output.status.success() => (edit_receipt(edit), Some(0)),
        Ok(output) => (
            format!(
                "edit failed on {}: {}{}",
                edit.file_path,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            Some(1),
        ),
        Err(error) => (
            format!("edit failed on {}: failed to run perl: {error}", edit.file_path),
            None,
        ),
    }
}

const FILE_FRESHNESS_SUFFIX: &str =
    "(file state is current in your context — no need to Read it back)";

/// Edit receipt when a file reread is not needed.
fn edit_receipt(edit: &Edit) -> String {
    if edit.replace_all {
        // <https://github.com/zai-org/ZCode/blob/328c1a0c0ffaa5a4f65e8fa199af5e4c20706e5f/apps/zcode-cli/packages/core/src/tool/handlers/edit.ts#L78>
        format!(
            "The file {} has been updated. All occurrences were successfully replaced. \
             {FILE_FRESHNESS_SUFFIX}",
            edit.file_path
        )
    } else {
        format!(
            "The file {} has been updated successfully. {FILE_FRESHNESS_SUFFIX}",
            edit.file_path
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use std::fmt::Write as _;
    use std::os::unix::process::ExitStatusExt as _;

    use crate::sandbox::live::skip_unless_live;
    use macro_rules_attribute::apply;
    use std::time::Instant;

    /// A tool call for `name` carrying `arguments` as the provider sent them.
    fn call(name: &str, arguments: String) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: name.to_string(),
            arguments,
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
            caller: None,
            r#async: None,
        }
    }

    /// A `bash` tool call requesting `command`.
    fn bash_call(arguments: &str) -> FunctionToolCall {
        call("bash", arguments.to_string())
    }

    /// A policy over a root, an agent on it, and the unarmed tooling over
    /// both: the scaffolding every `execute` test needs.
    struct TestSetup {
        policy: Policy,
        agent: Agent,
        token: CancelToken,
    }

    impl TestSetup {
        /// A rig whose sandbox grants writes under `root`.
        fn new(root: impl AsRef<Path>) -> Self {
            let policy = Policy::new(root).unwrap();
            Self {
                agent: Agent::new("http://localhost:9", "key", "model", policy.clone()),
                policy,
                token: CancelToken::new(),
            }
        }

        /// The tooling `execute` takes: no cancel lever, no subagents.
        fn tools(&self) -> Tooling<'_> {
            Tooling {
                policy: &self.policy,
                cancel: &self.token,
                agents: None,
                template: &self.agent,
            }
        }
    }

    #[test]
    fn the_bash_definition_matches_claude_code() {
        let tool = serde_json::to_value(bash()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "bash");
        assert_eq!(tool["parameters"]["required"][0], "command");
        assert_eq!(
            tool["parameters"]["properties"]["command"]["description"],
            "The command to execute"
        );
        // A number, in milliseconds, capped at ten minutes.
        assert_eq!(tool["parameters"]["properties"]["timeout"]["type"], "number");
        assert_eq!(
            tool["parameters"]["properties"]["timeout"]["description"],
            "Optional timeout in milliseconds (max 600000)"
        );
        assert_eq!(tool["parameters"]["required"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn parse_bash_reads_the_command_field() {
        let bash = parse_bash(r#"{"command":"ls -la"}"#).unwrap();

        assert_eq!(bash.command, "ls -la");
        assert_eq!(bash.timeout, DEFAULT_BASH_TIMEOUT, "an absent timeout defaults");
    }

    #[test]
    fn parse_bash_refuses_a_sub_second_timeout() {
        for arguments in [
            r#"{"command":"ls","timeout":300}"#,
            r#"{"command":"ls","timeout":999}"#,
            r#"{"command":"ls","timeout":999.5}"#,
            r#"{"command":"ls","timeout":0}"#,
            r#"{"command":"ls","timeout":-5}"#,
        ] {
            let error = parse_bash(arguments).unwrap_err().to_string();
            assert!(
                error.contains("timeout is measured in milliseconds"),
                "{arguments}: {error}"
            );
        }
    }

    /// The schema says number, so a fractional timeout rounds; an out-of-range
    /// one clamps to [`MAX_BASH_TIMEOUT`], and a wrongly typed one is ignored
    /// in favor of [`DEFAULT_BASH_TIMEOUT`]. Expected values are milliseconds.
    #[test]
    fn parse_bash_reads_the_timeout_in_milliseconds() {
        for (arguments, millis) in [
            (r#"{"command":"cargo build","timeout":300000}"#, 300_000),
            (r#"{"command":"sleep 1","timeout":1500.7}"#, 1_501),
            (r#"{"command":"sleep 1","timeout":120000.0}"#, 120_000),
            (r#"{"command":"ls","timeout":"300"}"#, 120_000),
            (r#"{"command":"sleep 5","timeout":9000000}"#, 600_000),
        ] {
            let parsed = parse_bash(arguments).unwrap();
            assert_eq!(parsed.timeout, Duration::from_millis(millis), "{arguments}");
        }
    }

    #[test]
    fn parse_bash_rejects_non_json() {
        let error = parse_bash("not json").unwrap_err().to_string();

        assert!(error.contains("weren't JSON"), "{error}");
    }

    #[test]
    fn parse_bash_rejects_a_missing_command() {
        let error = parse_bash(r#"{"other":1}"#).unwrap_err().to_string();

        assert!(
            error.contains("The required parameter `command` is missing"),
            "{error}"
        );
    }

    /// A finished command's streams with a Unix wait status: 0 is a success,
    /// `code << 8` an exit code, a small number a signal.
    fn output(raw: i32, stdout: &str, stderr: &str) -> Output {
        Output {
            status: ExitStatus::from_raw(raw),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    /// The tool result for a finished command, with no exit-one exemption.
    fn framed(raw: i32, stdout: &str, stderr: &str) -> String {
        let output = output(raw, stdout, stderr);
        command_text(&combined_output(&output), output.status, false)
    }

    #[test]
    fn command_result_passes_successful_output_through() {
        assert_eq!(framed(0, "hi\n", ""), "hi\n");
    }

    #[test]
    fn command_result_reports_a_silent_success_explicitly() {
        assert_eq!(framed(0, "", ""), "(Bash completed with no output)");
    }

    #[test]
    fn command_result_appends_the_exit_code_after_a_failure() {
        // stderr paragraphs below stdout, exit code as the final line.
        assert_eq!(framed(1 << 8, "hi\n", "boom\n"), "hi\n\nboom\nExit code 1");
    }

    #[test]
    fn command_result_marks_a_silent_failure_with_just_the_exit_code() {
        assert_eq!(framed(1 << 8, "", ""), "Exit code 1");
    }

    #[test]
    fn command_result_reports_a_signal_death_as_its_conventional_code() {
        assert_eq!(framed(9, "", ""), "Exit code 137");
    }

    /// stderr reads as its own paragraph below stdout
    #[test]
    fn command_result_paragraphs_stderr_below_stdout() {
        let output = output(0, "no trailing newline", "warning\n");
        assert_eq!(combined_output(&output), "no trailing newline\nwarning\n");
    }

    /// The search-and-test family's "no match" exit is a success.
    #[test]
    fn an_exit_one_from_the_search_family_is_success() {
        assert!(exit_one_is_success("grep needle file"));
        assert!(exit_one_is_success("/usr/bin/rg needle"));
        assert!(exit_one_is_success("git diff --stat"));
        assert!(exit_one_is_success("git grep needle"));
        assert!(!exit_one_is_success("git push"));
        assert!(!exit_one_is_success("cargo test"));

        // The exemption only forgives an exit of exactly 1.
        let forgiven = output(1 << 8, "", "");
        assert_eq!(
            command_text("", forgiven.status, exit_one_is_success("grep x")),
            "(Bash completed with no output)"
        );
        let failed = output(2 << 8, "", "");
        assert_eq!(
            command_text("", failed.status, exit_one_is_success("grep x")),
            "Exit code 2"
        );
    }

    /// A success past the inline cap spills to a file and previews the start.
    #[test]
    fn an_oversized_success_spills_to_a_file() {
        let dir = std::env::temp_dir().join("tart");
        let text = "x".repeat(BASH_SUCCESS_CAP + 1);
        let output = output(0, &text, "");
        let framed = command_text(&combined_output(&output), output.status, false);

        let lines: Vec<&str> = framed.lines().collect();
        assert_eq!(lines[0], "<persisted-output>", "{framed}");
        let (size, saved) = lines[1].split_once(". Full output saved to: ").unwrap();
        assert_eq!(size, "Output too large (29.3KB)", "{framed}");
        assert_eq!(lines[2], "", "{framed}");
        assert_eq!(lines[3], "Preview (first 2KB):", "{framed}");
        assert_eq!(lines[4].chars().count(), SPILL_PREVIEW, "{framed}");
        assert!(lines[4].chars().all(|c| c == 'x'), "{framed}");
        assert_eq!(lines[5], "...", "{framed}");
        assert_eq!(lines[6], "</persisted-output>", "{framed}");
        assert_eq!(lines.len(), 7, "{framed}");
        // The spill lives in tart's throwaway directory, not the temp root.
        let path = std::path::Path::new(saved);
        assert_eq!(path.parent(), Some(dir.as_path()), "{framed}");
        // However the environment spells the temp root, the spill resolves
        // inside the one the sandbox grants for reading.
        let granted = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        assert!(
            std::fs::canonicalize(path).unwrap().starts_with(&granted),
            "{framed}"
        );
        // The spill file holds the whole output for a later `read`, readable
        // only by its owner.
        let file = std::fs::read_to_string(path).unwrap();
        assert_eq!(file, text);
        let mode = std::fs::metadata(path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "{framed}");
        let _ = std::fs::remove_file(path);
    }

    /// The web tools and manual runs return their text inline however large, so a
    /// fetched page is never silently replaced by a scratch-file pointer.
    #[test]
    fn inline_framing_never_spills() {
        let text = "x".repeat(BASH_SUCCESS_CAP + 1);
        let output = output(0, &text, "");
        let framed = command_text_inline(&combined_output(&output), output.status, false);
        assert_eq!(framed, text);
    }

    /// A failure past the inline cap shrinks to a head-and-tail excerpt.
    #[test]
    fn an_oversized_failure_excerpts_head_and_tail() {
        let text = format!(
            "{}zzzz{}",
            "a".repeat(BASH_FAILURE_CAP),
            "b".repeat(BASH_FAILURE_CAP)
        );
        let output = output(1 << 8, &text, "");
        let framed = command_text(&combined_output(&output), output.status, false);

        let (excerpted, code) = framed.split_once("\nExit code ").unwrap();
        assert_eq!(code, "1");
        // The excerpt is `bounded` at the failure cap, so the seam carries the
        // same truncation marker the display caps use.
        let (head, rest) = excerpted.split_once('\n').unwrap();
        let (marker, tail) = rest.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first and last 4 KB shown]", "{framed}");
        assert_eq!(head.chars().count(), BASH_FAILURE_CAP / 2);
        assert!(head.chars().all(|c| c == 'a'), "{head}");
        assert!(tail.chars().all(|c| c == 'b'), "{tail}");
    }

    #[test]
    fn timeout_text_appends_claude_code_kill_lines() {
        assert_eq!(
            timeout_text("partial\ntail\n", DEFAULT_BASH_TIMEOUT),
            "partial\ntail\nExit code 137\nCommand timed out after 2m 0s"
        );
        assert_eq!(
            timeout_text("", MAX_BASH_TIMEOUT),
            "Exit code 137\nCommand timed out after 10m 0s"
        );
        // Sub-minute durations carry no minute marker.
        assert_eq!(
            timeout_text("", Duration::from_millis(1000)),
            "Exit code 137\nCommand timed out after 1s"
        );
        assert_eq!(duration_text(Duration::from_millis(59_999)), "59s");
        assert_eq!(duration_text(Duration::from_secs(60)), "1m 0s");
    }

    /// A multi-byte head must not split a character at the cut.
    #[test]
    fn head_cap_cuts_on_a_character_boundary() {
        let wide = "語".repeat(4_000); // 12 KB of 3-byte characters
        let capped = head_cap(&wide, 1024);
        let (kept, marker) = capped.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first 1 KB shown]");
        assert!(kept.chars().all(|c| c == '語'), "cut inside a character");
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

    #[test]
    fn bounded_keeps_both_ends_and_marks_the_middle() {
        assert_eq!(bounded("hi\n".to_string(), 10), "hi\n");

        let text = "abcdef".repeat(1024); // 6 KB
        let capped = bounded(text.clone(), 2048);
        let (head, rest) = capped.split_once('\n').unwrap();
        let (marker, tail) = rest.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first and last 1 KB shown]");
        assert_eq!(head, &text[..1024]);
        assert_eq!(tail, &text[text.len() - 1024..]);

        // Multi-byte characters must not split at either cut.
        let wide = "語".repeat(4_000); // 12 KB of 3-byte characters
        let capped = bounded(wide, 2048);
        let (head, rest) = capped.split_once('\n').unwrap();
        let (marker, tail) = rest.split_once('\n').unwrap();
        assert_eq!(marker, "[truncated; first and last 1 KB shown]");
        assert!(head.chars().all(|c| c == '語'), "cut inside a character");
        assert!(tail.chars().all(|c| c == '語'), "cut inside a character");
    }

    /// Manual runs reuse the bash tool's framing; these run unsandboxed
    #[test]
    fn manual_command_frames_success_and_failure() {
        assert_eq!(manual_command("echo hi", &CancelToken::new()), "hi\n");
        assert_eq!(
            manual_command("true", &CancelToken::new()),
            "(Bash completed with no output)"
        );
        assert_eq!(manual_command("false", &CancelToken::new()), "Exit code 1");
        assert_eq!(
            manual_command("echo boom >&2; exit 3", &CancelToken::new()),
            "boom\nExit code 3"
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

    /// An untouched token leaves the command to finish on its own.
    #[test]
    fn manual_command_runs_to_completion_without_a_cancel() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");

        let run = run_watched(&mut command, None, &CancelToken::new()).unwrap();

        assert_eq!(run.killed, None);
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

    /// These drive `run_watched` with plain commands, so they run without the sandbox
    #[test]
    fn run_watched_kills_a_command_that_outruns_the_deadline() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        let started = Instant::now();

        let run = run_watched(
            &mut command,
            Some(Duration::from_millis(300)),
            &CancelToken::new(),
        )
        .unwrap();

        assert_eq!(run.killed, Some(KillReason::Timeout));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!run.output.status.success());
    }

    #[test]
    fn run_watched_kills_a_command_the_moment_the_token_fires() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        let token = CancelToken::new();
        let trip = {
            let token = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                token.cancel();
            })
        };
        let started = Instant::now();

        let run = run_watched(&mut command, Some(Duration::from_secs(60)), &token).unwrap();

        assert_eq!(run.killed, Some(KillReason::Cancelled));
        assert!(started.elapsed() < Duration::from_secs(5));
        trip.join().unwrap();
    }

    #[test]
    fn run_watched_kills_the_whole_process_group() {
        // The backgrounded sleeps outlive bash and hold the output pipe; only a
        // group kill frees the capture, so returning promptly proves they died.
        let mut command = Command::new("/bin/bash");
        command.arg("-c").arg("sleep 9871 & sleep 9871 & wait");
        let started = Instant::now();

        let run = run_watched(
            &mut command,
            Some(Duration::from_millis(300)),
            &CancelToken::new(),
        )
        .unwrap();

        assert_eq!(run.killed, Some(KillReason::Timeout));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// A fast command finishes un-killed and successful, and the sender's drop
    /// joins the watchdog at once rather than letting it sleep out the timeout.
    #[test]
    fn run_watched_wakes_the_watchdog_when_the_command_finishes_early() {
        let mut command = Command::new("/bin/echo");
        command.arg("hi");
        let started = Instant::now();

        let run =
            run_watched(&mut command, Some(DEFAULT_BASH_TIMEOUT), &CancelToken::new()).unwrap();

        assert_eq!(run.killed, None);
        assert_eq!(combined_output(&run.output), "hi\n");
        assert!(run.output.status.success());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Live: reaches `sandbox-exec`, so it only passes outside a nested sandbox.
    #[apply(skip_unless_live!)]
    #[test]
    fn execute_reports_command_then_output() {
        let rig = TestSetup::new(std::env::current_dir().unwrap());
        let tools = rig.tools();
        let events = std::cell::RefCell::new(Vec::new());
        let output = execute(&bash_call(r#"{"command":"echo hi"}"#), &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert_eq!(output, "hi\n");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart {
                    id,
                    name,
                    arguments,
                },
                Progress::ToolOutput {
                    output,
                    exit: Some(0),
                    ..
                }
            ] if id == "call_0"
                && name == "bash"
                && arguments == r#"{"command":"echo hi"}"#
                && output == "hi\n"
        ));

        // A failure carries its exit code as the trailing line.
        assert_eq!(
            execute(&bash_call(r#"{"command":"false"}"#), &tools, &|_| {}),
            "Exit code 1"
        );
        assert_eq!(
            execute(&bash_call(r#"{"command":"true"}"#), &tools, &|_| {}),
            "(Bash completed with no output)"
        );
        // The search family's "no match" exit of 1 reads as success, and is
        // reported as such so the front end paints the box green too.
        events.borrow_mut().clear();
        assert_eq!(
            execute(
                &bash_call(r#"{"command":"grep needle /dev/null"}"#),
                &tools,
                &|progress| events.borrow_mut().push(progress)
            ),
            "(Bash completed with no output)"
        );
        assert!(matches!(
            events.borrow().as_slice(),
            [_, Progress::ToolOutput { exit: Some(0), .. }]
        ));
    }

    /// An output past the cap reaches the model whole but the front end capped,
    /// with the marker the model's own history carries: expanded view shows
    /// exactly what the model saw.
    #[test]
    fn oversized_output_reaches_the_front_end_capped() {
        let events = std::cell::RefCell::new(Vec::new());
        let result = traced(
            &bash_call(r#"{"command":"echo hi"}"#),
            &|progress| events.borrow_mut().push(progress),
            || {
                let text = "x".repeat(70_000);
                (text.clone(), text, Some(0))
            },
        );

        assert_eq!(result.len(), 70_000, "the model sees the whole text");
        assert!(
            matches!(
                events.borrow().as_slice(),
                [Progress::ToolStart { .. }, Progress::ToolOutput { output, .. }]
                    if output.len() < 70_000
                        && output.contains("[truncated; first and last 32 KB shown]")
            ),
            "the event's copy is capped like the model's history"
        );
    }

    #[test]
    fn execute_answers_an_unknown_tool_with_output_not_failure() {
        let rig = TestSetup::new(std::env::current_dir().unwrap());
        let tools = rig.tools();
        let mut call = bash_call(r#"{"command":"ls"}"#);
        call.name = "rm".to_string();
        let events = std::cell::RefCell::new(Vec::new());

        // Only the model can fix calling a tool that does not exist, so the
        // error is its output
        let output = execute(&call, &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert!(
            output.contains("<tool_use_error>Tool not found: rm</tool_use_error>"),
            "{output}"
        );
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, .. },
                Progress::ToolOutput { output, .. }
            ] if name == "rm" && output.contains("Tool not found")
        ));
    }

    #[test]
    fn chat_mode_denies_the_sandboxed_tools() {
        let mut rig = TestSetup::new(std::env::current_dir().unwrap());
        rig.agent.set_mode(ChatMode::Chat);
        let tools = rig.tools();
        for name in ["bash", "read", "edit", "spawn_agent", "check_agent"] {
            let mut call = bash_call(r#"{"command":"echo hi"}"#);
            call.name = name.to_string();
            let events = std::cell::RefCell::new(Vec::new());
            let output = execute(&call, &tools, &|progress| {
                events.borrow_mut().push(progress);
            });

            assert!(
                output.contains(&format!("the {name} tool is not available in chat mode")),
                "{output}"
            );
            assert!(
                matches!(
                    events.borrow().as_slice(),
                    [
                        Progress::ToolStart { .. },
                        Progress::ToolOutput { output, .. }
                    ] if output.contains("not available in chat mode")
                ),
                "the denial frames as one tool exchange: {:?}",
                events.borrow()
            );
        }
    }

    /// The name guard denies `spawn_agent` even with a registry armed: the
    /// denial must not depend on the tools happening to be unarmed. Live-safe:
    /// the guard fires before the registry is touched.
    #[test]
    fn chat_mode_denies_spawn_even_when_armed() {
        let mut rig = TestSetup::new(std::env::current_dir().unwrap());
        rig.agent.set_mode(ChatMode::Chat);
        let agents = Agents::new(|_, _| ());
        rig.agent.set_subagents(std::sync::Arc::new(agents.clone()));
        let tools = Tooling {
            policy: &rig.policy,
            cancel: &rig.token,
            agents: Some(&agents),
            template: &rig.agent,
        };
        let mut call = bash_call(r#"{"task":"do things"}"#);
        call.name = "spawn_agent".to_string();

        let output = execute(&call, &tools, &|_| {});

        assert!(
            output.contains("the spawn_agent tool is not available in chat mode"),
            "{output}"
        );
    }

    /// A temporary file holding `contents`, removed when the guard drops.
    fn scratch(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    /// An `edit` tool call replacing `old` with `new` in `path`, in the
    /// schema's `file_path` shape.
    fn edit_call(path: &Path, old: &str, new: &str) -> FunctionToolCall {
        call(
            "edit",
            serde_json::json!({
                "file_path": path, "old_string": old, "new_string": new
            })
            .to_string(),
        )
    }

    /// One parsed edit call for `path`, as its JSON arguments would have built it.
    fn an_edit(path: &Path, old: &str, new: &str, replace_all: bool) -> Edit {
        Edit {
            file_path: path.display().to_string(),
            old_string: old.to_string(),
            new_string: new.to_string(),
            replace_all,
        }
    }

    /// Drive [`EDIT_PROGRAM`] with a plain, unsandboxed perl command.
    fn perl_edit(path: &Path, old: &str, new: &str, replace_all: bool) -> String {
        spawn_perl(
            &an_edit(path, old, new, replace_all),
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
            format!(
                "The file {} has been updated successfully. (file state is current in your \
                 context \u{2014} no need to Read it back)",
                file.path().display()
            )
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
            format!(
                "The file {} has been updated. All occurrences were successfully replaced. \
                 (file state is current in your context \u{2014} no need to Read it back)",
                file.path().display()
            )
        );
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "b b b\n");
    }

    #[test]
    fn perl_edit_deletes_via_an_empty_new_string() {
        let file = scratch("keep\ndrop me\nkeep\n");
        let output = perl_edit(file.path(), "drop me\n", "", false);

        assert!(output.contains("updated successfully"), "{output}");
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

        assert!(first.join().unwrap().contains("updated successfully"));
        assert!(second.join().unwrap().contains("updated successfully"));
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
        let spawn_edit = |old: &str, new: &str| {
            let call = edit_call(file.path(), old, new);
            std::thread::spawn(move || {
                let rig = TestSetup::new(std::env::temp_dir());
                execute(&call, &rig.tools(), &|_| {})
            })
        };
        let first = spawn_edit("UNO", "uno");
        let second = spawn_edit("DOS", "dos");

        assert!(first.join().unwrap().contains("updated successfully"));
        assert!(second.join().unwrap().contains("updated successfully"));
        assert_eq!(
            std::fs::read_to_string(file.path()).unwrap(),
            "one uno alpha\ntwo dos beta\n"
        );
    }

    #[test]
    fn edit_definition_matches_claude_code() {
        let tool = serde_json::to_value(edit()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "edit");
        assert_eq!(tool["parameters"]["required"][0], "file_path");
        // The old name is gone from the schema.
        assert!(tool["parameters"]["properties"]["path"].is_null());
        assert_eq!(tool["parameters"]["required"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn parse_edit_reads_the_file_path() {
        let edit =
            parse_edit(r#"{"file_path":"src/main.rs","old_string":"a","new_string":"b"}"#).unwrap();
        assert_eq!(edit.file_path, "src/main.rs");

        let error = parse_edit(r#"{"old_string":"a","new_string":"b"}"#)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("The required parameter `file_path` is missing"),
            "{error}"
        );
    }

    #[test]
    fn edit_failures_speak_claude_code() {
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let file = scratch("a\nb\na\n");

        // An identical pair refuses
        let (identical, _) = apply_edit(&an_edit(file.path(), "a", "a", false), &policy);
        assert_eq!(
            identical,
            "<tool_use_error>No changes to make: old_string and new_string are exactly the \
             same.</tool_use_error>"
        );

        // A no-match echoes the string, as Claude Code does, so quoting slips show.
        let (missing, _) = apply_edit(&an_edit(file.path(), "z\n", "y\n", false), &policy);
        assert_eq!(
            missing,
            "<tool_use_error>String to replace not found in file.\nString: z\n</tool_use_error>"
        );

        // An ambiguous match without replace_all reports the count and the fix.
        let (ambiguous, _) = apply_edit(&an_edit(file.path(), "a", "c", false), &policy);
        assert!(ambiguous.starts_with(
            "<tool_use_error>Found 2 matches of the string to replace, but replace_all is \
                 false."
        ));
        assert!(ambiguous.ends_with("instance.\nString: a</tool_use_error>"));

        // A missing file names the working directory.
        let absent = std::env::temp_dir().join("tart-edit-does-not-exist");
        let (missing_file, _) = apply_edit(&an_edit(&absent, "a", "b", false), &policy);
        assert_eq!(
            missing_file,
            format!(
                "<tool_use_error>File does not exist. Note: your current working directory is {}.</tool_use_error>",
                std::env::current_dir().unwrap().display()
            )
        );
    }

    #[test]
    fn read_definition_matches_claude_code() {
        let tool = serde_json::to_value(read()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "read");
        assert_eq!(tool["parameters"]["required"][0], "file_path");
        assert!(tool["parameters"]["properties"]["offset"].is_object());
        assert!(tool["parameters"]["properties"]["limit"].is_object());
        // The old absolute-bound names are gone from the schema.
        assert!(tool["parameters"]["properties"]["path"].is_null());
        assert!(tool["parameters"]["properties"]["start_line"].is_null());
    }

    #[test]
    fn parse_read_reads_the_path_and_bounds() {
        let read = parse_read(r#"{"file_path":"src/main.rs","offset":10,"limit":41}"#).unwrap();
        assert_eq!(read.file_path, "src/main.rs");
        assert_eq!(read.bounds(), (Some(10), Some(50)));

        // Either bound alone reads from the top or to the end.
        let offset_only = parse_read(r#"{"file_path":"src/main.rs","offset":20}"#).unwrap();
        assert_eq!(offset_only.bounds(), (Some(20), None));
        let limit_only = parse_read(r#"{"file_path":"src/main.rs","limit":5}"#).unwrap();
        assert_eq!(limit_only.bounds(), (None, Some(5)));

        let whole = parse_read(r#"{"file_path":"src/main.rs"}"#).unwrap();
        assert_eq!(whole.bounds(), (None, None));

        // Zeros count as omitted rather than inverting the range.
        let zero_offset =
            parse_read(r#"{"file_path":"src/main.rs","offset":0,"limit":5}"#).unwrap();
        assert_eq!(zero_offset.bounds(), (None, Some(5)));
        let zero_limit =
            parse_read(r#"{"file_path":"src/main.rs","offset":10,"limit":0}"#).unwrap();
        assert_eq!(zero_limit.bounds(), (Some(10), None));
    }

    #[test]
    fn parse_spawn_takes_the_task_and_rejects_its_absence() {
        assert_eq!(
            parse_spawn(r#"{"task":"count the tests"}"#).unwrap().task,
            "count the tests"
        );

        let error = parse_spawn("{}").unwrap_err().to_string();
        assert!(
            error.contains("The required parameter `task` is missing"),
            "{error}"
        );
    }

    #[test]
    fn parse_check_takes_an_integer_id_and_rejects_anything_else() {
        assert_eq!(parse_check(r#"{"id":2}"#).unwrap().id, 2);

        let error = parse_check(r#"{"id":"two"}"#).unwrap_err().to_string();
        assert!(error.contains("check_agent needs an integer 'id'"), "{error}");
    }

    /// A `read` tool call for `path`, optionally bounded to a line range, in
    /// the schema's offset/limit shape.
    fn read_call(path: &Path, start_line: Option<u64>, end_line: Option<u64>) -> FunctionToolCall {
        // The inclusive bounds become a 1-based offset and a line count.
        let limit = match (start_line, end_line) {
            (Some(start), Some(end)) => end.checked_sub(start).map(|span| span + 1),
            (None, Some(end)) => Some(end),
            _ => None,
        };
        call(
            "read",
            serde_json::json!({
                "file_path": path,
                "offset": start_line,
                "limit": limit,
            })
            .to_string(),
        )
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
        let rig = TestSetup::new(std::env::temp_dir());
        let tools = rig.tools();
        let events = std::cell::RefCell::new(Vec::new());

        let whole = execute(&read_call(file.path(), None, None), &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert!(whole.starts_with("     1\tline 1\n"), "{whole}");
        assert!(whole.ends_with("    30\tline 30\n"), "{whole}");
        assert_eq!(whole.lines().count(), 30);
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, arguments, .. },
                Progress::ToolOutput { .. }
            ] if name == "read"
                && arguments
                    == &serde_json::json!({"file_path": file.path(), "offset": null,
                                           "limit": null})
                        .to_string()
        ));

        // A bounded read's arguments ride along untouched.
        events.borrow_mut().clear();
        let range = execute(&read_call(file.path(), Some(10), Some(12)), &tools, &|progress| {
            events.borrow_mut().push(progress);
        });
        assert_eq!(range, "    10\tline 10\n    11\tline 11\n    12\tline 12\n");
        assert!(matches!(
            events.borrow().as_slice(),
            [Progress::ToolStart { name, arguments, .. }, _]
                if name == "read"
                    && arguments
                        == &serde_json::json!({
                            "file_path": file.path(),
                            "offset": 10,
                            "limit": 3
                        })
                        .to_string()
        ));

        let tail = execute(&read_call(file.path(), Some(28), None), &tools, &|_| {});
        assert!(tail.starts_with("    28\tline 28\n"), "{tail}");
        assert_eq!(tail.lines().count(), 3);

        let missing = std::env::temp_dir().join("tart-read-does-not-exist");
        let absent = execute(&read_call(&missing, None, None), &tools, &|_| {});
        assert!(
            absent.contains("File does not exist. Note: your current working directory is"),
            "{absent}"
        );
    }

    #[test]
    fn a_malformed_call_is_output_not_failure() {
        let rig = TestSetup::new(std::env::temp_dir());
        let tools = rig.tools();
        let call = call("read", r#"{"start_line": 1}"#.to_string());
        let events = std::cell::RefCell::new(Vec::new());

        let output = execute(&call, &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert!(
            output.contains(
                "<tool_use_error>InputValidationError: read failed due to the following \
                 issue:\nThe required parameter `file_path` is missing</tool_use_error>"
            ),
            "{output}"
        );
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, .. },
                Progress::ToolOutput { output, .. }
            ] if name == "read" && output.contains("`file_path` is missing")
        ));
    }
}
