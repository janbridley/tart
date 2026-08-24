use std::path::Path;
use std::process::Output;

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};

use crate::{Progress, sandbox::Policy};

/// Perform a raw string find-and-replace operation, holding a lock for thread safety..
///
/// This string contains perl source code to perform the required work, dispatching a
/// platform independent flock to ensure concurrent agents cannot collide.
/// Exits 1 with a warning, file untouched, or when the match count is wrong.
const EDIT_PROGRAM: &str = include_str!("data/edit.pl");

/// Emulate `cat -n … | sed -n …` , using a shared `flock` with the edit tool.
const READ_PROGRAM: &str = include_str!("data/read.pl");

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
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: "bash".to_string(),
        description: Some(
            "Run a bash command in a sandbox (writes restricted to granted roots, no network) \
            and return its combined stdout/stderr"
                .to_string(),
        ),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"]
        })),
        strict: None,
    })
}

/// The read tool; files are read under the caller's [`Policy`].
#[must_use]
pub(crate) fn read() -> Tool {
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: "read".to_string(),
        description: Some(
            "Read a file with line numbers (cat -n style) in a sandbox (reads restricted to \
            granted roots); optionally pass start_line/end_line (1-based, inclusive) to read \
            a range"
                .to_string(),
        ),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to read"},
                "start_line": {"type": "integer", "description": "First line to read (1-based); omit to start at the top"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive); omit to read to the end"}
            },
            "required": ["path"]
        })),
        strict: None,
    })
}

/// The edit tool; replacements execute under the caller's [`Policy`].
#[must_use]
pub(crate) fn edit() -> Tool {
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: "edit".to_string(),
        description: Some(
            "Replace an exact string in an existing file. old_string must match the file exactly, including whitespace and \
            newlines, and occur exactly once unless replace_all is true: include surrounding \
            lines to make it unique. An empty new_string deletes old_string. The file must \
            already exist and be valid UTF-8, so use bash to create files. Prefer this tool \
            over bash for changing existing files"
                .to_string(),
        ),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the existing file to edit"},
                "old_string": {"type": "string", "description": "Text to replace; must match exactly and be unique unless replace_all"},
                "new_string": {"type": "string", "description": "Replacement text; empty deletes old_string"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of one unique match"}
            },
            "required": ["path", "old_string", "new_string"]
        })),
        strict: None,
    })
}

/// Extract the command from a bash tool call's JSON arguments.
fn parse_command(arguments: &str) -> anyhow::Result<String> {
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))?;
    args["command"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("tool call missing 'command'"))
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
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))?;
    Ok(Read {
        path: args["path"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("tool call missing 'path'"))?,
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
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))?;
    let field = |name: &str| {
        args[name]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow::anyhow!("tool call missing '{name}'"))
    };
    Ok(Edit {
        path: field("path")?,
        old_string: field("old_string")?,
        new_string: field("new_string")?,
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
pub(crate) fn execute<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    on_progress: &F,
) -> anyhow::Result<String> {
    match call.name.as_str() {
        "bash" => run_bash(call, policy, on_progress),
        "read" => run_read(call, policy, on_progress),
        "edit" => run_edit(call, policy, on_progress),
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

/// The tool result for one finished command: its stdout then its stderr, prefixed
/// with `[exit N]` when it failed, or `done` when a success printed nothing.
fn command_result(output: &Output) -> String {
    let text = combined_output(output);

    if output.status.success() {
        return if text.is_empty() { "done".to_string() } else { text };
    }

    let status = output.status.code().map_or("signal".into(), |c| c.to_string());

    let separator = if text.is_empty() { "" } else { "\n" };
    format!("[exit {status}]{separator}{text}")
}

/// Run one bash tool call under `policy`, reporting its steps to `on_progress`.
fn run_bash<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    on_progress: &F,
) -> anyhow::Result<String> {
    let command = parse_command(&call.arguments)?;
    on_progress(Progress::ToolStart {
        id: call.call_id.clone(),
        name: "Bash",
        digest: command.clone(),
    });
    // A failure to launch comes back as an error string rather than a `Result`,
    // so the output can be handed straight back to the model.
    let spawned = policy.command("/bin/bash").arg("-c").arg(&command).output();
    let (result, output, exit) = match &spawned {
        Ok(spawned) => (
            command_result(spawned),
            combined_output(spawned),
            spawned.status.code(),
        ),
        Err(error) => {
            let text = format!("error: {error}");
            (text.clone(), text, None)
        }
    };
    on_progress(Progress::ToolOutput {
        id: call.call_id.clone(),
        output,
        exit,
    });
    Ok(result)
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
fn run_read<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    on_progress: &F,
) -> anyhow::Result<String> {
    let read = parse_read(&call.arguments)?;
    on_progress(Progress::ToolStart {
        id: call.call_id.clone(),
        name: "Read",
        digest: read_digest(&read),
    });
    let mut command = policy.command("/usr/bin/perl");
    command
        .arg("-e")
        .arg(numbered_read(read.start_line, read.end_line))
        .arg("--")
        .arg(&read.path);
    // A failure to launch comes back as an error string for the model to deal with.
    let (output, exit) = match &command.output() {
        Ok(spawned) => (combined_output(spawned), spawned.status.code()),
        Err(error) => (format!("error: {error}"), None),
    };
    on_progress(Progress::ToolOutput {
        id: call.call_id.clone(),
        output: output.clone(),
        exit,
    });
    Ok(output)
}

/// Run one edit tool call: report the target, apply it, and report the outcome.
///
/// As with bash, edit *failures* (an unreadable file, no or ambiguous match, a
/// sandbox denial) are not errors: their message is content the model can act
/// on and retry.
fn run_edit<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    on_progress: &F,
) -> anyhow::Result<String> {
    let edit = parse_edit(&call.arguments)?;
    on_progress(Progress::ToolStart {
        id: call.call_id.clone(),
        name: "Edit",
        digest: edit.path.clone(),
    });
    let (result, exit) = apply_edit(&edit, policy);
    on_progress(Progress::ToolOutput {
        id: call.call_id.clone(),
        output: result.clone(),
        exit,
    });
    Ok(result)
}

/// Apply one parsed edit under `policy`, returning outcome message and the exit code.
///
/// We pre-check that the edit is valid in rust for performance, though the perl script
/// verifies to ensure we don't run into TOCTOU issues between here and the lock.
///
/// The exit is `None` whenever no process ran: pre-check refusals join spawn
/// errors in reporting "nothing exited" — the box still colors red.
fn apply_edit(edit: &Edit, policy: &Policy) -> (String, Option<i32>) {
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
    spawn_perl(edit, &mut policy.command("/usr/bin/perl"))
}

/// Run [`EDIT_PROGRAM`] through an already-configured `perl` command and map
/// its exit status to the message the model sees: perl's report on success,
/// its warning as retryable content otherwise.
///
/// Split out so tests can drive the program with a plain command, exercising
/// its locking and matching semantics without the sandbox.
fn spawn_perl(edit: &Edit, cmd: &mut std::process::Command) -> (String, Option<i32>) {
    let path = Path::new(&edit.path);
    cmd.arg("-e")
        .arg(EDIT_PROGRAM)
        .arg("--")
        .arg(&edit.path)
        .env("TART_OLD", &edit.old_string)
        .env("TART_NEW", &edit.new_string)
        .envs(edit.replace_all.then_some(("TART_ALL", "1")));
    match cmd.output() {
        Ok(output) if output.status.success() => (
            String::from_utf8_lossy(&output.stdout).trim_end().to_string(),
            Some(0),
        ),
        Ok(output) => (
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

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
    fn parse_command_reads_the_command_field() {
        assert_eq!(parse_command(r#"{"command":"ls -la"}"#).unwrap(), "ls -la");
    }

    #[test]
    fn parse_command_rejects_non_json() {
        let error = parse_command("not json").unwrap_err().to_string();

        assert!(error.contains("weren't JSON"), "{error}");
    }

    #[test]
    fn parse_command_rejects_a_missing_command() {
        let error = parse_command(r#"{"other":1}"#).unwrap_err().to_string();

        assert!(error.contains("missing 'command'"), "{error}");
    }

    /// A finished command's captured streams, with a Unix wait status: 0 is a
    /// success, `code << 8` an exit code, a small number a signal.
    fn command_output(raw: i32, stdout: &str, stderr: &str) -> Output {
        use std::os::unix::process::ExitStatusExt;

        Output {
            status: std::process::ExitStatus::from_raw(raw),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn command_result_passes_successful_output_through() {
        let output = command_output(0, "hi\n", "");

        assert_eq!(command_result(&output), "hi\n");
    }

    #[test]
    fn command_result_reports_a_silent_success_as_done() {
        let output = command_output(0, "", "");

        assert_eq!(command_result(&output), "done");
    }

    #[test]
    fn command_result_prefixes_a_failure_with_its_exit_code() {
        let output = command_output(1 << 8, "hi\n", "boom\n");

        assert_eq!(command_result(&output), "[exit 1]\nhi\nboom\n");
    }

    #[test]
    fn command_result_marks_a_silent_failure_with_just_the_exit_code() {
        let output = command_output(1 << 8, "", "");

        assert_eq!(command_result(&output), "[exit 1]");
    }

    #[test]
    fn command_result_marks_a_signal_death() {
        let output = command_output(9, "", "");

        assert_eq!(command_result(&output), "[exit signal]");
    }

    /// Live: reaches `sandbox-exec`, so it only passes outside a nested sandbox.
    #[test]
    fn execute_reports_command_then_output() {
        let policy = Policy::new(std::env::current_dir().unwrap()).unwrap();
        let events = std::cell::RefCell::new(Vec::new());
        let output = execute(&bash_call(r#"{"command":"echo hi"}"#), &policy, &|progress| {
            events.borrow_mut().push(progress);
        })
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
            execute(&bash_call(r#"{"command":"false"}"#), &policy, &|_| {}).unwrap(),
            "[exit 1]"
        );
        assert_eq!(
            execute(&bash_call(r#"{"command":"true"}"#), &policy, &|_| {}).unwrap(),
            "done"
        );
    }

    #[test]
    fn execute_rejects_unknown_tool_names() {
        let policy = Policy::new(std::env::current_dir().unwrap()).unwrap();
        let mut call = bash_call(r#"{"command":"ls"}"#);
        call.name = "rm".to_string();

        let error = execute(&call, &policy, &|_| {}).unwrap_err().to_string();

        assert!(error.contains("unknown tool"), "{error}");
    }

    /// A scratch file removed when the guard drops, so parallel tests don't
    /// share state.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        /// Create the guard under the temp directory.
        fn new(tag: &str, contents: &str) -> Self {
            let path = std::env::temp_dir().join(format!("tart-edit-{tag}-{}", std::process::id()));
            std::fs::write(&path, contents).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
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
        let file = Scratch::new("literal", "line1: cost $5.00 (a)\nline2: b.*x [y]\nline3\n");
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
        let file = Scratch::new("replace-all", "a a a\n");
        let output = perl_edit(file.path(), "a", "b", true);

        assert_eq!(
            output,
            format!("edited {}: 3 replacement(s)", file.path().display())
        );
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "b b b\n");
    }

    #[test]
    fn perl_edit_deletes_via_an_empty_new_string() {
        let file = Scratch::new("delete", "keep\ndrop me\nkeep\n");
        let output = perl_edit(file.path(), "drop me\n", "", false);

        assert!(output.contains("1 replacement(s)"), "{output}");
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "keep\nkeep\n");
    }

    #[test]
    fn perl_edit_reports_a_missing_match_and_leaves_the_file_untouched() {
        let file = Scratch::new("missing-match", "alpha beta\n");
        let output = perl_edit(file.path(), "gamma", "delta", false);

        assert!(output.contains("not found"), "{output}");
        assert_eq!(std::fs::read_to_string(file.path()).unwrap(), "alpha beta\n");
    }

    #[test]
    fn perl_edit_reports_an_ambiguous_match_without_replace_all() {
        let file = Scratch::new("ambiguous", "x x x\n");
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
        let file = Scratch::new("concurrent-perl", "AA eleven\nmid\nBB twelve\n");
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
    #[test]
    fn concurrent_edits_to_one_file_both_apply() {
        let file = Scratch::new("concurrent", "one UNO alpha\ntwo DOS beta\n");
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let spawn_edit = |old: &str, new: &str| {
            let call = edit_call(file.path(), old, new);
            let policy = policy.clone();
            std::thread::spawn(move || execute(&call, &policy, &|_| {}).unwrap())
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
        let file = Scratch::new("read-lock", "one\ntwo\n");
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
    #[test]
    fn read_returns_numbered_contents_and_ranges() {
        let mut contents = String::new();
        for line in 1..=30 {
            contents.push_str("line ");
            contents.push_str(&line.to_string());
            contents.push('\n');
        }
        let file = Scratch::new("read", &contents);
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let events = std::cell::RefCell::new(Vec::new());

        let whole = execute(&read_call(file.path(), None, None), &policy, &|progress| {
            events.borrow_mut().push(progress);
        })
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
        let range = execute(
            &read_call(file.path(), Some(10), Some(12)),
            &policy,
            &|progress| {
                events.borrow_mut().push(progress);
            },
        )
        .unwrap();
        assert_eq!(range, "    10\tline 10\n    11\tline 11\n    12\tline 12\n");
        assert!(matches!(
            events.borrow().as_slice(),
            [Progress::ToolStart { digest, .. }, _]
                if digest == &format!("{}:10-12", file.path().display())
        ));

        let tail = execute(&read_call(file.path(), Some(28), None), &policy, &|_| {}).unwrap();
        assert!(tail.starts_with("    28\tline 28\n"), "{tail}");
        assert_eq!(tail.lines().count(), 3);

        let missing = std::env::temp_dir().join("tart-read-does-not-exist");
        let absent = execute(&read_call(&missing, None, None), &policy, &|_| {}).unwrap();
        assert!(absent.contains("No such file or directory"), "{absent}");
    }
}
