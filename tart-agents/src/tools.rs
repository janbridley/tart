use std::path::Path;

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};

use crate::{Progress, sandbox::Policy};

/// Perform a raw string find-and-replace operation, holding a lock for thread safety..
///
/// This string contains perl source code to perform the required work, dispatching a
/// platform independent flock to ensure concurrent agents cannot collide.
/// Exits 1 with a warning, file untouched, or when the match count is wrong.
const EDIT_PROGRAM: &str = include_str!("data/edit.pl");

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
        "edit" => run_edit(call, policy, on_progress),
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

/// Run one bash tool call under `policy`, reporting its steps to `on_progress`.
fn run_bash<F: Fn(Progress)>(
    call: &FunctionToolCall,
    policy: &Policy,
    on_progress: &F,
) -> anyhow::Result<String> {
    let command = parse_command(&call.arguments)?;
    on_progress(Progress::Command(command.clone()));
    // A failure to launch comes back as an error string rather than a `Result`,
    // so the output can be handed straight back to the model.
    let output = policy
        .command("/bin/bash")
        .arg("-c")
        .arg(&command)
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
        );
    on_progress(Progress::CommandOutput(output.clone()));
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
    on_progress(Progress::Command(format!("edit {}", edit.path)));
    let result = apply_edit(&edit, policy);
    on_progress(Progress::CommandOutput(result.clone()));
    Ok(result)
}

/// Apply one parsed edit under `policy`, returning the outcome message.
///
/// We pre-check that the edit is valid in rust for performance, though the perl script
/// verifies to ensure we don't run into TOCTOU issues between here and the lock.
fn apply_edit(edit: &Edit, policy: &Policy) -> String {
    let path = Path::new(&edit.path);
    if edit.old_string.is_empty() {
        return format!("edit: old_string must not be empty: {}", path.display());
    }
    if edit.old_string == edit.new_string {
        return format!(
            "edit: old_string and new_string are identical: {}",
            path.display()
        );
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => return format!("edit: cannot read {}: {error}", path.display()),
    };
    let count = content.matches(&edit.old_string).count();
    if count == 0 {
        return format!(
            "edit: old_string not found in {}; the match must be exact, including whitespace",
            path.display()
        );
    }
    if count > 1 && !edit.replace_all {
        return format!(
            "edit: old_string matches {count} times in {}; pass replace_all or include more \
            surrounding lines to make it unique",
            path.display()
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
fn spawn_perl(edit: &Edit, cmd: &mut std::process::Command) -> String {
    let path = Path::new(&edit.path);
    cmd.arg("-e")
        .arg(EDIT_PROGRAM)
        .arg("--")
        .arg(&edit.path)
        .env("TART_OLD", &edit.old_string)
        .env("TART_NEW", &edit.new_string)
        .envs(edit.replace_all.then_some(("TART_ALL", "1")));
    match cmd.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim_end().to_string()
        }
        Ok(output) => format!(
            "edit failed on {}: {}{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("edit failed on {}: failed to run perl: {error}", path.display()),
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
            [Progress::Command(command), Progress::CommandOutput(output)]
                if command == "echo hi" && output == "hi\n"
        ));
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
}
