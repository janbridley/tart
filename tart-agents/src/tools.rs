use std::path::Path;

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};

use crate::{Progress, sandbox::Policy};

/// Replace the first instance of `$TART_OLD` with `$TART_NEW`.
///
/// `\Q...\E` quotes the old text so it matches literally.
const EDIT_PROGRAM: &str = "s/\\Q$ENV{TART_OLD}\\E/$ENV{TART_NEW}/";

/// [`EDIT_PROGRAM`] with `/g` to replace every occurrence.
const EDIT_PROGRAM_ALL: &str = "s/\\Q$ENV{TART_OLD}\\E/$ENV{TART_NEW}/g";

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
            "Replace an exact string in an existing file. old_string must match the file exactly, including whitespace and
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
    let expected = if edit.replace_all {
        content.replace(&edit.old_string, &edit.new_string)
    } else {
        content.replacen(&edit.old_string, &edit.new_string, 1)
    };

    let ran = policy
        .command("/usr/bin/perl")
        .arg("-0777")
        .arg("-i")
        .arg("-pe")
        .arg(if edit.replace_all { EDIT_PROGRAM_ALL } else { EDIT_PROGRAM })
        .arg("--")
        .arg(&edit.path)
        .env("TART_OLD", &edit.old_string)
        .env("TART_NEW", &edit.new_string)
        .output();
    if let (Ok(_), Ok(updated)) = (&ran, std::fs::read_to_string(path))
        && updated == expected
    {
        return format!("edited {}: {count} replacement(s)", path.display());
    }
    let detail = match &ran {
        Ok(output) => format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("failed to run perl: {error}"),
    };
    format!("edit failed on {}: {detail}", path.display())
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
}
