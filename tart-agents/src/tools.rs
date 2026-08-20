use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};

use crate::Progress;

/// Basic, *unsandboxed* bash tool. TODO: REPLACE!!
#[must_use]
pub(crate) fn bash() -> Tool {
    Tool::Function(FunctionTool {
        defer_loading: None,
        name: "bash".to_string(),
        description: Some("Run a bash command and return its stdout/stderr".to_string()),
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

/// Extract the command from a bash tool call's JSON arguments.
fn parse_command(arguments: &str) -> anyhow::Result<String> {
    let args: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|error| anyhow::anyhow!("tool arguments weren't JSON: {error}"))?;
    args["command"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("tool call missing 'command'"))
}

/// Run one tool call, report each step to `on_progress`, and return output to the model
///
/// Tool *failures* (a non-zero exit) are not errors here: their output is content that
/// the model should see.
pub(crate) fn execute<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> anyhow::Result<String> {
    if call.name != "bash" {
        anyhow::bail!("unknown tool: {}", call.name);
    }
    let command = parse_command(&call.arguments)?;
    on_progress(Progress::Command(command.clone()));
    let output = run_bash(&command);
    on_progress(Progress::CommandOutput(output.clone()));
    Ok(output)
}

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

    #[test]
    fn execute_reports_command_then_output() {
        let events = std::cell::RefCell::new(Vec::new());
        let output = execute(&bash_call(r#"{"command":"echo hi"}"#), &|progress| {
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
        let mut call = bash_call(r#"{"command":"ls"}"#);
        call.name = "rm".to_string();

        let error = execute(&call, &|_| {}).unwrap_err().to_string();

        assert!(error.contains("unknown tool"), "{error}");
    }
}
