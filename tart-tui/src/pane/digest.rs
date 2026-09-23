//! Formatting tool-calls nicely for the TUI.

use std::sync::LazyLock;

use itertools::Itertools;
use serde_json::Value;

use super::{ONE_LINE_CAP, clip_line};

/// `cd <dir> && ` prefixes are stripped from the output of shell commands.
static CWD_CD_PREFIXES: LazyLock<Vec<String>> = LazyLock::new(|| match std::env::current_dir() {
    Ok(dir) => ["", "\"", "'"]
        .into_iter()
        .map(|quote| format!("cd {quote}{}{quote} && ", dir.display()))
        .collect(),
    Err(_) => Vec::new(),
});

/// A bash call's command with a leading `cd` into `pwd` dropped for clarity.
fn preprocess_bash_command(command: &str) -> String {
    CWD_CD_PREFIXES
        .iter()
        .find_map(|prefix| command.strip_prefix(prefix.as_str()))
        .map(str::trim_start)
        .filter(|rest| !rest.is_empty())
        .unwrap_or(command)
        .to_string()
}

/// The box header for a run of calls to one tool: the display name, then the
/// calls' digests joined with `", "`, capped to one line.
pub(crate) fn tool_header(name: &str, arguments: &[String]) -> String {
    let digest = match name {
        "read" | "edit" => group_paths(name, arguments),
        "check_agent" => group_ids(arguments),
        _ => arguments.iter().map(|raw| argument(name, raw)).join(", "),
    };
    format!("{}({})", display_name(name), clip_line(&digest, ONE_LINE_CAP))
}

/// The wire name as shown: each underscore-separated word capitalized and
/// spaced, e.g. `bash` -> `Bash`, `check_agent` -> `Check Agent`.
fn display_name(name: &str) -> String {
    name.split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            let first = chars.next().unwrap_or_default().to_ascii_uppercase();
            format!("{first}{}", chars.as_str())
        })
        .join(" ")
}

/// One call's digest: the field that names what it did, or the raw arguments
/// when they do not parse or the tool is unknown.
pub(crate) fn argument(name: &str, raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|args| match name {
            "bash" => args["command"].as_str().map(preprocess_bash_command),
            "fetch" => args["url"].as_str().map(str::to_string),
            "read" | "edit" => args["file_path"]
                .as_str()
                .or_else(|| args["path"].as_str())
                .map(str::to_string),
            // The subagent pair: the task spawned, and the id checked on.
            "spawn_agent" => args["task"].as_str().map(str::to_string),
            "check_agent" => args["id"].as_u64().map(|id| id.to_string()),
            "search" => args["query"].as_str().map(|query| {
                let news = args["news"].as_bool() == Some(true);
                format!("{query}{}", if news { " [news]" } else { "" })
            }),
            _ => None,
        })
        .unwrap_or_else(|| raw.to_string())
}

/// One `edit`/`read` call's `(path, span)`; `None` when the arguments name no path.
fn parts(raw: &str) -> Option<(String, (u64, u64))> {
    let args: Value = serde_json::from_str(raw).ok()?;
    let path = args["file_path"]
        .as_str()
        .or_else(|| args["path"].as_str())?
        .to_string();
    // Zeros count as omitted, matching how the reader itself treats them.
    let offset = args["offset"].as_u64().filter(|&offset| offset > 0);
    let limit = args["limit"].as_u64().filter(|&limit| limit > 0);
    let (start, end) = match (offset, limit) {
        (Some(offset), Some(limit)) => (offset, offset.saturating_add(limit).saturating_sub(1)),
        (Some(offset), None) => (offset, u64::MAX),
        (None, Some(limit)) => (0, limit),
        (None, None) => (0, u64::MAX),
    };
    Some((path, (start, end)))
}

/// Group several `edit`/`read` calls grouped per path, each path once.
fn group_paths(name: &str, arguments: &[String]) -> String {
    let mut grouped: Vec<(String, Vec<(u64, u64)>)> = Vec::new();
    let mut loose: Vec<String> = Vec::new();
    for raw in arguments {
        let Some((path, bounds)) = parts(raw) else {
            loose.push(raw.clone());
            continue;
        };
        match grouped.iter_mut().find(|(known, _)| known == &path) {
            Some((_, spans)) => spans.push(bounds),
            None => grouped.push((path, vec![bounds])),
        }
    }
    grouped
        .iter()
        .map(|(path, spans)| match name {
            "edit" if spans.len() > 1 => format!("{path} × {}", spans.len()),
            "edit" => path.clone(),
            _ => coalesce(path, spans),
        })
        .chain(loose)
        .join(", ")
}

/// A run of `check_agent` calls, each id named once with its repeat count
/// from the second check on, in first-checked order.
fn group_ids(arguments: &[String]) -> String {
    let mut grouped: Vec<(String, usize)> = Vec::new();
    for digest in arguments.iter().map(|raw| argument("check_agent", raw)) {
        match grouped.iter_mut().find(|(known, _)| *known == digest) {
            Some((_, count)) => *count += 1,
            None => grouped.push((digest, 1)),
        }
    }
    grouped
        .into_iter()
        .map(|(digest, count)| match count {
            more @ 2.. => format!("{digest} × {more}"),
            _ => digest,
        })
        .join(", ")
}

/// A read path's spans, sorted and merged where adjacent or overlapping.
fn coalesce(path: &str, spans: &[(u64, u64)]) -> String {
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in spans.iter().copied().sorted() {
        match merged.last_mut() {
            Some(last) if last.1.saturating_add(1) >= start => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    if merged == [(0, u64::MAX)] {
        return path.to_string();
    }
    format!("{path}:{}", merged.iter().copied().map(span).join(","))
}

/// One range as `start-end`, an open end (`0`, `u64::MAX`) rendered bare.
fn span((start, end): (u64, u64)) -> String {
    match (start, end) {
        (_, u64::MAX) => format!("{start}-"),
        (0, _) => format!("-{end}"),
        _ => format!("{start}-{end}"),
    }
}

/// One child-agent tool call as it reads inside the agent box's header.
pub(crate) fn child_call(name: &str, raw: &str) -> String {
    format!("{}({})", display_name(name), argument(name, raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lone_calls_digest_their_naming_field() {
        let a12 = r#"{"path":"a.rs","start_line":1,"end_line":2}"#;
        let news = r#"{"query":"elections","news":true,"max_results":3}"#;
        for (name, raw, expected) in [
            ("bash", r#"{"command":"ls -la"}"#, "Bash(ls -la)"),
            ("edit", r#"{"path":"src/main.rs"}"#, "Edit(src/main.rs)"),
            ("edit", r#"{"file_path":"src/main.rs"}"#, "Edit(src/main.rs)"),
            ("fetch", r#"{"url":"http://x"}"#, "Fetch(http://x)"),
            ("search", r#"{"query":"rust regex"}"#, "Search(rust regex)"),
            // Reads in Claude Code's shape: file_path with offset/limit.
            ("read", r#"{"file_path":"src/main.rs"}"#, "Read(src/main.rs)"),
            (
                "read",
                r#"{"file_path":"a.rs","offset":1,"limit":2}"#,
                "Read(a.rs:1-2)",
            ),
            ("read", r#"{"file_path":"a.rs","offset":28}"#, "Read(a.rs:28-)"),
            ("read", r#"{"file_path":"a.rs","limit":12}"#, "Read(a.rs:-12)"),
            // Zeros count as omitted, so they do not invert the span.
            (
                "read",
                r#"{"file_path":"a.rs","offset":0,"limit":12}"#,
                "Read(a.rs:-12)",
            ),
            // The subagent pair: the task spawned, the id checked on.
            (
                "spawn_agent",
                r#"{"task":"find the flaky test"}"#,
                "Spawn Agent(find the flaky test)",
            ),
            ("check_agent", r#"{"id":2}"#, "Check Agent(2)"),
            // A news search tags its query; the rest of the arguments stay out.
            ("search", news, "Search(elections [news])"),
            ("read", r#"{"path":"src/main.rs"}"#, "Read(src/main.rs)"),
            ("read", a12, "Read(a.rs)"),
            // Odd names keep their lead and an empty one stays empty; an
            // underscore names a word break.
            ("Bash", r#"{"command":"ls"}"#, r#"Bash({"command":"ls"})"#),
            ("_private", r#"{"path":"src"}"#, r#"Private({"path":"src"})"#),
            ("", "loose", "(loose)"),
            // Unparseable arguments and missing fields degrade to the raw string.
            ("bash", "not json", "Bash(not json)"),
            ("read", r#"{"start_line":1}"#, r#"Read({"start_line":1})"#),
            ("edit", r#"{"old_string":"a"}"#, r#"Edit({"old_string":"a"})"#),
            ("search", r#"{"news":true}"#, r#"Search({"news":true})"#),
            ("fetch", r#"{"raw":true}"#, r#"Fetch({"raw":true})"#),
        ] {
            assert_eq!(tool_header(name, &[raw.to_string()]), expected);
        }
    }

    #[test]
    fn runs_of_calls_group_or_join() {
        let a12 = r#"{"file_path":"a.rs","offset":1,"limit":2}"#;
        let (x, a, b) = (r#"{"path":"x.py"}"#, r#"{"path":"a.rs"}"#, r#"{"path":"b.rs"}"#);
        let a10_20 = r#"{"file_path":"a.rs","offset":10,"limit":11}"#;
        let a15_30 = r#"{"file_path":"a.rs","offset":15,"limit":16}"#;
        let (a5, a60, head12) = (
            r#"{"file_path":"a.rs","offset":5}"#,
            r#"{"file_path":"a.rs","offset":60}"#,
            r#"{"file_path":"a.rs","limit":12}"#,
        );
        let calls: &[(&str, &[&str], &str)] = &[
            // The tools without paths join plainly, identical or not.
            ("fetch", &[r#"{"url":"u"}"#, r#"{"url":"u"}"#], "Fetch(u, u)"),
            // One file, several adjacent ranges -> they coalesce into one span.
            (
                "read",
                &[
                    r#"{"file_path":"README.md","offset":1,"limit":10}"#,
                    r#"{"file_path":"README.md","offset":11,"limit":10}"#,
                    r#"{"file_path":"README.md","offset":21,"limit":10}"#,
                ],
                "Read(README.md:1-30)",
            ),
            // Overlapping ranges merge into the one span covering them.
            ("read", &[a10_20, a15_30], "Read(a.rs:10-30)"),
            // Ranges render in range order, whatever the call order.
            ("read", &[a10_20, a12], "Read(a.rs:1-2,10-20)"),
            // Two open tails union into the earlier one.
            ("read", &[a5, a60], "Read(a.rs:5-)"),
            // An open head and an open tail cover the file between them.
            ("read", &[head12, a5], "Read(a.rs)"),
            // Interleaved files keep first-call order; open tails ride along.
            (
                "read",
                &[
                    r#"{"file_path":"b.rs","offset":1,"limit":2}"#,
                    r#"{"file_path":"a.rs","offset":5}"#,
                    r#"{"file_path":"b.rs","offset":4,"limit":2}"#,
                ],
                "Read(b.rs:1-2,4-5, a.rs:5-)",
            ),
            // A whole-file read subsumes its file's bounds, before or after.
            (
                "read",
                &[
                    r#"{"file_path":"a.rs","offset":10,"limit":11}"#,
                    r#"{"file_path":"b.rs","offset":1,"limit":2}"#,
                    r#"{"file_path":"a.rs"}"#,
                ],
                "Read(a.rs, b.rs:1-2)",
            ),
            // Repeated bounds collapse to one; unparsed company ends up raw.
            (
                "read",
                &[a12, a12, r#"{"start_line":3}"#, "not json"],
                r#"Read(a.rs:1-2, {"start_line":3}, not json)"#,
            ),
            // Edits name each path once, counting repeats from the second on.
            ("edit", &[x, x, x], "Edit(x.py × 3)"),
            ("edit", &[b, a, a], "Edit(b.rs, a.rs × 2)"),
            ("edit", &[x, b], "Edit(x.py, b.rs)"),
            ("edit", &[x, "not json"], "Edit(x.py, not json)"),
            // Checks name each id once, counting repeats the same way, in
            // first-checked order; unparsed arguments ride along raw.
            (
                "check_agent",
                &[r#"{"id":1}"#, r#"{"id":1}"#, r#"{"id":1}"#],
                "Check Agent(1 × 3)",
            ),
            (
                "check_agent",
                &[r#"{"id":2}"#, r#"{"id":1}"#, r#"{"id":2}"#],
                "Check Agent(2 × 2, 1)",
            ),
            ("check_agent", &[r#"{"id":7}"#], "Check Agent(7)"),
            (
                "check_agent",
                &[r#"{"id":1}"#, "not json"],
                "Check Agent(1, not json)",
            ),
        ];
        for &(name, raws, expected) in calls {
            let arguments = raws.iter().map(ToString::to_string).collect::<Vec<_>>();
            assert_eq!(tool_header(name, &arguments), expected);
        }
    }

    /// The whole header caps to one line of 60 characters plus the ellipsis;
    /// a multi-line digest keeps its first line only.
    #[test]
    fn headers_cap_to_one_line() {
        // The clip reserves one cell for its ellipsis, so a 60-cell cap
        // keeps 59 characters.
        assert_eq!(
            tool_header("bash", &[format!(r#"{{"command":"{}"}}"#, "x".repeat(90))]),
            format!("Bash({}…)", "x".repeat(ONE_LINE_CAP - 1))
        );
        assert_eq!(
            tool_header("bash", &[r#"{"command":"echo hi\necho bye"}"#.to_string()]),
            "Bash(echo hi)"
        );
    }

    #[test]
    fn bash_digests_drop_the_cwd_cd() {
        let cwd = std::env::current_dir().expect("valid").display().to_string();
        let bash = |command: &str| {
            let raw = serde_json::json!({ "command": command }).to_string();
            tool_header("bash", &[raw])
        };
        assert_eq!(bash(&format!("cd {cwd} && cargo test")), "Bash(cargo test)");
        assert_eq!(bash(&format!("cd {cwd} &&  ls")), "Bash(ls)");
        assert_eq!(bash(&format!("cd \"{cwd}\" && make")), "Bash(make)");
        assert_eq!(bash(&format!("cd '{cwd}' && make check")), "Bash(make check)");
        assert_eq!(bash("cd /tmp && ls"), "Bash(cd /tmp && ls)");
        assert_eq!(bash(&format!("cd {cwd}")), format!("Bash(cd {cwd})"));
    }
}
