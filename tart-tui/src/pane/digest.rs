//! Formatting tool-calls nicely for the TUI.

use itertools::Itertools;
use serde_json::Value;

/// The most characters one line keeps before the ellipsis.
const LINE_CAP: usize = 60;

/// The box header for a run of calls to one tool: the display name, then the
/// calls' digests joined with `", "`, capped to one line.
pub(crate) fn tool_header(name: &str, arguments: &[String]) -> String {
    let digest = match name {
        "read" | "edit" => group_paths(name, arguments),
        _ => arguments.iter().map(|raw| argument(name, raw)).join(", "),
    };
    format!("{}({})", display_name(name), one_line(&digest))
}

/// The wire name as shown: its first ASCII letter uppercased, e.g. `Bash`.
fn display_name(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

/// One call's digest: the field that names what it did, or the raw arguments
/// when they do not parse or the tool is unknown.
pub(crate) fn argument(name: &str, raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|args| match name {
            "bash" => args["command"].as_str().map(str::to_string),
            "fetch" => args["url"].as_str().map(str::to_string),
            "read" | "edit" => args["path"].as_str().map(str::to_string),
            // The subagent pair: the task spawned, and the id waited on.
            "spawn" => args["task"].as_str().map(str::to_string),
            "wait" => args["id"].as_u64().map(|id| id.to_string()),
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
    let path = args["path"].as_str()?.to_string();
    let side = |key: &str, or: u64| args[key].as_u64().unwrap_or(or);
    Some((path, (side("start_line", 0), side("end_line", u64::MAX))))
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

/// The first line of `text`, capped with an ellipsis.
fn one_line(text: &str) -> String {
    let line = text.split('\n').next().unwrap_or_default();
    match line.char_indices().nth(LINE_CAP) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line.to_string(),
    }
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
            ("fetch", r#"{"url":"http://x"}"#, "Fetch(http://x)"),
            ("search", r#"{"query":"rust regex"}"#, "Search(rust regex)"),
            // A news search tags its query; the rest of the arguments stay out.
            ("search", news, "Search(elections [news])"),
            // A read carries its bounds: whole-file, closed, or open at an end.
            ("read", r#"{"path":"src/main.rs"}"#, "Read(src/main.rs)"),
            ("read", a12, "Read(a.rs:1-2)"),
            ("read", r#"{"path":"a.rs","start_line":28}"#, "Read(a.rs:28-)"),
            ("read", r#"{"path":"a.rs","end_line":12}"#, "Read(a.rs:-12)"),
            // Odd names keep their lead and an empty one stays empty.
            ("Bash", r#"{"command":"ls"}"#, r#"Bash({"command":"ls"})"#),
            ("_private", r#"{"path":"src"}"#, r#"_private({"path":"src"})"#),
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
        let a12 = r#"{"path":"a.rs","start_line":1,"end_line":2}"#;
        let (x, a, b) = (r#"{"path":"x.py"}"#, r#"{"path":"a.rs"}"#, r#"{"path":"b.rs"}"#);
        let a10_20 = r#"{"path":"a.rs","start_line":10,"end_line":20}"#;
        let a15_30 = r#"{"path":"a.rs","start_line":15,"end_line":30}"#;
        let (a5, a60, head12) = (
            r#"{"path":"a.rs","start_line":5}"#,
            r#"{"path":"a.rs","start_line":60}"#,
            r#"{"path":"a.rs","end_line":12}"#,
        );
        let calls: &[(&str, &[&str], &str)] = &[
            // The tools without paths join plainly, identical or not.
            ("fetch", &[r#"{"url":"u"}"#, r#"{"url":"u"}"#], "Fetch(u, u)"),
            // One file, several adjacent ranges -> they coalesce into one span.
            (
                "read",
                &[
                    r#"{"path":"README.md","start_line":1,"end_line":10}"#,
                    r#"{"path":"README.md","start_line":11,"end_line":20}"#,
                    r#"{"path":"README.md","start_line":21,"end_line":30}"#,
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
                    r#"{"path":"b.rs","start_line":1,"end_line":2}"#,
                    r#"{"path":"a.rs","start_line":5}"#,
                    r#"{"path":"b.rs","start_line":4,"end_line":5}"#,
                ],
                "Read(b.rs:1-2,4-5, a.rs:5-)",
            ),
            // A whole-file read subsumes its file's bounds, before or after.
            (
                "read",
                &[
                    r#"{"path":"a.rs","start_line":10,"end_line":20}"#,
                    r#"{"path":"b.rs","start_line":1,"end_line":2}"#,
                    r#"{"path":"a.rs"}"#,
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
        assert_eq!(
            tool_header("bash", &[format!(r#"{{"command":"{}"}}"#, "x".repeat(90))]),
            format!("Bash({}…)", "x".repeat(LINE_CAP))
        );
        assert_eq!(
            tool_header("bash", &[r#"{"command":"echo hi\necho bye"}"#.to_string()]),
            "Bash(echo hi)"
        );
    }
}
