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
fn argument(name: &str, raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|args| match name {
            "bash" => args["command"].as_str().map(str::to_string),
            "fetch" => args["url"].as_str().map(str::to_string),
            "search" => args["query"].as_str().map(|query| {
                let news = args["news"].as_bool() == Some(true);
                format!("{query}{}", if news { " [news]" } else { "" })
            }),
            _ => None,
        })
        .unwrap_or_else(|| raw.to_string())
}

/// A read call's bounds as `start-end`, `start-`, or `-end`; `None` reads the whole file.
fn bounds(args: &Value) -> Option<String> {
    let side = |key: &str| args[key].as_u64().map(|n| format!("{n}")).unwrap_or_default();
    let bounds = format!("{}-{}", side("start_line"), side("end_line"));
    if bounds == "-" { None } else { Some(bounds) }
}

/// One `edit`/`read` call's `(path, share)`, a read's share being its bounds;
/// `None` when the arguments name no path.
fn parts(name: &str, raw: &str) -> Option<(String, Option<String>)> {
    let args: Value = serde_json::from_str(raw).ok()?;
    let path = args["path"].as_str()?.to_string();
    Some((path, if name == "read" { bounds(&args) } else { None }))
}

/// Several `edit`/`read` calls grouped per path, each path once in
/// first-call order — reads trailing their unmerged bounds unless subsumed by
/// a whole-file read, edits counting repeats as ` × N` — with calls naming no
/// path falling in at the end as their raw strings.
fn group_paths(name: &str, arguments: &[String]) -> String {
    let mut grouped: Vec<(String, Vec<Option<String>>)> = Vec::new();
    let mut loose: Vec<String> = Vec::new();
    for raw in arguments {
        let Some((path, share)) = parts(name, raw) else {
            loose.push(raw.clone());
            continue;
        };
        match grouped.iter_mut().find(|(known, _)| known == &path) {
            Some((_, shares)) => shares.push(share),
            None => grouped.push((path, vec![share])),
        }
    }
    grouped
        .iter()
        .map(|(path, shares)| match name {
            "edit" if shares.len() > 1 => format!("{path} × {}", shares.len()),
            "edit" => path.clone(),
            _ if shares.iter().any(Option::is_none) => path.clone(),
            _ => format!("{path}:{}", shares.iter().flatten().join(",")),
        })
        .chain(loose)
        .join(", ")
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
        let calls: &[(&str, &[&str], &str)] = &[
            // The tools without paths join plainly, identical or not.
            ("fetch", &[r#"{"url":"u"}"#, r#"{"url":"u"}"#], "Fetch(u, u)"),
            // One file, several ranges — unmerged, repeats included.
            (
                "read",
                &[
                    r#"{"path":"README.md","start_line":1,"end_line":10}"#,
                    r#"{"path":"README.md","start_line":11,"end_line":20}"#,
                    r#"{"path":"README.md","start_line":21,"end_line":30}"#,
                ],
                "Read(README.md:1-10,11-20,21-30)",
            ),
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
            // Reads never count repeats; unparsed company ends up raw.
            (
                "read",
                &[a12, a12, r#"{"start_line":3}"#, "not json"],
                r#"Read(a.rs:1-2,1-2, {"start_line":3}, not json)"#,
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
