//! Formatting tool-calls nicely for the TUI.

/// The most characters one line keeps before the ellipsis.
const LINE_CAP: usize = 60;

/// The box header for a run of calls to one tool: the display name, then the
/// calls' digests joined with `", "`, capped to one line.
pub(crate) fn tool_header(name: &str, arguments: &[String]) -> String {
    let digest = arguments
        .iter()
        .map(|raw| argument(name, raw))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({})", display_name(name), one_line(&digest))
}

/// The wire name as shown: its first ASCII letter uppercased, e.g. `Bash`.
fn display_name(name: &str) -> String {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut shown = first.to_ascii_uppercase().to_string();
    shown.push_str(chars.as_str());
    shown
}

/// One call's digest: the field that names what it did, falling back to the
/// raw arguments when they do not parse or the tool is unknown.
fn argument(name: &str, raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|args| match name {
            "bash" => args["command"].as_str().map(str::to_string),
            "read" => Some(read_digest(&args)).filter(|digest| !digest.is_empty()),
            "edit" => args["path"].as_str().map(str::to_string),
            "search" => search_digest(&args),
            "fetch" => args["url"].as_str().map(str::to_string),
            _ => None,
        })
        .unwrap_or_else(|| raw.to_string())
}

/// A read call's digest: the path, with any bounds as `path:start-end`.
fn read_digest(args: &serde_json::Value) -> String {
    let Some(path) = args["path"].as_str() else {
        return String::new();
    };
    match (args["start_line"].as_u64(), args["end_line"].as_u64()) {
        (None, None) => path.to_string(),
        (Some(start), Some(end)) => format!("{path}:{start}-{end}"),
        (Some(start), None) => format!("{path}:{start}-"),
        (None, Some(end)) => format!("{path}:-{end}"),
    }
}

/// A search call's digest: the query, tagged ` [news]` for a news search.
fn search_digest(args: &serde_json::Value) -> Option<String> {
    let query = args["query"].as_str()?;
    Some(if args["news"].as_bool().unwrap_or(false) {
        format!("{query} [news]")
    } else {
        query.to_string()
    })
}

/// The first line of `text`, capped with an ellipsis.
fn one_line(text: &str) -> String {
    let line = text.split('\n').next().unwrap_or_default();
    let mut capped = line.chars().take(LINE_CAP).collect::<String>();
    if line.chars().nth(LINE_CAP).is_some() {
        capped.push('…');
    }
    capped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each tool's header shows the field that names what it did.
    #[test]
    fn headers_show_each_tools_interesting_field() {
        assert_eq!(
            tool_header("bash", &[r#"{"command":"ls -la"}"#.to_string()]),
            "Bash(ls -la)"
        );
        assert_eq!(
            tool_header("edit", &[r#"{"path":"src/main.rs"}"#.to_string()]),
            "Edit(src/main.rs)"
        );
        assert_eq!(
            tool_header("fetch", &[r#"{"url":"https://example.com"}"#.to_string()]),
            "Fetch(https://example.com)"
        );
        assert_eq!(
            tool_header("search", &[r#"{"query":"rust regex"}"#.to_string()]),
            "Search(rust regex)"
        );
        // A news search tags its query; the rest of the arguments stay out.
        assert_eq!(
            tool_header(
                "search",
                &[r#"{"query":"elections","news":true,"max_results":3}"#.to_string()]
            ),
            "Search(elections [news])"
        );
    }

    /// A read digests its path, with any bounds as `path:start-end`.
    #[test]
    fn read_digests_carry_their_bounds() {
        let read = |arguments: &str| tool_header("read", &[arguments.to_string()]);
        assert_eq!(read(r#"{"path":"src/main.rs"}"#), "Read(src/main.rs)");
        assert_eq!(
            read(r#"{"path":"src/main.rs","start_line":10,"end_line":50}"#),
            "Read(src/main.rs:10-50)"
        );
        // An open tail (or head) keeps its half of the bounds.
        assert_eq!(read(r#"{"path":"a.rs","start_line":28}"#), "Read(a.rs:28-)");
        assert_eq!(read(r#"{"path":"a.rs","end_line":12}"#), "Read(a.rs:-12)");
    }

    /// The display name is the wire name with its first ASCII letter up.
    #[test]
    fn display_names_uppercase_the_first_letter() {
        assert_eq!(display_name("bash"), "Bash");
        assert_eq!(display_name("fetch"), "Fetch");
        // Already capitalized stays put, and non-letters lead unchanged.
        assert_eq!(display_name("Bash"), "Bash");
        assert_eq!(display_name("_private"), "_private");
        assert_eq!(display_name(""), "");
    }

    /// Unparseable arguments, unknown tools, and missing fields all degrade
    /// to the raw arguments.
    #[test]
    fn unusable_arguments_degrade_to_the_raw_string() {
        for (name, raw) in [
            ("bash", "not json"),
            ("read", r#"{"start_line":1}"#),
            ("edit", r#"{"old_string":"a"}"#),
            ("search", r#"{"news":true}"#),
            ("fetch", r#"{"raw":true}"#),
            ("mcp", r#"{"path":"src"}"#),
        ] {
            assert_eq!(argument(name, raw), raw, "{name}");
        }
    }

    /// Several calls join with `", "`: a plain join, no range coalescing and
    /// no edit counting.
    #[test]
    fn several_calls_join_plainly() {
        let digests = |parts: &[&str]| parts.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(
            tool_header(
                "read",
                &digests(&[
                    r#"{"path":"a.rs","start_line":1,"end_line":10}"#,
                    r#"{"path":"a.rs","start_line":20,"end_line":30}"#
                ])
            ),
            "Read(a.rs:1-10, a.rs:20-30)"
        );
        assert_eq!(
            tool_header("edit", &digests(&[r#"{"path":"x.py"}"#, r#"{"path":"x.py"}"#])),
            "Edit(x.py, x.py)"
        );
    }

    /// The whole header caps to one line of 60 characters plus the ellipsis.
    #[test]
    fn headers_cap_to_one_line() {
        let long = tool_header("bash", &[format!(r#"{{"command":"{}"}}"#, "x".repeat(90))]);
        // `Bash(` + 60 kept characters + the ellipsis + `)`.
        assert!(
            long.starts_with(&format!("Bash({}", "x".repeat(LINE_CAP))),
            "{long}"
        );
        assert!(long.ends_with("…)"), "{long}");
        assert_eq!(
            long.chars().count(),
            "Bash(".len() + LINE_CAP + "…)".chars().count()
        );
        // A multi-line command shows its first line only: json unescapes
        // `\n` into a newline the cap cuts at.
        assert_eq!(
            tool_header("bash", &[r#"{"command":"echo hi\necho bye"}"#.to_string()]),
            "Bash(echo hi)"
        );
    }
}
