//! The search tool: ddgs's argv, its results file, and the rendering the model reads.

use std::ffi::OsString;
use std::path::Path;
use std::time::Duration;

use crate::Progress;
use crate::backends::{FunctionToolCall, Tool};
use crate::tools::{
    CancelToken, combined_output, command_text_inline, misuse, parse_arguments, run_watched,
    string_field, tool, traced, watched_failure,
};

use super::{temp_path, tool_binary, web_command};

/// The timeout every search runs under.
///
/// ddgs's `auto` backend can walk several engines before one answers, so a search needs
/// a fair amount of headroom.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Results returned when the model does not ask for a number.
const DEFAULT_SEARCH_RESULTS: u64 = 8;

/// Most results one search may return.
const MAX_SEARCH_RESULTS: u64 = 25;

/// The search tool's definition, offered only when a ddgs CLI is installed.
fn search_definition() -> Tool {
    tool(
        "search",
        "Search the web and return ranked results (title, url, snippet). \
        Runs ddgs outside the sandbox, so it has network access but \
        cannot read or write files",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "max_results": {
                    "type": "integer",
                    "description": "Results to return, 1-25; default 8"
                },
                "timelimit": {
                    "type": "string",
                    "description": "Only results from the past d(ay), w(eek), m(onth), or y(ear)"
                },
                "news": {
                    "type": "boolean",
                    "description": "Search news articles instead of web pages"
                }
            },
            "required": ["query"]
        }),
    )
}

/// The search tool; `None` when no ddgs CLI is installed, so the model is never
/// offered a tool that cannot run.
#[must_use]
pub(crate) fn search() -> Option<Tool> {
    tool_binary("TART_SEARCH_BIN", "ddgs").map(|_| search_definition())
}

/// One parsed search tool call.
#[derive(Debug)]
pub(super) struct Search {
    /// The query, passed to ddgs as a single argv element.
    pub query: String,
    /// Results to return, clamped to 1-25.
    pub max_results: u64,
    /// Recency window, one of the letters ddgs understands; `None` is any time.
    pub timelimit: Option<String>,
    /// Search news articles rather than web pages.
    pub news: bool,
}

/// Extract the fields from a search tool call's JSON arguments.
///
/// Optional fields fall back to ddgs's own defaults, wrong-typed values are
/// ignored, and `timelimit` is validated against the four letters ddgs accepts
/// so a nonsense value cannot become a CLI error.
pub(super) fn parse_search(arguments: &str) -> anyhow::Result<Search> {
    let args = parse_arguments(arguments)?;
    Ok(Search {
        query: string_field(&args, "search", "query")?,
        max_results: args["max_results"]
            .as_u64()
            .unwrap_or(DEFAULT_SEARCH_RESULTS)
            .clamp(1, MAX_SEARCH_RESULTS),
        timelimit: args["timelimit"]
            .as_str()
            .filter(|limit| matches!(*limit, "d" | "w" | "m" | "y"))
            .map(str::to_string),
        news: args["news"].as_bool().unwrap_or(false),
    })
}

/// The ddgs argv for one search: the subcommand, the query as a single argv
/// element, the result count, any recency window, and the results file.
fn ddgs_args(search: &Search, results: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        if search.news { "news" } else { "text" }.into(),
        "-q".into(),
        search.query.as_str().into(),
        "-m".into(),
        search.max_results.to_string().into(),
    ];
    if let Some(timelimit) = &search.timelimit {
        args.push("-t".into());
        args.push(timelimit.as_str().into());
    }
    args.push("-o".into());
    args.push(results.as_os_str().to_owned());
    args
}

/// Run one search tool call, reporting its steps to `on_progress`.
///
/// As with bash, a failure (no CLI, a rate-limited backend, a timeout) is
/// content for the model, not an error.
pub(in crate::tools) fn run_search<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> String {
    let search = match parse_search(&call.arguments) {
        Ok(search) => search,
        Err(error) => return misuse(call, on_progress, &error),
    };
    traced(call, on_progress, || {
        let Some(binary) = tool_binary("TART_SEARCH_BIN", "ddgs") else {
            let text = "search: no ddgs CLI found; install one with `uv tool install ddgs` \
                        or point TART_SEARCH_BIN at it"
                .to_string();
            return (text.clone(), text, None);
        };
        let results = temp_path("search", "json");
        let mut ddgs = web_command(&binary);
        ddgs.args(ddgs_args(&search, &results));
        let outcome = run_watched(&mut ddgs, Some(SEARCH_TIMEOUT), &CancelToken::new());
        // The results file is ours whatever happened: read it, then drop it.
        let json = std::fs::read_to_string(&results).ok();
        let _ = std::fs::remove_file(&results);
        match json.as_deref().and_then(|json| render_results(&search, json)) {
            Some(rendered) => (rendered.clone(), rendered, Some(0)),
            None => match outcome {
                Ok(run) => {
                    let exit = run.output.status.code();
                    // A success that rendered no results still reports the CLI's output.
                    let text = watched_failure(&run, SEARCH_TIMEOUT).unwrap_or_else(|| {
                        command_text_inline(&combined_output(&run.output), run.output.status, false)
                    });
                    (text.clone(), text, exit)
                }
                Err(error) => {
                    let text = format!("error: {error}");
                    (text.clone(), text, None)
                }
            },
        }
    })
}

/// A non-empty, trimmed string field of a results record.
///
/// Looked up with `get` rather than indexing: a missing key in a `Map` panics,
/// and news records carry no `href` for the url lookup to find.
fn field<'a>(
    record: &'a serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Option<&'a str> {
    record
        .get(name)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Turn ddgs's results file into the numbered list the model reads, or `None`.
fn render_results(search: &Search, json: &str) -> Option<String> {
    let Ok(serde_json::Value::Array(records)) = serde_json::from_str(json) else {
        return None;
    };
    if records.is_empty() {
        return Some(format!("no results for {:?}", search.query));
    }
    let mut rendered = format!(
        "{} results for {:?} (ddgs {})\n",
        records.len(),
        search.query,
        if search.news { "news" } else { "text" }
    );
    for (index, record) in records.iter().enumerate() {
        rendered.push_str(&render_result(index + 1, record.as_object()?));
    }
    Some(rendered)
}

/// One result: the title, an indented url, an indented date for news records,
/// and the snippet. Text records carry their address as `href`, news ones as `url`.
fn render_result(index: usize, record: &serde_json::Map<String, serde_json::Value>) -> String {
    let url = field(record, "href").or_else(|| field(record, "url"));
    let mut lines = vec![format!(
        "{index}. {}",
        field(record, "title").unwrap_or("(untitled)")
    )];
    for line in [url, field(record, "date"), field(record, "body")]
        .into_iter()
        .flatten()
    {
        lines.push(format!("   {line}"));
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn search_definition_requires_query() {
        let tool = serde_json::to_value(search_definition()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "search");
        assert_eq!(tool["parameters"]["required"][0], "query");
    }

    #[test]
    fn parse_search_reads_the_query_and_defaults_the_rest() {
        let search = parse_search(r#"{"query":"rust regex crate"}"#).unwrap();

        assert_eq!(search.query, "rust regex crate");
        assert_eq!(search.max_results, DEFAULT_SEARCH_RESULTS);
        assert_eq!(search.timelimit, None);
        assert!(!search.news);
    }

    #[test]
    fn parse_search_clamps_results_and_drops_unknown_timelimits() {
        let clamped =
            parse_search(r#"{"query":"q","max_results":9000,"timelimit":"century","news":true}"#)
                .unwrap();
        assert_eq!(clamped.max_results, MAX_SEARCH_RESULTS);
        assert_eq!(clamped.timelimit, None);
        assert!(clamped.news);

        let window = parse_search(r#"{"query":"q","timelimit":"w"}"#).unwrap();
        assert_eq!(window.timelimit.as_deref(), Some("w"));
    }

    #[test]
    fn parse_search_rejects_a_missing_query() {
        let error = parse_search(r#"{"timelimit":"d"}"#).unwrap_err().to_string();

        assert!(
            error.contains("The required parameter `query` is missing"),
            "{error}"
        );
    }

    #[test]
    fn ddgs_args_pass_the_subcommand_query_bounds_and_results_file() {
        let search = parse_search(r#"{"query":"rust web","max_results":5}"#).unwrap();
        let results = PathBuf::from("/tmp/tart-search.json");
        let text: Vec<OsString> = vec![
            "text".into(),
            "-q".into(),
            "rust web".into(),
            "-m".into(),
            "5".into(),
            "-o".into(),
            "/tmp/tart-search.json".into(),
        ];

        assert_eq!(ddgs_args(&search, &results), text);

        let news = parse_search(r#"{"query":"q","news":true,"timelimit":"d"}"#).unwrap();
        let news_args: Vec<OsString> = vec![
            "news".into(),
            "-q".into(),
            "q".into(),
            "-m".into(),
            "8".into(),
            "-t".into(),
            "d".into(),
            "-o".into(),
            "/tmp/tart-search.json".into(),
        ];

        assert_eq!(ddgs_args(&news, &results), news_args);
    }

    #[test]
    fn render_results_formats_text_and_news_records() {
        let search = parse_search(r#"{"query":"rust"}"#).unwrap();
        let json = r#"[
            {"title":"Rust","href":"https://www.rust-lang.org","body":"  A systems language  "},
            {"title":"News","url":"https://example.com/n","date":"2026-08-01","body":"Today"}
        ]"#;

        let rendered = render_results(&search, json).unwrap();

        assert!(
            rendered.starts_with("2 results for \"rust\" (ddgs text)\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("1. Rust\n   https://www.rust-lang.org\n   A systems language\n")
        );
        assert!(rendered.contains("2. News\n   https://example.com/n\n   2026-08-01\n   Today\n"));
    }

    #[test]
    fn render_results_reports_no_results_and_rejects_non_json() {
        let search = parse_search(r#"{"query":"q"}"#).unwrap();

        assert!(render_results(&search, "[]").unwrap().contains("no results"));
        // ddgs's own failure mode: nothing written, its error on stdout.
        assert!(render_results(&search, "RatelimitException: ...").is_none());
        // The contract is an array; anything else is not ours to interpret.
        assert!(render_results(&search, r#"{"results":[]}"#).is_none());
    }
}
