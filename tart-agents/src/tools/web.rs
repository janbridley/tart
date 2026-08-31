//! The web tools: `search` and `fetch`.
//!
//! These are the only tools that run **outside** the seatbelt sandbox.
//! Processing stays out of Rust where it can: ddgs writes structured JSON we
//! only render, and the r.jina.ai reader turns a page into markdown before
//! curl hands it back, so no HTML is ever parsed here.

use std::ffi::OsString;
use std::net::{IpAddr, ToSocketAddrs};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_openai::types::responses::{FunctionToolCall, Tool};

use super::{
    CancelToken, WatchedRun, combined_output, command_text, parse_arguments, run_watched,
    string_field, timeout_text, tool, traced,
};
use crate::Progress;

/// Where a `ddgs` (or compatible) CLI is looked for, after `TART_SEARCH_BIN`.
const SEARCH_DEFAULT: &str = "ddgs";

/// The reader binary, overridable with `TART_FETCH_BIN`.
const FETCH_DEFAULT: &str = "/usr/bin/curl";

/// The timeout every search runs under: ddgs's `auto` backend can walk several
/// engines before one answers, so a search needs more headroom than a command.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(60);

/// The timeout every fetch runs under; curl is told to give up just before, so
/// its own exit message usually beats the watchdog's kill.
const FETCH_TIMEOUT: Duration = Duration::from_secs(45);

/// Seconds curl may spend on one fetch, as an argv element.
const CURL_MAX_TIME: &str = "40";

/// Results returned when the model does not ask for a number.
const DEFAULT_SEARCH_RESULTS: u64 = 8;

/// Most results one search may return; more is noise the model cannot use.
const MAX_SEARCH_RESULTS: u64 = 25;

/// The reader service: it renders a page as markdown so curl hands back text.
const READER: &str = "https://r.jina.ai/";

/// A browser-ish `User-Agent`: many sites serve nothing to `curl/x.y`.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
     (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

/// The `--write-out` trailer raw fetches append after the body, so where
/// redirects actually landed can be checked. Written with curl's `\n` escapes;
/// [`FINAL_URL_MARKER`] is what they become in the captured output.
const FINAL_URL: &str = "\\n[tart-url]\\n%{url_effective}";

/// Where that trailer starts in a captured body.
const FINAL_URL_MARKER: &str = "\n[tart-url]\n";

/// The search tool's definition, offered only when a ddgs CLI is installed.
fn search_definition() -> Tool {
    tool(
        "search",
        "Search the web and return ranked results (title, url, snippet). Runs the \
        locally installed ddgs CLI outside the sandbox, so it has network access but \
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
    search_binary().map(|_| search_definition())
}

/// The fetch tool's definition, offered only when a curl binary exists.
fn fetch_definition() -> Tool {
    tool(
        "fetch",
        "Read one web page as markdown (title, source url, then the text) through the \
        r.jina.ai reader service, which strips scripts, styles, and markup. When the \
        reader errors (i.e. rate limit or auth) retry with raw=true, which fetches the URL \
        directly and suits JSON or plain-text endpoints. Runs outside the sandbox, so it \
        has network access but cannot read or write files; refuses non-public hosts",
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The absolute http(s) URL to read"},
                "raw": {
                    "type": "boolean",
                    "description": "Fetch the URL directly instead of through the reader; default false"
                }
            },
            "required": ["url"]
        }),
    )
}

/// The fetch tool; `None` when no curl binary exists.
#[must_use]
pub(crate) fn fetch() -> Option<Tool> {
    fetch_binary().map(|_| fetch_definition())
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
        query: string_field(&args, "query")?,
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

/// One parsed fetch tool call.
#[derive(Debug)]
pub(super) struct Fetch {
    /// The URL to read, validated before anything runs.
    pub url: String,
    /// Fetch the URL directly rather than through the reader service.
    pub raw: bool,
}

/// Extract the fields from a fetch tool call's JSON arguments.
///
/// `raw` is optional and defaults to the reader mode.
pub(super) fn parse_fetch(arguments: &str) -> anyhow::Result<Fetch> {
    let args = parse_arguments(arguments)?;
    Ok(Fetch {
        url: string_field(&args, "url")?,
        raw: args["raw"].as_bool().unwrap_or(false),
    })
}

/// Locate the search CLI: the `TART_SEARCH_BIN` override, `~/.local/bin/ddgs`
/// (where `uv tool install ddgs` puts it), then `PATH`.
fn search_binary() -> Option<PathBuf> {
    let pinned = std::env::var_os("TART_SEARCH_BIN").map(PathBuf::from);
    let installed = std::env::var_os("HOME").map(|home| {
        let mut path = PathBuf::from(home);
        path.push(".local/bin/ddgs");
        path
    });
    pinned
        .into_iter()
        .chain(installed)
        .find(|path| is_executable(path))
        .or_else(|| find_on_path(SEARCH_DEFAULT))
}

/// Locate the reader CLI: the `TART_FETCH_BIN` override, then the system curl.
fn fetch_binary() -> Option<PathBuf> {
    let pinned = std::env::var_os("TART_FETCH_BIN").map(PathBuf::from);
    pinned
        .into_iter()
        .chain([PathBuf::from(FETCH_DEFAULT)])
        .find(|path| is_executable(path))
}

/// The first executable `name` on `PATH`, if any.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

/// Whether `path` is a file any user may execute.
fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && (meta.permissions().mode() & 0o111) != 0)
}

/// A command for one of the web binaries: cleared like the sandboxed tools, so
/// nothing in our environment can reach the child, with only the variables a
/// CLI needs to run re-added.
fn web_command(binary: PathBuf) -> Command {
    let mut command = Command::new(binary);
    command.env_clear();
    if let Some(home) = std::env::var_os("HOME") {
        command.env("HOME", home);
    }
    if let Some(temp) = std::env::var_os("TMPDIR") {
        command.env("TMPDIR", temp);
    }
    if let Some(path) = std::env::var_os("PATH") {
        command.env("PATH", path);
    }
    command
}

/// A fresh path under the system temp directory for one search's results.
///
/// ddgs writes structured results only to a file, never to stdout; the counter
/// keeps concurrent searches apart without adding a dependency.
fn results_path() -> PathBuf {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tart-search-{}-{call}.json", std::process::id()))
}

/// The ddgs argv for one search: the subcommand, the query as a single argv
/// element, the result count, any recency window, and the results file.
fn ddgs_args(search: &Search, results: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        if search.news { "news" } else { "text" }.into(),
        "-q".into(),
        search.query.clone().into(),
        "-m".into(),
        search.max_results.to_string().into(),
    ];
    if let Some(timelimit) = &search.timelimit {
        args.push("-t".into());
        args.push(timelimit.clone().into());
    }
    args.push("-o".into());
    args.push(results.as_os_str().to_owned());
    args
}

/// Run one search tool call, reporting its steps to `on_progress`.
///
/// As with bash, a failure (no CLI, a rate-limited backend, a timeout) is
/// content for the model, not an error.
pub(super) fn run_search<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> anyhow::Result<String> {
    let search = parse_search(&call.arguments)?;
    Ok(traced(call, on_progress, || {
        let Some(binary) = search_binary() else {
            let text = "search: no ddgs CLI found; install one with `uv tool install ddgs` \
                        or point TART_SEARCH_BIN at it"
                .to_string();
            return (text.clone(), text, None);
        };
        let results = results_path();
        let mut ddgs = web_command(binary);
        ddgs.args(ddgs_args(&search, &results));
        let outcome = run_watched(&mut ddgs, Some(SEARCH_TIMEOUT), &CancelToken::new());
        // The results file is ours whatever happened: read it, then drop it.
        let json = std::fs::read_to_string(&results).ok();
        let _ = std::fs::remove_file(&results);
        match json.as_deref().and_then(|json| render_results(&search, json)) {
            Some(rendered) => (rendered.clone(), rendered, Some(0)),
            None => match outcome {
                Ok(run) => {
                    let WatchedRun { output, killed } = run;
                    let text = combined_output(&output);
                    let exit = output.status.code();
                    if killed.is_some() {
                        let marked = timeout_text(&text, SEARCH_TIMEOUT);
                        (marked.clone(), marked, exit)
                    } else {
                        (command_text(&text, output.status), text, exit)
                    }
                }
                Err(error) => {
                    let text = format!("error: {error}");
                    (text.clone(), text, None)
                }
            },
        }
    }))
}

/// Reject URLs that are not plain public web addresses.
///
/// Anything but `http`/`https` is refused outright for security.
fn check_url(url: &str) -> Result<String, String> {
    let url = url.trim();
    if url.is_empty() || url.len() >= 2_048 {
        return Err("fetch: url is empty or longer than 2048 characters".to_string());
    }
    if url.bytes().any(|byte| byte.is_ascii_control() || byte == b' ') {
        return Err("fetch: url contains whitespace or control characters".to_string());
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return Err(format!("fetch: url has no scheme: {url}"));
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(format!("fetch: only http and https are supported, not {scheme}"));
    }
    let host = authority_host(rest);
    if host.is_empty() {
        return Err(format!("fetch: url has no host: {url}"));
    }
    if is_private_host(host) {
        return Err(format!("fetch: refusing non-public host {host}"));
    }
    Ok(url.to_string())
}

/// The host in a URL's authority: userinfo dropped, port dropped, IPv6
/// unbracketed.
fn authority_host(rest: &str) -> &str {
    let host_port = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    if let Some(inner) = host_port.strip_prefix('[') {
        // A bracketed IPv6 literal, with the port (if any) after the `]`.
        return inner.split(']').next().unwrap_or_default();
    }
    // A name or IPv4 with an optional port; a tail that is not all digits is
    // part of the host, not a port.
    match host_port.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        _ => host_port,
    }
}

/// Whether a host names this machine or a private network rather than the web.
///
/// Names are resolved and every address checked, so a public-looking name that
/// answers for a private record is refused too.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the host is lowercased on the first line, so `.local` is already case-insensitive"
)]
fn is_private_host(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        // A zone ID names a link-local interface address (`[fe80::1%en0]`); it
        // never parses as one, and curl would still understand it, so refuse
        // it lexically before the resolver ever runs.
        || host.contains('%')
        || host.parse::<IpAddr>().is_ok_and(|ip| !is_global(ip))
        || resolves_to_private(&host)
}

/// Whether any address `host` resolves to is not globally routable.
///
/// Every record is checked: a rebinder answers with a public and a private
/// address together, and either may come back first. A name that fails to
/// resolve stays public: curl shares this resolver, so an unresolvable name
/// fails there with the better error. Resolution blocks this thread for as
/// long as the resolver takes, like the curl call it guards.
fn resolves_to_private(host: &str) -> bool {
    (host, 0)
        .to_socket_addrs()
        .is_ok_and(|mut addrs| addrs.any(|addr| !is_global(addr.ip())))
}

/// Whether an IP address is globally routable.
///
/// TODO: replace w/ `IpAddr::is_global` when stabilized.
fn is_global(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, _, _] = ip.octets();
            !ip.is_loopback()
                && !ip.is_private()
                && !ip.is_link_local()
                && !ip.is_multicast()
                && !ip.is_broadcast()
                && !ip.is_unspecified()
                && !(a == 100 && b & 0b1100_0000 == 0b0100_0000) // 100.64/10, shared
        }
        IpAddr::V6(ip) => {
            let [first, ..] = ip.segments();
            !ip.is_loopback()
                && !ip.is_multicast()
                && !ip.is_unspecified()
                && first & 0xfe00 != 0xfc00 // fc00::/7, unique local
                && first & 0xffc0 != 0xfe80 // fe80::/10, link local
        }
    }
}

/// The curl argv for one fetch.
///
/// `-q` comes first so a `~/.curlrc` cannot add flags of its own. Reader mode
/// omits `-f` because the reader's error bodies are content the model can act
/// on; raw mode takes it, plus the protocols held across redirects, a browser
/// `User-Agent` (sites 403 curl's own), and the final-URL trailer.
fn fetch_args(fetch: &Fetch, url: &str) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "-q".into(),
        "-sS".into(),
        "-L".into(),
        "--max-redirs".into(),
        "5".into(),
        "--connect-timeout".into(),
        "10".into(),
        "--max-time".into(),
        CURL_MAX_TIME.into(),
        "--compressed".into(),
    ];
    if fetch.raw {
        args.extend([
            "-f".into(),
            "--proto".into(),
            "=http,https".into(),
            "--proto-redir".into(),
            "=http,https".into(),
            "--max-filesize".into(),
            "2000000".into(),
            "-A".into(),
            USER_AGENT.into(),
            "-w".into(),
            FINAL_URL.into(),
            "--".into(),
            url.into(),
        ]);
    } else {
        if let Some(key) = std::env::var_os("TART_JINA_KEY") {
            args.extend([
                "-H".into(),
                format!("Authorization: Bearer {}", key.to_string_lossy()).into(),
            ]);
        }
        args.extend([
            "--proto".into(),
            "=https".into(),
            "--".into(),
            format!("{READER}{url}").into(),
        ]);
    }
    args
}

/// Run one fetch tool call, reporting its steps to `on_progress`.
///
/// Like `search` this runs outside the sandbox.
pub(super) fn run_fetch<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> anyhow::Result<String> {
    let fetch = parse_fetch(&call.arguments)?;
    Ok(traced(call, on_progress, || {
        let Some(binary) = fetch_binary() else {
            let text = format!("fetch: no curl found at {FETCH_DEFAULT}; set TART_FETCH_BIN");
            return (text.clone(), text, None);
        };
        // The guard is content visible to the model, not an error.
        let url = match check_url(&fetch.url) {
            Ok(url) => url,
            Err(text) => return (text.clone(), text, None),
        };
        let mut curl = web_command(binary);
        curl.args(fetch_args(&fetch, &url));
        match run_watched(&mut curl, Some(FETCH_TIMEOUT), &CancelToken::new()) {
            Err(error) => {
                let text = format!("error: {error}");
                (text.clone(), text, None)
            }
            Ok(WatchedRun { output, killed: Some(_) }) => {
                let marked = timeout_text(&combined_output(&output), FETCH_TIMEOUT);
                (marked.clone(), marked, output.status.code())
            }
            Ok(WatchedRun { output, .. }) => {
                let exit = output.status.code();
                let text = combined_output(&output);
                if !output.status.success() {
                    return (command_text(&text, output.status), text, exit);
                }
                // Success: split and check where redirects landed.
                let checked = match separate_final_url(&text) {
                    Some((body, final_url)) => match private_redirect(final_url) {
                        Some(refused) => return (refused.clone(), refused, exit),
                        None => body,
                    },
                    None => text.as_str(),
                };
                let text = checked.to_string();
                (command_text(&text, output.status), text, exit)
            }
        }
    }))
}

/// Split curl's final-URL trailer off a captured raw fetch, when present.
///
/// The trailer is the last thing curl writes to stdout, so the *last* marker
/// wins: a page that happens to contain the marker text cannot lose its own
/// tail. Only the first whitespace-free word is the URL; anything behind it is
/// stderr arriving after the trailer.
fn separate_final_url(text: &str) -> Option<(&str, &str)> {
    let at = text.rfind(FINAL_URL_MARKER)?;
    let final_url = text[at + FINAL_URL_MARKER.len()..].split_whitespace().next()?;
    Some((&text[..at], final_url))
}

/// The refusal for a final URL that landed on a private host, if it did.
fn private_redirect(final_url: &str) -> Option<String> {
    let rest = final_url.split_once("://").map_or(final_url, |(_, rest)| rest);
    is_private_host(authority_host(rest))
        .then(|| format!("fetch: refusing redirect to non-public host {final_url}"))
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
/// and the snippet. Text records carry their address as `href`, news ones as
/// `url`.
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
    use crate::Agent;
    use crate::sandbox::Policy;
    use crate::sandbox::live::skip_unless_networked;
    use crate::tools::{Tooling, execute};
    use macro_rules_attribute::apply;

    /// A tool call for `name` with raw JSON `arguments`.
    fn call(name: &str, arguments: &str) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
        }
    }

    #[test]
    fn search_definition_requires_query() {
        let tool = serde_json::to_value(search_definition()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "search");
        assert_eq!(tool["parameters"]["required"][0], "query");
    }

    #[test]
    fn search_is_offered_exactly_when_the_cli_is_installed() {
        assert_eq!(search().is_some(), search_binary().is_some());
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

        assert!(error.contains("missing 'query'"), "{error}");
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

    #[test]
    fn results_paths_are_fresh_and_under_the_temporary_directory() {
        let first = results_path();
        let second = results_path();

        assert!(first.starts_with(std::env::temp_dir()));
        assert!(first.extension().is_some_and(|extension| extension == "json"));
        assert_ne!(first, second);
    }

    #[test]
    fn fetch_definition_requires_url() {
        let tool = serde_json::to_value(fetch_definition()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "fetch");
        assert_eq!(tool["parameters"]["required"][0], "url");
    }

    #[test]
    fn fetch_is_offered_exactly_when_the_cli_is_installed() {
        assert_eq!(fetch().is_some(), fetch_binary().is_some());
    }

    #[test]
    fn parse_fetch_reads_the_url_and_defaults_to_the_reader() {
        let fetch = parse_fetch(r#"{"url":"https://example.com"}"#).unwrap();

        assert_eq!(fetch.url, "https://example.com");
        assert!(!fetch.raw);

        let raw = parse_fetch(r#"{"url":"https://api.example.com/v1","raw":true}"#).unwrap();
        assert!(raw.raw);
    }

    #[test]
    fn parse_fetch_rejects_a_missing_url() {
        let error = parse_fetch(r#"{"raw":true}"#).unwrap_err().to_string();

        assert!(error.contains("missing 'url'"), "{error}");
    }

    #[test]
    fn check_url_accepts_public_web_addresses() {
        for url in [
            "https://example.com/a?b=c#d",
            "HTTP://Example.COM/",
            "https://user:pass@example.com/",
            "http://192.0.2.10:8080/page",
            "https://[2001:db8::1]/",
        ] {
            assert_eq!(check_url(url).ok().as_deref(), Some(url), "{url}");
        }
    }

    #[test]
    fn check_url_refuses_other_schemes_private_hosts_and_malformed_urls() {
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://localhost",
            "data:text/html,hi",
            "javascript:alert(1)",
            "/etc/passwd",
            "example.com/no-scheme",
            "",
            "https://",
            "https://user@/",
            "https://localhost/x",
            "http://127.0.0.1:8080/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
            "http://[fe80::1]/",
            "https://printer.local/",
            "https://service.internal/",
            "https://foo.bar.localhost/",
            "https://example.com/ has space",
            "https://example.com/\u{7}",
            &format!("https://example.com/{}", "a".repeat(2_100)),
        ] {
            assert!(check_url(url).is_err(), "expected refusal: {url}");
        }
    }

    #[test]
    fn authority_host_drops_userinfo_ports_and_brackets() {
        assert_eq!(authority_host("example.com/path"), "example.com");
        assert_eq!(authority_host("user:pass@example.com:8080/x"), "example.com");
        assert_eq!(authority_host("[2001:db8::1]:443/x"), "2001:db8::1");
        assert_eq!(authority_host("example.com?q"), "example.com");
        assert_eq!(authority_host(""), "");
    }

    #[test]
    fn fetch_args_reader_mode_routes_through_the_reader() {
        // An exact argv only when no key is configured; a configured machine
        // gets one more `-H` pair and nothing else changes.
        if std::env::var_os("TART_JINA_KEY").is_some() {
            return;
        }
        let fetch = parse_fetch(r#"{"url":"https://example.com/page"}"#).unwrap();
        let expected: Vec<OsString> = vec![
            "-q".into(),
            "-sS".into(),
            "-L".into(),
            "--max-redirs".into(),
            "5".into(),
            "--connect-timeout".into(),
            "10".into(),
            "--max-time".into(),
            "40".into(),
            "--compressed".into(),
            "--proto".into(),
            "=https".into(),
            "--".into(),
            "https://r.jina.ai/https://example.com/page".into(),
        ];

        assert_eq!(fetch_args(&fetch, "https://example.com/page"), expected);
    }

    #[test]
    fn fetch_args_raw_mode_hits_the_url_directly() {
        let fetch = parse_fetch(r#"{"url":"https://example.com/x","raw":true}"#).unwrap();
        let expected: Vec<OsString> = vec![
            "-q".into(),
            "-sS".into(),
            "-L".into(),
            "--max-redirs".into(),
            "5".into(),
            "--connect-timeout".into(),
            "10".into(),
            "--max-time".into(),
            "40".into(),
            "--compressed".into(),
            "-f".into(),
            "--proto".into(),
            "=http,https".into(),
            "--proto-redir".into(),
            "=http,https".into(),
            "--max-filesize".into(),
            "2000000".into(),
            "-A".into(),
            USER_AGENT.into(),
            "-w".into(),
            FINAL_URL.into(),
            "--".into(),
            "https://example.com/x".into(),
        ];

        assert_eq!(fetch_args(&fetch, "https://example.com/x"), expected);
    }

    #[test]
    fn separate_final_url_takes_the_last_marker_and_first_word() {
        let captured = "page body\n[tart-url]\nnot this\n[tart-url]\nhttps://final.example/x \
                        curl: noise\n";

        let (body, final_url) = separate_final_url(captured).unwrap();

        // The newline before the marker belongs to the trailer, not the body.
        assert_eq!(body, "page body\n[tart-url]\nnot this");
        assert_eq!(final_url, "https://final.example/x");
        assert!(separate_final_url("no trailer").is_none());
    }

    #[test]
    fn private_redirect_flags_private_landings_only() {
        assert!(private_redirect("http://127.0.0.1:8080/x").is_some());
        assert!(private_redirect("http://printer.local/").is_some());
        assert!(private_redirect("https://example.com/final").is_none());
        assert!(private_redirect("https://[::1]/").is_some());
    }

    #[test]
    fn check_url_refuses_zone_ids_and_resolver_shorthand_addresses() {
        // A zone ID names a link-local interface address; it does not parse as
        // one, and curl would still fetch it, so it is refused lexically.
        for url in ["https://[fe80::1%25en0]/", "http://[fe80::1%en0]:8080/x"] {
            assert!(check_url(url).is_err(), "expected refusal: {url}");
        }

        // Integer and hex shorthands for 127.0.0.1 parse as no IP at all; the
        // resolver accepts them, so the record check refuses them.
        for url in ["http://2130706433/", "http://0x7f000001/"] {
            assert!(check_url(url).is_err(), "expected refusal: {url}");
        }
    }

    #[test]
    fn resolves_to_private_checks_every_record_and_fails_open() {
        // `localhost` is private through /etc/hosts alone: the resolver finds
        // it without any network, and every record it lists is loopback.
        assert!(resolves_to_private("localhost"));

        // A name the resolver rejects outright stays public: curl shares the
        // resolver, so it reports the failure better than a refusal here could.
        assert!(!resolves_to_private("not a hostname"));
    }

    /// Live: reaches the network and the system keychain, so it only passes
    /// outside a nested sandbox.
    #[apply(skip_unless_networked!)]
    #[test]
    fn run_search_returns_rendered_results() {
        let Some(_) = search_binary() else {
            return;
        };
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let token = CancelToken::new();
        let agent = Agent::new("http://localhost:9", "key", "model", policy.clone());
        let tools = Tooling {
            policy: &policy,
            cancel: &token,
            agents: None,
            template: &agent,
        };
        let events = std::cell::RefCell::new(Vec::new());
        let request = call(
            "search",
            r#"{"query":"rust programming language","max_results":3}"#,
        );

        let output = execute(&request, &tools, &|progress| {
            events.borrow_mut().push(progress);
        })
        .unwrap();

        assert!(
            output.contains("results for \"rust programming language\""),
            "{output}"
        );
        assert!(output.contains("1. "), "{output}");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, arguments, .. },
                Progress::ToolOutput { exit: Some(0), .. }
            ] if name == "search"
                && arguments
                    == r#"{"query":"rust programming language","max_results":3}"#
        ));
    }

    /// Live: reaches the network, so it needs connectivity. Raw mode, so it
    /// stands on example.com alone and not on the reader service.
    #[apply(skip_unless_networked!)]
    #[test]
    fn run_fetch_returns_the_page() {
        let Some(_) = fetch_binary() else {
            return;
        };
        let policy = Policy::new(std::env::temp_dir()).unwrap();
        let token = CancelToken::new();
        let agent = Agent::new("http://localhost:9", "key", "model", policy.clone());
        let tools = Tooling {
            policy: &policy,
            cancel: &token,
            agents: None,
            template: &agent,
        };
        let events = std::cell::RefCell::new(Vec::new());
        let request = call("fetch", r#"{"url":"https://example.com","raw":true}"#);

        let output = execute(&request, &tools, &|progress| {
            events.borrow_mut().push(progress);
        })
        .unwrap();

        assert!(output.contains("Example Domain"), "{output}");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, arguments, .. },
                Progress::ToolOutput { exit: Some(0), .. }
            ] if name == "fetch" && arguments == r#"{"url":"https://example.com","raw":true}"#
        ));
    }
}
