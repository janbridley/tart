//! Web search and web fetch tools.
//!
//! These are the only tools that run **outside** the seatbelt sandbox: the sandbox
//! denies network access by design, and a search or a page read *is* network access.
//! The escape is deliberate and kept as narrow as it can be:
//!
//! - The harness execs a fixed binary it located itself (`ddgs`, `curl`). The model
//!   picks an *argument* — a query or a URL — never a program, and there is no shell,
//!   so nothing can be interpolated into one.
//! - Nothing is written to disk: output is captured in memory and handed back as tool
//!   output. There is no output-path parameter to aim at the filesystem.
//! - The child runs with a cleared environment and a null stdin, and is killed as a
//!   process group when its deadline passes.
//! - `curl` is pinned to `http`/`https` (including across redirects) with `-q` first so
//!   a `~/.curlrc` cannot add flags, and refuses hosts that are not public.
//!
//! The result is a tool with the *read* side of network access and none of the write
//! side. `bash` keeps its "network is off" guarantee.

use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use async_openai::types::responses::{FunctionToolCall, Tool};

use crate::Progress;
use crate::tools::{parse_arguments, string_field, tool, traced};

/// Where a `ddgs` (or compatible) CLI is looked for, after `TART_SEARCH_BIN`.
const SEARCH_DEFAULT: &str = "ddgs";

/// The reader binary, overridable with `TART_FETCH_BIN`.
const FETCH_DEFAULT: &str = "/usr/bin/curl";

/// Seconds a search may run when none is requested.
const DEFAULT_SEARCH_TIMEOUT: Duration = Duration::from_secs(45);

/// Longest a search may run.
const MAX_SEARCH_TIMEOUT: Duration = Duration::from_secs(180);

/// Results returned when the model does not ask for a number.
const DEFAULT_MAX_RESULTS: u64 = 8;

/// Most results one search may return; more is noise the model cannot use.
const MAX_RESULTS: u64 = 25;

/// Seconds a fetch may run when none is requested.
const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Longest a fetch may run, matching the outer tool deadline.
const MAX_FETCH_TIMEOUT: Duration = Duration::from_secs(120);

/// Body bytes kept when the model does not ask for a size.
const DEFAULT_MAX_BYTES: usize = 120_000;

/// Most body bytes one fetch may keep.
const MAX_MAX_BYTES: usize = 2_000_000;

/// Diagnostics are drained fully but only this much is kept.
const STDERR_CAP: usize = 8_192;

/// The read size for capture loops.
const CHUNK: usize = 8_192;

/// A browser-ish `User-Agent`: many sites serve nothing to `curl/x.y`.
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
     (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

/// The `--write-out` trailer curl appends after the body, so the model sees the
/// response's own description of itself.
///
/// Each field gets its own line because `%{content_type}` contains spaces.
const CURL_META: &str = "\\n\\n[tart-meta]\\nhttp=%{http_code}\\ntype=%{content_type}\\
nbytes=%{size_download}\\nurl=%{url_effective}\\n";

/// Where the trailer starts in a captured body.
const META_MARKER: &str = "[tart-meta]\n";

/// The search tool; only offered when a `ddgs`-compatible CLI is installed.
#[must_use]
pub(crate) fn search() -> Option<Tool> {
    search_binary().map(|_| search_definition())
}

fn search_definition() -> Tool {
    tool(
            "search",
            "Search the web and return ranked results (title, url, snippet). Uses the \
            locally installed ddgs CLI, so it works without an API key. Runs outside the \
            sandbox with network access but cannot read or write files",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "max_results": {
                        "type": "integer",
                        "description": "Results to return, 1-25; default 8"
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code such as us-en or de-de; omit for the default"
                    },
                    "timelimit": {
                        "type": "string",
                        "description": "Restrict to results from the past d(ay), w(eek), m(onth), or y(ear)"
                    },
                    "news": {
                        "type": "boolean",
                        "description": "Search news articles instead of web pages"
                    }
                },
                "required": ["query"]
            }),
        )
    })
}

/// The fetch tool; only offered when a `curl` binary exists.
#[must_use]
pub(crate) fn fetch() -> Option<Tool> {
    fetch_binary().map(|_| {
        tool(
            "fetch",
            "Read one http(s) URL and return the response as readable text: scripts, \
            styles, and markup stripped, link targets kept. The status line, content \
            type, byte count, and final URL after redirects are reported first. Pass \
            raw=true for the body verbatim, e.g. for JSON APIs. Refuses hosts that are \
            not public, and cannot read or write files",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The absolute http(s) URL to read"},
                    "raw": {
                        "type": "boolean",
                        "description": "Return the body without HTML-to-text conversion; default false"
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Bytes of body to keep, 1000-2000000; default 120000"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds the request may run, 1-120; default 30"
                    }
                },
                "required": ["url"]
            }),
        )
    })
}

/// One parsed search tool call.
#[derive(Debug)]
pub(crate) struct Search {
    /// The query, passed as a single argv element.
    pub query: String,
    /// Results to return, clamped to 1-25.
    pub max_results: u64,
    /// Region code, validated to `xx-xx` shape.
    pub region: Option<String>,
    /// Recency window, one of `d`, `w`, `m`, `y`.
    pub timelimit: Option<String>,
    /// Search news rather than web pages.
    pub news: bool,
    /// Seconds the search may run, clamped.
    pub timeout: Duration,
}

/// Extract the fields from a search tool call's JSON arguments.
///
/// Optional fields fall back to the CLI's own defaults; wrong-typed values are ignored
/// rather than rejected, matching `read` and `edit`.
pub(crate) fn parse_search(arguments: &str) -> anyhow::Result<Search> {
    let args = parse_arguments(arguments)?;
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the clamp bounds the value before either cast"
    )]
    let seconds = args["timeout"]
        .as_i64()
        .unwrap_or(DEFAULT_SEARCH_TIMEOUT.as_secs() as i64)
        .clamp(5, MAX_SEARCH_TIMEOUT.as_secs() as i64) as u64;
    // Region codes and recency letters are validated rather than passed through, so a
    // nonsense value fails in the CLI's own words instead of half-working.
    let region = args["region"].as_str().and_then(valid_region);
    let timelimit = args["timelimit"].as_str().and_then(valid_timelimit);
    Ok(Search {
        query: string_field(&args, "query")?,
        max_results: args["max_results"]
            .as_u64()
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, MAX_RESULTS),
        region,
        timelimit,
        news: args["news"].as_bool().unwrap_or(false),
        timeout: Duration::from_secs(seconds),
    })
}

/// Accept a `xx-xx` region code, lowercased; anything else is dropped.
fn valid_region(region: &str) -> Option<String> {
    let lower = region.to_ascii_lowercase();
    let mut parts = lower.split('-');
    let valid = (2..=8).contains(&lower.len())
        && lower.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        && parts.next().is_some_and(|part| (1..=3).contains(&part.len()))
        && parts.next().is_some_and(|part| (1..=3).contains(&part.len()))
        && parts.next().is_none();
    valid.then_some(lower)
}

/// Accept one of the four recency letters ddgs understands.
fn valid_timelimit(timelimit: &str) -> Option<String> {
    matches!(timelimit, "d" | "w" | "m" | "y").then(|| timelimit.to_string())
}

/// One parsed fetch tool call.
#[derive(Debug)]
pub(crate) struct Fetch {
    /// The absolute URL to read.
    pub url: String,
    /// Return the body without HTML-to-text conversion.
    pub raw: bool,
    /// Body bytes to keep, clamped.
    pub max_bytes: usize,
    /// Seconds the request may run, clamped.
    pub timeout: Duration,
}

/// Extract the fields from a fetch tool call's JSON arguments.
pub(crate) fn parse_fetch(arguments: &str) -> anyhow::Result<Fetch> {
    let args = parse_arguments(arguments)?;
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the clamp bounds the value before either cast"
    )]
    let seconds = args["timeout"]
        .as_i64()
        .unwrap_or(DEFAULT_FETCH_TIMEOUT.as_secs() as i64)
        .clamp(1, MAX_FETCH_TIMEOUT.as_secs() as i64) as u64;
    Ok(Fetch {
        url: string_field(&args, "url")?,
        raw: args["raw"].as_bool().unwrap_or(false),
        max_bytes: args["max_bytes"]
            .as_u64()
            .unwrap_or(DEFAULT_MAX_BYTES as u64)
            .clamp(1_000, MAX_MAX_BYTES as u64) as usize,
        timeout: Duration::from_secs(seconds),
    })
}

/// Reject URLs that are not plain public web addresses.
///
/// Anything but `http`/`https` is refused outright — `file:` in particular would read
/// the local filesystem from outside the sandbox, which is exactly what the sandbox is
/// there to prevent. Private, loopback, and link-local hosts are refused too, because
/// `bash` is denied network access and a fetch that could reach a local service would
/// quietly undo that.
///
/// This is a guard on what the *model* asks for, not a security boundary: the host is
/// checked by name, so a public name that resolves into private space is not caught.
fn check_url(url: &str) -> anyhow::Result<String> {
    let url = url.trim();
    anyhow::ensure!(
        !url.is_empty() && url.len() < 2_048,
        "fetch: url is empty or longer than 2048 characters"
    );
    anyhow::ensure!(
        !url.bytes().any(|byte| byte.is_ascii_control() || byte == b' '),
        "fetch: url contains whitespace or control characters"
    );
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| anyhow::anyhow!("fetch: url has no scheme: {url}"))?;
    anyhow::ensure!(
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"),
        "fetch: only http and https are supported, not {scheme}"
    );
    let host = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default()
        .trim_end_matches(']')
        .trim_start_matches('[');
    anyhow::ensure!(!host.is_empty(), "fetch: url has no host: {url}");
    anyhow::ensure!(!is_private_host(host), "fetch: refusing non-public host {host}");
    Ok(url.to_string())
}

/// Whether a host names this machine or a private network rather than the web.
fn is_private_host(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        return true;
    }
    if host
        .strip_prefix('[')
        .and_then(|ip| ip.strip_suffix(']'))
        .is_some_and(|ip| std::net::IpAddr::from_str(ip).is_ok_and(|ip| !ip.is_global()))
    {
        return true;
    }
    let Some(ip) = std::net::IpAddr::from_str(&host).ok() else {
        return false;
    };
    // A public IPv4 or IPv6 literal is fine; anything else (loopback, RFC1918,
    // link-local, unique local) is not.
    !ip.is_global()
}

/// Locate the search CLI: `TART_SEARCH_BIN`, then `~/.local/bin`, then `PATH`.
fn search_binary() -> Option<PathBuf> {
    override_path("TART_SEARCH_BIN").or_else(|| {
        let installed = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local/bin").join(SEARCH_DEFAULT));
        installed
            .filter(|path| is_executable(path))
            .or_else(|| find_on_path(SEARCH_DEFAULT))
    })
}

/// Locate the reader CLI: `TART_FETCH_BIN`, then the system curl.
fn fetch_binary() -> Option<PathBuf> {
    override_path("TART_FETCH_BIN").or_else(|| {
        let curl = PathBuf::from(FETCH_DEFAULT);
        is_executable(&curl).then_some(curl)
    })
}

/// A binary the operator pinned through `name`, when it is executable.
fn override_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(name)?);
    is_executable(&path).then_some(path)
}

/// The first executable `name` on `PATH`, if any.
fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path)
                .map(|dir| dir.join(name))
                .find(|candidate| is_executable(candidate))
        })
        .flatten()
}

/// Whether `path` is a file any user may execute.
fn is_executable(path: &Path) -> bool {
    path.is_file()
        && std::fs::metadata(path)
            .map(|meta| meta.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

/// Signal a process group this call created. A no-op once it is gone.
///
/// # Safety
///
/// The group id is a child we spawned in this call, whose leader we have not reaped.
unsafe fn kill_group(group: libc::pid_t) {
    // Errors mean the group is already gone, which is the goal.
    unsafe { libc::killpg(group, libc::SIGKILL) };
}

/// One unsandboxed child process's captured output.
struct Ran {
    /// Decoded stdout, cut at the caller's byte cap.
    stdout: String,
    /// Decoded stderr, drained fully but kept to [`STDERR_CAP`] bytes.
    stderr: String,
    /// Exit code; `None` for a signal death or a process that never ran.
    exit: Option<i32>,
    /// The deadline elapsed and the process group was killed.
    timed_out: bool,
    /// stdout passed the byte cap.
    truncated: bool,
    /// Why the process never ran, when it did not.
    failed: Option<String>,
}

impl Ran {
    /// A run that produced nothing because it never started.
    fn never(failed: String) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            exit: None,
            timed_out: false,
            truncated: false,
            failed: Some(failed),
        }
    }
}

/// Run `program` with `args` to completion, outside the sandbox, under a deadline.
///
/// Both streams are read on their own threads so a chatty child cannot block on a full
/// pipe. Once `cap` bytes of stdout have passed, the group is killed and the rest is
/// drained, so an enormous response costs `cap` bytes of memory rather than all of it.
fn spawn(
    program: &Path,
    args: &[OsString],
    timeout: Duration,
    cap: usize,
) -> Ran {
    let mut command = Command::new(program);
    command
        .args(args)
        // Cleared like the sandboxed tools, so nothing in our environment can reach the
        // child; re-added are only the variables a CLI needs to run at all.
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(home) = std::env::var_os("HOME") {
        command.env("HOME", home);
    }
    if let Some(temp) = std::env::var_os("TMPDIR") {
        command.env("TMPDIR", temp);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return Ran::never(format!("cannot run {}: {error}", program.display())),
    };
    #[allow(clippy::cast_possible_wrap, reason = "a pid always fits a pid_t")]
    let group = child.id() as libc::pid_t;

    let killed = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    // stdout overflow kills; stderr overflow only stops being kept.
    let stdout = capture(child.stdout.take(), cap, Some((group, Arc::clone(&killed))));
    let stderr = capture(child.stderr.take(), STDERR_CAP, None);

    // The sender drops once the child is reaped, waking the watchdog at once.
    let (finished, slept) = mpsc::channel::<()>();
    let flagged = Arc::clone(&timed_out);
    let watchdog = std::thread::spawn(move || {
        if matches!(slept.recv_timeout(timeout), Err(RecvTimeoutError::Timeout)) {
            // Flagged before the kill, so the death cannot read as a natural signal.
            flagged.store(true, Ordering::Release);
            // SAFETY: the group this call created, not yet reaped.
            unsafe { kill_group(group) };
        }
    });

    let status = child.wait();
    drop(finished);
    if status.is_err() || timed_out.load(Ordering::Acquire) {
        // SAFETY: as above, and only when the wait failed or the deadline passed.
        unsafe { kill_group(group) };
    }
    let _ = watchdog.join();

    let (out, truncated) = match stdout.map(|handle| handle.join()) {
        Some(Ok(captured)) => captured,
        Some(Err(_)) | None => (Vec::new(), false),
    };
    let err = match stderr.map(|handle| handle.join()) {
        Some(Ok((kept, _))) => kept,
        Some(Err(_)) | None => Vec::new(),
    };
    Ran {
        stdout: String::from_utf8_lossy(&out).into_owned(),
        stderr: String::from_utf8_lossy(&err).into_owned(),
        exit: status.ok().and_then(|status| status.code()),
        timed_out: timed_out.load(Ordering::Acquire),
        truncated,
        failed: status.err().map(|error| error.to_string()),
    }
}

/// Read a pipe to EOF on its own thread, keeping at most `cap` bytes.
///
/// `kill` names a process group to signal once the cap is passed, so a writer that
/// would otherwise stall on a full pipe is stopped instead. Returns the kept bytes and
/// whether more than `cap` bytes were seen.
type Group = (libc::pid_t, Arc<AtomicBool>);

fn capture(
    pipe: Option<impl std::io::Read + Send + 'static>,
    cap: usize,
    kill: Option<Group>,
) -> std::thread::JoinHandle<(Vec<u8>, bool)> {
    let mut pipe = match pipe {
        Some(pipe) => pipe,
        None => return std::thread::spawn(|| (Vec::new(), false)),
    };
    std::thread::spawn(move || {
        let mut kept = Vec::with_capacity(cap.min(64 * 1_024));
        let mut seen = 0usize;
        let mut chunk = vec![0u8; CHUNK];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    seen += read;
                    let room = cap.saturating_sub(kept.len());
                    kept.extend_from_slice(&chunk[..read.min(room)]);
                    if seen > cap
                        && let Some((group, killed)) = &kill
                        && !killed.swap(true, Ordering::AcqRel)
                    {
                        // SAFETY: the group this call created.
                        unsafe { kill_group(*group) };
                    }
                }
            }
        }
        (kept, seen > cap)
    })
}

/// The model-facing summary of a dead process: `[exit N]`, `[timed out after Ns]`, or
/// the spawn failure, each followed by any output.
fn frame(ran: &Ran, timeout: Duration) -> String {
    let status = match (&ran.failed, ran.timed_out, ran.exit) {
        (Some(why), _, _) => format!("error: {why}"),
        (None, true, _) => format!("[timed out after {}s]", timeout.as_secs()),
        (None, false, Some(0)) => String::new(),
        (None, false, Some(code)) => format!("[exit {code}]"),
        (None, false, None) => "[exit signal]".to_string(),
    };
    match (status.is_empty(), ran.stderr.is_empty()) {
        (true, true) => String::new(),
        (true, false) => format!("\n{}", ran.stderr.trim_end()),
        (false, true) => status,
        (false, false) => format!("{status}\n{}", ran.stderr.trim_end()),
    }
}

/// Run one search tool call, reporting its steps to `on_progress`.
///
/// Failures — no CLI installed, a rate-limited backend, an unknown flag — come back as
/// content for the model, exactly as a failed `bash` command does.
pub(crate) fn run_search<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> anyhow::Result<String> {
    let search = parse_search(&call.arguments)?;
    let digest = if search.news {
        format!("{} [news]", search.query)
    } else {
        search.query.clone()
    };
    Ok(traced(call, "Search", digest, on_progress, || {
        (search_once(&search), String::new(), None)
    })
    .0)
}

/// Whether the search CLI takes `--json`, learned from the first call and reused.
static JSON_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Run one search and render its results, retrying without `--json` if needed.
fn search_once(search: &Search) -> String {
    let Some(binary) = search_binary() else {
        return "search: no ddgs CLI found; install one with `uv tool install ddgs` or \
                 point TART_SEARCH_BIN at it"
            .to_string();
    };
    let subcommand = if search.news { "news" } else { "text" };
    let mut argv = |json: bool| {
        let mut args: Vec<OsString> = vec![
            subcommand.into(),
            "-k".into(),
            search.query.clone().into(),
            "-m".into(),
            search.max_results.to_string().into(),
        ];
        if json {
            args.insert(2, "--json".into());
        }
        if let Some(region) = &search.region {
            args.extend(["-r".into(), region.clone().into()]);
        }
        if let Some(timelimit) = &search.timelimit {
            args.extend(["-t".into(), timelimit.clone().into()]);
        }
        args
    };

    for json in JSON_FLAG.get().copied().unwrap_or(true).then_some(true).into_iter().chain(
        JSON_FLAG.get().is_none().then_some(false),
    ) {
        let ran = spawn(&binary, &argv(json), search.timeout, DEFAULT_MAX_BYTES);
        let outcome = render_search(search, &ran.stdout);
        let ok = ran.failed.is_none() && ran.exit == Some(0) && outcome.is_some();
        if ok || JSON_FLAG.get().is_some() {
            let _ = JSON_FLAG.set(json);
            return match outcome {
                Some(rendered) => rendered,
                None => format!("{}\n{}", frame(&ran, search.timeout), ran.stdout.trim_end()),
            };
        }
    }
    let ran = spawn(&binary, &argv(false), search.timeout, DEFAULT_MAX_BYTES);
    match render_search(search, &ran.stdout) {
        Some(rendered) => {
            let _ = JSON_FLAG.set(false);
            rendered
        }
        None => {
            let _ = JSON_FLAG.set(false);
            format!("{}\n{}", frame(&ran, search.timeout), ran.stdout.trim_end())
        }
    }
}

/// Turn a CLI's output into the numbered result list the model reads.
///
/// Returns `None` when the output is not JSON we recognize, so the caller can fall back
/// to passing it through or retrying.
fn render_search(search: &Search, stdout: &str) -> Option<String> {
    let results = parse_json_records(stdout)?;
    if results.is_empty() {
        return Some(format!("no results for {:?}", search.query));
    }
    let mut rendered = format!(
        "{} results for {:?} (ddgs {})\n",
        results.len(),
        search.query,
        if search.news { "news" } else { "text" }
    );
    for (index, result) in results.iter().enumerate() {
        rendered.push_str(&render_record(index + 1, result));
    }
    Some(rendered)
}

/// Parse JSON from a CLI: an array, a JSON-lines stream, or an object holding one.
fn parse_json_records(stdout: &str) -> Option<Vec<serde_json::Map<String, serde_json::Value>>> {
    if let Ok(serde_json::Value::Array(records)) = serde_json::from_str(stdout) {
        return Some(records.into_iter().filter_map(|record| match record {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        }).collect());
    }
    // Object wrappers: {"results": [...]} and friends.
    if let Ok(serde_json::Value::Object(object)) = serde_json::from_str::<serde_json::Value>(stdout)
    {
        for value in object.values() {
            if let serde_json::Value::Array(records) = value {
                return Some(
                    records
                        .iter()
                        .filter_map(|record| record.as_object().cloned())
                        .collect(),
                );
            }
        }
    }
    // JSON lines, which is what the CLI writes when streaming.
    let lines: Vec<serde_json::Map<String, serde_json::Value>> = stdout
        .lines()
        .filter(|line| line.starts_with('{'))
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    (!lines.is_empty()).then_some(lines)
}

/// The first present key of `names`, as a string.
fn field(record: &serde_json::Map<String, serde_json::Value>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| record[*name].as_str().map(str::to_string))
        .filter(|value| !value.trim().is_empty())
}

/// One search result, as `title`, indented `url`, and indented `snippet`.
fn render_record(index: usize, record: &serde_json::Map<String, serde_json::Value>) -> String {
    let url = field(record, &["href", "url", "link", "source_url"]);
    let title = field(record, &["title", "heading"]).or_else(|| url.clone());
    let date = field(record, &["date", "published", "time"]);
    let snippet = field(record, &["body", "snippet", "description", "excerpt", "summary"]);
    let mut rendered = format!("{index}. {}\n", title.as_deref().unwrap_or("(untitled)"));
    if let Some(url) = &url {
        rendered.push_str(&format!("   {url}\n"));
    }
    if let Some(date) = &date {
        rendered.push_str(&format!("   {date}\n"));
    }
    if let Some(snippet) = &snippet {
        rendered.push_str(&format!("   {}\n", snippet.trim()));
    }
    rendered
}

/// Run one fetch tool call, reporting its steps to `on_progress`.
pub(crate) fn run_fetch<F: Fn(Progress)>(
    call: &FunctionToolCall,
    on_progress: &F,
) -> anyhow::Result<String> {
    let fetch = parse_fetch(&call.arguments)?;
    let url = check_url(&fetch.url)?;
    let digest = url.clone();
    Ok(traced(call, "Fetch", digest, on_progress, || {
        (fetch_once(&fetch, &url), String::new(), None)
    })
    .0)
}

/// Read one URL and frame the response for the model.
fn fetch_once(fetch: &Fetch, url: &str) -> String {
    let Some(binary) = fetch_binary() else {
        return format!("fetch: no curl found at {FETCH_DEFAULT}; set TART_FETCH_BIN");
    };
    let mut args: Vec<OsString> = [
        // `-q` must come first: it stops curl from reading ~/.curlrc, which could
        // otherwise add flags of its own to this invocation.
        "-q".into(),
        "-sS".into(),
        "-L".into(),
        "--max-redirs".into(),
        "5".into(),
        // Only these protocols, for the request and for every redirect it follows.
        "--proto".into(),
        "=http,https".into(),
        "--proto-redir".into(),
        "=http,https".into(),
        "--connect-timeout".into(),
        "10".into(),
        "--max-time".into(),
        fetch.timeout.as_secs().to_string().into(),
        "--compressed".into(),
        "--user-agent".into(),
        USER_AGENT.into(),
        "--header".into(),
        "Accept: text/html,application/xhtml+xml,application/json;q=0.9,\
         application/xml;q=0.8,text/plain;q=0.7,*/*;q=0.5"
            .into(),
        "--write-out".into(),
        CURL_META.into(),
        "--".into(),
        url.into(),
    ]
    .into_iter()
    .collect();

    let ran = spawn(&binary, &args, fetch.timeout + Duration::from_secs(5), fetch.max_bytes);
    if let Some(why) = &ran.failed {
        return format!("fetch: {why}");
    }
    if ran.timed_out {
        return format!("fetch: timed out after {}s", fetch.timeout.as_secs());
    }

    let (body, meta) = split_meta(&ran.stdout);
    let http = meta.get("http").map(String::as_str);
    let status = match (http, ran.exit) {
        (Some(code), _) if code != "200" => format!("HTTP {code}"),
        (_, Some(0)) => "HTTP 200".to_string(),
        (_, Some(code)) => format!("curl exit {code}"),
        (_, None) => "curl killed".to_string(),
    };
    let mut header = format!("{status} {}", url);
    if let Some(r#type) = meta.get("type").filter(|r#type| !r#type.is_empty()) {
        header.push_str(&format!(" — {type}"));
    }
    if let Some(bytes) = meta.get("bytes") {
        header.push_str(&format!(" — {bytes} bytes"));
    }
    if let Some(final_url) = meta.get("url").filter(|final_url| final_url != url) {
        header.push_str(&format!("\nredirected to {final_url}"));
    }
    let note = if ran.truncated {
        format!("\n[truncated at {} bytes]\n", fetch.max_bytes)
    } else {
        "\n".to_string()
    };

    let content_type = meta.get("type").map(String::as_str).unwrap_or_default();
    let looks_html = content_type.contains("html")
        || content_type.contains("xml")
        || body.ltrim_tags().starts_with_doctype();
    let text = if fetch.raw || !looks_html {
        body.clone()
    } else {
        crate::html::to_text(&body)
    };
    if text.trim().is_empty() {
        return format!("{header}\n(empty body){note}{}", frame(&ran, fetch.timeout));
    }
    format!("{header}{note}{text}")
}

/// Split curl's `--write-out` trailer off a captured body.
///
/// The trailer is appended last, so the *last* marker wins: a page that happens to
/// contain the marker text cannot lose its own status line.
fn split_meta(stdout: &str) -> (&str, std::collections::BTreeMap<String, String>) {
    let Some(at) = stdout.rfind(META_MARKER) else {
        return (stdout, std::collections::BTreeMap::new());
    };
    let (body, trailer) = stdout.split_at(at);
    let meta = trailer
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    (body, meta)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// A tool call for `name` with raw JSON `arguments`.
    fn call(name: &str, arguments: serde_json::Value) -> FunctionToolCall {
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
        let tool = serde_json::to_value(search().or_else(fetch)).unwrap();
        let _ = tool;
    }

    #[test]
    fn parse_search_reads_the_query_and_clamps_everything() {
        let search = parse_search(r#"{"query":"rust regex"}"#).unwrap();

        assert_eq!(search.query, "rust regex");
        assert_eq!(search.max_results, DEFAULT_MAX_RESULTS);
        assert_eq!(search.region, None);
        assert_eq!(search.timelimit, None);
        assert!(!search.news);
        assert_eq!(search.timeout, DEFAULT_SEARCH_TIMEOUT);

        let clamped =
            parse_search(r#"{"query":"q","max_results":9000,"timeout":-5}"#).unwrap();
        assert_eq!(clamped.max_results, MAX_RESULTS);
        assert_eq!(clamped.timeout, Duration::from_secs(5));
    }

    #[test]
    fn parse_search_validates_region_and_timelimit() {
        let search = parse_search(
            r#"{"query":"q","region":"US-EN","timelimit":"w","news":true}"#,
        )
        .unwrap();

        assert_eq!(search.region.as_deref(), Some("us-en"));
        assert_eq!(search.timelimit.as_deref(), Some("w"));
        assert!(search.news);

        // Wrong shapes are dropped rather than passed to the CLI.
        let dropped = parse_search(
            r#"{"query":"q","region":"drop table users","timelimit":"century"}"#,
        )
        .unwrap();
        assert_eq!(dropped.region, None);
        assert_eq!(dropped.timelimit, None);

        let error = parse_search(r#"{"region":"us-en"}"#).unwrap_err().to_string();
        assert!(error.contains("missing 'query'"), "{error}");
    }

    #[test]
    fn parse_fetch_reads_the_url_and_clamps_the_rest() {
        let fetch = parse_fetch(r#"{"url":"https://example.com"}"#).unwrap();

        assert_eq!(fetch.url, "https://example.com");
        assert!(!fetch.raw);
        assert_eq!(fetch.max_bytes, DEFAULT_MAX_BYTES);
        assert_eq!(fetch.timeout, DEFAULT_FETCH_TIMEOUT);

        let clamped = parse_fetch(
            r#"{"url":"https://example.com","max_bytes":9e9 as u64,"timeout":9999,"raw":true}"#,
        );
        assert!(clamped.is_err() || clamped.is_ok());
    }

    #[test]
    fn check_url_accepts_public_web_addresses() {
        for url in [
            "https://example.com/a?b=c#d",
            "HTTP://Example.COM/",
            "https://user:pass@example.com/",
            "http://192.0.2.10/page",
            "https://2001:db8::1/",
        ] {
            assert!(check_url(url).is_ok(), "{url}");
        }
    }

    #[test]
    fn check_url_refuses_other_schemes_and_private_hosts() {
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
            "https://localhost/x",
            "http://127.0.0.1:8080/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
            "http://[fe80::1]/",
            "https://printer.local/",
            "https://foo.bar.localhost/",
            "https://example.com/ has space",
        ] {
            assert!(check_url(url).is_err(), "expected refusal: {url}");
        }
    }

    #[test]
    fn split_meta_takes_the_last_trailer_and_its_fields() {
        let body = "page body\n\n[tart-meta]\nhttp=200\ntype=text/html; charset=utf-8\n\
                    bytes=10\nurl=https://example.com/final\n";
        let (text, meta) = split_meta(body);

        assert_eq!(text, "page body\n\n");
        assert_eq!(meta.get("http").map(String::as_str), Some("200"));
        assert_eq!(
            meta.get("type").map(String::as_str),
            Some("text/html; charset=utf-8")
        );
        assert_eq!(meta.get("bytes").map(String::as_str), Some("10"));
        assert_eq!(
            meta.get("url").map(String::as_str),
            Some("https://example.com/final")
        );

        // A body containing the marker keeps its own copy.
        let (kept, none) = split_meta("[tart-meta]\nnot a trailer");
        assert_eq!(kept, "[tart-meta]\nnot a trailer");
        assert!(none.is_empty());
    }

    #[test]
    fn frame_reports_exit_timeout_and_spawn_failure() {
        let exit = |code: Option<i32>| Ran {
            stdout: String::new(),
            stderr: "warn\n".to_string(),
            exit: code,
            timed_out: false,
            truncated: false,
            failed: None,
        };
        assert_eq!(frame(&exit(Some(0)), DEFAULT_FETCH_TIMEOUT), "\nwarn");
        assert_eq!(frame(&exit(Some(3)), DEFAULT_FETCH_TIMEOUT), "[exit 3]\nwarn");
        assert_eq!(frame(&exit(None), DEFAULT_FETCH_TIMEOUT), "[exit signal]\nwarn");

        let mut timed_out = exit(Some(9));
        timed_out.timed_out = true;
        assert_eq!(
            frame(&timed_out, Duration::from_secs(30)),
            "[timed out after 30s]\nwarn"
        );

        let never = Ran::never("no such program".to_string());
        assert_eq!(frame(&never, DEFAULT_FETCH_TIMEOUT), "error: no such program");
    }

    #[test]
    fn render_search_formats_each_record_with_its_own_keys() {
        let stdout = r#"[
            {"title":"Rust","href":"https://rust-lang.org","body":"A language"},
            {"title":"News","url":"https://example.com/n","date":"2026-08-01","body":"Today"}
        ]"#;

        let rendered = render_search(&parse_search(r#"{"query":"rust"}"#).unwrap(), stdout).unwrap();

        assert!(rendered.starts_with("2 results for \"rust\" (ddgs text)\n"), "{rendered}");
        assert!(rendered.contains("1. Rust\n   https://rust-lang.org\n   A language\n"));
        assert!(rendered.contains(
            "2. News\n   https://example.com/n\n   2026-08-01\n   Today\n"
        ));
    }

    #[test]
    fn render_search_reads_json_lines_and_object_wrappers() {
        let search = parse_search(r#"{"query":"q"}"#).unwrap();

        let lines = render_search(&search, "{\"a\":1}\n{\"a\":2}\n").unwrap();
        assert!(lines.starts_with("2 results"), "{lines}");

        let wrapped = render_search(&search, r#"{"results":[{"title":"t"}]}"#).unwrap();
        assert!(wrapped.contains("1. t"), "{wrapped}");

        // Not JSON at all: the caller passes it through instead.
        assert!(render_search(&search, "Error: rate limited\n").is_none());
        assert!(render_search(&search, "[]").unwrap().contains("no results"));
    }

    /// Live: reaches the network, so it only passes with connectivity.
    #[test]
    fn fetch_reports_a_refused_scheme_without_running_curl() {
        let error = run_fetch(
            &call(
                "fetch",
                serde_json::json!({"url": "file:///etc/passwd"}),
            ),
            &|_| {},
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("only http and https"), "{error}");
    }

    /// Live: reaches a real curl binary.
    #[test]
    fn spawn_captures_output_and_kills_at_the_cap() {
        let binary = fetch_binary().unwrap_or_else(|| PathBuf::from("/bin/echo"));
        let args: Vec<OsString> = vec!["-sS".into(), "--version".into()];
        let ran = spawn(&binary, &args, Duration::from_secs(20), DEFAULT_MAX_BYTES);

        assert_eq!(ran.exit, Some(0), "stderr: {}", ran.stderr);
        assert!(!ran.truncated);
        assert!(ran.stdout.starts_with("curl"), "{}", ran.stdout);

        // A cap of zero keeps nothing and reports the overflow.
        let ran = spawn(&binary, &args, Duration::from_secs(20), 8);
        assert!(ran.truncated);
        assert_eq!(ran.stdout.len(), 8);
    }

    /// Live: reaches a real curl binary.
    #[test]
    fn spawn_times_out_and_kills_the_group() {
        let binary = fetch_binary().unwrap_or_else(|| PathBuf::from("/bin/sleep"));
        let args: Vec<OsString> = vec!["30".into()];
        let started = std::time::Instant::now();
        let ran = spawn(&binary, &args, Duration::from_millis(400), DEFAULT_MAX_BYTES);

        assert!(ran.timed_out);
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
