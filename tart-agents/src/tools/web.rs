//! The web tools: `search` and `fetch`.
//!
//! These are the only tools that run **outside** the seatbelt sandbox.
//! Processing stays out of Rust where it can: ddgs writes structured JSON we
//! only render, and the r.jina.ai reader turns a page into markdown before
//! curl hands it back, so no HTML is ever parsed here.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::net::{IpAddr, ToSocketAddrs};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::backends::{FunctionToolCall, Tool};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;

use super::{
    CancelToken, WatchedRun, combined_output, command_text_inline, head_cap, misuse,
    parse_arguments, run_watched, string_field, timeout_text, tool, traced,
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

/// The poppler text extractor looked for on `PATH`.
const PDF_DEFAULT_EXTRACTOR: &str = "pdftotext";

/// The most pages one PDF fetch may extract.
const PDF_MAX_PAGES: usize = 100;

/// The size cap on one fetched PDF.
const PDF_MAX_BYTES: u64 = 20 * 1024 * 1024; // 20 MiB

/// The longest PDF text handed to the model inline, in bytes.
const PDF_TEXT_CAP: usize = 150_000;

/// The timeout the poppler tools run under; a text layer should not need more.
const PDF_EXTRACT_TIMEOUT: Duration = Duration::from_secs(30);

/// The most pages the OCR fallback may chew: it renders and then OCRs each
/// page, seconds apiece, so scanned documents are bounded harder than text.
const PDF_OCR_PAGES: usize = 10;

/// The rendering resolution the OCR fallback feeds ocrs, in dots per inch.
const PDF_OCR_DPI: &str = "150";

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
        has network access but cannot read or write files; refuses non-public hosts. \
        PDF addresses (a .pdf path, or an arxiv /pdf/ paper) are fetched directly and \
        their text layer extracted locally as markdown with <!-- Page N --> markers; when \
        the layer is empty (scanned images) the pages fall back to local OCR: poppler's \
        pdftoppm renders them and ocrs reads them, when its models are cached",
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
        url: string_field(&args, "fetch", "url")?,
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

/// Locate the PDF text extractor: the `TART_PDFTOTEXT_BIN` override, then `PATH`.
fn pdftotext_binary() -> Option<PathBuf> {
    let pinned = std::env::var_os("TART_PDFTOTEXT_BIN").map(PathBuf::from);
    pinned
        .into_iter()
        .find(|path| is_executable(path))
        .or_else(|| find_on_path(PDF_DEFAULT_EXTRACTOR))
}

/// Locate pdfinfo for page counts and metadata.
fn pdfinfo_binary() -> Option<PathBuf> {
    let pinned = std::env::var_os("TART_PDFINFO_BIN").map(PathBuf::from);
    pinned
        .into_iter()
        .find(|path| is_executable(path))
        .or_else(|| find_on_path("pdfinfo"))
}

/// Locate the page renderer poppler ships, overridable with
/// `TART_PDFTOPPM_BIN`: the OCR fallback's input.
fn pdftoppm_binary() -> Option<PathBuf> {
    let pinned = std::env::var_os("TART_PDFTOPPM_BIN").map(PathBuf::from);
    pinned
        .into_iter()
        .find(|path| is_executable(path))
        .or_else(|| find_on_path("pdftoppm"))
}

/// The OCR engine ocrs provides, with its models from `TART_OCR_MODELS` or ocrs cache.
fn ocr_engine() -> Option<OcrEngine> {
    let dir = match std::env::var_os("TART_OCR_MODELS") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(std::env::var_os("HOME")?).join(".cache/ocrs"),
    };
    let model = |stem: &str| -> Option<Model> {
        ["onnx", "rten"]
            .iter()
            .find_map(|ext| Model::load_file(dir.join(format!("{stem}.{ext}"))).ok())
    };
    let detection = model("text-detection")?;
    let recognition = model("text-recognition")?;
    OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection),
        recognition_model: Some(recognition),
        ..OcrEngineParams::default()
    })
    .ok()
}

/// One rendered page as raw RGB plus its size, from the PPM (P6) pixels pdftoppm writes
fn read_ppm(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut rest = bytes.strip_prefix(b"P6")?;
    let mut header = [0u32; 3];
    for field in &mut header {
        loop {
            while rest.first().is_some_and(u8::is_ascii_whitespace) {
                rest = &rest[1..];
            }
            if rest.first() == Some(&b'#') {
                let end = rest.iter().position(|byte| *byte == b'\n')?;
                rest = &rest[end + 1..];
            } else {
                break;
            }
        }
        let end = rest.iter().position(|byte| !byte.is_ascii_digit())?;
        *field = std::str::from_utf8(&rest[..end]).ok()?.parse().ok()?;
        rest = &rest[end..];
    }
    let [width, height, maxval] = header;
    // Exactly one whitespace separates the header from the pixel run.
    rest = rest.strip_prefix(b"\n").or_else(|| rest.strip_prefix(b" "))?;
    let size = width as usize * height as usize * 3;
    (maxval == 255 && rest.len() == size).then(|| (rest.to_vec(), width, height))
}

/// Read one rendered page's text with the engine, lines joined by newlines.
fn ocr_page(engine: &OcrEngine, page: &Path) -> Option<String> {
    let (rgb, width, height) = read_ppm(&std::fs::read(page).ok()?)?;
    let source = ImageSource::from_bytes(&rgb, (width, height)).ok()?;
    let input = engine.prepare_input(source).ok()?;
    let words = engine.detect_words(&input).ok()?;
    let lines = engine.find_text_lines(&input, &words);
    let text = engine.recognize_text(&input, &lines).ok()?;
    let joined = text
        .iter()
        .flatten()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    (!joined.is_empty()).then_some(joined)
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

/// A command for one of the web binaries, with a clean working environment.
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

/// A fresh path under the system temp directory for one PDF fetch's files.
fn pdf_path(extension: &str) -> PathBuf {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tart-pdf-{}-{call}.{extension}", std::process::id()))
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
pub(super) fn run_search<F: Fn(Progress)>(call: &FunctionToolCall, on_progress: &F) -> String {
    let search = match parse_search(&call.arguments) {
        Ok(search) => search,
        Err(error) => return misuse(call, on_progress, &error),
    };
    traced(call, on_progress, || {
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
                        (command_text_inline(&text, output.status, false), text, exit)
                    }
                }
                Err(error) => {
                    let text = format!("error: {error}");
                    (text.clone(), text, None)
                }
            },
        }
    })
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

/// The curl command that fetches one PDF, writing output to `dest` instead of stdout
fn pdf_curl(binary: PathBuf, dest: &Path, url: &str) -> Command {
    let mut curl = web_command(binary);
    curl.args([
        "-q", // must be separate or we try and fail to read .curlrc
        "-fsSL",
        "--max-redirs",
        "5",
        "--connect-timeout",
        "10",
        "--max-time",
        CURL_MAX_TIME,
        "--compressed",
        "--proto",
        "=http,https",
        "--proto-redir",
        "=http,https",
        "--max-filesize",
    ])
    .arg(PDF_MAX_BYTES.to_string())
    .args(["-A", USER_AGENT, "-w", FINAL_URL, "-o"])
    .arg(dest)
    .arg("--")
    .arg(url);
    curl
}

/// The pdftotext command: keep the layout, bound the pages, write to stdout.
fn pdftotext_command(binary: PathBuf, downloaded: &Path, last_page: usize) -> Command {
    let mut pdftotext = web_command(binary);
    pdftotext
        .args(["-layout", "-enc", "UTF-8", "-f", "1", "-l"])
        .arg(last_page.to_string())
        .arg(downloaded)
        .arg("-");
    pdftotext
}

/// The pdftoppm command that renders a PDF's first pages as PNGs under `prefix`
fn pdftoppm_command(binary: PathBuf, downloaded: &Path, prefix: &Path, last: usize) -> Command {
    let mut pdftoppm = web_command(binary);
    pdftoppm
        .args(["-r", PDF_OCR_DPI, "-f", "1", "-l"])
        .arg(last.to_string())
        .arg(downloaded)
        .arg(prefix);
    pdftoppm
}

/// The page images pdftoppm wrote under `prefix`, in page order.
fn rendered_pages(prefix: &Path) -> Vec<PathBuf> {
    let stem = prefix
        .file_name()
        .map_or_else(String::new, |stem| stem.to_string_lossy().into_owned());
    let mut pages: Vec<(usize, PathBuf)> =
        std::fs::read_dir(prefix.parent().unwrap_or_else(|| Path::new(".")))
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension().is_some_and(|ext| ext == "ppm")
                    && path
                        .file_name()
                        .is_some_and(|name| name.to_string_lossy().starts_with(&stem))
            })
            .filter_map(|path| {
                let number = path.file_stem()?.to_string_lossy();
                let number = number.rsplit_once('-')?.1.parse::<usize>().ok()?;
                Some((number, path))
            })
            .collect();
    pages.sort_by_key(|(number, _)| *number);
    pages.into_iter().map(|(_, path)| path).collect()
}

/// OCR a PDF whose text layer is empty: render the first pages with poppler
/// and read them with ocrs, one page per run, joined by form feeds so
/// [`mark_pages`] numbers OCR'd pages as it numbers extracted ones.
///
/// `None` when a piece is missing, a step fails, or no text emerges: the
/// caller then reports the document as unreadable rather than guessing.
fn pdf_ocr(downloaded: &Path) -> Option<(String, usize)> {
    let renderer = pdftoppm_binary()?;
    let engine = ocr_engine()?;
    let prefix = pdf_path("page");
    let mut pdftoppm = pdftoppm_command(renderer, downloaded, &prefix, PDF_OCR_PAGES);
    let rendered = run_watched(&mut pdftoppm, Some(PDF_EXTRACT_TIMEOUT), &CancelToken::new())
        .is_ok_and(|run| run.output.status.success());
    if !rendered {
        return None;
    }
    let mut text = Vec::new();
    for page in rendered_pages(&prefix) {
        // The page is ours once read; a page OCR cannot read stays empty, so
        // the numbering around it survives.
        let read = ocr_page(&engine, &page);
        let _ = std::fs::remove_file(&page);
        text.push(read.unwrap_or_default());
    }
    let (body, pages) = mark_pages(&text.join("\x0c"));
    (!body.is_empty()).then_some((body, pages))
}

/// Run one fetch tool call, reporting its steps to `on_progress`.
///
/// Like `search` this runs outside the sandbox.
pub(super) fn run_fetch<F: Fn(Progress)>(call: &FunctionToolCall, on_progress: &F) -> String {
    let fetch = match parse_fetch(&call.arguments) {
        Ok(fetch) => fetch,
        Err(error) => return misuse(call, on_progress, &error),
    };
    traced(call, on_progress, || {
        let Some(binary) = fetch_binary() else {
            let text = format!("fetch: no curl found at {FETCH_DEFAULT}; set TART_FETCH_BIN");
            return (text.clone(), text, None);
        };
        // The guard is content visible to the model, not an error.
        let url = match check_url(&fetch.url) {
            Ok(url) => url,
            Err(text) => return (text.clone(), text, None),
        };
        // A PDF address bypasses the reader service.
        if is_pdf_url(&url) {
            return pdf_fetch(binary, &url);
        }
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
                    return (command_text_inline(&text, output.status, false), text, exit);
                }
                // Success: split and check where redirects landed.
                if let Some((body, final_url)) = separate_final_url(&text) {
                    if let Some(refused) = private_redirect(final_url) {
                        (refused.clone(), refused, exit)
                    } else {
                        let text = body.to_string();
                        (command_text_inline(&text, output.status, false), text, exit)
                    }
                } else {
                    (command_text_inline(&text, output.status, false), text, exit)
                }
            }
        }
    })
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

/// Whether a URL names a PDF: a `.pdf` path, or an arxiv paper address.
fn is_pdf_url(url: &str) -> bool {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    path.to_ascii_lowercase().ends_with(".pdf") || arxiv_paper_id(path).is_some()
}

/// The arxiv paper id a URL path names, as the `/pdf/<id>` pair in it: an
/// id starts with a digit and contains a dot, so `/pdf/manual` stays a page.
fn arxiv_paper_id(path: &str) -> Option<&str> {
    let mut previous = "";
    for segment in path.split('/') {
        if previous == "pdf"
            && segment.starts_with(|c: char| c.is_ascii_digit())
            && segment.contains('.')
        {
            return Some(segment);
        }
        previous = segment;
    }
    None
}

/// A readable title for a PDF address.
fn pdf_title(url: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    if let Some(id) = arxiv_paper_id(path) {
        return format!("arxiv-{id}");
    }
    let name = path.rsplit('/').next().unwrap_or_default();
    let stem = name
        .strip_suffix(".pdf")
        .or_else(|| name.strip_suffix(".PDF"))
        .unwrap_or(name);
    let spaced = stem.replace(['_', '-'], " ");
    let trimmed = spaced.trim();
    if trimmed.is_empty() {
        "document".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Turn pdftotext's form-feed separated pages into marker-delimited text.
///
/// Returns the marked text and the page count it covers.
fn mark_pages(text: &str) -> (String, usize) {
    let trimmed = text.trim_matches('\x0c');
    if trimmed.is_empty() {
        return (String::new(), 0);
    }
    let mut pages = trimmed.split('\x0c');
    let mut rendered = format!("<!-- Page 1 -->\n{}", pages.next().unwrap_or_default().trim_end());
    let mut count = 1;
    for page in pages {
        count += 1;
        let _ = write!(rendered, "\n\n<!-- Page {count} -->\n\n");
        rendered.push_str(page.trim_end());
    }
    (rendered, count)
}

/// One `Key: value` field of pdfinfo's report, when present and non-empty.
fn pdfinfo_field(report: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    report.lines().find_map(|line| {
        let value = line.strip_prefix(&prefix)?.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// pdfinfo's report on `downloaded`, or nothing when it is not installed.
fn pdf_report(downloaded: &Path) -> String {
    let Some(binary) = pdfinfo_binary() else {
        return String::new();
    };
    let mut pdfinfo = web_command(binary);
    pdfinfo.arg(downloaded);
    run_watched(&mut pdfinfo, Some(PDF_EXTRACT_TIMEOUT), &CancelToken::new())
        .ok()
        .map(|run| combined_output(&run.output))
        .unwrap_or_default()
}

/// Extract one downloaded PDF's text layer and wrap it as markdown.
///
/// The document is titled from pdfinfo's metadata or the address, its pages
/// marked `<!-- Page N -->`, and the extraction bounded to [`PDF_MAX_PAGES`]
/// with the truncation said out loud (after pi-web-access's wrapper).
fn pdf_document(extractor: &Path, downloaded: &Path, url: &str) -> Result<String, String> {
    let mut pdftotext = pdftotext_command(extractor.to_path_buf(), downloaded, PDF_MAX_PAGES);
    let run = match run_watched(&mut pdftotext, Some(PDF_EXTRACT_TIMEOUT), &CancelToken::new()) {
        Ok(run) => run,
        Err(error) => return Err(format!("error: {error}")),
    };
    if !run.output.status.success() {
        return Err(combined_output(&run.output));
    }
    let text = combined_output(&run.output);
    let (body, extracted) = mark_pages(&text);
    // An empty layer means scanned images: fall back to local OCR before
    // giving up, so a scanned paper reads like any other.
    let (body, extracted, ocr) = if body.is_empty() {
        match pdf_ocr(downloaded) {
            Some((body, extracted)) => (body, extracted, true),
            None => {
                return Err("no text layer: the PDF is scanned images, and ocrs's models \
                     are missing: `cargo install ocrs-cli` and run it once to fetch them"
                    .to_string());
            }
        }
    } else {
        (body, extracted, false)
    };
    let report = pdf_report(downloaded);
    let title = pdfinfo_field(&report, "Title").unwrap_or_else(|| pdf_title(url));
    let pages = pdfinfo_field(&report, "Pages")
        .and_then(|pages| pages.parse::<usize>().ok())
        .filter(|pages| *pages > 0)
        .unwrap_or(extracted);
    let bound = if ocr { PDF_OCR_PAGES } else { PDF_MAX_PAGES };
    let truncated = pages > bound;
    let mut document = format!("# {title}\n\n> Source: {url}\n> Pages: {pages}");
    if truncated {
        let _ = write!(document, " (extracted first {bound})");
    }
    if ocr {
        document.push_str("\n> OCR: ocrs");
    }
    if let Some(author) = pdfinfo_field(&report, "Author") {
        let _ = write!(document, "\n> Author: {author}");
    }
    document.push_str("\n\n---\n\n");
    document.push_str(&body);
    if truncated {
        let _ = write!(
            document,
            "\n\n---\n\n*[Truncated: only first {bound} of {pages} pages extracted]*"
        );
    }
    // The full document stays on disk for re-reading; only the inline copy is capped.
    let saved = pdf_path("md");
    if std::fs::write(&saved, &document).is_ok() {
        let _ = write!(document, "\n\nSaved to: {}", saved.display());
    }
    // Move the inline copy: under `head_cap`, we would only have cloned the full doc
    Ok(if document.len() > PDF_TEXT_CAP {
        head_cap(&document, PDF_TEXT_CAP)
    } else {
        document
    })
}

/// The downloaded file's first kilobyte and its full size, for the magic check.
fn pdf_head(path: &Path) -> std::io::Result<(Vec<u8>, u64)> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut head = Vec::with_capacity(1024);
    file.take(1024).read_to_end(&mut head)?;
    Ok((head, size))
}

/// Fetch a PDF address and return its text layer as markdown.
///
/// The bytes go to a temp file under curl's raw-mode protections, are
/// verified as a PDF before anything parses them, and are extracted by the
/// locally installed poppler tools.
fn pdf_fetch(curl_binary: PathBuf, url: &str) -> (String, String, Option<i32>) {
    let Some(extractor) = pdftotext_binary() else {
        let text = "fetch: no pdftotext found; install poppler with `brew install poppler` \
                    or point TART_PDFTOTEXT_BIN at it"
            .to_string();
        return (text.clone(), text, None);
    };
    let downloaded = pdf_path("pdf");
    let mut curl = pdf_curl(curl_binary, &downloaded, url);
    let run = match run_watched(&mut curl, Some(FETCH_TIMEOUT), &CancelToken::new()) {
        Ok(run) => run,
        Err(error) => {
            let _ = std::fs::remove_file(&downloaded);
            let text = format!("error: {error}");
            return (text.clone(), text, None);
        }
    };
    let WatchedRun { output, killed } = run;
    if killed.is_some() || !output.status.success() {
        let _ = std::fs::remove_file(&downloaded);
        let text = combined_output(&output);
        let marked = if killed.is_some() {
            timeout_text(&text, FETCH_TIMEOUT)
        } else {
            command_text_inline(&text, output.status, false)
        };
        let exit = output.status.code();
        return (marked.clone(), marked, exit);
    }
    // Redirects: under `-o` the trailer is all curl wrote to stdout.
    let captured = combined_output(&output);
    if let Some((_, final_url)) = separate_final_url(&captured)
        && let Some(refused) = private_redirect(final_url)
    {
        let _ = std::fs::remove_file(&downloaded);
        let exit = output.status.code();
        return (refused.clone(), refused, exit);
    }
    // Magic bytes within the first kilobyte: never hand poppler anything else.
    let verdict = match pdf_head(&downloaded) {
        Ok((head, _)) if head.windows(5).any(|w| w == b"%PDF-") => Ok(()),
        Ok((_, size)) => Err(format!(
            "fetch: {url} served {size} bytes that are not a PDF (no %PDF- header)"
        )),
        Err(error) => Err(format!("error: {error}")),
    };
    if let Err(text) = verdict {
        let _ = std::fs::remove_file(&downloaded);
        return (text.clone(), text, Some(0));
    }
    let document = pdf_document(&extractor, &downloaded, url);
    let _ = std::fs::remove_file(&downloaded);
    match document {
        Ok(text) => (text.clone(), text, Some(0)),
        Err(error) => {
            let text = format!("fetch: PDF extraction failed: {error}");
            (text.clone(), text, None)
        }
    }
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

    /// A minimal two-page PDF with a text layer: page one a heading over a paragraph,
    /// page two a lone heading, and a correct xref table throughout.
    fn fixture_pdf() -> Vec<u8> {
        let contents = [
            concat!(
                "BT /F1 14 Tf 72 720 Td (Tart PDF Fetch Test) Tj ET\n",
                "BT /F1 11 Tf 14 TL 72 696 Td (The text layer is extracted by poppler ",
                "with no) Tj T* (network reader and no OCR involved at all.) Tj ET\n",
            ),
            "BT /F1 12 Tf 72 720 Td (Second Page Heading) Tj ET\n",
        ];
        let mut objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            Vec::new(), // The page tree, filled once every id is known.
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        let mut content_ids = Vec::new();
        for content in contents {
            content_ids.push(objects.len() + 1);
            let body = format!("<< /Length {} >>\nstream\n{content}endstream", content.len());
            objects.push(body.into_bytes());
        }
        let mut kids = Vec::new();
        for content_id in content_ids {
            kids.push(format!("{} 0 R", objects.len() + 1));
            let page = format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            );
            objects.push(page.into_bytes());
        }
        objects[1] = format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            kids.join(" "),
            kids.len()
        )
        .into_bytes();
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
            pdf.extend_from_slice(object);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref = pdf.len();
        pdf.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for offset in offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    /// A tool call for `name` with raw JSON `arguments`.
    fn call(name: &str, arguments: &str) -> FunctionToolCall {
        FunctionToolCall {
            namespace: None,
            name: name.to_string(),
            arguments: arguments.to_string(),
            call_id: "call_0".to_string(),
            id: Some("item_0".to_string()),
            status: None,
            caller: None,
            r#async: None,
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

    #[test]
    fn fetch_definition_requires_url() {
        let tool = serde_json::to_value(fetch_definition()).unwrap();

        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "fetch");
        assert_eq!(tool["parameters"]["required"][0], "url");
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

        assert!(
            error.contains("The required parameter `url` is missing"),
            "{error}"
        );
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
    fn is_pdf_url_matches_pdf_paths_and_arxiv_papers() {
        for url in [
            "https://example.com/paper.pdf",
            "https://example.com/PAPER.PDF?download=1",
            "https://arxiv.org/pdf/1234.01234",
            "https://arxiv.org/pdf/1234.01234v2#page=2",
        ] {
            assert!(is_pdf_url(url), "expected a PDF address: {url}");
        }
        for url in [
            "https://example.com/paper.pdfx",
            "https://example.com/a.pdf/b",
            "https://example.com/pdf/manual",
            "https://arxiv.org/abs/1234.01234",
            "https://example.com/",
        ] {
            assert!(!is_pdf_url(url), "expected a page address: {url}");
        }
    }

    #[test]
    fn pdf_title_derives_names_from_paths_and_arxiv_ids() {
        assert_eq!(
            pdf_title("https://example.com/some-report_v2.pdf?download=1"),
            "some report v2"
        );
        assert_eq!(
            pdf_title("https://arxiv.org/pdf/1234.01234v2"),
            "arxiv-1234.01234v2"
        );
        assert_eq!(pdf_title("https://example.com/My_PDF.PDF"), "My PDF");
        assert_eq!(pdf_title("https://example.com/"), "document");
    }

    #[test]
    fn mark_pages_numbers_empty_pages_and_drops_trailing_feeds() {
        let (rendered, pages) = mark_pages("first\x0c\x0cthird\x0c");

        assert_eq!(pages, 3);
        assert!(rendered.contains("<!-- Page 1 -->\nfirst"));
        assert!(rendered.contains("<!-- Page 2 -->"));
        assert!(rendered.contains("<!-- Page 3 -->\n\nthird"));

        assert_eq!(mark_pages(""), (String::new(), 0));
        assert_eq!(mark_pages("\x0c\x0c"), (String::new(), 0));
    }

    #[test]
    fn pdfinfo_field_reads_trimmed_nonempty_values() {
        let report = "Title:          A Paper\nPages:          2\nAuthor:   \
                      \nProducer: poppler";

        assert_eq!(pdfinfo_field(report, "Title").as_deref(), Some("A Paper"));
        assert_eq!(pdfinfo_field(report, "Pages").as_deref(), Some("2"));
        assert_eq!(pdfinfo_field(report, "Author"), None);
        assert_eq!(pdfinfo_field(report, "Missing"), None);
    }

    #[test]
    fn pdf_curl_fetches_to_the_file_under_the_pdf_cap() {
        let curl = pdf_curl(
            PathBuf::from("/usr/bin/curl"),
            Path::new("/tmp/tart-pdf.pdf"),
            "https://example.com/paper.pdf",
        );

        assert_eq!(curl.get_program(), "/usr/bin/curl");
        assert_eq!(
            curl.get_args().collect::<Vec<_>>(),
            [
                "-q",
                "-fsSL",
                "--max-redirs",
                "5",
                "--connect-timeout",
                "10",
                "--max-time",
                "40",
                "--compressed",
                "--proto",
                "=http,https",
                "--proto-redir",
                "=http,https",
                "--max-filesize",
                "20971520",
                "-A",
                USER_AGENT,
                "-w",
                FINAL_URL,
                "-o",
                "/tmp/tart-pdf.pdf",
                "--",
                "https://example.com/paper.pdf",
            ]
        );
    }

    #[test]
    fn pdftotext_command_keeps_layout_bounds_pages_writes_to_stdout() {
        let pdftotext = pdftotext_command(
            PathBuf::from("/opt/homebrew/bin/pdftotext"),
            Path::new("/tmp/tart-pdf.pdf"),
            PDF_MAX_PAGES,
        );

        assert_eq!(pdftotext.get_program(), "/opt/homebrew/bin/pdftotext");
        assert_eq!(
            pdftotext.get_args().collect::<Vec<_>>(),
            [
                "-layout",
                "-enc",
                "UTF-8",
                "-f",
                "1",
                "-l",
                "100",
                "/tmp/tart-pdf.pdf",
                "-"
            ]
        );
    }

    #[test]
    fn pdftoppm_command_renders_pngs_for_the_ocr_fallback() {
        let pdftoppm = pdftoppm_command(
            PathBuf::from("/opt/homebrew/bin/pdftoppm"),
            Path::new("/tmp/tart-pdf.pdf"),
            Path::new("/tmp/tart-pages"),
            PDF_OCR_PAGES,
        );

        assert_eq!(pdftoppm.get_program(), "/opt/homebrew/bin/pdftoppm");
        assert_eq!(
            pdftoppm.get_args().collect::<Vec<_>>(),
            [
                "-r",
                "150",
                "-f",
                "1",
                "-l",
                "10",
                "/tmp/tart-pdf.pdf",
                "/tmp/tart-pages"
            ]
        );
    }

    #[test]
    fn rendered_pages_orders_by_number_not_name() {
        let dir = std::env::temp_dir().join(format!("tart-pages-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "prefix-2.ppm",
            "prefix-10.ppm",
            "prefix-1.ppm",
            "other-1.ppm",
            "prefix-1.txt",
        ] {
            std::fs::write(dir.join(name), b"").unwrap();
        }

        let pages = rendered_pages(&dir.join("prefix"));

        let names: Vec<String> = pages
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(names, ["prefix-1.ppm", "prefix-2.ppm", "prefix-10.ppm"]);
    }

    #[test]
    fn read_ppm_parses_headers_and_rejects_bad_ones() {
        let ppm = b"P6\n# a comment\n4 2 255\n\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18";

        let (rgb, width, height) = read_ppm(ppm).unwrap();

        assert_eq!((width, height), (4, 2));
        assert_eq!(rgb, &ppm[ppm.len() - 24..]);
        assert!(read_ppm(b"P5\n1 1 255\n\x00").is_none());
        assert!(read_ppm(b"P6\n2 1 255\n\x00").is_none());
        assert!(read_ppm(b"P6\n2 1 65535\n\x00\x00\x00\x00\x00\x00").is_none());
    }

    /// Offline, but real poppler: skips on machines without the tools.
    #[test]
    fn pdf_document_extracts_pages_with_markers() {
        let Some(extractor) = pdftotext_binary() else {
            return;
        };
        let downloaded = pdf_path("pdf");
        std::fs::write(&downloaded, fixture_pdf()).unwrap();

        let document =
            pdf_document(&extractor, &downloaded, "https://example.com/tart-fixture.pdf");

        let _ = std::fs::remove_file(&downloaded);
        let document = document.unwrap();

        assert!(document.contains("# tart fixture\n"), "{document}");
        assert!(
            document.contains("> Source: https://example.com/tart-fixture.pdf\n"),
            "{document}"
        );
        assert!(document.contains("> Pages: 2"), "{document}");
        assert!(
            document.contains("<!-- Page 1 -->\nTart PDF Fetch Test"),
            "{document}"
        );
        assert!(
            document.contains("<!-- Page 2 -->\n\nSecond Page Heading"),
            "{document}"
        );
        assert!(document.contains("Saved to: "), "{document}");
        if let Some((_, saved)) = document.split_once("Saved to: ") {
            let _ = std::fs::remove_file(saved.trim());
        }
    }

    /// A scanned page: a crop of why-rust.png, the real text screenshot from ocrs tests
    #[test]
    fn pdf_document_ocrs_a_scanned_page_when_models_exist() {
        let Some(extractor) = pdftotext_binary() else {
            return;
        };
        if ocr_engine().is_none() {
            return;
        }
        let downloaded = pdf_path("pdf");
        std::fs::write(&downloaded, include_bytes!("../data/scanned.pdf")).unwrap();

        let document = pdf_document(&extractor, &downloaded, "https://example.com/scanned.pdf");

        let _ = std::fs::remove_file(&downloaded);
        let document = document.unwrap();

        assert!(document.contains("> OCR: ocrs"), "{document}");
        assert!(document.contains("> Pages: 1"), "{document}");
        assert!(document.contains("<!-- Page 1 -->"), "{document}");
        // Any one of the drawn words proving the engine really read the
        // raster, tolerant of OCR's spelling of the rest.
        assert!(
            document.contains("Why Rust")
                || document.contains("Performance")
                || document.contains("blazingly"),
            "{document}"
        );
        if let Some((_, saved)) = document.split_once("Saved to: ") {
            let _ = std::fs::remove_file(saved.trim());
        }
    }

    /// Live: reaches the network, so it needs connectivity, plus poppler.
    #[apply(skip_unless_networked!)]
    #[test]
    fn run_fetch_extracts_a_pdf_address() {
        let Some(_) = pdftotext_binary() else {
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
        let url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
        let arguments = format!(r#"{{"url":"{url}"}}"#);
        let request = call("fetch", &arguments);

        let output = execute(&request, &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert!(output.contains("<!-- Page 1 -->"), "{output}");
        assert!(output.contains("Dummy PDF"), "{output}");
        assert!(output.contains("> Pages: 1"), "{output}");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, .. },
                Progress::ToolOutput { exit: Some(0), .. }
            ] if name == "fetch"
        ));
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
        });

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
        });

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
