//! The web tools: `search` and `fetch`.
//!
//! These are the only tools that run **outside** the seatbelt sandbox.
//! `search` shells out to the locally installed ddgs CLI, which writes structured
//! JSON we render. `fetch` uses native http via `reqwest`, which pre-processes HTML to
//! surface LaTeX equations and then extracts the article body via `rs-trafilatura.`
//! PDFs are read via `poppler` and OCR'd using `tesseract` if necessary.

use std::io::{Read as _, Write as _};
use std::iter::successors;
use std::net::{IpAddr, ToSocketAddrs};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::redirect::Policy as Redirects;

use crate::backends::{FunctionToolCall, Tool};

use super::{
    CancelToken, WatchedRun, bounded, combined_output, create_private, misuse, parse_arguments,
    run_watched, string_field, tool, traced,
};
use crate::Progress;

mod preprocessing;

mod search;

pub(super) use search::run_search;
pub(crate) use search::search;

use dom_query::Document;
use preprocessing::math_to_text;

/// The total budget for one fetch: connect, redirects, and body, enforced by
/// the client rather than a watchdog now the transport is in-process.
const FETCH_TIMEOUT: Duration = Duration::from_secs(40);

/// Hops one fetch will follow before giving up.
const MAX_REDIRECTS: usize = 5;

/// Pages one fetch will read from a PDF.
const PDF_MAX_PAGES: usize = 60;

/// Thin (probably scanned) pages one fetch will OCR, at roughly 1.5s each.
const PDF_OCR_PAGES: usize = 20;

/// Fewer alphanumerics than this in a page's text layer marks it scanned.
const PDF_MIN_ALNUM: usize = 40;

/// Bytes of one PDF the model is handed, pages kept whole where they fit.
const PDF_MAX_BYTES: usize = 120_000;

/// Bytes searched for the `%PDF-` header, matching poppler's `headerSearchSize`
const PDF_HEADER_WINDOW: usize = 1024;

/// Bytes one fetch will download: papers and datasheets run well past a
/// couple of MB, so the body bounds itself.
const DOWNLOAD_CAP: usize = 32 * 1024 * 1024;

/// Bytes one extraction will process before refusing. Larger files are very slow.
const EXTRACT_CAP: usize = 5 * 1024 * 1024;

/// Maximum tag-nesting depth before a body is rejected as pathological.
const NESTING_CAP: i64 = 1_000;

/// Maximum colspan/rowspan value before a body is rejected.
const TABLE_SPAN_CAP: usize = 1_000;

/// Bytes of a non-PDF body handed back inline.
const RAW_MAX_BYTES: usize = 2_000_000;

/// The budget for every poppler/tesseract stage of one PDF.
const PDF_TIMEOUT: Duration = Duration::from_secs(150);

/// A browser-ish `User-Agent`.
///
/// The macOS Safari form from the MDN examples: <https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent>
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 \
     (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

/// The fetch tool's definition; the transport is in-process, so it is always offered.
fn fetch_definition() -> Tool {
    tool(
        "fetch",
        "Read one web page as markdown, extracted automatically by the \
        built-in reader (article body, without boilerplate).\
        An empty extraction \
        usually means the page needs JavaScript: retry with raw=true, which fetches \
        the URL directly and suits JSON or plain-text endpoints. PDF urls (papers, \
        product datasheets, scanned documents) are detected and read natively instead.\
        GitHub file, pull, and issue urls are fetched through their plain-text\
        and API views automatically.\
        Runs outside the sandbox, so it has network \
        access but cannot read or write files; refuses non-public hosts",
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The absolute http(s) URL to read"},
                "raw": {
                    "type": "boolean",
                    "description": "Fetch the URL directly with no extraction; default false"
                }
            },
            "required": ["url"]
        }),
    )
}

/// The fetch tool; always offered: the reader runs in-process, no CLI needed.
#[must_use]
pub(crate) fn fetch() -> Tool {
    fetch_definition()
}

/// One parsed fetch tool call.
#[derive(Debug)]
pub(super) struct Fetch {
    /// The URL to read, validated before anything runs.
    pub url: String,
    /// Fetch the URL directly with no local extraction.
    pub raw: bool,
}

/// Extract the fields from a fetch tool call's JSON arguments.
///
/// `raw` is optional and defaults to local extraction.
pub(super) fn parse_fetch(arguments: &str) -> anyhow::Result<Fetch> {
    let args = parse_arguments(arguments)?;
    Ok(Fetch {
        url: string_field(&args, "fetch", "url")?,
        raw: args["raw"].as_bool().unwrap_or(false),
    })
}

/// Locate a CLI the web tools shell out to: its `TART_*_BIN` override,
/// `~/.local/bin/<name>` (where `uv tool install` puts it), then `PATH`.
///
/// Looked up per call, never cached: the override must take effect from the
/// moment it is set, tests included.
fn tool_binary(override_var: &str, name: &str) -> Option<PathBuf> {
    let pinned = std::env::var_os(override_var).map(PathBuf::from);
    let installed =
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(format!(".local/bin/{name}")));
    pinned
        .into_iter()
        .chain(installed)
        .find(|path| is_executable(path))
        .or_else(|| find_on_path(name))
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
fn web_command(binary: &Path) -> Command {
    let mut command = Command::new(binary);
    command.env_clear();
    for name in ["HOME", "TMPDIR", "PATH"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
}

/// A fresh path under the system temp directory for one call's scratch file:
/// a search's ddgs results (ddgs writes them only to a file), a fetched body,
/// or a rendered page image.
fn temp_path(tool: &str, extension: &str) -> PathBuf {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tart-{tool}-{}-{call}.{extension}", std::process::id()))
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

/// The host in a URL's authority: userinfo dropped, port dropped, IPv6 unbracketed.
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
/// resolve stays public: the client shares this resolver, so an unresolvable
/// name fails there with the better error. Resolution blocks this thread for
/// as long as the resolver takes, like the fetch it guards.
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

/// An error with its causes on one line.
fn error_chain(error: &dyn std::error::Error) -> String {
    successors(Some(error), |error| error.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// The client every fetch shares, built on first use: connection pooling and
/// the one background thread come with the first fetch and stay for the
/// process. The build result is cached beside the client, so a failed build
/// reports the same error every time rather than panicking a lock. Nothing
/// here can be const: building spawns a thread and installs the TLS provider.
/// The blocking client hosts its own runtime and panics inside an async one,
/// so fetch only ever runs on plain threads, never the agent's.
static FETCH_CLIENT: LazyLock<Result<Client, String>> = LazyLock::new(|| {
    // Register a cryptography backend, otherwise reqwest rustls-no-provider panics.
    let _ = rustls::crypto::ring::default_provider().install_default();
    Client::builder()
        .user_agent(USER_AGENT)
        .redirect(redirect_policy())
        .connect_timeout(Duration::from_secs(10))
        .timeout(FETCH_TIMEOUT)
        .no_proxy()
        .build()
        .map_err(|error| format!("error: {}", error_chain(&error)))
});

/// The redirect rule: at most [`MAX_REDIRECTS`] hops, and none into a host
/// this machine or a private network owns, checked on every hop.
fn redirect_policy() -> Redirects {
    Redirects::custom(|attempt| {
        if let Some(refusal) = private_hop(attempt.url()) {
            attempt.error(refusal)
        } else if attempt.previous().len() >= MAX_REDIRECTS {
            attempt.error("too many redirects")
        } else {
            attempt.follow()
        }
    })
}

/// The refusal for one hop's url, when its host is not on the public web.
///
/// Names go through the same resolver check as the original url in
/// [`check_url`], so a rebinder answering for a private address is refused
/// mid-chain.
fn private_hop(url: &Url) -> Option<String> {
    let private = match url.host()? {
        url::Host::Domain(host) => is_private_host(host),
        url::Host::Ipv4(address) => !is_global(IpAddr::V4(address)),
        url::Host::Ipv6(address) => !is_global(IpAddr::V6(address)),
    };
    private.then(|| format!("fetch: refusing redirect to non-public host {url}"))
}

/// Whether a url's path looks like a PDF: a `.pdf`-named leaf or path segment.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "the leaf is lowercased first, so the extension test is already case-insensitive"
)]
fn url_hints_pdf(url: &str) -> bool {
    let path = url.split(['?', '#']).next().unwrap_or_default();
    let leaf = path.rsplit('/').next().unwrap_or_default().to_ascii_lowercase();
    leaf.ends_with(".pdf") || path.split('/').any(|seg| seg.eq_ignore_ascii_case("pdf"))
}

/// Whether a body starts like a PDF, in poppler's own header-search window.
fn is_pdf(body: &[u8]) -> bool {
    body[..body.len().min(PDF_HEADER_WINDOW)]
        .windows(5)
        .any(|window| window == b"%PDF-")
}

/// A stable plain-text or JSON view of a github.com page, when one exists.
///
/// - `github.com/{o}/{r}/blob/{ref}/{path}` -> `raw.githubusercontent.com/{o}/{r}/{ref}/{path}`
/// - `github.com/{o}/{r}/pull/{n}` -> `api.github.com/repos/{o}/{r}/pulls/{n}`
/// - `github.com/{o}/{r}/issues/{n}` -> `api.github.com/repos/{o}/{r}/issues/{n}`
///
/// `None` for every other URL, including views themselves, so rewrites apply only once.
fn github_text_view(url: &str) -> Option<String> {
    let url = Url::parse(url).ok()?;
    if url.host_str()? != "github.com" {
        return None;
    }
    let segments = url.path_segments()?.collect::<Vec<_>>();
    match segments.as_slice() {
        // A blob names a file: the raw host serves its bytes verbatim, query
        // and fragment dropped, path segments kept percent-encoded as parsed.
        [owner, repo, "blob", reference, path @ ..] if !path.is_empty() => Some(format!(
            "https://raw.githubusercontent.com/{owner}/{repo}/{reference}/{}",
            path.join("/")
        )),
        [owner, repo, kind @ ("pull" | "issues"), number] if number.parse::<u64>().is_ok() => {
            Some(format!(
                "https://api.github.com/repos/{owner}/{repo}/{}/{number}",
                if *kind == "pull" { "pulls" } else { "issues" }
            ))
        }
        _ => None,
    }
}

/// One successfully downloaded body, with the URL it landed on.
///
/// The landing URL is the post-redirect destination (`response.url()`), which
/// is more accurate for the page-type classifier than the request URL.
#[derive(Debug, PartialEq)]
struct DownloadedBody {
    /// The response bytes, verified under [`DOWNLOAD_CAP`].
    bytes: Vec<u8>,
    /// The URL the response actually landed on (after redirects).
    final_url: String,
}

impl DownloadedBody {
    /// Downloads one URL and captures the post-redirect landing URL.
    ///
    /// Fails on http errors, or if the body exceeds [`DOWNLOAD_CAP`], returning a
    /// string that gets passed through to the model.
    fn try_download(client: &Client, url: &str) -> Result<Self, String> {
        let response = client
            .get(url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| format!("error: {}", error_chain(&error)))?;
        // The landing URL: the response knows where redirects actually went.
        let final_url = response.url().to_string();
        let mut bytes = Vec::new();
        response
            .take(DOWNLOAD_CAP as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("error: {error}"))?;
        if bytes.len() > DOWNLOAD_CAP {
            return Err(format!("error: body over {DOWNLOAD_CAP} bytes"));
        }
        Ok(Self { bytes, final_url })
    }

    /// Whether the first kilobyte contains `%PDF-`.
    fn is_pdf(&self) -> bool {
        is_pdf(&self.bytes)
    }

    /// Returns the raw bytes as bounded text, up to [`RAW_MAX_BYTES`].
    fn raw_text(&self) -> String {
        body_text(&self.bytes)
    }

    /// Returns whether the tag nesting is shallow enough to process safely.
    fn nesting_ok(&self) -> bool {
        let html = String::from_utf8_lossy(&self.bytes);
        let doc = Document::from(html.as_ref());
        // The levels under `body`, one per generation, ending at the first
        // empty one. A level at the cap's index means the tree is deeper
        // than allowed, and `nth` stops right there rather than walking the
        // rest of a page it is about to refuse.
        let too_deep = successors(Some(doc.select("body").first()), |level| {
            let children = level.children();
            (!children.is_empty()).then_some(children)
        })
        .nth(NESTING_CAP as usize)
        .is_some();
        !too_deep
    }

    /// Returns whether no colspan/rowspan attribute claims an absurd value.
    ///
    /// This catches degenerate or placeholder text like `colspan="1000000000000000"`,
    /// which attempts to allocate too much memory. We allow up to [`TABLE_SPAN_CAP`].
    fn table_spans_ok(&self) -> bool {
        let text = String::from_utf8_lossy(&self.bytes);
        for attr in ["colspan", "rowspan"] {
            let mut search = 0;
            while let Some(pos) = text[search..].find(attr) {
                let rest = &text[search + pos + attr.len()..];
                let digits: String = rest
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(char::is_ascii_digit)
                    .collect();
                if let Ok(value) = digits.parse::<usize>()
                    && value > TABLE_SPAN_CAP
                {
                    return false;
                }
                search += pos + attr.len();
            }
        }
        true
    }

    /// Guards against pathological inputs before any parsing begins.
    ///
    /// Both the math pre-pass and the reader are superlinear in nesting depth, so we
    /// check to make sure that the page seems safe. Returns a model-facing hint
    /// when the body is unsafe to parse, or `None` when it can proceed.
    fn setup(&self) -> Option<String> {
        if self.bytes.len() > EXTRACT_CAP {
            return Some(raw_hint(&format!("body over {EXTRACT_CAP} bytes")));
        }
        if !self.nesting_ok() {
            return Some(raw_hint(&format!("page nesting exceeds {NESTING_CAP} tags")));
        }
        if !self.table_spans_ok() {
            return Some(raw_hint(&format!(
                "table colspan/rowspan exceeds {TABLE_SPAN_CAP}"
            )));
        }
        None
    }

    /// Produce the model-facing text for this body.
    ///
    /// PDF bytes go to poppler or tesseract; everything else gets HTML preprocessed
    /// and then send into the reader if [`raw=true`](raw_hint), otherwise returned raw.
    fn read(&self) -> String {
        if self.is_pdf() {
            return pdf_from_body(&self.bytes);
        }
        if let Some(hint) = self.setup() {
            return hint;
        }
        let markdown = extract_markdown(&self.bytes, &self.final_url);
        if markdown.trim().is_empty() {
            raw_hint("no content extracted; the page may need JavaScript or have no article body")
        } else {
            body_text(markdown.as_bytes())
        }
    }
}

/// Extract one downloaded PDF as text via the poppler/tesseract.
///
/// The body is written to a private temp file (0600) that poppler reads; pages with
/// little text are OCR'd by tesseract at 200 dpi, bounded by [`PDF_TIMEOUT`] per stage.
fn pdf_from_body(body: &[u8]) -> String {
    let pdf = temp_path("fetch", "pdf");
    let text = create_private(&pdf)
        .and_then(|mut file| file.write_all(body))
        .map_or_else(|error| format!("error: {error}"), |()| pdf_text(&pdf));
    let _ = std::fs::remove_file(&pdf);
    text
}

/// Extract one downloaded page as markdown.
///
/// NOTE: short (~100 char) entries are doubled - this is a bug in (rs)-trafilatura, but
/// is mostly harmless
fn extract_markdown(body: &[u8], url: &str) -> String {
    let options = rs_trafilatura::Options {
        output_markdown: true,
        include_links: true,
        url: Some(url.to_string()),
        ..rs_trafilatura::Options::default()
    };
    // Always run the math_to_text preprocessor on utf8 pages.
    match std::str::from_utf8(body) {
        Ok(html) => {
            let preprocessed = math_to_text(html);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rs_trafilatura::extract_with_options(&preprocessed, &options)
            }))
            .unwrap_or(Err(rs_trafilatura::Error::NoContent))
            .map(|result| extraction_text(result.content_markdown, result.content_text))
            .unwrap_or_default()
        }
        // Non-utf8 data cannot be handled by extract_with_options so we don't try
        // to preprocess it
        Err(_) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rs_trafilatura::extract_bytes_with_options(body, &options)
        }))
        .unwrap_or(Err(rs_trafilatura::Error::NoContent))
        .map(|result| extraction_text(result.content_markdown, result.content_text))
        .unwrap_or_default(),
    }
}

/// The model-facing text of one extraction result.
fn extraction_text(markdown: Option<String>, text: String) -> String {
    markdown.filter(|md| !md.trim().is_empty()).unwrap_or(text)
}

/// Formats the model-facing suggestion to bypass extraction and fetch raw bytes.
///
/// Every guard and extraction failure funnels through this single message
/// shape, so the model always sees the same remedy regardless of which
/// stage refused the page.
fn raw_hint(reason: &str) -> String {
    format!("fetch: {reason}; use `raw=true` for the unprocessed response")
}

/// A fetched non-PDF body as inline text, capped at [`RAW_MAX_BYTES`].
fn body_text(body: &[u8]) -> String {
    bounded(&String::from_utf8_lossy(body), RAW_MAX_BYTES)
}

/// The page count `pdfinfo` reports, when pdfinfo and the file cooperate.
fn page_count(pdf: &Path) -> Option<usize> {
    let mut info = web_command(&find_on_path("pdfinfo")?);
    info.arg(pdf);
    let run = run_watched(&mut info, Some(PDF_TIMEOUT), &CancelToken::new()).ok()?;
    String::from_utf8_lossy(&run.output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("Pages:")?.trim().parse().ok())
}

/// Render one page of `pdf` to a PNG at 200 dpi and OCR it, trimmed.
///
/// `None` when either processing stage fails.
fn ocr_page(pdf: &Path, page: usize, pdftoppm: &Path, tesseract: &Path) -> Option<String> {
    let png = temp_path("fetch", "png");
    let mut render = web_command(pdftoppm);
    render.args(format!("-singlefile -f {page} -l {page} -r 200 -png").split(' '));
    render.arg(pdf).arg(png.with_extension(""));
    let rendered = run_watched(&mut render, Some(PDF_TIMEOUT), &CancelToken::new())
        .is_ok_and(|run| run.output.status.success())
        && std::fs::metadata(&png).is_ok();
    let mut ocr = web_command(tesseract);
    ocr.arg(&png).arg("stdout");
    let text = if rendered {
        run_watched(&mut ocr, Some(PDF_TIMEOUT), &CancelToken::new())
            .ok()
            .and_then(|WatchedRun { output, .. }| {
                output
                    .status
                    .success()
                    .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
            })
    } else {
        None
    };
    // The image is ours on every path, unlike the body file `download` hands back.
    let _ = std::fs::remove_file(&png);
    text
}

/// Extract one PDF for the model: poppler's embedded text layer, tesseract
/// for the pages it left thin, page-marked and bounded, status in a header.
fn pdf_text(path: &Path) -> String {
    let Some(pdftotext) = find_on_path("pdftotext") else {
        return "fetch: PDF detected but pdftotext is not on PATH; install poppler \
                (brew install poppler)"
            .to_string();
    };
    let mut extract = web_command(&pdftotext);
    extract.args(format!("-enc UTF-8 -l {PDF_MAX_PAGES}").split(' '));
    extract.arg(path).arg("-");
    let Ok(WatchedRun { output, .. }) =
        run_watched(&mut extract, Some(PDF_TIMEOUT), &CancelToken::new())
    else {
        return "fetch: pdftotext could not be run".to_string();
    };
    if !output.status.success() {
        return format!("fetch: pdftotext failed:\n{}", combined_output(&output));
    }
    let mut texts: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split('\u{c}')
        .map(|page| page.trim().to_string())
        .collect();
    // One form feed after the last page leaves one trailing empty chunk.
    texts.pop_if(|page| page.is_empty());
    let mut pages = Pages::new(&mut texts);
    pages.ocr_thin(path);
    if let Some(total) = page_count(path) {
        pages.total = total;
    }
    pages.render()
}

/// One extracted PDF: the borrowed page texts, and the accounting of how they were read
struct Pages<'a> {
    /// The page texts in order, empty where nothing was extracted.
    text: &'a mut [String],
    /// The page count pdfinfo reported, or the pages held when it did not.
    total: usize,
    /// Thin pages the OCR ladder recovered.
    ocr: usize,
    /// Thin pages the OCR ladder failed on.
    failed: usize,
    /// Thin pages never attempted: no stack installed, or the cap reached.
    skipped: usize,
}

impl<'a> Pages<'a> {
    /// Wrap pages straight out of pdftotext, totals still unrefined.
    fn new(text: &'a mut [String]) -> Self {
        Self {
            total: text.len(),
            text,
            ocr: 0,
            failed: 0,
            skipped: 0,
        }
    }

    /// OCR the pages the text layer left thin, in place.
    fn ocr_thin(&mut self, pdf: &Path) {
        let stack = find_on_path("pdftoppm").zip(find_on_path("tesseract"));
        for (index, page) in self.text.iter_mut().enumerate() {
            if page.chars().filter(|c| c.is_alphanumeric()).count() >= PDF_MIN_ALNUM {
                continue;
            }
            if let Some((pdftoppm, tesseract)) = stack.as_ref()
                && self.ocr + self.failed < PDF_OCR_PAGES
            {
                if let Some(text) = ocr_page(pdf, index + 1, pdftoppm, tesseract) {
                    *page = text;
                    self.ocr += 1;
                } else {
                    self.failed += 1;
                }
            } else {
                self.skipped += 1;
            }
        }
    }

    /// The pages the text layer left thin, whatever happened to them after.
    fn scanned(&self) -> usize {
        self.ocr + self.failed + self.skipped
    }

    /// The page-marked, bounded rendering: a `[pdf: ...]` header carrying the
    /// status, then `--- page N ---` bodies the model can cite.
    fn render(&self) -> String {
        let mut parts = vec![match (self.scanned(), self.ocr + self.failed) {
            (0, _) => "embedded text".to_string(),
            (_, 0) => "no text layer; install tesseract (brew) to OCR it".to_string(),
            _ if self.scanned() == self.text.len() => {
                format!("scanned: {} pages OCR'd", self.ocr)
            }
            _ => {
                format!(
                    "embedded text on {}, {} OCR'd",
                    self.text.len() - self.scanned(),
                    self.ocr
                )
            }
        }];
        if self.failed > 0 {
            parts.push(format!("({} OCR failed)", self.failed));
        }
        // Leftover pages only count against the cap when a stack ran at all;
        // with none installed, the source line already says so.
        if self.skipped > 0 && self.ocr + self.failed > 0 {
            parts.push(format!("({} scanned pages left: OCR cap)", self.skipped));
        }
        let mut body = String::new();
        let mut shown = 0;
        let mut cut = false;
        // The 160 reserves the header line's own budget, which the cap forgets.
        let budget = PDF_MAX_BYTES.saturating_sub(160);
        for (index, page) in self.text.iter().enumerate() {
            let text = if page.is_empty() {
                "(no text extracted)"
            } else {
                page.as_str()
            };
            let marked = format!("\n--- page {} ---\n{text}", index + 1);
            // Keep whole pages while they fit; a first page too big for the
            // whole budget is cut on a character boundary rather than dropped.
            if body.len() + marked.len() > budget {
                if shown == 0 {
                    body.push_str(&bounded(&marked, budget));
                    shown = 1;
                    cut = true;
                }
                break;
            }
            body.push_str(&marked);
            shown += 1;
        }
        let note = if cut || shown < self.text.len() {
            format!("; first {shown} of {} pages shown (byte cap)", self.total)
        } else if shown < self.total {
            format!("; first {shown} of {} pages shown (page cap)", self.total)
        } else {
            String::new()
        };
        format!("[pdf: {} pages, {}{note}]\n{body}", self.total, parts.join(" "))
    }
}

/// Get the content for a single fetch, handling special case filetypes.
fn fetch_body(fetch: &Fetch, url: &str, client: &Client) -> Result<String, String> {
    if !fetch.raw && !url_hints_pdf(url) {
        return DownloadedBody::try_download(client, url).map(|body| body.read());
    }
    let body = DownloadedBody::try_download(client, url)?;
    if fetch.raw && !body.is_pdf() {
        // Raw non-PDF bodies are content, unprocessed.
        Ok(body.raw_text())
    } else {
        // PDF bytes (raw or not) go to poppler+OCR; hinted non-PDFs go to the reader.
        Ok(body.read())
    }
}

/// Run one fetch tool call, reporting its steps to `on_progress`.
///
/// Like `search` this runs outside the sandbox. Every path is content for the model:
/// text on success, or a refusal or error that explains it.
pub(super) fn run_fetch<F: Fn(Progress)>(call: &FunctionToolCall, on_progress: &F) -> String {
    let fetch = match parse_fetch(&call.arguments) {
        Ok(fetch) => fetch,
        Err(error) => return misuse(call, on_progress, &error),
    };
    traced(call, on_progress, || {
        let fetched = || -> Result<String, String> {
            // The guard is content visible to the model, not an error.
            let url = check_url(&fetch.url)?;
            let client = FETCH_CLIENT.clone()?;
            // A github.com page whose text view exists is fetched through it,
            // raw (the view is plain text or JSON, never HTML to extract);
            // a view that refuses falls back to the page itself. `raw=true`
            // skips the detour: the model asked for its URL, byte for byte.
            if !fetch.raw
                && let Some(view) = github_text_view(&url)
                && let raw = (Fetch { url: view.clone(), raw: true })
                && let Ok(text) = fetch_body(&raw, &view, &client)
            {
                return Ok(text);
            }
            fetch_body(&fetch, &url, &client)
        };
        match fetched() {
            Ok(text) => (text.clone(), text, Some(0)),
            Err(text) => (text.clone(), text, None),
        }
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use crate::Agent;
    use crate::sandbox::Policy;
    use crate::sandbox::live::{skip_unless_live, skip_unless_networked};
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
            caller: None,
            r#async: None,
        }
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
    fn http_client_builds() {
        assert!(FETCH_CLIENT.is_ok());
    }

    /// An error whose cause is another error, for the chain flattening.
    #[derive(Debug)]
    struct Wrapped(&'static str, std::io::Error);
    impl std::fmt::Display for Wrapped {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "wrapped: {}", self.0)
        }
    }
    impl std::error::Error for Wrapped {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.1)
        }
    }

    #[test]
    fn error_chain_joins_causes_with_colons() {
        let wrapped = Wrapped("hop", std::io::Error::other("refusing redirect"));
        assert_eq!(error_chain(&wrapped), "wrapped: hop: refusing redirect");
    }

    /// Serve `response` once on an ephemeral loopback port and get its address.
    fn serve_once(response: String) -> std::net::SocketAddr {
        use std::io::{Read as _, Write as _};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0; 4_096];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        address
    }

    /// Live: needs the loopback socket a nested sandbox denies.
    #[apply(skip_unless_live!)]
    #[test]
    fn download_reads_bodies_statuses_and_refuses_private_redirects() {
        let client = FETCH_CLIENT.clone().unwrap();

        let address = serve_once("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_string());
        let body = DownloadedBody::try_download(&client, &format!("http://{address}/x")).unwrap();
        assert_eq!(body.bytes, b"hello");

        let address = serve_once("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string());
        let error =
            DownloadedBody::try_download(&client, &format!("http://{address}/gone")).unwrap_err();
        assert!(error.contains("404"), "{error}");

        // The policy refusal happens on the hop, before any connection is
        // made to the private address, and its text survives reqwest's
        // error wrapper.
        let address = serve_once(
            "HTTP/1.1 302 Found\r\nLocation: http://10.0.0.5/x\r\nContent-Length: 0\r\n\r\n"
                .to_string(),
        );
        let error =
            DownloadedBody::try_download(&client, &format!("http://{address}/hop")).unwrap_err();
        assert!(error.contains("refusing redirect"), "{error}");
        assert!(error.contains("10.0.0.5"), "{error}");
    }

    #[test]
    fn private_hop_flags_private_redirects_only() {
        for url in [
            "http://127.0.0.1:8080/x",
            "http://10.0.0.5/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "https://printer.local/",
            "https://localhost/x",
        ] {
            let url = Url::parse(url).unwrap();
            assert!(private_hop(&url).is_some(), "expected refusal: {url}");
        }

        let url = Url::parse("https://example.com/final").unwrap();
        assert!(private_hop(&url).is_none());
    }

    #[test]
    fn url_hints_pdf_matches_pdf_leaves_and_segments() {
        for url in [
            "https://dl.dell.com/manuals/x/powerconnect-5548_setup%20guide2_en-us.pdf",
            "https://arxiv.org/pdf/2401.14196",
            "https://arxiv.org/pdf/2401.14196v2.pdf",
            "https://openreview.net/pdf?id=V6E3Yd6iic",
            "https://example.com/paper.PDF",
            "https://example.com/download.pdf?version=2",
        ] {
            assert!(url_hints_pdf(url), "expected a hint: {url}");
        }

        for url in [
            "https://example.com/",
            "https://example.com/pdf-tutorial",
            "https://example.com/lib/pdfjs/reader",
            "https://arxiv.org/abs/2401.14196",
        ] {
            assert!(!url_hints_pdf(url), "expected no hint: {url}");
        }
    }

    #[test]
    fn github_text_view_maps_blobs_pulls_and_issues() {
        assert_eq!(
            github_text_view(
                "https://github.com/glotzerlab/hoomd-workflow/blob/trunk/workflow.toml"
            )
            .as_deref(),
            Some("https://raw.githubusercontent.com/glotzerlab/hoomd-workflow/trunk/workflow.toml")
        );
        // Deep and percent-encoded paths survive the rebuild, query drops.
        assert_eq!(
            github_text_view("https://github.com/o/r/blob/main/a%20b/c.toml?plain=1").as_deref(),
            Some("https://raw.githubusercontent.com/o/r/main/a%20b/c.toml")
        );
        assert_eq!(
            github_text_view("https://github.com/janbridley/tart/pull/67").as_deref(),
            Some("https://api.github.com/repos/janbridley/tart/pulls/67")
        );
        assert_eq!(
            github_text_view("https://github.com/janbridley/tart/issues/68#issuecomment-1")
                .as_deref(),
            Some("https://api.github.com/repos/janbridley/tart/issues/68")
        );
    }

    #[test]
    fn github_text_view_leaves_everything_else_alone() {
        for url in [
            "https://github.com/janbridley/tart",
            "https://github.com/janbridley/tart/tree/main/src",
            "https://github.com/janbridley/tart/pull/67/files",
            "https://github.com/janbridley/tart/pull/67.diff",
            "https://github.com/janbridley/tart/pulls",
            "https://github.com/janbridley/tart/pull/not-a-number",
            "https://github.com/janbridley/tart/blob/trunk",
            "https://raw.githubusercontent.com/o/r/main/f",
            "https://api.github.com/repos/o/r",
            "https://gitlab.com/o/r/blob/main/f",
            "https://github.com/janbridley/tart/wiki",
        ] {
            assert!(github_text_view(url).is_none(), "expected no view: {url}");
        }
    }

    #[test]
    fn is_pdf_sniffs_the_magic_bytes() {
        assert!(is_pdf(b"%PDF-1.7\nrest is opaque\n"));
        assert!(!is_pdf(b"<html>not a pdf</html>"));
        // A damaged-but-readable file may pad junk before the header, as long
        // as the header still starts inside poppler's 1024-byte search window.
        assert!(is_pdf(b"junk-before-header\n%PDF-1.4\n"));
        let mut junky = vec![b'x'; PDF_HEADER_WINDOW - 5];
        junky.extend_from_slice(b"%PDF-");
        assert!(is_pdf(&junky));
        junky.insert(0, b'x');
        assert!(!is_pdf(&junky));
        assert!(!is_pdf(b""));
    }

    #[test]
    fn render_pdf_marks_pages_and_reports_status() {
        let mut text = vec!["Dell PowerConnect 5548".to_string(), "Page two".to_string()];
        let extracted = Pages {
            total: 2,
            text: &mut text,
            ocr: 0,
            failed: 0,
            skipped: 0,
        };
        let rendered = extracted.render();

        assert_eq!(rendered.lines().next().unwrap(), "[pdf: 2 pages, embedded text]");
        assert!(rendered.contains("\n--- page 1 ---\nDell PowerConnect 5548"));
        assert!(rendered.contains("\n--- page 2 ---\nPage two"));

        // All pages thin, none OCR'd: the honest install hint, not silence.
        let mut empty = vec![String::new(), String::new()];
        let thin = Pages {
            total: 2,
            text: &mut empty,
            ocr: 0,
            failed: 0,
            skipped: 2,
        };
        let rendered = thin.render();
        assert!(rendered.contains("no text layer; install tesseract"));
        assert!(rendered.contains("--- page 1 ---\n(no text extracted)"));

        // Scanned and OCR'd, one page left over the cap.
        let mut scanned = vec!["scanned".to_string(); 4];
        let scanned = Pages {
            total: 4,
            text: &mut scanned,
            ocr: 3,
            failed: 0,
            skipped: 1,
        };
        let rendered = scanned.render();
        assert!(rendered.contains("scanned: 3 pages OCR'd (1 scanned pages left: OCR cap)"));

        // Truncation is page-aligned and said so in the header.
        let mut big = vec!["x".repeat(6_000); 30];
        let big = Pages {
            total: 30,
            text: &mut big,
            ocr: 0,
            failed: 0,
            skipped: 0,
        };
        let rendered = big.render();
        assert!(rendered.contains("; first 19 of 30 pages shown (byte cap)"));
        assert!(!rendered.contains("--- page 20 ---"));
    }

    /// One of the real PDFs dropped in the workspace root, when present.
    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(name)
    }

    #[test]
    fn pdf_text_extracts_the_dell_guide_text_layer() {
        let pdf = fixture(
            "all-products_esuprt_ser_stor_net_esuprt_powerconnect_powerconnect-5548_setup%20guide2_en-us.pdf",
        );
        if !pdf.is_file() || find_on_path("pdftotext").is_none() {
            return;
        }

        let text = pdf_text(&pdf);

        assert!(text.starts_with("[pdf: 14 pages, embedded text]"), "{text}");
        assert!(text.contains("--- page 1 ---"), "{text}");
        assert!(text.contains("PowerConnect"), "{text}");
        assert!(text.contains("CAUTION"), "{text}");
    }

    /// Whether tesseract can actually load its language data, so the OCR test
    /// runs only where OCR can.
    fn ocr_ready() -> bool {
        let Some(tesseract) = find_on_path("tesseract") else {
            return false;
        };
        let mut probe = web_command(&tesseract);
        probe.arg("--list-langs");
        run_watched(&mut probe, Some(Duration::from_secs(10)), &CancelToken::new())
            .is_ok_and(|run| run.output.status.success())
    }

    /// Live: OCRs eight scanned pages, so it needs tesseract with its language
    /// data and the patience for ~20 seconds.
    #[test]
    fn pdf_text_ocrs_the_scanned_mailing() {
        let pdf = fixture("PublicWaterMassMailing.pdf");
        if !pdf.is_file() || !ocr_ready() {
            return;
        }

        let text = pdf_text(&pdf);

        assert!(
            text.starts_with("[pdf: 8 pages, scanned: 8 pages OCR'd]"),
            "{}",
            text.chars().take(400).collect::<String>()
        );
        assert!(
            text.contains("Missouri"),
            "{}",
            text.chars().take(400).collect::<String>()
        );
        assert!(text.contains("--- page 8 ---"));
    }

    #[test]
    fn extract_body_caps_extraction_input() {
        let big = vec![b'<'; 11 * 1024 * 1024];
        let body = DownloadedBody {
            bytes: big,
            final_url: "https://example.com".to_string(),
        };
        let out = body.read();
        assert!(
            out.contains(&format!("body over {EXTRACT_CAP} bytes")),
            "over-cap body must say the cap: {out}"
        );
        assert!(out.contains("raw=true"), "over-cap body must hint: {out}");
    }

    #[test]
    fn an_empty_markdown_some_falls_back_to_the_extracted_text() {
        // Real markdown wins over the text, as before.
        let kept = extraction_text(Some("# Title".to_string()), "plain".to_string());
        assert_eq!(kept, "# Title");

        // An empty or whitespace-only `Some` beside text: the text wins,
        // not the emptiness.
        let empty = extraction_text(Some(String::new()), "plain".to_string());
        assert_eq!(empty, "plain");
        let blank = extraction_text(Some("  \n".to_string()), "plain".to_string());
        assert_eq!(blank, "plain");

        // No markdown at all stays the text; an empty extraction stays empty.
        let none = extraction_text(None, "plain".to_string());
        assert_eq!(none, "plain");
        let barren = extraction_text(None, String::new());
        assert_eq!(barren, "");
    }

    #[test]
    fn the_nesting_guard_covers_non_utf8_bodies() {
        // One invalid byte kept the old strict-decode guard from ever
        // running; the lossy decode must refuse the page regardless.
        let mut deep =
            format!("<body>{}</body>", "<div>".repeat(NESTING_CAP as usize + 1)).into_bytes();
        deep.push(0xff);
        let body = DownloadedBody {
            bytes: deep,
            final_url: "https://example.com".to_string(),
        };
        let out = body.read();
        assert!(
            out.contains(&format!("page nesting exceeds {NESTING_CAP} tags")),
            "deep non-UTF-8 body must be refused: {out}"
        );
        assert!(out.contains("raw=true"), "{out}");

        // A shallow non-UTF-8 body still goes to the reader, not the guard.
        let shallow = DownloadedBody {
            bytes: b"<body>caf\xe9</body>".to_vec(),
            final_url: "https://example.com".to_string(),
        };
        let out = shallow.read();
        assert!(
            !out.contains("page nesting"),
            "shallow body must not refuse: {out}"
        );
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
        let Some(_) = tool_binary("TART_SEARCH_BIN", "ddgs") else {
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
    /// stands on example.com alone.
    #[apply(skip_unless_networked!)]
    #[test]
    fn run_fetch_returns_the_page() {
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

    #[apply(skip_unless_networked!)]
    #[test]
    fn run_fetch_extracts_a_page_to_markdown() {
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
        let request = call("fetch", r#"{"url":"https://example.com"}"#);

        let output = execute(&request, &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        // Extraction, not passthrough
        assert!(!output.trim().is_empty(), "must extract text: {output}");
        assert!(!output.contains("<html"), "{output}");
        assert!(!output.contains("<body"), "{output}");
        assert!(
            output.contains("domain") && output.len() > 50,
            "must be the example.com body text: {output}"
        );
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, arguments, .. },
                Progress::ToolOutput { exit: Some(0), .. }
            ] if name == "fetch" && arguments == r#"{"url":"https://example.com"}"#
        ));
    }

    #[apply(skip_unless_networked!)]
    #[test]
    fn run_fetch_extracts_a_pdf_url_natively() {
        if find_on_path("pdftotext").is_none() {
            return;
        }
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
            "fetch",
            r#"{"url":"https://dl.dell.com/manuals/all-products/esuprt_ser_stor_net/esuprt_powerconnect/powerconnect-5548_setup%20guide2_en-us.pdf"}"#,
        );

        let output = execute(&request, &tools, &|progress| {
            events.borrow_mut().push(progress);
        });

        assert!(output.starts_with("[pdf: 14 pages, embedded text]"), "{output}");
        assert!(output.contains("--- page 2 ---"), "{output}");
        assert!(output.contains("CAUTION"), "{output}");
        assert!(matches!(
            events.borrow().as_slice(),
            [
                Progress::ToolStart { name, .. },
                Progress::ToolOutput { exit: Some(0), .. }
            ] if name == "fetch"
        ));
    }
}
