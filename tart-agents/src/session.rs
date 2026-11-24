//! Serialize transcripts as gzipped jsonl to allow resumable tart sessions.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Context;
use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use time::OffsetDateTime;

use crate::backends::{InputItem, Item, Role};
use crate::history::{RecordLine, Transcript, is_message, stamp_utc_at};
use crate::progress::Progress;
#[cfg(not(test))]
use crate::usage;
use crate::usage::Ledger;

/// The directory every project's sessions record under:
/// `~/.config/tart/sessions`.
pub static SESSIONS_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("$HOME is not set; nowhere to keep session files");
    home.join(".config/tart/sessions")
});

/// The project name chat mode records under: its sessions live in
/// `~/.config/tart/sessions/CHAT/`, apart from every working directory's.
pub const CHAT_PROJECT: &str = "CHAT";

/// One session's file: gzipped JSON lines, one recorded item each, rewritten
/// whole after each turn through a sibling renamed into place, so a crash
/// leaves the previous write intact, never a torn file.
///
/// A recorded conversation resumes as the same request it recorded:
///
/// ```
/// use std::path::Path;
/// use tart_agents::{Session, Transcript, session};
/// # fn main() -> anyhow::Result<()> {
/// let root = tempfile::tempdir()?;
/// let transcript = Transcript::new()?;
/// transcript.push_user("hello".to_string())?;
/// transcript.push_assistant("hi".to_string())?;
/// let mut session = Session::start(root.path(), Path::new("/tmp/proj"));
/// session.record(&transcript)?;
///
/// // The resume flow: list the project's sessions, newest first, and open one.
/// let (path, opening) = session::list(root.path(), Path::new("/tmp/proj"))?.remove(0);
/// assert_eq!(opening, "hello");
/// let (resumed, _) = Session::open(root.path(), Path::new("/tmp/proj"), &path)?;
/// assert_eq!(resumed.request_items(), transcript.request_items());
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct Session {
    /// The sessions root, `~/.config/tart/sessions`.
    root: PathBuf,
    /// The directory-naming project, the working directory the session ran in.
    project: PathBuf,
    /// The file once created (or resumed); `None` until the first turn records.
    path: Option<PathBuf>,
}

impl Session {
    /// A fresh session for `project` under `root`.
    #[inline]
    #[must_use]
    pub fn start(root: &Path, project: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            project: project.to_path_buf(),
            path: None,
        }
    }

    /// This session's directory under the root, where its file is created.
    fn dir(&self) -> PathBuf {
        self.root.join(slug(&self.project))
    }

    /// Open a session: its lines up to the first damaged one, trimmed of any
    /// call the record never answered, with the file rewritten only when that
    /// left something out.
    #[inline]
    pub fn open(root: &Path, project: &Path, path: &Path) -> anyhow::Result<(Transcript, Self)> {
        let (mut lines, damaged) = read(path)?;
        if lines.is_empty() {
            anyhow::bail!("session file {} has no items", path.display());
        }
        let recorded = lines.len();
        trim_unpaired(&mut lines);
        if damaged || lines.len() != recorded {
            write(path, &lines)?;
        }
        let session = Self {
            root: root.to_path_buf(),
            project: project.to_path_buf(),
            path: Some(path.to_path_buf()),
        };
        // Tests hold the ledger's session slot themselves; production names
        // the live session here so rows self-tag without agent plumbing.
        #[cfg(not(test))]
        usage::set_session(session.stem());
        Ok((Transcript::from_lines(lines), session))
    }

    /// Open a sibling session at the same root and project as the current one.
    #[inline]
    pub fn reopen(&self, path: &Path) -> anyhow::Result<(Transcript, Self)> {
        Self::open(&self.root, &self.project, path)
    }

    /// Write the resumed conversation into `pane` or other container.
    #[inline]
    pub fn write_into<P: Extend<Progress>>(&self, pane: &mut P, transcript: &Transcript) {
        let name = self.stem();
        if name.is_empty() {
            return;
        }
        pane.extend([Progress::Note(format!("resumed {name}"))]);
        pane.extend(transcript.replay());
        if let Some((input, cached, output)) = Ledger::gauge_for(&name) {
            pane.extend([Progress::Usage { input, cached, output }]);
        }
    }

    /// Write the transcript to the session file, creating it once a user turn
    /// exists. The file is written whole: unchanged lines keep their stamps,
    /// and a crash leaves the previous write in place.
    #[inline]
    pub fn record(&mut self, transcript: &Transcript) -> anyhow::Result<()> {
        let lines = transcript.lines();
        // A session that has yet to hear from the user needs no file.
        if self.path.is_none() && !lines.iter().any(|line| is_message(&line.item, Role::User)) {
            return Ok(());
        }
        let path = if let Some(path) = &self.path {
            path.clone()
        } else {
            let dir = self.dir();
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let path = unused(&dir, &stamp_at(OffsetDateTime::now_utc()));
            self.path = Some(path.clone());
            #[cfg(not(test))]
            usage::set_session(self.stem());
            path
        };
        write(&path, &lines)
    }

    /// The live file's stem, the ledger's tag for this session's rounds.
    #[inline]
    #[must_use]
    pub fn stem(&self) -> String {
        self.path
            .as_deref()
            .and_then(Path::file_stem)
            .map_or_else(String::new, |stem| {
                stem.to_string_lossy().trim_end_matches(".jsonl").to_owned()
            })
    }

    /// Forget the current file; the next record starts a fresh session.
    #[inline]
    pub fn reset(&mut self) {
        self.path = None;
        #[cfg(not(test))]
        usage::clear_session();
    }
}

/// The per-project directory name for `path`: separators become `-`.
pub(crate) fn slug(path: &Path) -> String {
    let text = path.to_string_lossy();
    let stripped = text.strip_prefix('/').unwrap_or(&text);
    stripped.replace('/', "-")
}

/// Enumerate the project's sessions, newest first, each paired with the first
/// line of its opening user message.
#[inline]
pub fn list(root: &Path, project: &Path) -> anyhow::Result<Vec<(PathBuf, String)>> {
    let dir = root.join(slug(project));
    let entries =
        fs::read_dir(&dir).with_context(|| format!("no sessions found under {}", dir.display()))?;
    // A zero-length file is a crash between creation and the first line;
    // there is nothing to resume.
    let mut files: Vec<PathBuf> = entries
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("reading {}", dir.display()))?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.to_string_lossy().ends_with(".jsonl.gz")
                && fs::metadata(path).is_ok_and(|meta| meta.len() > 0)
        })
        .collect();
    if files.is_empty() {
        anyhow::bail!("no sessions found under {}", dir.display());
    }
    // Newest first: the file a picker's first hit should be.
    files.sort_by_key(|path| {
        std::cmp::Reverse((
            fs::metadata(path).and_then(|meta| meta.modified()).ok(),
            path.file_name().map(std::ffi::OsStr::to_owned),
        ))
    });
    Ok(files
        .into_iter()
        .map(|path| (path.clone(), opening(&path)))
        .collect())
}

/// The first line of the session in `path`'s opening user message, or empty
/// when it has none.
fn opening(path: &Path) -> String {
    read(path)
        .ok()
        .and_then(|(lines, _)| {
            Transcript::from_lines(lines)
                .user_turns()
                .first()
                .map(|(_, text)| text.lines().next().unwrap_or_default().to_string())
        })
        .unwrap_or_default()
}

/// The session's lines up to the first damaged one, and whether damage ended
/// them: a truncated or foreign file, or a line that is not a whole JSON
/// object. The turns before the damage survive; the file is left as it lies,
/// and [`Session::open`] decides whether the damage is worth a rewrite.
fn read(path: &Path) -> anyhow::Result<(Vec<RecordLine>, bool)> {
    let file =
        fs::File::open(path).with_context(|| format!("reading session {}", path.display()))?;
    let mut lines = Vec::new();
    for line in std::io::BufReader::new(MultiGzDecoder::new(file)).lines() {
        match line {
            // A blank line is nothing.
            Ok(line) if line.trim().is_empty() => {}
            Ok(line) => match serde_json::from_str::<RecordLine>(&line) {
                Ok(line) => lines.push(line),
                Err(_) => return Ok((lines, true)),
            },
            Err(_) => return Ok((lines, true)),
        }
    }
    Ok((lines, false))
}

/// Write `lines` to `path` as one gzip member, through a sibling of the
/// writer's own renamed into place: a reader sees the last complete write a
/// process left behind, never a torn one.
fn write(path: &Path, lines: &[RecordLine]) -> anyhow::Result<()> {
    let mut bytes = Vec::new();
    let mut member = GzEncoder::new(&mut bytes, Compression::new(6));
    for line in lines {
        serde_json::to_writer(&mut member, line)?;
        member.write_all(b"\n")?;
    }
    member.finish()?;
    let sibling = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&sibling, &bytes).with_context(|| format!("writing {}", sibling.display()))?;
    fs::rename(&sibling, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Drop trailing calls whose outputs never arrived, stamps with them.
fn trim_unpaired(lines: &mut Vec<RecordLine>) {
    let answered: HashSet<String> = lines
        .iter()
        .filter_map(|line| match &line.item {
            InputItem::Item(Item::FunctionCallOutput(output)) => output.call_id.clone(),
            _ => None,
        })
        .collect();
    while lines.last().is_some_and(|line| {
        matches!(&line.item, InputItem::Item(Item::FunctionCall(call)) if !answered.contains(call.call_id.as_str()))
    }) {
        lines.pop();
    }
}

/// The moment as `YYYYMMDD-HHMMSS` UTC, the session filename.
#[inline]
fn stamp_at(moment: OffsetDateTime) -> String {
    stamp_utc_at(moment).replace(['-', ':'], "").replace(' ', "-")
}

/// The first unused `stem.jsonl.gz` in `dir`, bumping a numeric suffix on collision.
fn unused(dir: &Path, stem: &str) -> PathBuf {
    let mut suffix = 0;
    loop {
        suffix += 1;
        let name = match suffix {
            1 => format!("{stem}.jsonl.gz"),
            n => format!("{stem}-{n}.jsonl.gz"),
        };
        let path = dir.join(name);
        if !path.exists() {
            return path;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use std::io::Read as _;

    /// The session's text, gzip members and all.
    fn text_of(path: &Path) -> String {
        let mut text = String::new();
        MultiGzDecoder::new(std::fs::File::open(path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        text
    }

    /// The `timestamp` each of a file's lines carries, in order.
    fn stamps(path: &Path) -> Vec<Option<String>> {
        text_of(path)
            .lines()
            .map(|line| serde_json::from_str::<RecordLine>(line).unwrap().timestamp)
            .collect()
    }

    /// The gzip member `text` makes.
    fn member(text: &str) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(text.as_bytes()).unwrap();
        encoder.finish().unwrap()
    }

    /// Write `text` to `path` as one gzip member.
    fn write_session(path: &Path, text: &str) {
        std::fs::write(path, member(text)).unwrap();
    }

    /// The transcript's items as one JSON line each, the text a member holds.
    fn json_lines(transcript: &Transcript) -> String {
        let mut text = String::new();
        for item in transcript.request_items() {
            text.push_str(&serde_json::to_value(item).unwrap().to_string());
            text.push('\n');
        }
        text
    }

    fn is_stamp(stamp: &str) -> bool {
        const SHAPE: &[u8] = b"0000-00-00 00:00:00";
        stamp.len() == SHAPE.len()
            && stamp.bytes().zip(SHAPE).all(|(byte, shape)| match shape {
                b'0' => byte.is_ascii_digit(),
                _ => byte == *shape,
            })
    }

    fn all_stamped(stamps: &[Option<String>]) -> bool {
        !stamps.is_empty() && stamps.iter().all(|stamp| stamp.as_deref().is_some_and(is_stamp))
    }

    /// A transcript holding one `hello`/`hi` turn.
    fn greeted() -> Transcript {
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        transcript
    }

    /// A session that recorded `greeted()` under `root` for `/tmp/proj`, with
    /// the transcript it recorded.
    fn recorded(root: &Path) -> (Transcript, Session) {
        let mut session = Session::start(root, Path::new("/tmp/proj"));
        let transcript = greeted();
        session.record(&transcript).unwrap();
        (transcript, session)
    }

    /// The moment `seconds` after the epoch.
    fn unix(seconds: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(seconds).unwrap()
    }

    #[test]
    fn slug_maps_separators_to_dashes() {
        assert_eq!(
            slug(Path::new("/Users/jenna/github/tart-no-slop")),
            "Users-jenna-github-tart-no-slop"
        );
        // Relative paths keep their shape; there is just no leading dash.
        assert_eq!(slug(Path::new("relative/dir")), "relative-dir");
        // The chat project stays verbatim: its own directory, nothing cwd-shaped.
        assert_eq!(slug(Path::new(CHAT_PROJECT)), "CHAT");
    }

    /// A chat session records under `<root>/CHAT/`, its own directory, and
    /// resumes with its own stored system prompt: the minimal CHAT preamble.
    #[test]
    fn chat_sessions_live_apart_and_resume_their_system_prompt() {
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new_with(crate::prompts::CHAT).unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        let mut session = Session::start(root.path(), Path::new(CHAT_PROJECT));
        session.record(&transcript).unwrap();
        assert_eq!(
            session.path.clone().unwrap().parent(),
            Some(root.path().join("CHAT").as_path()),
            "the chat file sits in the CHAT directory"
        );

        let (resumed, _) =
            Session::open(root.path(), Path::new(CHAT_PROJECT), &session.path.unwrap()).unwrap();
        let items = serde_json::to_value(resumed.request_items()).unwrap();
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], crate::prompts::CHAT);
    }

    /// File names are calendar dates that sort chronologically, and the stamps
    /// beside them human ones that do too, zero-padded.
    #[test]
    fn stamps_are_zero_padded_dates_that_sort_chronologically() {
        assert_eq!(stamp_at(unix(0)), "19700101-000000");
        assert_eq!(stamp_utc_at(unix(0)), "1970-01-01 00:00:00");
        assert_eq!(stamp_at(unix(86_400)), "19700102-000000");
        assert_eq!(stamp_at(unix(951_782_405)), "20000229-000005");
        assert_eq!(stamp_utc_at(unix(951_782_405)), "2000-02-29 00:00:05");
        assert!(stamp_at(unix(951_782_399)) < stamp_at(unix(951_782_400)));
        assert!(stamp_utc_at(unix(951_782_399)) < stamp_utc_at(unix(951_782_400)));
    }

    /// Every recorded line carries the stamp of its own flush, and the stamp
    /// stays on the record: no request ever carries it, a resume keeps it.
    #[test]
    fn recorded_lines_carry_their_own_stamps_off_the_request() {
        let root = tempfile::tempdir().unwrap();
        let (transcript, mut session) = recorded(root.path());
        let file = session.path.clone().unwrap();

        let first = stamps(&file);
        assert!(all_stamped(&first));
        assert_eq!(first.len(), 3);
        assert!(first.windows(2).all(|pair| pair[0] <= pair[1]));

        // The next turn's flush carries its own lines' stamps.
        transcript.push_user("again".to_string()).unwrap();
        transcript.push_assistant("there".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(stamps(&file).len(), 5);

        let (resumed, _) = Session::open(root.path(), Path::new("/tmp/proj"), &file).unwrap();
        for item in resumed.request_items() {
            assert!(serde_json::to_value(item).unwrap().get("timestamp").is_none());
        }
        assert!(resumed.lines().iter().all(|line| line.timestamp.is_some()));
    }

    /// Recording the same state twice rewrites the same bytes: not one byte
    /// differs; a resumed session keeps writing to the same file, and a
    /// cancelled turn's partial answer lands like any other.
    #[test]
    fn record_flushes_each_turn_once() {
        let root = tempfile::tempdir().unwrap();
        let (transcript, mut session) = recorded(root.path());
        let file = session.path.clone().unwrap();
        let was = fs::metadata(&file).unwrap().len();
        session.record(&transcript).unwrap();
        assert_eq!(
            fs::metadata(&file).unwrap().len(),
            was,
            "nothing fresh, nothing written"
        );

        transcript.push_user("again".to_string()).unwrap();
        transcript.push_assistant("there".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(text_of(&file).lines().count(), 5);

        transcript.push_user("cancel me".to_string()).unwrap();
        transcript.push_assistant("partial".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(text_of(&file).lines().count(), 7);
    }

    /// Opening a clean session touches nothing: same bytes, same mtime.
    #[test]
    fn a_clean_session_opens_untouched() {
        let root = tempfile::tempdir().unwrap();
        let (_, session) = recorded(root.path());
        let file = session.path.clone().unwrap();
        let (was, moment) = (
            std::fs::read(&file).unwrap(),
            fs::metadata(&file).unwrap().modified().unwrap(),
        );

        let _ = Session::open(root.path(), Path::new("/tmp/proj"), &file).unwrap();

        assert_eq!(std::fs::read(&file).unwrap(), was, "no rewrite, no conversion");
        assert_eq!(fs::metadata(&file).unwrap().modified().unwrap(), moment);
    }

    /// A `timestamp` or `usage` sibling of the wrong shape reads as absent,
    /// not damage: the line survives, the file is left as it lies.
    #[test]
    fn malformed_siblings_are_absent_not_damage() {
        let root = tempfile::tempdir().unwrap();
        let mangled = root.path().join("mangled.jsonl");
        let system = serde_json::to_value(&greeted().request_items()[0]).unwrap();
        let extra =
            r#"{"content":"run it","role":"user","timestamp":123,"usage":"lots","type":"message"}"#;
        write_session(&mangled, &format!("{system}\n{extra}\n"));

        let (resumed, _) = Session::open(root.path(), Path::new("/tmp/other"), &mangled).unwrap();

        assert_eq!(resumed.request_items().len(), 2);
        assert_eq!(resumed.lines()[1].timestamp, None, "the mangled siblings vanish");
        assert!(text_of(&mangled).contains(r#""timestamp":123"#));
    }

    #[test]
    fn record_never_writes_the_reminder() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let mut transcript = Transcript::new().unwrap();
        let mut session = Session::start(root.path(), project);

        transcript.set_reminder(Some("plan mode is on")).unwrap();
        transcript.push_user("look at the auth flow".to_string()).unwrap();
        transcript.push_assistant("here is the plan".to_string()).unwrap();
        session.record(&transcript).unwrap();

        let file = session.path.clone().unwrap();
        let text = text_of(&file);
        assert_eq!(text.lines().count(), 3, "system, user, assistant: {text}");
        assert!(
            !text.contains("plan mode is on"),
            "the file holds history only: {text}"
        );

        let (resumed, _) = Session::open(root.path(), project, &file).unwrap();
        assert!(
            !serde_json::to_string(&resumed.request_items())
                .unwrap()
                .contains("plan mode is on"),
            "a resumed transcript starts with no reminder"
        );
    }

    /// Reading a damaged file leaves it exactly as it lay: only `open` rewrites.
    #[test]
    fn load_leaves_a_damaged_file_as_it_lies() {
        let root = tempfile::tempdir().unwrap();
        let torn = root.path().join("torn.jsonl.gz");
        write_session(&torn, &json_lines(&greeted()));
        // Garbage after a whole member: the member's turns survive, and the
        // tail is damage past them.
        let mut ripped = std::fs::read(&torn).unwrap();
        ripped.extend_from_slice(b"not gzip");
        std::fs::write(&torn, &ripped).unwrap();

        let (lines, damaged) = read(&torn).unwrap();

        assert_eq!(lines.len(), 3);
        assert!(damaged, "the torn tail is damage past the prefix");
        assert_eq!(std::fs::read(&torn).unwrap(), ripped);
    }

    /// A torn tail ends the record, and an unpaired trailing call goes with
    /// its stamp; the repair keeps every earlier stamp, and the session
    /// appends from there.
    #[test]
    fn damaged_tails_end_the_record_and_orphans_are_trimmed() {
        let root = tempfile::tempdir().unwrap();
        let both = root.path().join("both.jsonl");
        let originals = ["2026-09-09 10:00:00", "2026-09-09 10:00:07"];
        let stamped = [
            r##"{"type":"message","role":"system","content":"# system","timestamp":"2026-09-09 10:00:00"}"##,
            r#"{"type":"message","role":"user","content":"run it","timestamp":"2026-09-09 10:00:07"}"#,
        ];
        // A well-formed but unpaired trailing call in its own member, then a
        // garbage tail: the damage the repair path trims back through.
        let orphan = r#"{"type":"function_call","name":"bash","arguments":"{\"command\":\"ls\"}","call_id":"call_0","id":"fc_0","status":"completed","timestamp":"2026-09-09 10:00:09"}"#;
        std::fs::write(
            &both,
            [
                member(&format!("{}\n", stamped.join("\n"))),
                member(&format!("{orphan}\n")),
                b"not gzip".to_vec(),
            ]
            .concat(),
        )
        .unwrap();

        let (resumed, mut session) =
            Session::open(root.path(), Path::new("/tmp/other"), &both).unwrap();

        assert_eq!(resumed.request_items().len(), 2);
        let after = text_of(&both);
        assert_eq!(after.lines().count(), 2, "system and turn only: {after}");
        assert!(after.ends_with('\n'), "every line is terminated");
        assert!(!after.contains("function_call"), "the orphan is gone: {after}");
        assert!(!after.contains("10:00:09"), "its metadata dies with it");
        assert_eq!(
            stamps(&both),
            originals.map(|stamp| Some(stamp.to_string())),
            "the repair preserves the recorded stamps"
        );

        // The repaired session appends from there like any other.
        resumed.push_user("again".to_string()).unwrap();
        resumed.push_assistant("ok".to_string()).unwrap();
        session.record(&resumed).unwrap();
        let (reopened, _) = Session::open(root.path(), Path::new("/tmp/other"), &both).unwrap();
        assert_eq!(reopened.request_items().len(), 4);
    }

    /// A member torn mid-stream keeps every whole line before the tear, with
    /// the stamps it recorded; the repair rewrites the file once, and it opens
    /// untouched from then on.
    #[test]
    fn a_torn_member_keeps_its_whole_lines() {
        let root = tempfile::tempdir().unwrap();
        let torn = root.path().join("torn.jsonl.gz");
        // Two hundred short, individually stamped turns in one member.
        let long: String = (0..200)
            .map(|n| {
                format!(
                    r#"{{"content":"line {n} with some padding to lower the ratio","role":"assistant","timestamp":"2026-09-09 10:{:02}:{:02}","type":"message"}}"#,
                    n / 60,
                    n % 60
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&torn, member(&long)).unwrap();
        // Cut inside the member: the lines whose bytes survived the cut load.
        let whole = std::fs::read(&torn).unwrap();
        let cut = &whole[..whole.len() / 2];
        std::fs::write(&torn, cut).unwrap();
        let (lines, damaged) = read(&torn).unwrap();
        let kept = lines.len();
        assert!(damaged, "the tear is damage");
        assert!(
            kept > 10 && kept < 200,
            "a prefix of whole lines survives the tear: {kept}"
        );
        assert_eq!(
            lines[1].timestamp.as_deref(),
            Some("2026-09-09 10:00:01"),
            "the survivors keep their stamps"
        );
        assert_eq!(std::fs::read(&torn).unwrap(), cut, "reading rewrites nothing");

        let (resumed, _) = Session::open(root.path(), Path::new("/tmp/other"), &torn).unwrap();
        assert_eq!(resumed.request_items().len(), kept);
        assert_eq!(
            text_of(&torn).lines().count(),
            kept,
            "the repair keeps the prefix"
        );
        let bytes = std::fs::read(&torn).unwrap();
        let _ = Session::open(root.path(), Path::new("/tmp/other"), &torn).unwrap();
        assert_eq!(
            std::fs::read(&torn).unwrap(),
            bytes,
            "repaired once, then untouched"
        );
    }

    #[test]
    fn list_orders_newest_first_and_extracts_the_opening_request() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let dir = root.path().join(slug(project));
        std::fs::create_dir_all(&dir).unwrap();
        let line = |role: &str, text: &str| {
            serde_json::json!({"type": "message", "role": role, "content": text}).to_string()
        };
        // A session that never got a message, and one whose opening runs long.
        let sessions = [
            (
                "20260101-000000",
                format!("{}\n", line("user", "from the old session\nwith a second line")),
            ),
            (
                "20260102-000000",
                format!("{}\n", line("user", "from the new session")),
            ),
            ("20260102-000001", String::new()),
            ("20260102-000002", format!("{}\n", line("user", &"x".repeat(80)))),
        ];
        for (stem, text) in &sessions {
            write_session(&dir.join(format!("{stem}.jsonl.gz")), text);
        }
        // The uncompressed format is gone: such a file is not a session.
        std::fs::write(dir.join("20260103-000000.jsonl"), "not a session\n").unwrap();

        let listed = list(root.path(), project).unwrap();

        // The opening text arrives raw and uncapped: formatting it is the
        // front end's business.
        assert_eq!(listed.len(), 4);
        assert!(listed[0].0.ends_with("20260102-000002.jsonl.gz"));
        assert_eq!(listed[0].1, "x".repeat(80));
        assert_eq!(listed[1].1, "");
        assert_eq!(listed[2].1, "from the new session");
        assert_eq!(listed[3].1, "from the old session");
    }

    /// A foreign file, or a directory without sessions, fails to resume naming
    /// what it found, and is left as it lies.
    #[test]
    fn a_foreign_file_or_empty_directory_errors_without_a_rewrite() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("notes.txt");
        std::fs::write(&file, "just some text\n").unwrap();

        let error = Session::open(root.path(), Path::new("/tmp/other"), &file)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no items"), "{error}");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "just some text\n");

        let error = list(root.path(), Path::new("/tmp/proj")).unwrap_err().to_string();
        assert!(error.contains("no sessions found"), "{error}");
    }

    #[test]
    fn a_session_without_messages_leaves_no_file() {
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new().unwrap();
        let mut session = Session::start(root.path(), Path::new("/tmp/proj"));

        // Recording before any message exists writes nothing: no file appears.
        session.record(&transcript).unwrap();
        session.record(&transcript).unwrap();
        assert!(session.path.is_none());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }

    /// A reset, as `/clear` and `/rewind` run it, abandons the file untouched;
    /// the next turn starts a fresh file, seeded with whatever the record kept,
    /// under the kept lines' original stamps.
    #[test]
    fn reset_starts_a_fresh_file_and_leaves_the_old_intact() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let transcript = Transcript::new().unwrap();
        for (user, answer) in [("one", "1"), ("two", "2")] {
            transcript.push_user(user.to_string()).unwrap();
            transcript.push_assistant(answer.to_string()).unwrap();
        }
        let mut session = Session::start(root.path(), project);
        session.record(&transcript).unwrap();
        let abandoned = session.path.clone().unwrap();
        let was = text_of(&abandoned);

        // The `/rewind` flow: the record cuts back to a turn's start, and the
        // session forgets its file. (`/clear` is the same with nothing kept.)
        transcript.rewind(transcript.user_turns()[1].0);
        session.reset();
        transcript.push_user("twice".to_string()).unwrap();
        transcript.push_assistant("again".to_string()).unwrap();
        session.record(&transcript).unwrap();

        let fresh = session.path.clone().unwrap();
        assert_ne!(fresh, abandoned);
        assert_eq!(text_of(&abandoned), was);
        let (resumed, mut resumed_session) = Session::open(root.path(), project, &fresh).unwrap();
        // The fresh file holds the system prompt, the kept turn, and the new
        // one: everything the rewound conversation continued from.
        assert_eq!(
            serde_json::to_value(resumed.request_items()).unwrap(),
            serde_json::to_value(transcript.request_items()).unwrap()
        );

        // The kept prefix re-flushes with its ORIGINAL stamps.
        let abandoned_stamps = stamps(&abandoned);
        let fresh_stamps = stamps(&fresh);
        assert_eq!(
            fresh_stamps[..3],
            abandoned_stamps[..3],
            "kept lines keep their stamps"
        );
        assert!(all_stamped(&fresh_stamps[3..]));

        resumed.push_user("three".to_string()).unwrap();
        resumed_session.record(&resumed).unwrap();
        // Five lines seeded the file; the resumed turn appends the sixth.
        assert_eq!(
            text_of(&fresh).lines().count(),
            6,
            "the fresh session appends from there"
        );
    }

    #[test]
    fn unused_bumps_a_suffix_on_collision() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("20260101-000000.jsonl.gz"), "").unwrap();

        let next = unused(dir.path(), "20260101-000000");

        assert_eq!(next, dir.path().join("20260101-000000-2.jsonl.gz"));
    }

    /// A resumed session writes itself into a pane: a note naming it, then
    /// the recorded turns. A session yet to open its file writes nothing.
    #[test]
    fn write_into_replays_the_turns_with_a_note() {
        let _held = crate::usage::tests::held_for_test();
        let _ledger = crate::usage::tests::ledger_root();
        let root = tempfile::tempdir().unwrap();
        let (transcript, session) = recorded(root.path());

        let mut painted = Vec::new();
        session.write_into(&mut painted, &transcript);

        assert_eq!(painted.len(), 3, "{painted:?}");
        assert!(
            matches!(&painted[0], Progress::Note(note) if note.ends_with(&session.stem())),
            "{painted:?}"
        );
        assert!(
            matches!(&painted[1], Progress::User(text) if text == "hello"),
            "{painted:?}"
        );
        assert!(
            matches!(&painted[2], Progress::Answer(text) if text == "hi"),
            "{painted:?}"
        );

        let mut fresh = Vec::new();
        Session::start(root.path(), Path::new("/tmp/proj")).write_into(&mut fresh, &transcript);
        assert!(fresh.is_empty(), "{fresh:?}");
    }

    /// A session file an earlier tart wrote — plain messages, calls with
    /// minted ids, reasoning with its ciphertext — loads, and every field of
    /// every line reaches the next request: the old record is the new request.
    #[test]
    fn a_recorded_session_replays_verbatim() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("20261124-151934.jsonl.gz");
        let fixture = include_str!("data/session-replay.jsonl");
        write_session(&path, fixture);

        let (transcript, _session) =
            Session::open(root.path(), Path::new("/tmp/proj"), &path).unwrap();
        let items = serde_json::to_value(transcript.request_items()).unwrap();

        for (recorded, item) in fixture.lines().zip(items.as_array().unwrap()) {
            let mut recorded: serde_json::Value = serde_json::from_str(recorded).unwrap();
            recorded.as_object_mut().unwrap().remove("timestamp");
            assert_eq!(item, &recorded, "{item}");
        }
        assert_eq!(
            items[4]["encrypted_content"],
            "gAAAAAAAAAB0ZXN0LWZpeHR1cmUtY2lwaGVydGV4dC1ub3QtcmVhbA"
        );
        assert_eq!(items[2]["call_id"], items[3]["call_id"]);
        // And the replay still renders its words and tool headers.
        assert!(!transcript.replay().is_empty());
    }
}
