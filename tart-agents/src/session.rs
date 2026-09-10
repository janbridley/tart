//! Serialize transcripts as jsonl files to allow resumable tart sessions.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Context;
use async_openai::types::responses::{InputItem, Item, Role};
use time::OffsetDateTime;

use crate::history::{LineMetadata, RecordLine, Transcript, stamp_utc_at};
#[cfg(not(test))]
use crate::usage;

pub static SESSIONS_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("$HOME is not set; nowhere to keep session files");
    home.join(".config/tart/sessions")
});

/// One session's JSONL file, appended to at turn boundaries.
#[derive(Debug)]
pub struct Session {
    /// The sessions root, `~/.config/tart/sessions`.
    root: PathBuf,
    /// The directory-naming project, the working directory the session ran in.
    project: PathBuf,
    /// The file once created (or resumed); `None` until the first flush.
    path: Option<PathBuf>,
    /// How many items have been written; the transcript is flushed past this.
    written: usize,
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
            written: 0,
        }
    }

    /// This session's directory under the root, where its file is created.
    fn dir(&self) -> PathBuf {
        self.root.join(slug(&self.project))
    }

    /// Open a session, returning its transcript and location data.
    #[inline]
    pub fn open(root: &Path, project: &Path, path: &Path) -> anyhow::Result<(Transcript, Self)> {
        let mut loaded = load(path)?;
        if loaded.lines.is_empty() {
            anyhow::bail!("session file {} has no items", path.display());
        }
        // Try and trim a broken transcript into one we can send back to a provider.
        let recorded = loaded.lines.len();
        trim_unpaired(&mut loaded.lines);
        // Rewrite the file ONLY if its damaged or missing entries
        if loaded.damaged || loaded.lines.len() != recorded {
            rewrite(path, &loaded.lines)?;
        }
        let written = loaded.lines.len();
        let transcript = Transcript::from_lines(loaded.lines);
        let session = Self {
            root: root.to_path_buf(),
            project: project.to_path_buf(),
            path: Some(path.to_path_buf()),
            written,
        };
        // Tests hold the ledger's session slot themselves; production names
        // the live session here so rows self-tag without agent plumbing.
        #[cfg(not(test))]
        usage::set_session(session.stem());
        Ok((transcript, session))
    }

    /// Open a sibiling session at the same root and project as the current one.
    #[inline]
    pub fn reopen(&self, path: &Path) -> anyhow::Result<(Transcript, Self)> {
        Self::open(&self.root, &self.project, path)
    }

    /// Append the transcript's items past `written`, creating the file if needed.
    #[inline]
    pub fn record(&mut self, transcript: &Transcript) -> anyhow::Result<()> {
        // A cleared record can end before the flushed prefix; flush from there.
        self.written = self.written.min(transcript.len());
        let fresh = transcript.recorded_after(self.written);
        // If the session is empty (no user input), we don't need to save a record.
        if self.path.is_none()
            && !fresh
                .iter()
                .any(|line| matches!(&line.item, InputItem::EasyMessage(m) if m.role == Role::User))
        {
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
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let moment = OffsetDateTime::now_utc();
        for line in &fresh {
            write_line(&mut file, &path, line, moment)?;
        }
        self.written += fresh.len();
        Ok(())
    }

    /// The live file's stem, the ledger's tag for this session's rounds.
    #[inline]
    #[must_use]
    pub fn stem(&self) -> String {
        self.path
            .as_deref()
            .and_then(Path::file_stem)
            .map_or_else(String::new, |stem| stem.to_string_lossy().into_owned())
    }

    /// Forget the current file; the next record starts a fresh session.
    #[inline]
    pub fn reset(&mut self) {
        self.path = None;
        self.written = 0;
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

/// Enumerate the project's session files, newest first, each paired with the
/// first line of its opening user message, uncapped
#[inline]
pub fn list(root: &Path, project: &Path) -> anyhow::Result<Vec<(PathBuf, String)>> {
    let dir = root.join(slug(project));
    let entries =
        fs::read_dir(&dir).with_context(|| format!("no sessions found under {}", dir.display()))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        // A zero-length file is a crash between creation and the first line;
        // there is nothing to resume.
        .filter(|path| fs::metadata(path).is_ok_and(|meta| meta.len() > 0))
        .collect();
    if files.is_empty() {
        anyhow::bail!("no sessions found under {}", dir.display());
    }
    // Newest sessions first: the file a picker's first hit should be.
    files.sort_by_key(|path| {
        let modified = fs::metadata(path).and_then(|meta| meta.modified()).ok();
        std::cmp::Reverse((modified, path.file_name().map(std::ffi::OsStr::to_owned)))
    });
    Ok(files
        .into_iter()
        .map(|path| (path.clone(), opening(&path)))
        .collect())
}

/// The first line of the session in `path`'s opening user message, or empty
/// when it never got one.
fn opening(path: &Path) -> String {
    std::fs::File::open(path)
        .ok()
        .and_then(|file| {
            std::io::BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .find_map(|line| first_user_text(&line))
        })
        .unwrap_or_default()
}

/// The first line of the first user message in one JSONL item line, if it is
/// one.
fn first_user_text(line: &str) -> Option<String> {
    let item: serde_json::Value = serde_json::from_str(line).ok()?;
    let text = (item["role"] == "user").then(|| item["content"].as_str())??;
    Some(text.split_once('\n').map_or(text, |(line, _)| line).to_string())
}

/// A session file as [`load`] read it: its lines, and whether it held damage
/// past them.
struct Loaded {
    /// Every line up to the first damaged one.
    lines: Vec<RecordLine>,
    /// A torn final write or an unparseable line ends the record early.
    damaged: bool,
}

/// The items in a session file, stopping at the first damaged line.
///
/// The file is left exactly as it lies; [`Session::open`] decides whether the
/// damage is worth a rewrite. A `timestamp` or `usage` sibling of the wrong
/// type or shape is treated as absent.
fn load(path: &Path) -> anyhow::Result<Loaded> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading session {}", path.display()))?;
    let mut lines = Vec::new();
    let mut good = 0;
    for line in text.split_inclusive('\n') {
        // A final line without its newline is a torn write.
        if !line.ends_with('\n') {
            break;
        }
        if line.trim().is_empty() {
            good += line.len();
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            break;
        };
        let meta = peel(&value);
        let Ok(item) = serde_json::from_value(value) else {
            break;
        };
        lines.push(RecordLine { item, meta });
        good += line.len();
    }
    Ok(Loaded { lines, damaged: good < text.len() })
}

/// The metadata sibling a session line carried: its `timestamp` as written,
/// absent when missing or malformed.
fn peel(value: &serde_json::Value) -> LineMetadata {
    LineMetadata {
        timestamp: value
            .get("timestamp")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
    }
}

/// Rewrite `path` with exactly `lines`, keeping each line's recorded metadata.
fn rewrite(path: &Path, lines: &[RecordLine]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path)
        .with_context(|| format!("rewriting session {}", path.display()))?;
    let moment = OffsetDateTime::now_utc();
    for line in lines {
        write_line(&mut file, path, line, moment)?;
    }
    Ok(())
}

/// Append `line` to `file` as one JSON line: the recorded stamp or one minted
/// at `moment`.
///
/// One write per line: a crash tears at most the last line.
fn write_line(
    file: &mut fs::File,
    path: &Path,
    line: &RecordLine,
    moment: OffsetDateTime,
) -> anyhow::Result<()> {
    let text = line_of(line, moment)?;
    let text = format!("{text}\n");
    file.write_all(text.as_bytes())
        .with_context(|| format!("writing {}", path.display()))
}

/// The JSON line `line` writes: its item's serialization with the recorded
/// stamp or one minted at `moment`, each key alphabetized in.
fn line_of(line: &RecordLine, moment: OffsetDateTime) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(&line.item)?;
    let object = value
        .as_object_mut()
        .context("an item serializes as a JSON object")?;
    let stamp = line
        .meta
        .timestamp
        .clone()
        .unwrap_or_else(|| stamp_utc_at(moment));
    object.insert("timestamp".to_string(), serde_json::Value::String(stamp));
    Ok(serde_json::to_string(&value)?)
}

/// Drop trailing calls whose outputs never arrived, metadata with them.
fn trim_unpaired(lines: &mut Vec<RecordLine>) {
    let answered: HashSet<String> = lines
        .iter()
        .filter_map(|line| match &line.item {
            InputItem::Item(Item::FunctionCallOutput(output)) => Some(output.call_id.clone()),
            _ => None,
        })
        .collect();
    while let Some(RecordLine {
        item: InputItem::Item(Item::FunctionCall(call)),
        ..
    }) = lines.last()
    {
        if answered.contains(call.call_id.as_str()) {
            break;
        }
        lines.pop();
    }
}

/// The moment as `YYYYMMDD-HHMMSS` UTC, the session filename.
#[inline]
fn stamp_at(moment: OffsetDateTime) -> String {
    stamp_utc_at(moment).replace(['-', ':'], "").replace(' ', "-")
}

/// The first unused `stem.jsonl` in `dir`, bumping a numeric suffix on collision.
fn unused(dir: &Path, stem: &str) -> PathBuf {
    let mut suffix = 0;
    loop {
        suffix += 1;
        let name = if suffix == 1 {
            format!("{stem}.jsonl")
        } else {
            format!("{stem}-{suffix}.jsonl")
        };
        let candidate = dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;
    use async_openai::types::responses::FunctionToolCall;

    /// A session file's line count.
    fn line_count(path: &Path) -> usize {
        std::fs::read_to_string(path).unwrap().lines().count()
    }

    /// Write `transcript`'s items to `path`, one JSON line each.
    fn write_session(path: &Path, transcript: &Transcript) {
        let mut text = String::new();
        for item in transcript.request_items() {
            text.push_str(&serde_json::to_string(&item).unwrap());
            text.push('\n');
        }
        std::fs::write(path, text).unwrap();
    }

    /// The parsed lines of a session file, in order.
    fn parsed_lines(path: &Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
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
        stamps.iter().all(|stamp| stamp.as_deref().is_some_and(is_stamp))
    }

    /// The `timestamp` each of a file's lines carries, in order: the
    /// production peel, so a test cannot disagree with the reader.
    fn stamps(path: &Path) -> Vec<Option<String>> {
        parsed_lines(path)
            .iter()
            .map(|line| peel(line).timestamp)
            .collect()
    }

    #[test]
    fn slug_maps_separators_to_dashes() {
        assert_eq!(
            slug(Path::new("/Users/jenna/github/tart-no-slop")),
            "Users-jenna-github-tart-no-slop"
        );
        // Relative paths keep their shape; there is just no leading dash.
        assert_eq!(slug(Path::new("relative/dir")), "relative-dir");
    }

    /// The moment `seconds` after the epoch.
    fn unix(seconds: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(seconds).unwrap()
    }

    #[test]
    fn stamps_are_calendar_dates_that_sort_chronologically() {
        assert_eq!(stamp_at(unix(0)), "19700101-000000");
        assert_eq!(stamp_at(unix(86_400)), "19700102-000000");
        // A leap day renders zero-padded
        assert_eq!(stamp_at(unix(951_782_400)), "20000229-000000");
        assert_eq!(stamp_at(unix(951_782_405)), "20000229-000005");
        assert!(stamp_at(unix(951_782_399)) < stamp_at(unix(951_782_400)));
    }

    #[test]
    fn utc_stamps_are_human_dates_that_sort_chronologically() {
        assert_eq!(stamp_utc_at(unix(0)), "1970-01-01 00:00:00");
        assert_eq!(stamp_utc_at(unix(86_400)), "1970-01-02 00:00:00");
        // A leap day renders zero-padded, every field in place
        assert_eq!(stamp_utc_at(unix(951_782_400)), "2000-02-29 00:00:00");
        assert_eq!(stamp_utc_at(unix(951_782_405)), "2000-02-29 00:00:05");
        assert!(stamp_utc_at(unix(951_782_399)) < stamp_utc_at(unix(951_782_400)));
    }

    #[test]
    fn recorded_lines_carry_their_own_stamps() {
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        let mut session = Session::start(root.path(), Path::new("/tmp/proj"));
        session.record(&transcript).unwrap();
        let file = session.path.clone().unwrap();

        // Every line is stamped with the moment of its recording.
        let recorded = stamps(&file);
        assert!(all_stamped(&recorded));
        assert_eq!(recorded.len(), 3);
        assert!(recorded.windows(2).all(|pair| pair[0] <= pair[1]));

        // The next turn's flush carries its own lines' stamps.
        transcript.push_user("again".to_string()).unwrap();
        transcript.push_assistant("there".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(stamps(&file).len(), 5);
    }

    #[test]
    fn recorded_stamps_stay_off_the_resumed_request() {
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        let mut session = Session::start(root.path(), Path::new("/tmp/proj"));
        session.record(&transcript).unwrap();

        let (resumed, _) = Session::open(
            root.path(),
            Path::new("/tmp/proj"),
            &session.path.clone().unwrap(),
        )
        .unwrap();

        // A key scan of the request: the stamp dies at parse, so no provider
        // ever sees it, while the record keeps it, ready to write back.
        for item in resumed.request_items() {
            assert!(serde_json::to_value(item).unwrap().get("timestamp").is_none());
        }
        let lines = resumed.recorded_after(0);
        assert!(lines.iter().all(|line| line.meta.timestamp.is_some()));
    }

    #[test]
    fn mixed_format_files_merge_cleanly() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join("legacy.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        write_session(&legacy, &transcript);

        // A new turn recorded onto an old file: the old lines stay as they
        // were, the new ones are stamped, and one file holds both.
        let (resumed, mut session) =
            Session::open(root.path(), Path::new("/tmp/other"), &legacy).unwrap();
        resumed.push_user("again".to_string()).unwrap();
        resumed.push_assistant("ok".to_string()).unwrap();
        session.record(&resumed).unwrap();

        let stamps = stamps(&legacy);
        assert_eq!(
            stamps[..3],
            [None, None, None],
            "the legacy lines load metadata-free"
        );
        assert!(all_stamped(&stamps[3..]));
        let (reopened, _) = Session::open(root.path(), Path::new("/tmp/other"), &legacy).unwrap();
        assert_eq!(reopened.request_items().len(), 5);
    }

    #[test]
    fn malformed_siblings_are_absent_not_damage() {
        let root = tempfile::tempdir().unwrap();
        let mangled = root.path().join("mangled.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        write_session(&mangled, &transcript);
        // A numeric timestamp and a string usage are ignored, not fatal.
        let extra = concat!(
            r#"{"content":"run it","role":"user","timestamp":123,"usage":"lots","type":"message"}"#,
            "\n"
        );
        std::fs::write(&mangled, std::fs::read_to_string(&mangled).unwrap() + extra).unwrap();

        let (resumed, _) = Session::open(root.path(), Path::new("/tmp/other"), &mangled).unwrap();

        assert_eq!(resumed.request_items().len(), 3);
        let lines = resumed.recorded_after(0);
        assert_eq!(
            lines[2].meta,
            LineMetadata::default(),
            "the mangled siblings vanish"
        );
        assert!(
            std::fs::read_to_string(&mangled)
                .unwrap()
                .contains(r#""timestamp":123"#)
        );
    }

    #[test]
    fn record_flushes_turns_once() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let transcript = Transcript::new().unwrap();
        let mut session = Session::start(root.path(), project);

        // The system prompt lands with the first turn that completes, and
        // recording the same state again writes nothing.
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        session.record(&transcript).unwrap();
        session.record(&transcript).unwrap();
        let file = session.path.clone().unwrap();
        assert_eq!(line_count(&file), 3);

        // A resumed session appends to the same file.
        transcript.push_user("again".to_string()).unwrap();
        transcript.push_assistant("there".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(line_count(&file), 5);

        // A cancelled turn keeps its partial answer, so it flushes like any other.
        transcript.push_user("cancel me".to_string()).unwrap();
        transcript.push_assistant("partial".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(line_count(&file), 7);

        let (resumed, _) = Session::open(root.path(), project, &file).unwrap();
        assert_eq!(
            serde_json::to_value(resumed.request_items()).unwrap(),
            serde_json::to_value(transcript.request_items()).unwrap()
        );
    }

    /// A record that shrank below the flushed prefix resyncs at its new end:
    /// what follows still flushes, rather than landing behind the cursor
    /// forever. (A `/clear` resets the cursor with it, so this only shows up
    /// when the two drift apart.)
    #[test]
    fn a_shrunk_record_still_flushes_what_follows() {
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        let mut session = Session::start(root.path(), Path::new("/tmp/proj"));
        session.record(&transcript).unwrap();
        let file = session.path.clone().unwrap();
        assert_eq!(line_count(&file), 3);

        // The clear leaves the cursor past the record's new end; the next
        // record resyncs to it, skipping the turn that landed behind.
        transcript.clear();
        transcript.push_user("again".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(line_count(&file), 3);
        // The turn after it flushes on top, instead of being skipped too.
        transcript.push_assistant("ok".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(line_count(&file), 4);
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

        let file = std::fs::read_to_string(session.path.clone().unwrap()).unwrap();
        assert_eq!(file.lines().count(), 3, "system, user, assistant: {file}");
        assert!(
            !file.contains("plan mode is on"),
            "the file holds history only: {file}"
        );

        let (resumed, _) =
            Session::open(root.path(), project, &session.path.clone().unwrap()).unwrap();
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
        let torn = root.path().join("torn.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        write_session(&torn, &transcript);
        let ripped = std::fs::read_to_string(&torn).unwrap() + r#"{"type":"message","role":"ass"#;
        std::fs::write(&torn, &ripped).unwrap();

        let loaded = load(&torn).unwrap();

        assert_eq!(loaded.lines.len(), 2);
        assert!(loaded.damaged, "the torn tail is damage past the prefix");
        assert_eq!(std::fs::read_to_string(&torn).unwrap(), ripped);
    }

    #[test]
    fn damaged_tails_end_the_record_and_orphans_are_trimmed() {
        let root = tempfile::tempdir().unwrap();
        let both = root.path().join("both.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        // A stamped record, distinct originals, so preservation shows.
        let moment = OffsetDateTime::from_unix_timestamp(1_000_000_000).unwrap();
        let originals = ["2026-09-09 10:00:00", "2026-09-09 10:00:07"];
        let mut stamped = String::new();
        for (index, line) in transcript.recorded_after(0).iter().enumerate() {
            let mut line = line.clone();
            line.meta = LineMetadata {
                timestamp: Some(originals[index].to_string()),
            };
            stamped.push_str(&line_of(&line, moment).unwrap());
            stamped.push('\n');
        }
        // A well-formed but unpaired trailing call, and a torn tail after it.
        let orphan = line_of(
            &RecordLine {
                item: InputItem::Item(Item::FunctionCall(FunctionToolCall {
                    namespace: None,
                    name: "bash".to_string(),
                    arguments: r#"{"command":"ls"}"#.to_string(),
                    call_id: "call_0".to_string(),
                    id: None,
                    status: None,
                })),
                meta: LineMetadata {
                    timestamp: Some("2026-09-09 10:00:09".to_string()),
                },
            },
            moment,
        )
        .unwrap();
        std::fs::write(&both, format!("{stamped}{orphan}\n{{\"type\":\"me")).unwrap();

        let (resumed, mut session) =
            Session::open(root.path(), Path::new("/tmp/other"), &both).unwrap();

        assert_eq!(resumed.request_items().len(), 2);
        let after = std::fs::read_to_string(&both).unwrap();
        assert_eq!(after.lines().count(), 2, "system and turn only: {after}");
        assert!(after.ends_with('\n'), "every line is terminated");
        assert!(!after.contains("function_call"), "the orphan is gone: {after}");
        assert!(!after.contains("10:00:09"), "its metadata dies with it");
        assert_eq!(
            stamps(&both),
            vec![Some(originals[0].to_string()), Some(originals[1].to_string())],
            "the repair preserves the recorded stamps"
        );

        // The repaired session appends from there like any other.
        resumed.push_user("again".to_string()).unwrap();
        resumed.push_assistant("ok".to_string()).unwrap();
        session.record(&resumed).unwrap();
        let (reopened, _) = Session::open(root.path(), Path::new("/tmp/other"), &both).unwrap();
        assert_eq!(reopened.request_items().len(), 4);
    }

    #[test]
    fn list_orders_newest_first_and_extracts_the_opening_request() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let dir = root.path().join(slug(project));
        std::fs::create_dir_all(&dir).unwrap();
        let old = Transcript::new().unwrap();
        old.push_user("from the old session\nwith a second line".to_string())
            .unwrap();
        write_session(&dir.join("20260101-000000.jsonl"), &old);
        let recent = Transcript::new().unwrap();
        recent.push_user("from the new session".to_string()).unwrap();
        write_session(&dir.join("20260102-000000.jsonl"), &recent);
        // A session that never got a message, and one whose opening runs long.
        write_session(&dir.join("20260102-000001.jsonl"), &Transcript::new().unwrap());
        let chatty = Transcript::new().unwrap();
        chatty.push_user("x".repeat(80)).unwrap();
        write_session(&dir.join("20260102-000002.jsonl"), &chatty);

        let listed = list(root.path(), project).unwrap();

        // The opening text arrives raw and uncapped: formatting it is the
        // front end's business.
        assert_eq!(listed.len(), 4);
        assert!(listed[0].0.ends_with("20260102-000002.jsonl"));
        assert_eq!(listed[0].1, "x".repeat(80));
        assert_eq!(listed[1].1, "");
        assert_eq!(listed[2].1, "from the new session");
        assert_eq!(listed[3].1, "from the old session");
    }

    #[test]
    fn a_foreign_file_fails_to_resume_without_being_rewritten() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("notes.txt");
        std::fs::write(&file, "just some text\n").unwrap();

        let error = Session::open(root.path(), Path::new("/tmp/other"), &file)
            .unwrap_err()
            .to_string();

        assert!(error.contains("no items"), "{error}");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "just some text\n");
    }

    #[test]
    fn list_without_sessions_errors_naming_the_directory() {
        let root = tempfile::tempdir().unwrap();

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
        let empty =
            std::fs::read_dir(root.path()).map_or(true, |mut entries| entries.next().is_none());
        assert!(empty);
    }

    /// A reset abandons the file untouched; the next turn starts a fresh one.
    #[test]
    fn reset_starts_a_fresh_file_and_leaves_the_old_intact() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        transcript.push_assistant("hi".to_string()).unwrap();
        let mut session = Session::start(root.path(), project);
        session.record(&transcript).unwrap();
        let abandoned = session.path.clone().unwrap();
        let was = std::fs::read_to_string(&abandoned).unwrap();

        // The `/clear` flow: the record empties back to its system items and
        // the session forgets its file.
        transcript.clear();
        session.reset();
        transcript.push_user("start over".to_string()).unwrap();
        transcript.push_assistant("fresh".to_string()).unwrap();
        session.record(&transcript).unwrap();

        let fresh = session.path.clone().unwrap();
        assert_ne!(fresh, abandoned);
        assert_eq!(line_count(&fresh), 3, "the fresh file holds the new turn only");
        assert_eq!(std::fs::read_to_string(&abandoned).unwrap(), was);

        let fresh_stamps = stamps(&fresh);
        assert_eq!(fresh_stamps[0], stamps(&abandoned)[0]);
        assert!(all_stamped(&fresh_stamps[1..]));
    }

    /// A rewind abandons the file like a clear does, but the next record is seeded with
    /// the preserved prefix from the pre-rewind session
    #[test]
    fn a_rewind_starts_a_fresh_file_from_the_kept_prefix() {
        let root = tempfile::tempdir().unwrap();
        let project = Path::new("/tmp/proj");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("one".to_string()).unwrap();
        transcript.push_assistant("1".to_string()).unwrap();
        transcript.push_user("two".to_string()).unwrap();
        transcript.push_assistant("2".to_string()).unwrap();
        let mut session = Session::start(root.path(), project);
        session.record(&transcript).unwrap();
        let abandoned = session.path.clone().unwrap();
        let was = std::fs::read_to_string(&abandoned).unwrap();

        // The `/rewind` flow: the record cuts back to a turn's start, and
        // the session forgets its file.
        transcript.rewind(transcript.user_turns()[1].0);
        session.reset();
        transcript.push_user("twice".to_string()).unwrap();
        transcript.push_assistant("again".to_string()).unwrap();
        session.record(&transcript).unwrap();

        let fresh = session.path.clone().unwrap();
        assert_ne!(fresh, abandoned);
        let (resumed, mut resumed_session) = Session::open(root.path(), project, &fresh).unwrap();
        // The fresh file holds the system prompt, the kept turn, and the new
        // one: everything the rewound conversation continued from.
        assert_eq!(resumed.request_items(), transcript.request_items());

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
        assert_eq!(line_count(&fresh), 6, "the fresh session appends from there");
        assert_eq!(std::fs::read_to_string(&abandoned).unwrap(), was);
    }

    #[test]
    fn unused_bumps_a_suffix_on_collision() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("20260101-000000.jsonl"), "").unwrap();

        let next = unused(dir.path(), "20260101-000000");

        assert_eq!(next, dir.path().join("20260101-000000-2.jsonl"));
    }

    #[test]
    fn the_resumed_stem_finds_its_ledger_gauge() {
        let _held = crate::usage::tests::held_for_test();
        let _ledger = crate::usage::tests::ledger_root();
        let root = tempfile::tempdir().unwrap();
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        let mut session = Session::start(root.path(), Path::new("/tmp/proj"));
        session.record(&transcript).unwrap();

        // What the `/resume` chooser derives: the file's stem.
        let stem = session.stem();
        let from_path = session
            .path
            .as_ref()
            .unwrap()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(stem, from_path, "the tag is the name the chooser shows");

        // The round the resumed conversation last billed under that tag.
        crate::usage::set_session(stem);
        crate::usage::tests::sample_usage().bill(crate::usage::MAIN_AGENT, "m");
        assert_eq!(
            crate::usage::Ledger::gauge_for(&from_path),
            Some((9001, 8214, 512)),
            "a resume finds the gauge its own session billed"
        );
    }
}
