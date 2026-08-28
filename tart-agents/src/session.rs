//! Serialize transcripts as jsonl files to allow resumable tart sessions.

use std::fs::{self, OpenOptions};
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Context;
use async_openai::types::responses::{InputItem, Item, Role};
use time::OffsetDateTime;

use crate::Transcript;

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
    pub fn open(root: &Path, project: &Path, path: &Path) -> anyhow::Result<(Transcript, Session)> {
        let mut items = load(path)?;
        if items.is_empty() {
            anyhow::bail!("session file {} has no items", path.display());
        }
        // Try and trim a broken transcript into one we can send back to a provider.
        let recorded = items.len();
        trim_unpaired(&mut items);
        if items.len() != recorded {
            save(path, &items)?;
        }
        let written = items.len();
        let transcript = Transcript::from_items(items);
        let session = Self {
            root: root.to_path_buf(),
            project: project.to_path_buf(),
            path: Some(path.to_path_buf()),
            written,
        };
        Ok((transcript, session))
    }

    /// Open a sibiling session at the same root and project as the current one.
    #[inline]
    pub fn reopen(&self, path: &Path) -> anyhow::Result<(Transcript, Session)> {
        Session::open(&self.root, &self.project, path)
    }

    /// Append the transcript's items past `written`, creating the file if needed.
    #[inline]
    pub fn record(&mut self, transcript: &Transcript) -> anyhow::Result<()> {
        let items = transcript.stored_items();
        // A cleared record can end before the flushed prefix; flush from there.
        self.written = self.written.min(items.len());
        // If the session is empty (no user input), we don't need to save a record.
        if self.path.is_none() && !items.iter().any(is_user_message) {
            return Ok(());
        }
        let Some(fresh) = items.get(self.written..) else {
            return Ok(());
        };
        let path = if let Some(path) = &self.path {
            path.clone()
        } else {
            let dir = self.dir();
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let path = unused(&dir, &stamp());
            self.path = Some(path.clone());
            path
        };
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        // One write per line: a crash tears at most the last line.
        for item in fresh {
            let line = serde_json::to_string(item)?;
            writeln!(file, "{line}").with_context(|| format!("writing {}", path.display()))?;
        }
        self.written = items.len();
        Ok(())
    }

    /// Forget the current file; the next record starts a fresh session.
    #[inline]
    pub fn reset(&mut self) {
        self.path = None;
        self.written = 0;
    }
}

/// The per-project directory name for `path`: separators become `-`.
fn slug(path: &Path) -> String {
    let text = path.to_string_lossy();
    let stripped = text.strip_prefix('/').unwrap_or(&text);
    stripped.replace('/', "-")
}

/// Whether the item records something the user said.
fn is_user_message(item: &InputItem) -> bool {
    matches!(item, InputItem::EasyMessage(message) if message.role == Role::User)
}

/// Enumerate the project's session files, newest first, as a nice label for the picker
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
        .map(|path| (path.clone(), label(&path)))
        .collect())
}

/// One line naming the session in `path`: its stamp, then its opening request.
fn label(path: &Path) -> String {
    let name = path
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let stamp = name.strip_suffix(".jsonl").unwrap_or(&name);
    // The lines are scanned, not slurped: a session's first message is at the
    // top of a file that can be long.
    let opening = std::fs::File::open(path)
        .map(|file| {
            std::io::BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .find_map(|line| first_user_text(&line))
        })
        .ok()
        .flatten();
    let opening = opening
        .as_deref()
        .map_or_else(|| "(no messages)".to_string(), one_line);
    format!("{stamp}  {opening}")
}

/// The content of the first user message in one JSONL item line, if it is one.
fn first_user_text(line: &str) -> Option<String> {
    let item: serde_json::Value = serde_json::from_str(line).ok()?;
    (item["role"] == "user").then(|| item["content"].as_str().map(str::to_string))?
}

/// The first line of `text`, capped with an ellipsis.
pub(crate) fn one_line(text: &str) -> String {
    let line = text.split('\n').next().unwrap_or_default();
    let mut capped = line.chars().take(60).collect::<String>();
    if line.chars().count() > 60 {
        capped.push('…');
    }
    capped
}

/// The items in a session file, stopping at the first damaged line.
fn load(path: &Path) -> anyhow::Result<Vec<InputItem>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading session {}", path.display()))?;
    let mut items = Vec::new();
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
        match serde_json::from_str::<InputItem>(line) {
            Ok(item) => {
                items.push(item);
                good += line.len();
            }
            Err(_) => break,
        }
    }
    // Rewrite only when something parsed, bad sesions should not attempt to write
    if good < text.len() && !items.is_empty() {
        fs::write(path, &text[..good])
            .with_context(|| format!("trimming session {}", path.display()))?;
    }
    Ok(items)
}

/// Rewrite `path` with exactly `items`, one JSON line each.
fn save(path: &Path, items: &[InputItem]) -> anyhow::Result<()> {
    let mut text = String::new();
    for item in items {
        text.push_str(&serde_json::to_string(item)?);
        text.push('\n');
    }
    fs::write(path, text).with_context(|| format!("trimming session {}", path.display()))
}

/// Drop trailing calls whose outputs never arrived.
fn trim_unpaired(items: &mut Vec<InputItem>) {
    let unanswered = |items: &[InputItem], item: &InputItem| {
        let InputItem::Item(Item::FunctionCall(call)) = item else {
            return false;
        };
        !items.iter().any(|other| {
            matches!(
                other,
                InputItem::Item(Item::FunctionCallOutput(output)) if output.call_id == call.call_id
            )
        })
    };
    while items.last().is_some_and(|item| unanswered(items, item)) {
        items.pop();
    }
}

/// The moment as `YYYYMMDD-HHMMSS` UTC.
#[inline]
fn stamp_at(moment: OffsetDateTime) -> String {
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        moment.year(),
        u8::from(moment.month()),
        moment.day(),
        moment.hour(),
        moment.minute(),
        moment.second()
    )
}

/// The current instant as [`stamp_at`] renders it, the session filename.
#[inline]
fn stamp() -> String {
    stamp_at(OffsetDateTime::now_utc())
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

    /// Write `transcript`'s items to `path`, one JSON line each.
    fn write_session(path: &Path, transcript: &Transcript) {
        let mut text = String::new();
        for item in &transcript.request_items() {
            text.push_str(&serde_json::to_string(item).unwrap());
            text.push('\n');
        }
        std::fs::write(path, text).unwrap();
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

    /// [`stamp_at`](super::stamp_at) for a moment `seconds` after the epoch.
    fn stamp_unix(seconds: i64) -> String {
        stamp_at(OffsetDateTime::from_unix_timestamp(seconds).unwrap())
    }

    #[test]
    fn stamps_are_calendar_dates_that_sort_chronologically() {
        assert_eq!(stamp_unix(0), "19700101-000000");
        assert_eq!(stamp_unix(86_400), "19700102-000000");
        // A leap day renders zero-padded
        assert_eq!(stamp_unix(951_782_400), "20000229-000000");
        assert_eq!(stamp_unix(951_782_405), "20000229-000005");
        assert!(stamp_unix(951_782_399) < stamp_unix(951_782_400));
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
        assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 3);

        // A resumed session appends to the same file.
        transcript.push_user("again".to_string()).unwrap();
        transcript.push_assistant("there".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 5);

        // A cancelled turn keeps its partial answer, so it flushes like any other.
        transcript.push_user("cancel me".to_string()).unwrap();
        transcript.push_assistant("partial".to_string()).unwrap();
        session.record(&transcript).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 7);

        let (resumed, _) = Session::open(root.path(), project, &file).unwrap();
        assert_eq!(
            serde_json::to_value(resumed.request_items()).unwrap(),
            serde_json::to_value(transcript.request_items()).unwrap()
        );
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

    #[test]
    fn damaged_tails_end_the_record_and_orphans_are_trimmed() {
        let root = tempfile::tempdir().unwrap();

        // A torn final line ends the record, and the file is rewritten as possible
        let torn = root.path().join("torn.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("hello".to_string()).unwrap();
        write_session(&torn, &transcript);
        let ripped = std::fs::read_to_string(&torn).unwrap() + r#"{"type":"message","role":"ass"#;
        std::fs::write(&torn, ripped).unwrap();
        let (resumed, _) = Session::open(root.path(), Path::new("/tmp/other"), &torn).unwrap();
        assert_eq!(resumed.request_items().len(), 2);
        let after = std::fs::read_to_string(&torn).unwrap();
        assert_eq!(after.lines().count(), 2);
        assert!(after.ends_with('\n'));

        // A well-formed but unpaired trailing call is trimmed from the record
        // *and* the file, so recording a new turn cannot strand it mid-file.
        let orphan = root.path().join("orphan.jsonl");
        let transcript = Transcript::new().unwrap();
        transcript.push_user("run it".to_string()).unwrap();
        write_session(&orphan, &transcript);
        let call = FunctionToolCall {
            namespace: None,
            name: "bash".to_string(),
            arguments: r#"{"command":"ls"}"#.to_string(),
            call_id: "call_0".to_string(),
            id: None,
            status: None,
        };
        let line = format!(
            "{}\n",
            serde_json::to_string(&InputItem::Item(Item::FunctionCall(call))).unwrap()
        );
        std::fs::write(&orphan, std::fs::read_to_string(&orphan).unwrap() + &line).unwrap();
        let (resumed, mut session) =
            Session::open(root.path(), Path::new("/tmp/other"), &orphan).unwrap();
        assert_eq!(
            serde_json::to_value(resumed.request_items())
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(
            !std::fs::read_to_string(&orphan)
                .unwrap()
                .contains("function_call")
        );
        resumed.push_user("again".to_string()).unwrap();
        resumed.push_assistant("ok".to_string()).unwrap();
        session.record(&resumed).unwrap();
        let (reopened, _) = Session::open(root.path(), Path::new("/tmp/other"), &orphan).unwrap();
        assert!(
            !serde_json::to_value(reopened.request_items())
                .unwrap()
                .to_string()
                .contains("function_call")
        );
    }

    #[test]
    fn list_orders_newest_first_and_labels_the_opening_request() {
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

        // Newest first, each labelled with its stamp and opening request — the
        // first line only, capped with an ellipsis.
        assert_eq!(listed.len(), 4);
        assert!(listed[0].0.ends_with("20260102-000002.jsonl"));
        assert_eq!(listed[0].1, format!("20260102-000002  {}…", "x".repeat(60)));
        assert_eq!(listed[1].1, "20260102-000001  (no messages)");
        assert_eq!(listed[2].1, "20260102-000000  from the new session");
        assert_eq!(listed[3].1, "20260101-000000  from the old session");
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

    #[test]
    fn unused_bumps_a_suffix_on_collision() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("20260101-000000.jsonl"), "").unwrap();

        let next = unused(dir.path(), "20260101-000000");

        assert_eq!(next, dir.path().join("20260101-000000-2.jsonl"));
    }
}
