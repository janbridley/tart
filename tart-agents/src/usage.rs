//! Append-only ledger of token usage per model and session.

use std::fs::{self, OpenOptions};
use std::io::{BufRead as _, Write as _};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::locked;

use anyhow::Context;
use async_openai::types::responses::ResponseUsage;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::debug;
use crate::history::stamp_utc_at;
use crate::session::slug;

/// The live session's file stem, as [`crate::Session`] last named it.
static SESSION: Mutex<Option<String>> = Mutex::new(None);

/// One response's five token counts, as the provider measured them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct TokenUsage {
    /// All input tokens, cache included.
    pub(crate) input: u64,
    /// The input tokens served from the prompt cache.
    pub(crate) cached: u64,
    /// The tokens the model generated.
    pub(crate) output: u64,
    /// The generated tokens spent reasoning.
    pub(crate) reasoning: u64,
    /// The provider's total, input plus output.
    pub(crate) total: u64,
}

impl TokenUsage {
    /// The usage a completed response reported, its five counts carried over:
    /// `cached` is `input_tokens_details.cached_tokens`, `reasoning` is
    /// `output_tokens_details.reasoning_tokens`.
    #[inline]
    #[must_use]
    pub(crate) fn extract(usage: &ResponseUsage) -> Self {
        Self {
            input: u64::from(usage.input_tokens),
            cached: u64::from(usage.input_tokens_details.cached_tokens),
            output: u64::from(usage.output_tokens),
            reasoning: u64::from(usage.output_tokens_details.reasoning_tokens),
            total: u64::from(usage.total_tokens),
        }
    }

    /// The gauge's three display counts: `(input, cached, output)`.
    #[inline]
    #[must_use]
    pub(crate) fn gauge(self) -> (u64, u64, u64) {
        (self.input, self.cached, self.output)
    }

    /// Append this usage to the ledger as one billed response
    ///
    /// NOTE: this is *not* guaranteed to be the exact bill quantity
    /// Cancelled sessions or other interruptions may be slightly off.
    pub(crate) fn bill(self, agent: &str, model: &str) {
        let entry = LedgerEntry {
            agent: agent.to_string(),
            model: model.to_string(),
            session: locked(&SESSION).clone(),
            project: project(),
            stamp: stamp_utc_at(OffsetDateTime::now_utc()),
            usage: self,
        };
        if let Err(error) = Ledger::append(&entry) {
            debug::log("usage ledger", || error.to_string());
        }
    }
}

/// The append-only token ledger: `usage.jsonl` beside the sessions, with one
/// [`LedgerEntry`] per billed response.
pub struct Ledger;

impl Ledger {
    /// Append `entry` as the ledger's next line.
    fn append(entry: &LedgerEntry) -> anyhow::Result<()> {
        let path = Self::path().context("a ledger root needs $HOME")?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let line = format!("{}\n", serde_json::to_string(entry)?);
        file.write_all(line.as_bytes())
            .with_context(|| format!("writing {}", path.display()))
    }

    /// The ledger's entries, oldest first
    ///
    /// Malformed lines are ignored to prevent corruption on crash.
    fn entries() -> impl Iterator<Item = LedgerEntry> {
        let file =
            std::fs::File::open(Self::path().unwrap_or_else(|| PathBuf::from("/dev/null"))).ok();
        let lines = file.map_or_else(Vec::new, |file| {
            std::io::BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .collect::<Vec<_>>()
        });
        lines.into_iter().filter_map(|line| LedgerEntry::parse(&line))
    }

    /// The gauge's counts for the newest entry billed for `stem`, when there
    /// is one: `(input, cached, output)`.
    #[must_use]
    #[inline]
    pub fn gauge_for(stem: &str) -> Option<(u64, u64, u64)> {
        Self::entries()
            .filter(|entry| entry.session.as_deref() == Some(stem))
            .map(|entry| entry.usage.gauge())
            .last()
    }

    /// The ledger's path.
    fn path() -> Option<PathBuf> {
        #[cfg(test)]
        if let Some(root) = locked(&tests::ROOT).clone() {
            return Some(root);
        }
        let root = std::env::var_os("HOME").map(PathBuf::from)?;
        Some(root.join(".config/tart/usage.jsonl"))
    }
}

/// Remember `stem` as the session whose rounds bill next.
#[inline]
pub(crate) fn set_session(stem: String) {
    *locked(&SESSION) = Some(stem);
}

/// Forget the live session: rounds bill untagged until one is named again.
#[cfg(not(test))]
#[inline]
pub(crate) fn clear_session() {
    *locked(&SESSION) = None;
}

/// One ledger entry: a billed response and everything it bills against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct LedgerEntry {
    /// The agent tag: `"main"`, or `"child-N"` for the Nth subagent.
    pub(crate) agent: String,
    /// The model that burned the tokens.
    pub(crate) model: String,
    /// The session the round ran in, when one was named.
    pub(crate) session: Option<String>,
    /// The project: the sessions-directory slug of the working directory.
    pub(crate) project: String,
    /// When the entry was billed, `YYYY-MM-DD HH:MM:SS` UTC.
    pub(crate) stamp: String,
    /// The usage itself, flattened into the entry's own keys.
    #[serde(flatten)]
    pub(crate) usage: TokenUsage,
}

impl LedgerEntry {
    /// The entry one ledger line held, when it is shaped like one.
    fn parse(line: &str) -> Option<LedgerEntry> {
        serde_json::from_str(line).ok()
    }
}

/// The project the billed round ran in: its sessions-directory slug.
fn project() -> String {
    std::env::current_dir().map_or_else(|_| String::new(), |dir| slug(&dir))
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    /// A test's ledger root, overriding the one beside the sessions.
    pub static ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);

    /// A sample spend, as a round that burned cache would report it.
    #[inline]
    #[must_use]
    pub fn sample_usage() -> TokenUsage {
        TokenUsage {
            input: 9001,
            cached: 8214,
            output: 512,
            reasoning: 96,
            total: 9513,
        }
    }

    /// Point the ledger at `root`.
    #[inline]
    pub fn set_root(root: PathBuf) {
        *locked(&ROOT) = Some(root);
    }

    /// The live session's ledger tag, as a test reads it back.
    #[inline]
    #[must_use]
    pub fn session() -> Option<String> {
        locked(&SESSION).clone()
    }

    /// A ledger root in a fresh tempdir, held for its test's life.
    pub struct LedgerRoot(tempfile::TempDir);

    impl Drop for LedgerRoot {
        fn drop(&mut self) {
            *locked(&ROOT) = None;
        }
    }

    /// Root the ledger for one test, under its held lock.
    pub fn ledger_root() -> LedgerRoot {
        let dir = tempfile::tempdir().unwrap();
        set_root(dir.path().join("usage.jsonl"));
        LedgerRoot(dir)
    }

    /// The lock that serializes tests sharing the ledger's global slots.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// Hold the ledger's globals for one test: the root and session slots are
    /// process-wide, so the tests that touch them run one at a time.
    #[inline]
    pub fn held_for_test() -> MutexGuard<'static, ()> {
        locked(&SERIAL)
    }

    impl Ledger {
        /// The ledger's newest entry, as only a test reads one back.
        #[inline]
        #[must_use]
        pub(crate) fn last_entry() -> Option<LedgerEntry> {
            Self::entries().last()
        }
    }

    #[test]
    fn billed_rows_carry_agent_model_session_and_spend() {
        let _held = held_for_test();
        let ledger = ledger_root();
        let path = ledger.0.path().join("usage.jsonl");
        set_session("20260910-091422".to_string());
        sample_usage().bill("child-2", "glm-4.7");

        let entry = Ledger::entries().last().expect("the entry landed");
        assert_eq!(entry.agent, "child-2");
        assert_eq!(entry.model, "glm-4.7");
        assert_eq!(entry.session.as_deref(), Some("20260910-091422"));
        assert_eq!(entry.usage, sample_usage());
        // The full entry parses back: project and stamp are fields, not just
        // file bytes.
        assert!(!entry.project.is_empty(), "the entry names its project");
        assert_eq!(
            entry.stamp.len(),
            19,
            "the stamp is a full UTC moment: {}",
            entry.stamp
        );
        let line = std::fs::read_to_string(&path).unwrap();
        assert!(line.ends_with('\n'), "one write per entry");
        assert!(
            line.contains("\"project\""),
            "the entry names its project: {line}"
        );
    }

    #[test]
    fn last_for_returns_the_newest_matching_row_only() {
        let _held = held_for_test();
        let _ledger = ledger_root();
        set_session("old".to_string());
        TokenUsage { total: 1, ..sample_usage() }.bill("main", "m");
        set_session("new".to_string());
        TokenUsage { total: 2, ..sample_usage() }.bill("main", "m");
        TokenUsage { total: 3, ..sample_usage() }.bill("main", "m");

        // The newest of the session's entries, and nothing for a stem with none.
        assert_eq!(
            Ledger::entries()
                .filter(|entry| entry.session.as_deref() == Some("new"))
                .map(|entry| entry.usage.total)
                .last(),
            Some(3)
        );
        assert_eq!(
            Ledger::entries()
                .filter(|entry| entry.session.as_deref() == Some("old"))
                .map(|entry| entry.usage.total)
                .last(),
            Some(1)
        );
        assert_eq!(Ledger::gauge_for("never"), None);
        assert_eq!(Ledger::gauge_for("new"), Some((9001, 8214, 512)));
    }

    #[test]
    fn a_torn_tail_reads_as_no_row_after_the_last_good_one() {
        let _held = held_for_test();
        let ledger = ledger_root();
        let path = ledger.0.path().join("usage.jsonl");
        set_session("torn".to_string());
        sample_usage().bill("main", "m");
        let torn = std::fs::read_to_string(&path).unwrap() + r#"{"agent":"ma"#;
        std::fs::write(&path, torn).unwrap();

        assert_eq!(
            Ledger::entries()
                .filter(|entry| entry.session.as_deref() == Some("torn"))
                .map(|entry| entry.usage)
                .last(),
            Some(sample_usage()),
            "the good entry stands"
        );
    }

    #[test]
    fn the_session_slot_round_trips() {
        let _held = held_for_test();
        set_session("slot".to_string());
        assert_eq!(session().as_deref(), Some("slot"));
    }

    /// A sample entry, as one billed round writes it.
    fn sample_entry() -> LedgerEntry {
        LedgerEntry {
            agent: "child-2".to_string(),
            model: "glm-5.3".to_string(),
            session: Some("20260910-091422".to_string()),
            project: "Users-jenna-github-tart".to_string(),
            stamp: "2026-09-10 09:14:22".to_string(),
            usage: sample_usage(),
        }
    }

    #[test]
    fn a_line_is_flat_and_round_trips() {
        let entry = sample_entry();
        let line = serde_json::to_string(&entry).unwrap();

        // The five counts ride at the top level, not nested under `usage`:
        // the flattened shape every ledger file already holds.
        assert!(line.contains(r#""input":9001"#), "{line}");
        assert!(!line.contains(r#""usage""#), "no nested object: {line}");
        assert_eq!(LedgerEntry::parse(&line), Some(entry));
    }

    #[test]
    fn a_row_written_by_the_old_hand_rolled_shape_still_parses() {
        // Byte-for-byte what the pre-derive writer produced, including the
        // alphabetized keys and the null session.
        let line = r#"{"agent":"main","cached":8214,"input":9001,"model":"glm-5.3","output":512,"project":"Users-jenna-github-tart","reasoning":96,"session":null,"stamp":"2026-09-10 09:14:22","total":9513}"#;

        let entry = LedgerEntry::parse(line).expect("an existing row parses");

        assert_eq!(entry.agent, "main");
        assert_eq!(entry.session, None, "a null session reads as absent");
        assert_eq!(entry.usage, sample_usage());
        // And a mistyped field still reads as no entry, never damage.
        assert_eq!(LedgerEntry::parse(&line.replace("9001", "\"many\"")), None);
    }
}
