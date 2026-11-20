//! Append-only ledger of token usage per model and session.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

use fixed::types::I32F32;

use crate::locked;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::debug;
use crate::history::stamp_utc_at;
use crate::session::slug;

/// The live session's file stem, as [`crate::Session`] last named it.
static SESSION: Mutex<Option<String>> = Mutex::new(None);

/// The ledger's `agent` tag for the main conversation's rows.
pub(crate) const MAIN_AGENT: &str = "main";

/// One response's five token counts, as the provider measured them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
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
    /// The gauge's three display counts: `(input, cached, output)`.
    #[inline]
    #[must_use]
    pub(crate) fn gauge(self) -> (u64, u64, u64) {
        (self.input, self.cached, self.output)
    }

    /// Display the session's usage on exit.
    ///
    /// ```text
    /// Token usage so far: total=411717 input=341493 (+ 5247872 cached) output=70224 (reasoning 37081)
    /// ```
    pub(crate) fn summary(self) -> anyhow::Result<String> {
        use std::fmt::Write as _;

        let fresh = self.input.saturating_sub(self.cached);
        let mut line = format!("Token usage so far: total={} input={fresh}", fresh + self.output);
        if self.cached > 0 {
            write!(line, " (+ {} cached)", self.cached)?;
        }
        write!(line, " output={}", self.output)?;
        if self.reasoning > 0 {
            write!(line, " (reasoning {})", self.reasoning)?;
        }
        Ok(line)
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

/// A session's spend is its rows added field-wise.
impl std::iter::Sum for TokenUsage {
    #[inline]
    fn sum<I: Iterator<Item = TokenUsage>>(iter: I) -> Self {
        iter.fold(Self::default(), |sum, usage| Self {
            input: sum.input + usage.input,
            cached: sum.cached + usage.cached,
            output: sum.output + usage.output,
            reasoning: sum.reasoning + usage.reasoning,
            total: sum.total + usage.total,
        })
    }
}

/// A price in US$ per 1M tokens, held exactly.
pub type Price = I32F32;

/// A cost in US dollars.
///
/// Values enter only through checked conversions, so unrepresentable amounts are None.
pub type Money = I32F32;

/// A model's pay-as-you-go prices: USD per 1M tokens.
///
/// Agents that are not pay-as-you-go do not require a [`Cost`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// Fresh input tokens.
    pub input: Price,
    /// Cached input tokens: the discounted rate cache reads bill at.
    pub cached: Price,
    /// Output tokens, reasoning included.
    pub output: Price,
}

impl Cost {
    /// `tokens` at `price` US$/1M, as [`Money`].
    fn part(tokens: u64, price: Price) -> Option<Money> {
        // Scale before multiply so that we don't drop any microdollars.
        let millions = I32F32::checked_from_num(tokens)? / I32F32::from_num(1_000_000);
        millions.checked_mul(price)
    }
}

/// [`Money`] as a dollar string: four decimals, shedding the hundredths of a
/// cent that read as noise, so a session's groups sum to its total on
/// screen. The figure is non-negative: prices are, and token counts are.
fn dollars(money: Money) -> String {
    // Count $0.0001s. Saturating, so a figure somehow past the type's range
    // still prints rather than panics.
    let ticks = money
        .saturating_mul(Money::from_num(10_000))
        .round()
        .to_num::<i64>();
    let mut text = format!("{}.{:04}", ticks / 10_000, ticks % 10_000);
    if text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('0') {
        text.pop();
    }
    format!("${text}")
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

    /// The ledger's entries, oldest first.
    ///
    /// A row that cannot be read (e.g. bad bytes, malformed JSON) is skipped.
    fn entries() -> impl Iterator<Item = LedgerEntry> {
        let text = Self::path()
            .and_then(|path| std::fs::read(path).ok())
            .map_or_else(String::new, |bytes| String::from_utf8_lossy(&bytes).into_owned());
        text.lines()
            .filter_map(LedgerEntry::parse)
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// The gauge's counts for the newest entry the main conversation billed for
    /// `stem`, when there is one: `(input, cached, output)`.
    ///
    /// Subagent rows are skipped: a child bills under the same session but its
    /// usage feeds the child badge, never the conversation's gauge.
    #[must_use]
    #[inline]
    pub fn gauge_for(stem: &str) -> Option<(u64, u64, u64)> {
        Self::entries()
            .filter(|entry| entry.agent == MAIN_AGENT && entry.session.as_deref() == Some(stem))
            .map(|entry| entry.usage.gauge())
            .last()
    }

    /// The session's whole spend in tokens, plus its estimated cost when configured.
    ///
    /// ```text
    /// Token usage so far: total=411717 input=341493 (+ 5247872 cached) output=70224 (reasoning 37081)
    /// Estimated cost: $2.10 @ glm-5.3 (input $1.00 + cached $0.10 + output $1.00)
    /// ```
    ///
    /// This counts all rows in the session, including subagents. Models without a price
    /// are named but do not give cost entries.
    #[must_use]
    #[inline]
    pub fn exit_summary(stem: &str, prices: &BTreeMap<String, Cost>) -> Option<String> {
        let by_model = Self::session_by_model(stem);
        let total: TokenUsage = by_model.values().copied().sum();
        if total.total == 0 {
            return None;
        }
        let mut line = total
            .summary()
            .inspect_err(|error| debug::log("usage ledger", || error.to_string()))
            .ok()?;
        if let Some(cost) = cost_line(&by_model, prices) {
            line.push('\n');
            line.push_str(&cost);
        }
        Some(line)
    }

    /// The session's rows summed per model, model names alphabetical.
    fn session_by_model(stem: &str) -> BTreeMap<String, TokenUsage> {
        let mut by_model = BTreeMap::new();
        for entry in Self::entries().filter(|entry| entry.session.as_deref() == Some(stem)) {
            let sum = by_model.entry(entry.model).or_insert_with(TokenUsage::default);
            *sum = TokenUsage {
                input: sum.input + entry.usage.input,
                cached: sum.cached + entry.usage.cached,
                output: sum.output + entry.usage.output,
                reasoning: sum.reasoning + entry.usage.reasoning,
                total: sum.total + entry.usage.total,
            };
        }
        by_model
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

/// The exit summary's estimated-cost line: one group per priced model, the
/// session's total in front, unpriced models named if present.
fn cost_line(
    by_model: &BTreeMap<String, TokenUsage>,
    prices: &BTreeMap<String, Cost>,
) -> Option<String> {
    let mut total = Money::ZERO;
    let mut groups = Vec::new();
    let mut unpriced = Vec::new();
    for (model, usage) in by_model {
        let Some(price) = prices.get(model) else {
            unpriced.push(model.clone());
            continue;
        };
        let fresh = usage.input.saturating_sub(usage.cached);
        let parts = [
            (fresh, price.input),
            (usage.cached, price.cached),
            (usage.output, price.output),
        ];
        let mut cost = Money::ZERO;
        let mut labels = [String::new(), String::new(), String::new()];
        for (slot, &(tokens, rate)) in labels.iter_mut().zip(&parts) {
            let part = Cost::part(tokens, rate)?;
            cost = cost.checked_add(part)?;
            *slot = dollars(part);
        }
        total = total.checked_add(cost)?;
        let [input, cached, output] = labels;
        groups.push((
            model,
            cost,
            format!("input {input} + cached {cached} + output {output}"),
        ));
    }
    if groups.is_empty() {
        return None;
    }
    let named = groups
        .iter()
        .map(|(model, cost, parts)| {
            if groups.len() > 1 {
                format!("{model} {} ({parts})", dollars(*cost))
            } else {
                format!("{model} ({parts})")
            }
        })
        .collect::<Vec<_>>()
        .join("; ");
    let mut line = format!("Estimated cost: {} @ {named}", dollars(total));
    if !unpriced.is_empty() {
        let named: Vec<String> = unpriced
            .iter()
            .map(|model| format!("{model} (no price set)"))
            .collect();
        line.push_str("; ");
        line.push_str(&named.join("; "));
    }
    Some(line)
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

    #[test]
    fn gauge_for_ignores_subagent_rows() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("shared".to_string());
        sample_usage().bill(MAIN_AGENT, "m");
        TokenUsage {
            input: 7,
            cached: 8,
            output: 9,
            reasoning: 1,
            total: 42,
        }
        .bill("child-2", "m");

        assert_eq!(
            Ledger::gauge_for("shared"),
            Some((9001, 8214, 512)),
            "the child's newer row is not the conversation's gauge"
        );

        // Child rows only: no gauge at all, never a child's numbers.
        set_session("child-only".to_string());
        TokenUsage {
            input: 1,
            cached: 2,
            output: 3,
            reasoning: 4,
            total: 5,
        }
        .bill("child-1", "m");
        assert_eq!(Ledger::gauge_for("child-only"), None);
    }

    #[test]
    fn a_row_after_bad_bytes_is_still_read() {
        use std::io::Write as _;

        let _held = held_for_test();
        let ledger = ledger_root();
        let path = ledger.0.path().join("usage.jsonl");
        set_session("bad-bytes".to_string());
        sample_usage().bill(MAIN_AGENT, "m");

        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"\xff\xfe not utf-8\n").unwrap();
        drop(file);
        TokenUsage {
            input: 1,
            cached: 2,
            output: 3,
            reasoning: 4,
            total: 5,
        }
        .bill(MAIN_AGENT, "m");

        assert_eq!(Ledger::entries().count(), 2, "both good rows are read");
        assert_eq!(
            Ledger::gauge_for("bad-bytes"),
            Some((1, 2, 3)),
            "the row past the bad bytes is the newest"
        );
    }

    #[test]
    fn the_summary_line_keeps_codexs_shape_with_bare_digits() {
        let usage = TokenUsage {
            input: 5_589_365,
            cached: 5_247_872,
            output: 70_224,
            reasoning: 37_081,
            total: 5_659_589,
        };

        assert_eq!(
            usage.summary().expect("writing into a String cannot fail"),
            "Token usage so far: total=411717 input=341493 \
             (+ 5247872 cached) output=70224 (reasoning 37081)"
        );
    }

    #[test]
    fn zero_cached_and_reasoning_segments_drop_away() {
        let usage = TokenUsage {
            input: 10,
            cached: 0,
            output: 2,
            reasoning: 0,
            total: 12,
        };

        // The same shape codex's own snapshot test holds.
        let line = usage.summary().expect("the line formats");
        assert_eq!(line, "Token usage so far: total=12 input=10 output=2");
    }

    #[test]
    fn exit_summary_sums_the_whole_session_children_included() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("sum".to_string());
        sample_usage().bill(MAIN_AGENT, "m");
        sample_usage().bill(MAIN_AGENT, "m");
        TokenUsage {
            input: 1000,
            cached: 1000,
            output: 50,
            reasoning: 10,
            total: 1050,
        }
        .bill("child-1", "m");
        // Another session's rows stay out of this one's line.
        set_session("elsewhere".to_string());
        sample_usage().bill(MAIN_AGENT, "m");

        assert_eq!(
            Ledger::exit_summary("sum", &no_prices()).as_deref(),
            Some(
                "Token usage so far: total=2648 input=1574 \
                 (+ 17428 cached) output=1074 (reasoning 202)"
            ),
            "both main rows and the child's row count"
        );
        // A child-only session still spent; the line reports spend, not the gauge.
        set_session("child-only".to_string());
        TokenUsage {
            input: 100,
            cached: 90,
            output: 10,
            reasoning: 2,
            total: 110,
        }
        .bill("child-1", "m");
        assert_eq!(
            Ledger::exit_summary("child-only", &no_prices()).as_deref(),
            Some("Token usage so far: total=20 input=10 (+ 90 cached) output=10 (reasoning 2)")
        );
        // A session that never billed prints nothing at all.
        assert_eq!(Ledger::exit_summary("never", &no_prices()), None);
        assert_eq!(Ledger::exit_summary("", &no_prices()), None);
    }

    /// No model priced: only the token line, never a zero-dollar estimate.
    fn no_prices() -> BTreeMap<String, Cost> {
        BTreeMap::new()
    }

    /// A price as the config spells it: a decimal string, never a float.
    fn price(text: &str) -> Price {
        text.parse().expect("a valid price")
    }

    /// The three rates, spelled as the config spells them.
    fn priced(input: &str, cached: &str, output: &str) -> Cost {
        Cost {
            input: price(input),
            cached: price(cached),
            output: price(output),
        }
    }

    /// `per_million` prices chosen so every part lands on an exact micro-dollar.
    fn glm_prices() -> BTreeMap<String, Cost> {
        BTreeMap::from([("glm-5.3".to_string(), priced("1", "0.1", "2"))])
    }

    #[test]
    fn a_priced_session_estimates_its_cost() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("priced".to_string());
        // Fresh 1M input + 1M cached + 0.5M output = $1.00 + $0.10 + $1.00.
        TokenUsage {
            input: 2_000_000,
            cached: 1_000_000,
            output: 500_000,
            reasoning: 0,
            total: 2_500_000,
        }
        .bill(MAIN_AGENT, "glm-5.3");

        assert_eq!(
            Ledger::exit_summary("priced", &glm_prices()).as_deref(),
            Some(
                "Token usage so far: total=1500000 input=1000000 \
                 (+ 1000000 cached) output=500000\n\
                 Estimated cost: $2.10 @ glm-5.3 (input $1.00 + cached $0.10 + output $1.00)"
            )
        );
    }

    #[test]
    fn unpriced_models_are_named_and_cost_nothing() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("mixed".to_string());
        sample_usage().bill(MAIN_AGENT, "glm-5.3");
        sample_usage().bill("child-1", "glm-4.7");

        // Only glm-5.3's spend prices out; glm-4.7 is named for its absence.
        assert_eq!(
            Ledger::exit_summary("mixed", &glm_prices())
                .expect("the summary formats")
                .lines()
                .nth(1)
                .expect("the cost line follows the token line"),
            "Estimated cost: $0.0026 @ glm-5.3 (input $0.0008 + cached $0.0008 + output $0.001); \
             glm-4.7 (no price set)"
        );
    }

    #[test]
    fn two_priced_models_split_the_estimate_between_them() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("split".to_string());
        sample_usage().bill(MAIN_AGENT, "glm-5.3");
        TokenUsage {
            input: 1_000_000,
            cached: 0,
            output: 1_000_000,
            reasoning: 0,
            total: 2_000_000,
        }
        .bill("child-1", "glm-5.3-flash");
        let prices = BTreeMap::from([
            ("glm-5.3".to_string(), priced("1", "0.1", "2")),
            ("glm-5.3-flash".to_string(), priced("0.5", "0.05", "1")),
        ]);

        // glm-5.3: $0.002632; glm-5.3-flash: 1M fresh + 1M output = $1.50.
        // Alphabetical order, each model's own sum after its name.
        assert_eq!(
            Ledger::exit_summary("split", &prices)
                .expect("the summary formats")
                .lines()
                .nth(1)
                .expect("the cost line follows the token line"),
            "Estimated cost: $1.5026 @ glm-5.3 $0.0026 (input $0.0008 + cached $0.0008 + output $0.001); \
             glm-5.3-flash $1.50 (input $0.50 + cached $0.00 + output $1.00)"
        );
    }

    #[test]
    fn sub_cent_estimates_keep_four_decimals() {
        let _held = held_for_test();
        let _ledger = ledger_root();

        set_session("cheap".to_string());
        sample_usage().bill(MAIN_AGENT, "glm-5.3");

        // 787 + 821.4 + 1024 micro-dollars = $0.002632, not "$0.00".
        let line = Ledger::exit_summary("cheap", &glm_prices()).expect("the summary formats");
        assert!(
            line.lines()
                .nth(1)
                .expect("the cost line")
                .starts_with("Estimated cost: $0.0026"),
            "{line}"
        );
    }

    #[test]
    fn a_part_outside_the_envelope_is_declined() {
        // A token count beyond the type's range cannot be priced at all.
        assert_eq!(Cost::part(u64::MAX, price("1")), None);
        // Nor can a count × price whose product leaves the range: 2e9 tokens
        // fit, $2M/1M times their millions does not.
        assert_eq!(Cost::part(2_000_000_000, price("2000000")), None);
        // The same count in-envelope prices: 2e9 tokens at $1/1M is $2,000.
        assert_eq!(Cost::part(2_000_000_000, price("1")), Some(price("2000")));
    }
}
