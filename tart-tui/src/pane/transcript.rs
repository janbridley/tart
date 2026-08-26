//! The transcript: the message log, its entries, and the wrap cache.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[cfg(test)]
use crate::testutil::texts;

use super::DIM_STYLE;
use super::wrap::wrap_lines;

/// Stands in for a hidden thinking run.
const THINKING_HIDDEN: &str = "[Thinking… ctrl+t to toggle]";

/// A tool box's collapsed output keeps this many head and tail lines.
const TOOL_HEAD: usize = 3;
const TOOL_TAIL: usize = 2;
/// Standin for a tool that produced no output.
const TOOL_NO_OUTPUT: &str = "(no output)";
/// The bullet and name of a tool box header: running, succeeded, or failed.
const TOOL_RUNNING: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
const TOOL_OK: Style = Style::new().fg(Color::Green).add_modifier(Modifier::BOLD);
const TOOL_ERR: Style = Style::new().fg(Color::Red).add_modifier(Modifier::BOLD);

/// The row rendered in place of a hidden thinking run.
fn thinking_placeholder() -> Line<'static> {
    Line::from(Span::styled(THINKING_HIDDEN, DIM_STYLE))
}

/// One dim `⎿` output row of a tool box.
fn tool_row(text: &str) -> Line<'static> {
    Line::from(Span::styled(format!("  ⎿ {text}"), DIM_STYLE))
}

/// A status header and then the output rendered for a running tool call
///
/// Running calls show just the header with a dim ellipsis; finished ones add
/// their output rows, collapsed to [`TOOL_HEAD`] head and [`TOOL_TAIL`] tail
/// lines around a count of the hidden middle unless `expanded`. A call a later
/// one superseded only renders its header.
fn tool_lines(tool: &ToolCall, expanded: bool) -> Vec<Line<'static>> {
    let status = match (&tool.output, tool.exit) {
        (None, _) => TOOL_RUNNING,
        (_, Some(0)) => TOOL_OK,
        _ => TOOL_ERR,
    };

    let mut header = vec![
        Span::styled("● ", status),
        Span::styled(tool.name, status),
        Span::styled(format!("({})", tool.digest), DIM_STYLE),
    ];

    let Some(output) = &tool.output else {
        header.push(Span::styled(" …", DIM_STYLE));
        return vec![Line::from(header)];
    };

    if let Some(c) = tool.exit.filter(|&c| c != 0) {
        header.push(Span::styled(format!(" exit {c}"), TOOL_ERR));
    }

    // A superseded box stays folded down to its header; Ctrl+O governs only
    // the boxes still standing.
    if tool.superseded {
        return vec![Line::from(header)];
    }

    let lines: Vec<_> = output.lines().collect();
    let limit = TOOL_HEAD + TOOL_TAIL;

    let body: Vec<Line<'static>> = match lines.as_slice() {
        [] => vec![tool_row(TOOL_NO_OUTPUT)],
        _ if !expanded && lines.len() > limit => {
            let hidden = lines.len() - limit;
            lines[..TOOL_HEAD]
                .iter()
                .copied()
                .map(tool_row)
                .chain(std::iter::once(tool_row(&format!(
                    "… +{hidden} lines (ctrl+o to expand)"
                ))))
                .chain(lines[lines.len() - TOOL_TAIL..].iter().copied().map(tool_row))
                .collect()
        }
        _ => lines.into_iter().map(tool_row).collect(),
    };

    [vec![Line::from(header)], body].concat()
}

/// The display lines a run of entries renders as.
fn entry_lines(entries: &[Entry], expanded: bool) -> Vec<Line<'static>> {
    entries
        .iter()
        .flat_map(|entry| match entry {
            Entry::Text(line) => std::iter::once(line.clone()).collect::<Vec<_>>(),
            Entry::Tool(tool) => tool_lines(tool, expanded),
        })
        .collect()
}

/// One response's chain-of-thought, as a message-index range into `Transcript::messages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ThinkingRun {
    /// First message index of the run.
    start: usize,
    /// One past the last thinking message; equals `messages.len()` while the run is
    /// still the transcript tail.
    end: usize,
}

/// One tool invocation, updated in place when its output arrives.
#[derive(Clone)]
struct ToolCall {
    /// Pairs the finishing `ToolOutput` with this start.
    id: String,
    /// Display name: `Bash`, `Read`, or `Edit`.
    name: &'static str,
    /// Argument digest, e.g. `ls -la` or `src/main.rs:10-50`.
    digest: String,
    /// Combined output; `None` while the call is still running.
    output: Option<String>,
    /// The process exit code, for the status color.
    exit: Option<i32>,
    /// A later invocation folded this box down to its header line.
    superseded: bool,
}

/// One transcript message: a text line or a live tool invocation.
#[derive(Clone)]
enum Entry {
    Text(Line<'static>),
    Tool(ToolCall),
}

/// One view's message log, kept pre-wrapped to the display width:
///
/// Call `sync` to re-wrap the new content. The current response's thinking lives in
/// [`Transcript::run`]. While hidden its messages render as a placeholder block, and
/// are drained when the next response begins.
#[derive(Default)]
pub(crate) struct Transcript {
    messages: Vec<Entry>,
    /// Whether the last message is still open for `append` runs.
    open: bool,
    rows: Vec<Line<'static>>,
    /// (width, how many *visible* messages `rows` already contains).
    cache: (usize, usize),
    /// The current response's chain-of-thought, if one has begun. Tool starts
    /// rotate it below themselves, so it always rides under the tool boxes.
    run: Option<ThinkingRun>,
    /// Whether the thinking run renders; sticky across turns. Starts hidden.
    show_thinking: bool,
    /// Whether tool outputs render in full; sticky, flipped by Ctrl+O. Starts
    /// collapsed.
    show_tool_output: bool,
}

impl Transcript {
    /// Append a committed line, ending any append-run.
    pub(crate) fn push(&mut self, line: impl Into<Line<'static>>) {
        self.messages.push(Entry::Text(line.into()));
        self.open = false;
    }

    /// Record a tool invocation's start; it renders as a running header until
    /// finished. The call supersedes every finished box before it, folding each
    /// down to its header line, and rotates the thinking run below itself so
    /// the chain-of-thought always renders under the tool boxes.
    pub(crate) fn start_tool(&mut self, id: String, name: &'static str, digest: String) {
        let fold = self.messages.iter().position(
            |entry| matches!(entry, Entry::Tool(tool) if tool.output.is_some() && !tool.superseded),
        );
        // The rotation below also moves the run's messages, so the stale-row
        // span starts at whichever of the two comes first.
        let run_start = self.run.filter(|run| run.start < run.end).map(|run| run.start);
        let stale = [fold, run_start].into_iter().flatten().min();
        // Rewind before flipping the flags: the stale-row count must reflect the
        // bodies the flags are about to hide.
        if let Some(index) = stale {
            self.rewind(index);
        }
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && tool.output.is_some()
            {
                tool.superseded = true;
            }
        }
        self.messages.push(Entry::Tool(ToolCall {
            id,
            name,
            digest,
            output: None,
            exit: None,
            superseded: false,
        }));
        // Late thinking fragments then extend the run in place, under the
        // boxes, instead of splicing back above them; an empty run moves only
        // its markers, so thinking that starts after the call still opens
        // below the box.
        if let Some(run) = &mut self.run {
            let span = run.end - run.start;
            self.messages[run.start..].rotate_left(span);
            run.end = self.messages.len();
            run.start = run.end - span;
        }
        self.open = false;
    }

    /// Fill in the pending invocation with `id`, then refold from its box.
    pub(crate) fn finish_tool(&mut self, id: &str, output: String, exit: Option<i32>) {
        let Some(index) = self.messages.iter().rposition(
            |entry| matches!(entry, Entry::Tool(tool) if tool.output.is_none() && tool.id == id),
        ) else {
            return;
        };
        // The box folds differently once tools finished, so the rows `sync` folded
        // `messages[index..]` are stale. Rewind the fold point so the next `sync`
        // refolds just the tail.
        self.rewind(index);
        let Some(Entry::Tool(tool)) = self.messages.get_mut(index) else {
            return;
        };
        tool.output = Some(output);
        tool.exit = exit;
    }

    /// Drop the cached rows of `messages[index..]` so the next `sync` refolds
    /// them. A hidden thinking run inside the span forces a full refold: the
    /// placeholder means the raw row count would not match what `sync` folded.
    fn rewind(&mut self, index: usize) {
        if self.cache.0 == 0 || index >= self.cache.1 {
            return;
        }
        if (index..self.cache.1).any(|i| self.thinking_hidden(i)) {
            // Hidden thinking requires us to refold EVERYTHING for correctness.
            self.rows.clear();
            self.cache.1 = 0;
        } else {
            let stale = wrap_lines(
                &entry_lines(&self.messages[index..self.cache.1], self.show_tool_output),
                self.cache.0,
            )
            .len();
            self.rows.truncate(self.rows.len() - stale);
            self.cache.1 = index;
        }
    }

    /// Resolve every still-running invocation as failed to prevent stuck boxes.
    pub(crate) fn fail_pending(&mut self, reason: &str) {
        let mut failed = false;
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && tool.output.is_none()
            {
                tool.output = Some(reason.to_string());
                tool.exit = None;
                failed = true;
            }
        }
        if failed {
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// Expand or collapse every tool output.
    pub(crate) fn toggle_expand(&mut self) {
        if self.messages.iter().any(|entry| matches!(entry, Entry::Tool(_))) {
            self.show_tool_output = !self.show_tool_output;
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// The last message when it is text, for append-run gluing.
    fn text_last(&self) -> Option<&Line<'static>> {
        match self.messages.last() {
            Some(Entry::Text(line)) => Some(line),
            _ => None,
        }
    }

    /// The last message when it is text, mutably, for span extension.
    fn text_last_mut(&mut self) -> Option<&mut Line<'static>> {
        match self.messages.last_mut() {
            Some(Entry::Text(line)) => Some(line),
            _ => None,
        }
    }

    /// Append a streaming fragment, gluing onto the previous fragment while style matches.
    ///
    /// Newlines in the text end the current line.
    pub(crate) fn append(&mut self, span: &Span<'static>) {
        if self.run.is_some_and(|run| run.end == self.messages.len()) {
            self.break_line();
        }
        self.append_lines(&span.content, span.style);
    }

    /// Append one fragment's text: every `\n` ends the line, and an empty part
    ///
    /// Newlines trailing the fragment stay pending, and only render once a following
    /// newline closes the empty whitespace.
    fn append_lines(&mut self, content: &str, style: Style) {
        if content.is_empty() {
            return;
        }
        let count = content.split('\n').count();
        for (i, part) in content.split('\n').enumerate() {
            (i > 0).then(|| self.break_line());
            if !part.is_empty() {
                self.append_fragment(Span::styled(part.to_string(), style));
            } else if !self.open && i + 1 < count {
                // A line already broke before us and another newline follows.
                self.append_fragment(Span::styled(String::new(), style));
            }
        }
    }

    /// Glue one unbroken fragment onto the transcript.
    fn append_fragment(&mut self, span: Span<'static>) {
        let glue = self.open
            && self
                .text_last()
                .is_some_and(|line| line.spans.last().is_some_and(|last| last.style == span.style));
        if glue {
            // The cache already counted the line being extended: drop its stale rows
            // and hand the message back for the next sync (unless it is hidden
            // thinking, whose rows were never in `rows`)
            if self.cache.1 == self.messages.len()
                && self.cache.0 > 0
                && !self.thinking_hidden(self.messages.len() - 1)
            {
                let stale = wrap_lines(
                    &entry_lines(&self.messages[self.messages.len() - 1..], self.show_tool_output),
                    self.cache.0,
                )
                .len();
                self.rows.truncate(self.rows.len() - stale);
                self.cache.1 -= 1;
            }
            if let Some(last) = self.text_last_mut().and_then(|l| l.spans.last_mut()) {
                // Extend the last matching span if available to save memory.
                last.content.to_mut().push_str(&span.content);
            }
        } else {
            self.messages.push(Entry::Text(Line::from(span)));
            self.open = true;
        }
    }

    /// End the current append-run; later appends start a fresh line.
    fn break_line(&mut self) {
        self.open = false;
    }

    /// Append a chain-of-thought fragment into the current thinking run.
    pub(crate) fn append_thinking(&mut self, span: &Span<'static>) {
        // Open a run if none exists yet (e.g. thinking after a `/clear`).
        let at = self.messages.len();
        self.run.get_or_insert(ThinkingRun { start: at, end: at });
        // The answer already started: move the fragment back above it
        let late = self.run.is_some_and(|run| run.end < self.messages.len());
        if late && span.content.split('\n').all(str::is_empty) {
            // Nothing to splice back: leave the wrap cache and the answer run alone
            return;
        }
        let before = self.messages.len();
        // Skip gluing thinking if we have a late thinking fragment or the run is empty
        if late || self.run.is_some_and(|run| run.start == run.end) {
            self.break_line();
        }
        self.append_lines(&span.content, span.style);
        let Some(run) = &mut self.run else {
            return;
        };
        if late {
            let end = run.end;
            let count = self.messages.len() - before;
            // Rotate the fresh messages back above the tail that followed the run.
            self.messages[end..].rotate_left(before - end);
            run.end = end + count;
            // Rewrap from the splice point, dropping only the rows the already-folded
            // tail occupied.
            if self.cache.0 > 0 && self.cache.1 > end {
                let stale = wrap_lines(
                    &entry_lines(
                        &self.messages[run.end..self.cache.1 + count],
                        self.show_tool_output,
                    ),
                    self.cache.0,
                )
                .len();
                self.rows.truncate(self.rows.len() - stale);
            }
            self.cache.1 = self.cache.1.min(end);
        } else {
            run.end = self.messages.len();
        }
    }

    /// Show or hide the current response's chain-of-thought.
    pub(crate) fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
        if self.run.is_some_and(|run| run.start < run.end) {
            // The run's rows sit mid-`rows`; rebuild from scratch.
            self.rows.clear();
            self.cache.1 = 0;
        }
    }

    /// Retire the previous response's thinking and open a fresh, empty run.
    pub(crate) fn begin_response(&mut self) {
        if let Some(run) = self.run.take()
            && run.start < run.end
        {
            self.messages.drain(run.start..run.end);
        }
        // Rows that included the drained messages are stale; rewrapping once
        // per turn is fine.
        self.rows.clear();
        self.cache.1 = 0;
        // Never glue onto the retired turn's tail.
        self.open = false;
        let at = self.messages.len();
        self.run = Some(ThinkingRun { start: at, end: at });
    }

    /// Roll the log back to `entries` messages, planting a fresh thinking run.
    pub(crate) fn restore_to(&mut self, entries: usize) {
        self.messages.truncate(entries);
        self.rows.clear();
        self.cache.1 = 0;
        self.run = Some(ThinkingRun { start: entries, end: entries });
        self.open = false;
    }

    /// Messages so far; the point a cancelled turn rewinds to.
    pub(crate) fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Whether message `i` belongs to the hidden thinking run.
    fn thinking_hidden(&self, i: usize) -> bool {
        !self.show_thinking && self.run.is_some_and(|run| i >= run.start && i < run.end)
    }

    /// Drop every message and reset our caches, persisting the thinking preference
    pub(crate) fn clear(&mut self) {
        self.messages.clear();
        self.rows.clear();
        self.cache.1 = 0;
        self.open = false;
        self.run = None;
    }

    /// Wrap the visible messages not yet folded into `rows`; a width change rewraps
    /// everything. A hidden thinking run renders as its placeholder.
    pub(crate) fn sync(&mut self, width: usize) -> &[Line<'static>] {
        if self.cache.0 != width {
            self.rows.clear();
            self.cache = (width, 0);
        }
        let expanded = self.show_tool_output;
        let done = self.cache.1;
        if let Some(run) = &self.run
            && !self.show_thinking
            && run.start < run.end
        {
            let (start, end) = (run.start, run.end);
            self.rows.extend(wrap_lines(
                &entry_lines(&self.messages[done.min(start)..start], expanded),
                width,
            ));
            // If the fold point is before the run we need to render, otherwise skip.
            if start >= done {
                self.rows.extend(wrap_lines(&[thinking_placeholder()], width));
            }
            self.rows.extend(wrap_lines(
                &entry_lines(&self.messages[end.max(done)..], expanded),
                width,
            ));
        } else {
            self.rows
                .extend(wrap_lines(&entry_lines(&self.messages[done..], expanded), width));
        }
        self.cache.1 = self.messages.len();
        &self.rows
    }

    /// The wrapped rows; current as of the last `sync`.
    pub(crate) fn rows(&self) -> &[Line<'static>] {
        &self.rows
    }

    /// The text of the transcript's plain messages, tool boxes aside.
    #[cfg(test)]
    pub(crate) fn message_texts(&self) -> Vec<String> {
        self.messages
            .iter()
            .filter_map(|entry| match entry {
                Entry::Text(line) => Some(
                    line.spans
                        .iter()
                        .map(|span| span.content.as_ref())
                        .collect::<String>(),
                ),
                Entry::Tool(_) => None,
            })
            .collect()
    }

    /// The cached rows equal a full re-wrap of every message at the wrapped width.
    #[cfg(test)]
    pub(crate) fn assert_rows_match_full_rewrap(&self) {
        assert_eq!(
            texts(&self.rows),
            texts(&wrap_lines(
                &entry_lines(&self.messages, self.show_tool_output),
                self.cache.0
            ))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The messages the transcript renders: the log with a hidden run
    /// replaced by its placeholder row.
    fn visible(t: &Transcript) -> Vec<Line<'static>> {
        let mut entries = t.messages.clone();
        if let Some(run) = t.run
            && !t.show_thinking
            && run.start < run.end
        {
            entries.splice(
                run.start..run.end,
                std::iter::once(Entry::Text(thinking_placeholder())),
            );
        }
        entry_lines(&entries, t.show_tool_output)
    }

    /// Start a pending `Bash(echo hi)` invocation, as the pane would on a
    /// `ToolStart` event.
    fn start_bash(t: &mut Transcript, id: &str) {
        t.start_tool(id.to_string(), "Bash", "echo hi".to_string());
    }

    /// Fragments of one streaming run land on one growing line, and a style change
    /// starts a new one without any caller-side state.
    #[test]
    fn appends_glue_by_style() {
        let dim = Style::new().fg(Color::DarkGray);
        let mut transcript = Transcript::default();
        transcript.push(Line::from("prompt"));
        transcript.append(&Span::raw("Hel"));
        transcript.append(&Span::raw("lo "));
        transcript.append(&Span::raw("world"));
        transcript.append(&Span::styled(" (thinking)", dim));
        transcript.append(&Span::styled(" more", dim));
        assert_eq!(
            transcript.message_texts(),
            ["prompt", "Hello world", " (thinking) more"]
        );

        // `push` and `break_line` both end the run.
        transcript.push(Line::from("committed"));
        transcript.append(&Span::raw("after"));
        transcript.break_line();
        transcript.append(&Span::raw("again"));
        assert_eq!(
            transcript.message_texts(),
            [
                "prompt",
                "Hello world",
                " (thinking) more",
                "committed",
                "after",
                "again"
            ]
        );
    }

    /// Blank lines should survive streaming.
    #[test]
    fn blank_lines_render_across_fragments() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking(&Span::styled("hmm\n\nhmm", DIM_STYLE));
        t.append(&Span::raw("one\n\ntwo\n")); // blank inside one fragment
        t.append(&Span::raw("three\n")); // a lone trailing newline…
        t.append(&Span::raw("\nfour\n")); // …that a leading `\n` doubles into a blank
        t.append(&Span::raw("five\n")); // another lone newline…
        t.append(&Span::raw("six")); // …only breaks the line
        t.append(&Span::raw("\n\neven")); // after open text: first `\n` ends it, second blanks
        assert_eq!(
            t.message_texts(),
            [
                "hmm", "", "hmm", "one", "", "two", "three", "", "four", "five", "six", "", "even"
            ]
        );
        t.sync(40);
        // Thinking renders hidden by default: its placeholder stands in.
        assert_eq!(
            texts(&t.rows),
            [
                THINKING_HIDDEN,
                "one",
                "",
                "two",
                "three",
                "",
                "four",
                "five",
                "six",
                "",
                "even"
            ]
        );
    }

    /// Whatever pushes, appends, or width changes happen between renders, the cached
    /// rows always equal a full re-wrap
    #[test]
    fn wrap_cache_always_matches_a_full_rewrap() {
        let mut transcript = Transcript::default();
        for i in 0..5 {
            transcript.push(Line::from(format!("message {i} aaaa bbbb cccc dddd")));
        }
        let assert_fresh = |transcript: &Transcript| {
            assert_eq!(
                texts(&transcript.rows),
                texts(&wrap_lines(
                    &entry_lines(&transcript.messages, transcript.show_tool_output),
                    transcript.cache.0
                ))
            );
        };
        transcript.sync(20);
        assert_eq!(transcript.cache, (20, 5));
        assert_fresh(&transcript);

        transcript.push(Line::from("tail")); // between renders
        transcript.sync(20);
        assert_fresh(&transcript);

        transcript.sync(80); // width change rebuilds
        assert_eq!(transcript.cache, (80, 6));
        assert_fresh(&transcript);

        transcript.append(&Span::raw("streaming aaaa bbbb")); // glued run
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.append(&Span::raw(" cccc dddd"));
        transcript.sync(80);
        assert_fresh(&transcript);

        // Tool boxes mutate mid-log: the running header, then the finished block.
        start_bash(&mut transcript, "call_0");
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.finish_tool("call_0", "one\ntwo\nthree\n".to_string(), Some(0));
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.toggle_expand();
        transcript.sync(80);
        assert_fresh(&transcript);

        transcript.clear(); // hidden pane: re-push to the same count
        for i in 0..6 {
            transcript.push(Line::from(format!("fresh {i}")));
        }
        transcript.sync(80);
        assert_fresh(&transcript);
        assert!(!texts(&transcript.rows).iter().any(|row| row.contains("aaaa")));
    }

    #[test]
    fn wrap_cache_matches_a_full_rewrap_while_hidden() {
        let mut t = Transcript::default();
        assert!(!t.show_thinking, "thinking starts hidden");
        let assert_fresh = |t: &Transcript| {
            assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(t), t.cache.0)));
        };

        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled("hmm aaaa bbbb", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" cccc dddd", DIM_STYLE)); // glued
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled("line two\nline three", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);
        t.append(&Span::raw("the answer aaaa bbbb"));
        t.sync(20);
        assert_fresh(&t);

        // A tool box lands after the answer, finishes, and round-two reasoning
        // splices back above both.
        start_bash(&mut t, "c0");
        t.sync(20);
        assert_fresh(&t);
        t.finish_tool("c0", "out aaaa".to_string(), Some(0));
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" mid", DIM_STYLE));
        t.sync(20);
        assert_fresh(&t);

        t.sync(80); // width change rebuilds
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" late", DIM_STYLE)); // splices above the answer
        t.sync(80);
        assert_fresh(&t);

        t.toggle_thinking(); // reveal
        t.sync(80);
        assert_fresh(&t);
        t.append_thinking(&Span::styled(" more", DIM_STYLE));
        t.sync(80);
        assert_fresh(&t);
        t.toggle_thinking(); // and hide again
        t.sync(80);
        assert_fresh(&t);

        t.begin_response(); // retirement drains the run
        t.sync(80);
        assert_fresh(&t);

        t.clear();
        t.sync(80);
        assert_fresh(&t);
    }

    #[test]
    fn toggle_hides_then_shows_the_latest_thinking() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        // Long enough to wrap: hiding must shrink the row count.
        let reasoning = "secret reasoning ".repeat(6);
        t.append_thinking(&Span::styled(reasoning, DIM_STYLE));
        t.append(&Span::raw("the answer"));

        t.toggle_thinking(); // reveal
        t.sync(40);
        let shown = texts(&t.rows);
        assert!(shown.iter().any(|row| row.contains("secret reasoning")));
        assert!(shown.iter().any(|row| row.contains("the answer")));

        t.toggle_thinking(); // hide: a placeholder replaces the reasoning
        t.sync(40);
        let hidden = texts(&t.rows);
        assert!(hidden.len() < shown.len());
        assert!(hidden.iter().any(|row| row.contains("Thinking")));
        assert!(hidden.iter().all(|row| !row.contains("secret reasoning")));
        assert!(hidden.iter().any(|row| row.contains("the answer")));

        t.toggle_thinking(); // and back: the rewrap is byte-identical
        t.sync(40);
        assert_eq!(texts(&t.rows), shown);
    }

    #[test]
    fn retirement_drains_old_thinking_only() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ one"));
        t.begin_response();
        t.append_thinking(&Span::styled("old reasoning", DIM_STYLE));
        t.append(&Span::raw("old answer"));
        t.push(Line::from("❯ two"));
        t.begin_response();
        let messages = t.message_texts();
        assert!(messages.iter().all(|m| !m.contains("old reasoning")));
        assert!(messages.iter().any(|m| m.contains("old answer")));
        assert!(messages.iter().any(|m| m.contains("❯ two")));

        t.append_thinking(&Span::styled("new reasoning", DIM_STYLE));
        t.append(&Span::raw("new answer"));
        t.push(Line::from("❯ three"));
        t.begin_response();
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().all(|r| !r.contains("reasoning")));
        assert!(!rows.iter().any(|r| r.contains("Thinking")), "empty run");
        assert!(rows.iter().any(|r| r.contains("old answer")));
        assert!(rows.iter().any(|r| r.contains("new answer")));
    }

    /// The dim error line the `Failed` path appends must never end up inside the drain
    #[test]
    fn error_line_survives_retirement() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking(&Span::styled("doomed reasoning", DIM_STYLE));
        // Same dim style as the thinking: without the append boundary it
        // would glue into the run and drain away with it.
        t.append(&Span::styled("boom: network down", DIM_STYLE));
        assert_eq!(t.message_texts(), ["doomed reasoning", "boom: network down"]);
        t.begin_response();
        let messages = t.message_texts();
        assert!(messages.iter().any(|m| m.contains("boom")));
        assert!(messages.iter().all(|m| !m.contains("doomed")));
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("boom")));
    }

    #[test]
    fn late_thinking_stays_contiguous() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.append_thinking(&Span::styled("t2", DIM_STYLE));
        assert_eq!(t.message_texts(), ["t1", "t2", "a1"]);
        let run = t.run.expect("run");
        assert_eq!((run.start, run.end), (0, 2));

        t.sync(40); // hidden (default): both thinking messages collapse
        let hidden = texts(&t.rows);
        assert_eq!(hidden.len(), 2); // placeholder + answer
        assert!(hidden.iter().any(|r| r.contains("Thinking")));
        assert!(hidden.iter().all(|r| !r.contains("t1") && !r.contains("t2")));

        t.toggle_thinking();
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));

        t.begin_response();
        assert_eq!(t.message_texts(), ["a1"]);
    }

    /// Gluing a hidden thinking fragment must not truncate the cached rows of
    /// the visible messages around it.
    #[test]
    fn hidden_glue_does_not_eat_visible_rows() {
        let mut t = Transcript::default();
        t.push(Line::from("prompt"));
        t.begin_response();
        t.append_thinking(&Span::styled("abc", DIM_STYLE));
        t.sync(20);
        t.append_thinking(&Span::styled("def", DIM_STYLE)); // glue, cache primed
        t.sync(20);
        t.append(&Span::raw("answer"));
        t.sync(20);
        assert_eq!(
            texts(&t.rows),
            texts(&wrap_lines(
                &[Line::from("prompt"), thinking_placeholder(), Line::from("answer")],
                20
            ))
        );
    }

    /// Make sure late fragments (reasoning AFTER text) rewrap only the spliced data.
    #[test]
    fn late_fragments_rewrap_from_the_splice_point() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.sync(40);
        t.append_thinking(&Span::styled("t2", DIM_STYLE)); // late, tail folded
        assert_eq!(t.cache, (40, 2), "the pre-run messages stay cached");
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));

        // A second late fragment with an unsynced glued answer in between.
        t.append(&Span::raw(" a2"));
        t.append_thinking(&Span::styled("t3", DIM_STYLE));
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "t2", "t3", "a1 a2"]);
    }

    /// A late fragment without text shouldn't break the cache or current run.
    #[test]
    fn empty_late_fragments_change_nothing() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        t.sync(40);
        let (rows, cache) = (texts(&t.rows), t.cache);
        t.append_thinking(&Span::styled("\n\n", DIM_STYLE));
        assert_eq!(t.cache, cache);
        assert_eq!(texts(&t.rows), rows);
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "a1"]);
        // The answer is still open: later text joins its message.
        t.append(&Span::raw(" more"));
        t.sync(40);
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "a1 more"]);
    }

    /// Tool boxes render, update in place, collapse, and color by outcome.
    #[test]
    fn tool_calls_render_update_and_collapse() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ run it"));
        t.begin_response();
        start_bash(&mut t, "call_0");
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("● Bash(echo hi) …")));

        t.finish_tool("call_0", "hi\n".to_string(), Some(0));
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("● Bash(echo hi)")));
        assert!(!rows.iter().any(|row| row.contains('…')));
        assert!(rows.iter().any(|row| row.contains("⎿ hi")));

        // A long output collapses to head + count + tail; Ctrl+O expands it.
        start_bash(&mut t, "call_1");
        let mut long = String::new();
        for i in 0..20 {
            long.push_str("line ");
            long.push_str(&i.to_string());
            long.push('\n');
        }
        t.finish_tool("call_1", long, Some(0));
        t.sync(40);
        let collapsed = texts(&t.rows);
        assert!(
            collapsed
                .iter()
                .any(|row| row.contains("… +15 lines (ctrl+o to expand)"))
        );
        assert!(!collapsed.iter().any(|row| row.contains("line 10")));

        t.toggle_expand();
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains("line 10")));
        t.toggle_expand();
        t.sync(40);
        assert!(!texts(&t.rows).iter().any(|row| row.contains("line 10")));

        // A failure shows its code, an empty output says so, and an unknown id
        // finishes nothing.
        start_bash(&mut t, "call_2");
        t.finish_tool("call_2", String::new(), Some(1));
        t.finish_tool("call_404", "nope".to_string(), Some(0));
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("exit 1")));
        assert!(rows.iter().any(|row| row.contains(TOOL_NO_OUTPUT)));
        assert!(!rows.iter().any(|row| row.contains("nope")));
    }

    /// A new call folds every finished box before it down to its header line,
    /// Ctrl+O or not; the newest box keeps its body until it is folded too.
    #[test]
    fn new_calls_fold_finished_boxes_to_their_headers() {
        let mut t = Transcript::default();
        let assert_fresh = |t: &Transcript| {
            assert_eq!(
                texts(&t.rows),
                texts(&wrap_lines(
                    &entry_lines(&t.messages, t.show_tool_output),
                    t.cache.0
                ))
            );
        };
        t.push(Line::from("❯ run it"));
        t.begin_response();
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "one\ntwo\nthree\n".to_string(), Some(0));
        t.sync(40);
        assert_fresh(&t);
        assert!(texts(&t.rows).iter().any(|row| row.contains("⎿ one")));

        // The second call folds the first: only the two headers render.
        t.start_tool("call_1".to_string(), "Bash", "ls -la".to_string());
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("● Bash(echo hi)")));
        assert!(rows.iter().any(|row| row.contains("● Bash(ls -la) …")));
        assert!(!rows.iter().any(|row| row.contains('⎿')));

        // The newest box keeps its collapsed body; the folded one stays hidden.
        t.finish_tool(
            "call_1",
            "out aaaa\nbbbb\ncccc\ndddd\neeee\nffff\n".to_string(),
            Some(0),
        );
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("⎿ out aaaa")));
        assert!(!rows.iter().any(|row| row.contains("⎿ dddd")));
        assert!(!rows.iter().any(|row| row.contains("⎿ one")));

        // Ctrl+O expands the standing box only; the folded one stays a header.
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("⎿ dddd")));
        assert!(!rows.iter().any(|row| row.contains("⎿ one")));
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        assert!(!texts(&t.rows).iter().any(|row| row.contains("⎿ dddd")));
    }

    /// A failed generation resolves its still-running boxes instead of
    /// leaving them pending forever.
    #[test]
    fn failed_generations_resolve_pending_tools() {
        let mut t = Transcript::default();
        start_bash(&mut t, "call_0");
        t.sync(40);
        assert!(texts(&t.rows).iter().any(|row| row.contains('…')));

        t.fail_pending("generation panicked");
        t.sync(40);
        let rows = texts(&t.rows);
        assert!(rows.iter().any(|row| row.contains("generation panicked")));
        assert!(!rows.iter().any(|row| row.contains('…')));
    }

    /// Text streamed after a tool box never glues onto it.
    #[test]
    fn appends_after_a_tool_start_a_new_line() {
        let mut t = Transcript::default();
        t.append(&Span::raw("answer"));
        start_bash(&mut t, "call_0");
        t.append(&Span::raw("more"));
        assert_eq!(t.message_texts(), ["answer", "more"]);
    }

    /// A tool start rotates the thinking run below the new box, and later
    /// fragments extend it there; the cache stays honest through toggles and
    /// width changes.
    #[test]
    fn thinking_rides_below_tool_boxes() {
        let mut t = Transcript::default();
        let assert_fresh = |t: &Transcript| {
            assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(t), t.cache.0)));
        };
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append_thinking(&Span::styled("t1", DIM_STYLE));
        t.append(&Span::raw("a1"));
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "out\n".to_string(), Some(0));
        t.sync(40);
        assert_fresh(&t);
        // t1 rotated below the box when the call started, placeholder and all.
        assert_eq!(t.message_texts(), ["❯ go", "a1", "t1"]);
        assert!(texts(&t.rows).iter().any(|row| row.contains("Thinking")));

        t.append_thinking(&Span::styled("t2", DIM_STYLE)); // extends the run below the box
        start_bash(&mut t, "call_1"); // a second box stacks above the run
        t.sync(40);
        assert_fresh(&t);
        assert_eq!(t.message_texts(), ["❯ go", "a1", "t1", "t2"]);
        // Every header must precede the placeholder in the rendered rows.
        let rows = texts(&t.rows);
        let last_box = rows
            .iter()
            .rposition(|row| row.contains("Bash"))
            .expect("box headers");
        let think = rows
            .iter()
            .position(|row| row.contains("Thinking"))
            .expect("placeholder");
        assert!(last_box < think, "{rows:?}");

        t.toggle_thinking();
        t.sync(40);
        assert_fresh(&t);
        t.toggle_expand();
        t.sync(40);
        assert_fresh(&t);
        t.sync(80);
        assert_fresh(&t);
    }

    #[test]
    fn empty_runs_and_clear_reset_state() {
        let mut t = Transcript::default();
        t.push(Line::from("prompt"));
        t.begin_response();
        t.sync(20);
        // An empty run renders no placeholder; toggling it changes nothing.
        let rows = texts(&t.rows);
        assert_eq!(rows, ["prompt"]);
        t.toggle_thinking();
        t.sync(20);
        assert_eq!(texts(&t.rows), rows);

        t.append_thinking(&Span::styled("reasoning", DIM_STYLE));
        t.clear();
        assert!(t.messages.is_empty() && t.rows.is_empty());
        assert!(t.run.is_none());
        // Toggled on above: `clear` keeps the preference rather than the default.
        assert!(t.show_thinking, "clear keeps the sticky preference");

        // Thinking after a clear lazily re-opens a drainable run.
        t.append_thinking(&Span::styled("again", DIM_STYLE));
        assert!(t.run.is_some_and(|run| run.start == 0 && run.end == 1));
        t.begin_response();
        assert!(t.messages.is_empty(), "the re-opened run drains");
    }
}
