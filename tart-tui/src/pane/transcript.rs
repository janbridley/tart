//! The transcript: the message log, its entries, and the wrap cache.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use tart_agents::merge_digests;

#[cfg(test)]
use crate::testutil::texts;

use super::markdown;
use super::wrap::wrap_lines;
use super::{DIM_STYLE, HIGHLIGHT_STYLE};

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

/// The highlighted row rendered in place of a hidden thinking run.
fn thinking_placeholder() -> Line<'static> {
    Line::from(Span::styled(THINKING_HIDDEN, HIGHLIGHT_STYLE))
}

/// One dim `⎿` output row of a tool box.
fn tool_row(text: &str) -> Line<'static> {
    Line::from(Span::styled(format!("  ⎿ {text}"), DIM_STYLE))
}

/// The `⎿` gutter row counting a collapsed middle, with the count and the
/// expand hint highlighted.
fn tool_hint_row(hidden: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ⎿ ", DIM_STYLE),
        Span::styled(format!("… +{hidden} lines (ctrl+o to expand)"), HIGHLIGHT_STYLE),
    ])
}

/// A status header and then the output rendered for a running tool call
///
/// Running calls show just the header with a dim ellipsis; finished ones add
/// their output rows, collapsed to [`TOOL_HEAD`] head and [`TOOL_TAIL`] tail
/// lines around a count of the hidden middle unless `expanded`. A call a later
/// one superseded only renders its header.
fn tool_lines(tool: &ToolCall, expanded: bool) -> Vec<Line<'static>> {
    let status = match (tool.running, tool.exit) {
        (true, _) => TOOL_RUNNING,
        (_, Some(0)) => TOOL_OK,
        _ => TOOL_ERR,
    };

    let digest = Span::styled(
        format!("({})", merge_digests(tool.name, &tool.digests)),
        DIM_STYLE,
    );
    let mut header = vec![
        Span::styled("● ", status),
        Span::styled(tool.name, status),
        digest,
    ];

    if tool.running {
        header.push(Span::styled(" …", DIM_STYLE));
    }
    if !tool.running
        && let Some(c) = tool.exit.filter(|&c| c != 0)
    {
        header.push(Span::styled(format!(" exit {c}"), TOOL_ERR));
    }

    // A superseded box stays folded down to its header; Ctrl+O governs only
    // the boxes still standing. A merged box that reopens keeps its previous
    // output under the running header, so edits can be drawn in place.
    if tool.superseded {
        return vec![Line::from(header)];
    }

    let Some(output) = &tool.output else {
        return vec![Line::from(header)];
    };

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
                .chain(std::iter::once(tool_hint_row(hidden)))
                .chain(lines[lines.len() - TOOL_TAIL..].iter().copied().map(tool_row))
                .collect()
        }
        _ => lines.into_iter().map(tool_row).collect(),
    };

    [vec![Line::from(header)], body].concat()
}

impl Entry {
    /// The display lines the entry renders as. Stale entries render immediately
    fn lines(&self, expanded: bool, thinking: bool) -> Vec<Line<'static>> {
        match self {
            Entry::Text(line) => vec![line.clone()],
            Entry::Tool(tool) => tool_lines(tool, expanded),
            Entry::Answer { raw, width, lines } => {
                if *width == 0 {
                    markdown::render(raw, 0)
                } else {
                    lines.clone()
                }
            }
            Entry::Thinking { raw } => thinking_lines(raw, thinking),
        }
    }
}

/// The lines a thinking block renders: dim when shown or a placeholder otherwise.
fn thinking_lines(raw: &str, shown: bool) -> Vec<Line<'static>> {
    let dim = |text: &str| {
        let mut line = Line::from(text.to_owned());
        line.style = DIM_STYLE;
        line
    };
    if !shown {
        return raw
            .is_empty()
            .then(Vec::new)
            .unwrap_or_else(|| vec![thinking_placeholder()]);
    }
    let mut parts = raw.split('\n').collect::<Vec<_>>();
    while parts.last() == Some(&"") {
        parts.pop();
    }
    let mut lines = Vec::with_capacity(parts.len());
    let mut blank = false;
    for part in parts {
        if part.is_empty() {
            if !blank {
                lines.push(dim(""));
            }
            blank = true;
        } else {
            lines.push(dim(part));
            blank = false;
        }
    }
    lines
}

/// One tool invocation, updated in place when its output arrives.
#[derive(Clone)]
struct ToolCall {
    /// Pairs the finishing `ToolOutput` with this start.
    id: String,
    /// Display name: `Bash`, `Read`, or `Edit`.
    name: &'static str,
    /// The argument digest of each call in the run, e.g. `ls -la` or `main.rs:10-50`
    digests: Vec<String>,
    /// Whether a call is in flight; `false` once finished, which always fills `output`
    running: bool,
    /// Combined output; `None` until the first call of the run finishes.
    output: Option<String>,
    /// The process exit code, for the status color.
    exit: Option<i32>,
    /// A later invocation folded this box down to its header line.
    superseded: bool,
}

/// One transcript message: a text line, a live tool invocation, a streamed
/// answer, or the turn's thinking block.
#[derive(Clone)]
enum Entry {
    Text(Line<'static>),
    Tool(ToolCall),
    /// The model's answer text, rendered as markdown by `sync`.
    Answer {
        /// Every fragment verbatim, with renderer-defined newline style.
        raw: String,
        /// The width `lines` were rendered at, or 0 if stale.
        width: usize,
        /// The rows [`markdown::render`] produced for `raw` at `width`.
        lines: Vec<Line<'static>>,
    },
    /// The current turn's chain-of-thought, which grows in place.
    Thinking {
        /// Every fragment verbatim; only `\n` breaks a line.
        raw: String,
    },
}

/// One view's message log, kept pre-wrapped to the display width:
///
/// Call `sync` to re-wrap new content; it first re-renders any stale answer.
/// The current turn's thinking lives in one [`Entry::Thinking`] block.
#[derive(Default)]
pub(crate) struct Transcript {
    messages: Vec<Entry>,
    rows: Vec<Line<'static>>,
    /// Rows each folded message contributes, aligned with `messages` up to `cache.1`
    folds: Vec<usize>,
    /// (width, how many *visible* messages `rows` already contains).
    cache: (usize, usize),
    /// Whether the thinking block renders; sticky across turns. Starts hidden.
    show_thinking: bool,
    /// Whether tool outputs render in full; sticky, flipped by Ctrl+O. Starts
    /// collapsed.
    show_tool_output: bool,
}

impl Transcript {
    /// Append a committed line.
    pub(crate) fn push(&mut self, line: impl Into<Line<'static>>) {
        self.messages.push(Entry::Text(line.into()));
    }

    /// Record a tool invocation's start; it renders as a running header until
    /// finished. A `Read` or `Edit` following its own kind merges into that
    /// trailing finished box instead of stacking a fresh one. Otherwise the
    /// call supersedes every finished box before it, folding each down to its
    /// header line, and moves the thinking block below itself so the
    /// chain-of-thought always renders under the tool boxes.
    pub(crate) fn start_tool(&mut self, id: String, name: &'static str, digest: String) {
        if self.merge_into_tail(&id, name, &digest) {
            return;
        }
        let fold = self.messages.iter().position(
            |entry| matches!(entry, Entry::Tool(tool) if !tool.running && !tool.superseded),
        );
        let think = self
            .messages
            .iter()
            .position(|entry| matches!(entry, Entry::Thinking { .. }));
        let stale = [fold, think].into_iter().flatten().min();
        // Rewind before flipping the flags
        if let Some(index) = stale {
            self.rewind(index);
        }
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && !tool.running
            {
                tool.superseded = true;
            }
        }
        self.messages.push(Entry::Tool(ToolCall {
            id,
            name,
            digests: vec![digest],
            running: true,
            output: None,
            exit: None,
            superseded: false,
        }));
        // The thinking block rides below the boxes: pull it to the tail.
        if let Some(at) = think {
            let entry = self.messages.remove(at);
            self.messages.push(entry);
        }
    }

    /// Fold a same-kind call into a trailing finished box.
    fn merge_into_tail(&mut self, id: &str, name: &'static str, digest: &str) -> bool {
        // The thinking block rides below the boxes, so step past it to the fresh entry
        let Some(index) = self
            .messages
            .iter()
            .rposition(|entry| !matches!(entry, Entry::Thinking { .. }))
        else {
            return false;
        };
        if !matches!(name, "Read" | "Edit")
            || !matches!(&self.messages[index],
                Entry::Tool(tool) if tool.name == name && !tool.running)
        {
            return false;
        }
        self.rewind(index);
        if let Entry::Tool(tool) = &mut self.messages[index] {
            tool.digests.push(digest.to_owned());
            id.clone_into(&mut tool.id);
            tool.running = true;
            return true;
        }
        false
    }

    /// Fill in the pending invocation with `id`, then refold from its box.
    pub(crate) fn finish_tool(&mut self, id: &str, output: String, exit: Option<i32>) {
        let Some(index) = self
            .messages
            .iter()
            .rposition(|entry| matches!(entry, Entry::Tool(tool) if tool.running && tool.id == id))
        else {
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
        tool.running = false;
    }

    /// Drop the cached rows of `messages[index..]` so the next `sync` refolds
    /// them. Each fold slot mirrors its message's rows exactly.
    fn rewind(&mut self, index: usize) {
        if self.cache.0 == 0 || index >= self.cache.1 {
            return;
        }
        let stale: usize = self.folds[index..self.cache.1].iter().sum();
        self.rows.truncate(self.rows.len() - stale);
        self.cache.1 = index;
        self.folds.truncate(index);
    }

    /// Resolve every still-running invocation as failed to prevent stuck boxes.
    pub(crate) fn fail_pending(&mut self, reason: &str) {
        let mut failed = false;
        for entry in &mut self.messages {
            if let Entry::Tool(tool) = entry
                && tool.running
            {
                tool.output = Some(reason.to_string());
                tool.exit = None;
                tool.running = false;
                failed = true;
            }
        }
        if failed {
            self.rows.clear();
            self.folds.clear();
            self.cache.1 = 0;
        }
    }

    /// Expand or collapse every tool output.
    pub(crate) fn toggle_expand(&mut self) {
        if self.messages.iter().any(|entry| matches!(entry, Entry::Tool(_))) {
            self.show_tool_output = !self.show_tool_output;
            self.rows.clear();
            self.folds.clear();
            self.cache.1 = 0;
        }
    }

    /// Append a fragment of the model's answer, leaving its rendering stale.
    ///
    /// The fragment only grows `raw` and rewinds the wrap cache to the segment.
    /// `sync` re-renders at the next frame, so any number of fragments coalesce into
    /// one render, and one that restyles rows above (a fence opening, a `**` closing)
    /// is just part of that pass. Any other entry closes the segment; the next
    /// fragment opens a fresh one.
    pub(crate) fn append(&mut self, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        if let Some(Entry::Answer { raw, width, .. }) = self.messages.last_mut() {
            raw.push_str(fragment);
            // The folded rows still show the old rendering.
            *width = 0;
        } else {
            self.messages.push(Entry::Answer {
                raw: fragment.to_owned(),
                width: 0,
                lines: Vec::new(),
            });
        }
        self.rewind(self.messages.len() - 1);
    }

    /// Append the dim error line a failed turn leaves: newlines end lines,
    /// interior blanks materialize, and a trailing newline drops away.
    pub(crate) fn append_span(&mut self, span: &Span<'static>) {
        for line in span.content.lines() {
            let mut entry = Line::from(line.to_owned());
            entry.style = span.style;
            self.push(entry);
        }
    }

    /// Append a chain-of-thought fragment into the turn's thinking block.
    pub(crate) fn append_thinking(&mut self, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        let at = if let Some(at) = self
            .messages
            .iter()
            .position(|entry| matches!(entry, Entry::Thinking { .. }))
        {
            // The block renders where it sits; growing it restyles from
            // there, hidden or shown.
            if let Some(Entry::Thinking { raw }) = self.messages.get_mut(at) {
                raw.push_str(fragment);
            }
            at
        } else {
            // Reasoning precedes the answer it explains: open the block just
            // above a trailing answer, else at the tail.
            let at = match self.messages.last() {
                Some(Entry::Answer { .. }) => self.messages.len() - 1,
                _ => self.messages.len(),
            };
            self.messages
                .insert(at, Entry::Thinking { raw: fragment.to_owned() });
            at
        };
        self.rewind(at);
    }

    /// Show or hide the turn's chain-of-thought.
    pub(crate) fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
        if self
            .messages
            .iter()
            .any(|entry| matches!(entry, Entry::Thinking { .. }))
        {
            // The block's rows change wholesale; rebuild from scratch.
            self.rows.clear();
            self.folds.clear();
            self.cache.1 = 0;
        }
    }

    /// Retire the previous turn's thinking block, keeping its answers.
    pub(crate) fn begin_response(&mut self) {
        self.messages
            .retain(|entry| !matches!(entry, Entry::Thinking { .. }));
        // Rows that included the drained block are stale; rewrapping once
        // per turn is fine.
        self.rows.clear();
        self.folds.clear();
        self.cache.1 = 0;
    }

    /// Drop every message and reset our caches, persisting the thinking preference
    pub(crate) fn clear(&mut self) {
        self.messages.clear();
        self.rows.clear();
        self.folds.clear();
        self.cache.1 = 0;
    }

    /// Wrap the visible messages not yet folded into `rows`; a width change
    /// rewraps everything, and a stale answer re-renders first.
    pub(crate) fn sync(&mut self, width: usize) -> &[Line<'static>] {
        if self.cache.0 != width {
            self.rows.clear();
            self.folds.clear();
            self.cache = (width, 0);
        }
        let expanded = self.show_tool_output;
        let done = self.cache.1;
        for entry in &mut self.messages[done..] {
            if let Entry::Answer { raw, width: at, lines } = entry
                && *at != width
            {
                *lines = markdown::render(raw, width);
                *at = width;
            }
        }
        for index in done..self.messages.len() {
            self.fold_entry(index, width, expanded);
        }
        self.cache.1 = self.messages.len();
        &self.rows
    }

    /// Wrap one message into the row cache, recording its row count. A
    /// separated entry's blank row counts among its own, keeping `folds`
    /// aligned with `messages` through rewinds.
    fn fold_entry(&mut self, index: usize, width: usize, expanded: bool) {
        let separated = self.separated(index);
        let wrapped = match &self.messages[index] {
            // An answer wraps its rendering without cloning
            Entry::Answer { lines, .. } => wrap_lines(lines, width),
            Entry::Thinking { raw } => wrap_lines(&thinking_lines(raw, self.show_thinking), width),
            entry => wrap_lines(&entry.lines(expanded, self.show_thinking), width),
        };
        if separated {
            self.folds.push(wrapped.len() + 1);
            self.rows.push(Line::from(""));
        } else {
            self.folds.push(wrapped.len());
        }
        self.rows.extend(wrapped);
    }

    /// Whether a blank row leads `messages[index]` and we should emit a newline.
    fn separated(&self, index: usize) -> bool {
        // true for an answer, false for a tool box, `None` for anything else.
        let kind = |entry: &Entry| {
            matches!(entry, Entry::Answer { .. }).then_some(true).or(matches!(
                entry,
                Entry::Tool(_)
            )
            .then_some(false))
        };
        let Some(current) = kind(&self.messages[index]) else {
            return false;
        };
        self.messages[..index]
            .iter()
            .rev()
            .find(|entry| !matches!(entry, Entry::Thinking { .. }))
            .and_then(kind)
            .is_some_and(|before| before != current)
    }

    /// The wrapped rows; current as of the last `sync`.
    pub(crate) fn rows(&self) -> &[Line<'static>] {
        &self.rows
    }

    /// The text of the transcript's plain messages, tool boxes aside; an answer
    /// contributes one string per rendered line.
    #[cfg(test)]
    pub(crate) fn message_texts(&self) -> Vec<String> {
        let line_text = |line: &Line<'static>| {
            line.spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>()
        };
        self.messages
            .iter()
            .flat_map(|entry| match entry {
                Entry::Text(line) => vec![line_text(line)],
                Entry::Answer { raw, .. } => {
                    markdown::render(raw, 0).iter().map(line_text).collect()
                }
                Entry::Thinking { raw } => {
                    thinking_lines(raw, true).iter().map(line_text).collect()
                }
                Entry::Tool(_) => vec![],
            })
            .collect()
    }

    /// The cached rows equal a full re-wrap of every message at the wrapped width
    #[cfg(test)]
    pub(crate) fn assert_rows_match_full_rewrap(&self) {
        let mut full = Vec::new();
        for index in 0..self.messages.len() {
            if self.separated(index) {
                full.push(Line::from(""));
            }
            full.extend(self.messages[index].lines(self.show_tool_output, self.show_thinking));
        }
        assert_eq!(texts(&self.rows), texts(&wrap_lines(&full, self.cache.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The messages the transcript renders: each entry's display lines, including separators
    fn visible(t: &Transcript) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for (index, entry) in t.messages.iter().enumerate() {
            if t.separated(index) {
                lines.push(Line::from(""));
            }
            lines.extend(entry.lines(t.show_tool_output, t.show_thinking));
        }
        lines
    }

    /// Start a pending `Bash(echo hi)` invocation, as the pane would on a `ToolStart`
    fn start_bash(t: &mut Transcript, id: &str) {
        t.start_tool(id.to_string(), "Bash", "echo hi".to_string());
    }

    /// Start a pending `Read(digest)` invocation, as the pane would on a `ToolStart`
    fn start_read(t: &mut Transcript, id: &str, digest: &str) {
        t.start_tool(id.to_string(), "Read", digest.to_string());
    }

    /// A blank row separates an answer from an adjacent tool box, in either order
    #[test]
    fn answers_and_tool_boxes_get_a_blank_between() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append("the answer");
        t.sync(40);
        assert_eq!(texts(t.rows()), ["❯ go", "the answer"]);

        // A tool following the answer leads with air.
        start_bash(&mut t, "call_0");
        t.sync(40);
        assert_eq!(texts(t.rows()), ["❯ go", "the answer", "", "● Bash(echo hi) …"]);

        // An answer following the box does too, the riding thinking aside.
        t.finish_tool("call_0", "hi\n".to_string(), Some(0));
        t.append_thinking("why");
        t.append("more");
        t.sync(40);
        assert_eq!(
            texts(t.rows()),
            [
                "❯ go",
                "the answer",
                "",
                "● Bash(echo hi)",
                "  ⎿ hi",
                THINKING_HIDDEN,
                "",
                "more",
            ]
        );
        t.assert_rows_match_full_rewrap();

        // Consecutive boxes and plain lines stack without air.
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "hi\n".to_string(), Some(0));
        start_bash(&mut t, "call_1");
        t.sync(40);
        assert_eq!(texts(t.rows()), ["❯ go", "● Bash(echo hi)", "● Bash(echo hi) …"]);
    }

    #[test]
    fn same_kind_runs_merge_into_one_box() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();

        start_read(&mut t, "r0", "a.rs:1-10");
        t.finish_tool("r0", "one\n".to_string(), Some(0));
        t.append_thinking("mid-run reasoning"); // rides below the boxes
        start_read(&mut t, "r1", "a.rs:20-30");
        t.sync(40);
        let rows = texts(t.rows());
        assert_eq!(rows.iter().filter(|row| row.contains("Read(")).count(), 1);
        assert!(rows.iter().any(|row| row.contains("● Read(a.rs:1-10,20-30) …")));
        t.assert_rows_match_full_rewrap();

        // The newest output lands on the merged box.
        t.finish_tool("r1", "fresh\n".to_string(), Some(0));
        t.sync(40);
        assert!(texts(t.rows()).iter().any(|row| row.contains("⎿ fresh")));

        // An answer between calls breaks the run: a second box.
        t.append("done reading");
        start_read(&mut t, "r2", "b.rs:1-5");
        t.sync(40);
        assert_eq!(
            texts(t.rows()).iter().filter(|row| row.contains("Read(")).count(),
            2
        );

        // A box still running never absorbs the next call.
        start_read(&mut t, "r3", "b.rs:9-9");
        t.sync(40);
        assert_eq!(
            texts(t.rows()).iter().filter(|row| row.contains("Read(")).count(),
            3
        );

        // Bash never merges with the reads.
        start_bash(&mut t, "b0");
        t.sync(40);
        let rows = texts(t.rows());
        assert_eq!(rows.iter().filter(|row| row.contains("Read(")).count(), 3);
        assert!(rows.iter().any(|row| row.contains("● Bash(echo hi) …")));
    }

    #[test]
    fn merged_boxes_reopen_without_flicker() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        start_read(&mut t, "r0", "a.rs:1-10");
        t.finish_tool("r0", "one\ntwo\nthree\n".to_string(), Some(0));
        t.sync(40);
        let settled = texts(t.rows());
        assert_eq!(settled.len(), 5, "{settled:?}"); // prompt, header, three rows

        start_read(&mut t, "r1", "a.rs:20-30");
        t.sync(40);
        let reopened = texts(t.rows());
        assert_eq!(reopened.len(), settled.len(), "{reopened:?}");
        assert!(
            reopened
                .iter()
                .any(|row| row.contains("● Read(a.rs:1-10,20-30) …"))
        );
        assert!(
            reopened.iter().any(|row| row.contains("⎿ three")),
            "stale output stays"
        );

        t.finish_tool("r1", "fresh\n".to_string(), Some(0));
        t.sync(40);
        let swapped = texts(t.rows());
        assert!(swapped.iter().any(|row| row.contains("⎿ fresh")));
        assert!(!swapped.iter().any(|row| row.contains("⎿ three")));
    }

    /// Answer fragments join one growing markdown segment; the styled error
    /// path still glues by style; a push closes the segment.
    #[test]
    fn appends_glue_by_style() {
        let dim = Style::new().fg(Color::DarkGray);
        let mut transcript = Transcript::default();
        transcript.push(Line::from("prompt"));
        transcript.append("Hel");
        transcript.append("lo ");
        transcript.append("world");
        assert_eq!(transcript.message_texts(), ["prompt", "Hello world"]);

        // The dim error path lands line by line, never into the answer.
        transcript.append_span(&Span::styled(" (thinking) more", dim));
        assert_eq!(
            transcript.message_texts(),
            ["prompt", "Hello world", " (thinking) more"]
        );

        // A push ends the segment; later fragments join their own new one.
        transcript.push(Line::from("committed"));
        transcript.append("after");
        transcript.append("again");
        assert_eq!(
            transcript.message_texts(),
            [
                "prompt",
                "Hello world",
                " (thinking) more",
                "committed",
                "afteragain"
            ]
        );
    }

    /// The answer renders as markdown: markers style away, and the styled
    /// rows flow through the wrap cache like any others.
    #[test]
    fn answers_render_as_markdown() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ hi"));
        t.begin_response();
        t.append("Here is **bold** and `code`:");
        t.sync(40);
        let rows = texts(t.rows());
        assert!(rows.iter().any(|row| row.contains("Here is bold and code:")));
        assert!(rows.iter().all(|row| !row.contains("**")));
        t.assert_rows_match_full_rewrap();
    }

    /// An append defers its rendering to the next sync.
    #[test]
    fn appends_render_at_the_next_sync() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append("plain");
        t.sync(40);
        assert_eq!(texts(t.rows()), ["❯ go", "plain"]);
        t.append(" more");
        // The append rewound the segment's rows until sync refolds them.
        assert_eq!(texts(t.rows()), ["❯ go"]);
        t.sync(40);
        assert_eq!(texts(t.rows()), ["❯ go", "plain more"]);
        t.assert_rows_match_full_rewrap();
    }

    /// A segment restyles wholesale when a late fragment opens markdown the
    /// earlier plain rows lacked; the text itself is unchanged, so only the
    /// styling arrives.
    #[test]
    fn a_marker_restyles_the_whole_segment() {
        let mut t = Transcript::default();
        t.append("plain so far");
        t.sync(40);
        assert_eq!(t.message_texts(), ["plain so far"]);
        t.append(" now **bold**");
        t.sync(40);
        assert_eq!(t.message_texts(), ["plain so far now bold"]);
        t.assert_rows_match_full_rewrap();
    }

    /// A wrapped numbered item keeps its hanging indent through the wrap
    /// cache: the answer pre-wraps at the pane's width, each continuation
    /// aligned under its item's text, and the cache's re-wrap passes the
    /// rows through untouched.
    #[test]
    fn wrapped_list_items_keep_their_indent_through_sync() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ list"));
        t.begin_response();
        t.append("1. one two three four five six seven eight nine ten");
        t.sync(24);
        assert_eq!(
            texts(t.rows()),
            [
                "❯ list",
                "1. one two three four ",
                "   five six seven eight ",
                "   nine ten",
            ]
        );
        t.assert_rows_match_full_rewrap();
    }

    /// A probe reproducing a reported mis-render: a long markdown answer
    /// streamed in small fragments must finish fully styled.
    #[test]
    fn a_long_markdown_answer_streams_styled() {
        let raw = "Here's an example markdown document:\n\n\
# Quarterly Project Report\n\n\
## Overview\n\n\
This document summarizes the progress of the **Orion Project** during Q3 2025.\n\
The team completed *twelve* milestones, including the ~~beta launch~~ (moved to Q4)\n\
and the analytics dashboard rewrite.\n\n\
## Table of Contents\n\n\
1. [Overview](#overview)\n\n\
## Metrics\n\n\
| Metric | Q2 2025 | Q3 2025 | Change |\n| --- | ---: | ---: | ---: |\n\
| Active users | 12,400 | 18,950 | +52.8% |\n\n\
## Team Updates\n\n\
- **Frontend**: Shipped the new design system\n\
  - Completed zero-downtime cutover\n\n\
## Technical Notes\n\n\
```python\ndef get_cached(key: str, ttl: int = 300):\n    return value\n```\n\n\
> Note: The caching layer is now the single hottest path in production.\n\n\
## Links and References\n\n\
- Full dashboard: [Grafana](https://grafana.example.com)\n\n\
---\n\n\
*Last fed: this morning.*";
        let mut t = Transcript::default();
        t.push(Line::from("❯ probe"));
        t.begin_response();
        // Token-drip: fragments at awkward boundaries, including inside
        // the fence and inside ** pairs.
        let mut start = 0;
        while start < raw.len() {
            let end = raw[start..]
                .char_indices()
                .nth(37)
                .map_or(raw.len(), |(i, _)| start + i);
            t.append(&raw[start..end]);
            start = end;
        }
        t.sync(80);
        let rows = texts(t.rows());
        let flat = rows.join("\n");
        assert!(flat.contains("Quarterly Project Report"), "{flat}");
        assert!(!flat.contains("**"), "{flat}");
        assert!(!flat.contains("# "), "{flat}");
        assert!(
            flat.contains("• Full dashboard: Grafana (https://grafana.example.com)"),
            "{flat}"
        );
        assert!(flat.contains("Orion Project"), "{flat}");
        t.assert_rows_match_full_rewrap();
    }

    /// A late fragment can change the segment's row count
    #[test]
    fn a_row_count_changing_append_keeps_the_cache_honest() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append("plain");
        t.sync(40);
        // The fence opens: the segment grows from one row to several.
        t.append("\n\n```\ncode line one\ncode line two");
        t.sync(40);
        t.assert_rows_match_full_rewrap();
        let rows = texts(t.rows());
        assert_eq!(rows.first(), Some(&"❯ go".to_string()));
        assert_eq!(rows.iter().filter(|row| row.as_str() == "plain").count(), 1);
        assert!(rows.iter().any(|row| row == "code line one"));

        // Growth the other way: a list marker arrives and turns two plain
        // rows into a list block; the cache must stay exact through it.
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.append("intro\n\noutro");
        t.sync(40);
        t.append("\n\n- item");
        t.sync(40);
        t.assert_rows_match_full_rewrap();
        assert_eq!(texts(t.rows()), ["❯ go", "intro", "", "outro", "", "• item"]);
    }

    /// Fragments accumulate into the open segment; a tool box or a push closes it
    #[test]
    fn answer_segments_close_on_boundaries() {
        let mut t = Transcript::default();
        t.append("one ");
        t.append("two");
        start_bash(&mut t, "call_0");
        t.append("three");
        t.push(Line::from("committed"));
        t.append("four");
        assert_eq!(t.message_texts(), ["one two", "three", "committed", "four"]);
        t.append("");
        assert_eq!(t.message_texts().len(), 4);
    }

    /// Late thinking extends the block above the open answer, which then
    /// continues in place.
    #[test]
    fn late_thinking_extends_above_the_open_answer() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking("t1\n");
        t.append("a1");
        t.append_thinking("t2");
        t.append(" a2");
        assert_eq!(t.message_texts(), ["t1", "t2", "a1 a2"]);
    }

    /// Reasoning that starts only after the answer still opens above it
    #[test]
    fn fresh_thinking_opens_above_a_trailing_answer() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ go"));
        t.begin_response();
        t.append("a1");
        t.append_thinking("t1\n");
        t.append(" more");
        assert_eq!(t.message_texts(), ["❯ go", "t1", "a1 more"]);
    }

    /// Blank lines should survive streaming.
    #[test]
    fn blank_lines_render_across_fragments() {
        let mut t = Transcript::default();
        t.begin_response();
        t.append_thinking("hmm\n\nhmm");
        t.append("one\n\ntwo\n"); // blank inside one fragment
        t.append("three\n"); // a lone trailing newline…
        t.append("\nfour\n"); // …that a leading `\n` doubles into a blank
        t.append("five\n"); // another lone newline…
        t.append("six"); // …only breaks the line
        t.append("\n\neven"); // after open text: first `\n` ends it, second blanks
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
            transcript.assert_rows_match_full_rewrap();
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

        transcript.append("streaming aaaa bbbb"); // glued run
        transcript.sync(80);
        assert_fresh(&transcript);
        transcript.append(" cccc dddd");
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
        t.append_thinking("hmm aaaa bbbb");
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking(" cccc dddd"); // glued
        t.sync(20);
        assert_fresh(&t);
        t.append_thinking("line two\nline three");
        t.sync(20);
        assert_fresh(&t);
        t.append("the answer aaaa bbbb");
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
        t.append_thinking(" mid");
        t.sync(20);
        assert_fresh(&t);

        t.sync(80); // width change rebuilds
        assert_fresh(&t);
        t.append_thinking(" late"); // splices above the answer
        t.sync(80);
        assert_fresh(&t);

        t.toggle_thinking(); // reveal
        t.sync(80);
        assert_fresh(&t);
        t.append_thinking(" more");
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
        t.append_thinking(&reasoning);
        t.append("the answer");

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
        t.append_thinking("old reasoning");
        t.append("old answer");
        t.push(Line::from("❯ two"));
        t.begin_response();
        let messages = t.message_texts();
        assert!(messages.iter().all(|m| !m.contains("old reasoning")));
        assert!(messages.iter().any(|m| m.contains("old answer")));
        assert!(messages.iter().any(|m| m.contains("❯ two")));

        t.append_thinking("new reasoning");
        t.append("new answer");
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
        t.append_thinking("doomed reasoning");
        // Same dim style as the thinking: the block drains, the error line
        // is its own entry and stays.
        t.append_span(&Span::styled("boom: network down", DIM_STYLE));
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
        t.append_thinking("t1\n");
        t.append("a1");
        t.append_thinking("t2");
        assert_eq!(t.message_texts(), ["t1", "t2", "a1"]);

        t.sync(40); // hidden (default): the block collapses
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
        t.append_thinking("abc");
        t.sync(20);
        t.append_thinking("def"); // glue, cache primed
        t.sync(20);
        t.append("answer");
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
        t.append_thinking("t1\n");
        t.append("a1");
        t.sync(40);
        t.append_thinking("t2\n"); // late, tail folded
        assert_eq!(t.cache, (40, 1), "only the echo stays cached");
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));

        // A second late fragment with an unsynced glued answer in between.
        t.append(" a2");
        t.append_thinking("t3\n");
        t.sync(40);
        assert_eq!(texts(&t.rows), texts(&wrap_lines(&visible(&t), 40)));
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "t2", "t3", "a1 a2"]);
    }

    /// A trailing blank in thinking stays pending: it renders nothing until
    /// later text confirms it.
    #[test]
    fn empty_late_fragments_change_nothing_settled() {
        let mut t = Transcript::default();
        t.push(Line::from("❯ echo"));
        t.begin_response();
        t.append_thinking("t1\n");
        t.append("a1");
        t.sync(40);
        let rows = texts(&t.rows);
        t.append_thinking("\n\n");
        t.sync(40);
        assert_eq!(texts(&t.rows), rows);
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "a1"]);
        // Later text confirms the blank, and the answer stays open.
        t.append_thinking("x");
        t.append(" more");
        t.sync(40);
        assert_eq!(t.message_texts(), ["❯ echo", "t1", "", "x", "a1 more"]);
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
            t.assert_rows_match_full_rewrap();
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
        t.append("answer");
        start_bash(&mut t, "call_0");
        t.append("more");
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
        t.append_thinking("t1\n");
        t.append("a1");
        start_bash(&mut t, "call_0");
        t.finish_tool("call_0", "out\n".to_string(), Some(0));
        t.sync(40);
        assert_fresh(&t);
        // t1 rotated below the box when the call started, placeholder and all.
        assert_eq!(t.message_texts(), ["❯ go", "a1", "t1"]);
        assert!(texts(&t.rows).iter().any(|row| row.contains("Thinking")));

        t.append_thinking("t2"); // extends the run below the box
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

        t.append_thinking("reasoning");
        t.clear();
        assert!(t.messages.is_empty() && t.rows.is_empty());
        // Toggled on above: `clear` keeps the preference rather than the default.
        assert!(t.show_thinking, "clear keeps the sticky preference");

        // Thinking after a clear lazily opens a drainable block.
        t.append_thinking("again");
        t.begin_response();
        assert!(t.messages.is_empty(), "the re-opened block drains");
    }
}
