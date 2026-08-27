//! Shared test helpers: render-to-string over the ratatui test backend.

use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::{Frame, Terminal};

/// Render two frames and hand back the terminal.
///
/// Two frames because the app always polls between draws, and the scroll
/// that the first post-resize frame parks only settles on the second.
fn frames(mut render: impl FnMut(&mut Frame, Rect), (w, h): (u16, u16)) -> Terminal<TestBackend> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    for _ in 0..2 {
        terminal
            .draw(|frame| {
                let area = frame.area();
                render(frame, area);
            })
            .unwrap();
    }
    terminal
}

/// Render two frames (see [`frames`]) and return the terminal's text.
pub(crate) fn draw(render: impl FnMut(&mut Frame, Rect), size: (u16, u16)) -> String {
    frames(render, size)
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol().to_string())
        .collect()
}

/// Concatenate the text of rendered lines.
pub(crate) fn texts(lines: &[Line<'static>]) -> Vec<String> {
    lines
        .iter()
        .map(|line| line.spans.iter().map(|span| span.content.as_ref()).collect())
        .collect()
}

/// Render two frames (see [`frames`]) and map each cell: `#` where its
/// background is `Color::DarkGray` — the selection band — and `.` elsewhere,
/// one output line per terminal row.
pub(crate) fn draw_backgrounds(render: impl FnMut(&mut Frame, Rect), (w, h): (u16, u16)) -> String {
    let terminal = frames(render, (w, h));
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| if buf[(x, y)].bg == Color::DarkGray { '#' } else { '.' })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One line's spans as `(text, style)` pairs with the line style patched in .
pub(crate) fn segments(lines: &[Line<'static>]) -> Vec<Vec<(String, Style)>> {
    lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| (span.content.as_ref().to_string(), line.style.patch(span.style)))
                .collect()
        })
        .collect()
}

/// Render two frames (see [`frames`]) and map each cell to one legend char by
/// its style. One output line per terminal row, so a styled word reads as a run of its
/// legend char:
///
/// - `B` bold · `I` italic · `S` crossed-out
/// - `c` yellow fg (inline code) · `d` dark-gray fg (dim)
/// - `b` blue fg · `C` cyan fg
/// - `.` anything else
pub(crate) fn draw_styles(render: impl FnMut(&mut Frame, Rect), (w, h): (u16, u16)) -> String {
    let terminal = frames(render, (w, h));
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    let cell = &buf[(x, y)];
                    let add = cell.modifier;
                    if add.contains(Modifier::BOLD) {
                        'B'
                    } else if add.contains(Modifier::ITALIC) {
                        'I'
                    } else if add.contains(Modifier::CROSSED_OUT) {
                        'S'
                    } else {
                        match cell.fg {
                            Color::Yellow => 'c',
                            Color::DarkGray => 'd',
                            Color::Blue => 'b',
                            Color::Cyan => 'C',
                            _ => '.',
                        }
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}
