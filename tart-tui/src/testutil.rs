//! Shared test helpers: render-to-string over the ratatui test backend.

use ratatui::backend::TestBackend;
use ratatui::buffer::Cell;
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

/// Render two frames (see [`frames`]) and map each cell to `char_of`, one
/// output line per terminal row.
fn grid(
    render: impl FnMut(&mut Frame, Rect),
    (w, h): (u16, u16),
    char_of: impl Fn(&Cell) -> char,
) -> String {
    let terminal = frames(render, (w, h));
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| (0..w).map(|x| char_of(&buf[(x, y)])).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render two frames (see [`frames`]) and map each cell: `#` where its
/// background is `Color::DarkGray` and `.` elsewhere.
pub(crate) fn draw_backgrounds(render: impl FnMut(&mut Frame, Rect), size: (u16, u16)) -> String {
    grid(
        render,
        size,
        |cell| if cell.bg == Color::DarkGray { '#' } else { '.' },
    )
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
/// - `d` dark-gray fg (dim) · `c` yellow fg (plan mode's rules)
/// - `b` blue fg · `C` cyan fg · `m` magenta fg (the `!` mode's frame)
/// - `r` light-red fg · `l` light-blue fg (links) · `y` light-yellow fg
///   (inline code) · `p` light-magenta fg
/// - `.` anything else
pub(crate) fn draw_styles(render: impl FnMut(&mut Frame, Rect), size: (u16, u16)) -> String {
    grid(render, size, |cell| {
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
                Color::Magenta => 'm',
                Color::LightRed => 'r',
                Color::LightBlue => 'l',
                Color::LightYellow => 'y',
                Color::LightMagenta => 'p',
                _ => '.',
            }
        }
    })
}
