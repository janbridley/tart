//! Shared test helpers: render-to-string over the ratatui test backend.

use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::{Frame, Terminal};

/// Render two frames and return the terminal's text.
///
/// Two frames because the app always polls between draws, and the scroll
/// that the first post-resize frame parks only settles on the second.
pub(crate) fn draw(mut render: impl FnMut(&mut Frame, Rect), (w, h): (u16, u16)) -> String {
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
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol().to_string())
        .collect()
}
