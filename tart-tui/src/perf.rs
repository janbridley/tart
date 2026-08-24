//! Render statistics for the `/perf` stats line.

use std::collections::VecDeque;
use std::time::Duration;

use ratatui::buffer::Buffer;

const WINDOW: usize = 10;

#[derive(Default)]
pub struct Perf {
    /// Paint durations of the last ten frames, for the rolling average.
    recent: VecDeque<Duration>,
}

impl Perf {
    /// Fold in one finished frame and format the stats line.
    pub fn frame(&mut self, paint: Duration, buf: &Buffer) -> String {
        self.recent.push_back(paint);
        if self.recent.len() > WINDOW {
            self.recent.pop_front();
        }
        let avg = self.recent.iter().sum::<Duration>() / self.recent.len() as u32;
        let total = buf.content.len();
        let cells = buf.content.iter().filter(|c| c.symbol() != " ").count();
        format!(" paint avg {avg:.1?} · cells {cells}/{total} ")
    }
}
