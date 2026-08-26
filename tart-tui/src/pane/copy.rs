//! Copy mode and scrollback cursor.

use ratatui::crossterm::event::KeyCode;
use ratatui::text::Line;

/// Cursor in copy mode: `row`/`col` address a cell in the wrapped rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct CopyCursor {
    pub(crate) row: usize,
    pub(crate) col: usize,
    /// Selection start cell; `None` until Space anchors one.
    pub(crate) anchor: Option<(usize, usize)>,
    /// `usize::MAX` means we start at the end, and position is clamped on render
    pub(crate) top: usize,
    /// Rows the last render showed.
    pub(crate) visible: usize,
}

impl CopyCursor {
    /// Enter copy mode with the cursor on the last row.
    pub(crate) fn enter(rows_len: usize) -> Self {
        Self {
            row: rows_len.saturating_sub(1),
            col: 0,
            anchor: None,
            top: usize::MAX,
            visible: 0,
        }
    }
}

/// The cursor after one key step, clamped to the transcript edges.
pub(crate) fn moved(rows: &[Line<'static>], cursor: CopyCursor, key: KeyCode) -> CopyCursor {
    if rows.is_empty() {
        return CopyCursor { row: 0, col: 0, ..cursor };
    }
    let last = rows.len() - 1;
    let mut c = cursor;
    match key {
        KeyCode::Up => c.row = c.row.saturating_sub(1),
        KeyCode::Down => c.row = (c.row + 1).min(last),
        KeyCode::Left => {
            if c.col > 0 {
                c.col -= 1;
            } else if c.row > 0 {
                c.row -= 1;
                c.col = rows[c.row].width().saturating_sub(1);
            }
        }
        KeyCode::Right => {
            if c.col + 1 < rows[c.row].width() {
                c.col += 1;
            } else if c.row < last {
                c.row += 1;
                c.col = 0;
            }
        }
        KeyCode::PageUp => c.row = c.row.saturating_sub(c.visible.max(1)),
        KeyCode::PageDown => c.row = (c.row + c.visible.max(1)).min(last),
        KeyCode::Home => c.row = 0,
        KeyCode::End => c.row = last,
        _ => return cursor,
    }
    c.col = c.col.min(rows[c.row].width().saturating_sub(1));
    c
}

/// First transcript row to render, used to anchor the prompt viewport.
pub(crate) fn window_top(rows_len: usize, visible: usize, anchor: Option<(usize, usize)>) -> usize {
    let max_top = rows_len.saturating_sub(visible);
    anchor.map_or(max_top, |(row, top)| {
        top.min(max_top)
            .min(row)
            .max(row.saturating_add(1).saturating_sub(visible))
    })
}

/// A cell clamped into the wrapped rows: the last row, that row's last cell.
#[inline]
pub(crate) fn clamp_cell(rows: &[Line<'static>], row: usize, col: usize) -> (usize, usize) {
    let row = row.min(rows.len().saturating_sub(1));
    let col = col.min(rows.get(row).map_or(0, |row| row.width().saturating_sub(1)));
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pane::wrap::wrap_lines;

    #[test]
    fn viewport_and_copy_cursor_math() {
        assert_eq!(window_top(10, 5, None), 5); // live pins the tail
        assert_eq!(window_top(10, 5, Some((9, usize::MAX))), 5); // enter
        assert_eq!(window_top(10, 5, Some((3, 5))), 3); // dragged up
        assert_eq!(window_top(10, 5, Some((8, 2))), 4); // dragged down
        assert_eq!(window_top(10, 5, Some((4, 2))), 2); // inside: stays

        let rows = wrap_lines(&[Line::from("abc"), Line::from("de")], 10);
        let cur = |row, col| CopyCursor {
            row,
            col,
            anchor: None,
            top: 0,
            visible: 2,
        };
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::Right), cur(0, 1));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Left), cur(0, 2)); // wraps
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::PageUp), cur(0, 0));
        assert_eq!(moved(&rows, cur(0, 0), KeyCode::PageDown), cur(1, 0));
        assert_eq!(moved(&rows, cur(1, 0), KeyCode::Char('x')), cur(1, 0));

        // A move reshapes a selection; it never drops the anchor.
        let mut anchored = cur(0, 0);
        anchored.anchor = Some((1, 1));
        assert_eq!(moved(&rows, anchored, KeyCode::Right).anchor, Some((1, 1)));
    }
}
