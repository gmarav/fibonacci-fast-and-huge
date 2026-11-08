//! Fibonacci logo with a classical golden-rectangle spiral.
//!
//! The spiral is the classical Fibonacci construction: eight quarter arcs on a
//! `34 x 21` golden rectangle, from the two `1 x 1` squares out to the `21 x 21`
//! square. The word `FIBONACCI` is drawn behind the curve with a 5x7 block font
//! and a vertical blue gradient.
//!
//! The animation grows the curve from the inside out and reveals the background
//! word from top to bottom across all letters at once.

use std::io::Write;

/// Logo canvas width in character cells.
pub const COLS: usize = 68;
/// Logo canvas height in character cells.
pub const ROWS: usize = 22;
/// Braille subpixel columns (two per character cell).
const PIXELS_X: usize = COLS * 2;
/// Braille subpixel rows (four per character cell).
const PIXELS_Y: usize = ROWS * 4;

/// Left margin shared with the formula banner.
const INDENT: &str = "   ";

/// Base stroke thickness in subpixels.
const THICKNESS: f64 = 1.35;
/// Aspect ratio of the classical rectangle. The Braille 2x4 grid already
/// compensates for the terminal cell shape, so `1.0` gives round arcs.
const ASPECT_RATIO: f64 = 1.0;
/// Golden rectangle width in the spiral's own units.
const RECT_WIDTH: f64 = 34.0;
/// Golden rectangle height in the spiral's own units.
const RECT_HEIGHT: f64 = 21.0;
/// Extra clearance around the curve where the background word is suppressed.
const HALO_MARGIN: f64 = 1.6;

/// Word drawn behind the spiral.
const WORD: &str = "FIBONACCI";
/// Width of one glyph in the 5x7 font.
const GLYPH_WIDTH: usize = 5;
/// Height of one glyph in the 5x7 font.
const GLYPH_HEIGHT: usize = 7;
/// Blank columns between two glyphs.
const GLYPH_GAP: usize = 2;
/// Word width as a fraction of the fitted spiral width.
const TEXT_WIDTH_RATIO: f64 = 0.55;
/// Word height in Braille subpixel rows.
const TEXT_HEIGHT_PX: i32 = 16;

/// Quarter arcs `(center_x, center_y, radius, start_deg, end_deg)`.
///
/// The coordinates follow the classical golden-rectangle construction:
/// two 1x1 squares, then 2, 3, 5, 8, 13 and finally 21.
const ARCS: [(f64, f64, f64, f64, f64); 8] = [
    (25.0, 6.0, 1.0, 180.0, 270.0),
    (25.0, 6.0, 1.0, 270.0, 360.0),
    (24.0, 6.0, 2.0, 0.0, 90.0),
    (24.0, 5.0, 3.0, 90.0, 180.0),
    (26.0, 5.0, 5.0, 180.0, 270.0),
    (26.0, 8.0, 8.0, 270.0, 360.0),
    (21.0, 8.0, 13.0, 0.0, 90.0),
    (21.0, 0.0, 21.0, 90.0, 180.0),
];

/// Spiral gradient: bright core, gold, amber, dark amber, bronze.
const SPIRAL_STOPS: [(f64, (u8, u8, u8)); 5] = [
    (0.00, (255, 255, 225)),
    (0.22, (255, 215, 0)),
    (0.50, (255, 145, 0)),
    (0.78, (205, 75, 0)),
    (1.00, (120, 35, 5)),
];

/// Background word gradient: steel indigo at the top to twilight blue.
const WORD_STOPS: [(f64, (u8, u8, u8)); 3] = [
    (0.00, (105, 140, 185)),
    (0.50, (60, 82, 118)),
    (1.00, (32, 44, 68)),
];

/// Braille dot bits for a subpixel row (`0` = top) and column (`0` = left).
const DOT_MAP: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

/// Number of animation frames.
const ANIMATION_FRAMES: usize = 36;
/// Delay between animation frames in milliseconds.
const ANIMATION_DELAY_MS: u64 = 20;

/// One packed Braille cell: `None` is an ASCII space, `Some` carries the glyph
/// and its gradient color.
type LogoCell = Option<(char, (u8, u8, u8))>;

/// ANSI 24-bit foreground prefix for `rgb`.
fn rgb_sgr(rgb: (u8, u8, u8)) -> String {
    let (r, g, b) = rgb;
    format!("\x1b[38;2;{r};{g};{b}m")
}

/// Samples a color stop table at normalised position `t`.
fn gradient(stops: &[(f64, (u8, u8, u8))], t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    for index in 0..stops.len() - 1 {
        let (start_t, start_rgb) = stops[index];
        let (end_t, end_rgb) = stops[index + 1];
        if start_t <= t && t <= end_t {
            let fraction = (t - start_t) / (end_t - start_t);
            let mix = |from: u8, to: u8| {
                (f64::from(from) + fraction * (f64::from(to) - f64::from(from))) as u8
            };
            return (
                mix(start_rgb.0, end_rgb.0),
                mix(start_rgb.1, end_rgb.1),
                mix(start_rgb.2, end_rgb.2),
            );
        }
    }
    stops[stops.len() - 1].1
}

fn spiral_color(t: f64) -> (u8, u8, u8) {
    gradient(&SPIRAL_STOPS, t)
}

fn word_color(t: f64) -> (u8, u8, u8) {
    gradient(&WORD_STOPS, t)
}

/// 5x7 glyph rows for the background word.
fn glyph(ch: u8) -> [&'static str; GLYPH_HEIGHT] {
    match ch {
        b'F' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "10000",
        ],
        b'I' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "11111",
        ],
        b'B' => [
            "11110", "10001", "10001", "11110", "10001", "10001", "11110",
        ],
        b'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        b'N' => [
            "10001", "11001", "10101", "10011", "10001", "10001", "10001",
        ],
        b'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        b'C' => [
            "01111", "10000", "10000", "10000", "10000", "10000", "01111",
        ],
        _ => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
    }
}

/// Fitted spiral curve and background word placement.
struct SpiralGeometry {
    /// Scaled pixel coordinates `(x, y)` and the normalised path parameter.
    points: Vec<(f64, f64, f64)>,
    /// Halo of the complete curve. It is fixed, so the word is revealed once
    /// from top to bottom and never disappears as the spiral grows.
    halo: Vec<Vec<bool>>,
    text_x0: i32,
    text_y0: i32,
    text_width: i32,
    text_height: i32,
}

impl SpiralGeometry {
    fn new() -> Self {
        let mut points = Vec::new();
        let total_len: f64 = ARCS
            .iter()
            .map(|arc| arc.2 * std::f64::consts::FRAC_PI_2)
            .sum();
        let mut acc_len = 0.0;

        for &(cx, cy, radius, start_deg, end_deg) in &ARCS {
            let arc_len = radius * std::f64::consts::FRAC_PI_2;
            let steps = ((radius * 150.0) as usize).max(300);
            for step in 0..=steps {
                let fraction = step as f64 / steps as f64;
                let degrees = start_deg + fraction * (end_deg - start_deg);
                let radians = degrees.to_radians();
                let x = cx + radius * radians.cos();
                let y = RECT_HEIGHT - (cy + radius * radians.sin());
                let t_norm = (acc_len + fraction * arc_len) / total_len;
                points.push((x, y * ASPECT_RATIO, t_norm));
            }
            acc_len += arc_len;
        }

        let max_brush = THICKNESS * 1.35;
        let pad = max_brush + 1.5;
        let spiral_height = RECT_HEIGHT * ASPECT_RATIO;
        let scale = ((PIXELS_X as f64 - 2.0 * pad) / RECT_WIDTH)
            .min((PIXELS_Y as f64 - 2.0 * pad) / spiral_height);
        let actual_width = RECT_WIDTH * scale;
        let actual_height = spiral_height * scale;
        let offset_x = (PIXELS_X as f64 - actual_width) / 2.0;
        let offset_y = (PIXELS_Y as f64 - actual_height) / 2.0;

        for point in &mut points {
            point.0 = point.0 * scale + offset_x;
            point.1 = point.1 * scale + offset_y;
        }

        // The halo is built from the complete curve so a word row that has
        // already appeared never vanishes when the spiral reaches it later.
        let max_brush = THICKNESS * 1.35;
        let mut halo = vec![vec![false; PIXELS_X]; PIXELS_Y];
        for &(sx, sy, t_norm) in &points {
            let thickness = max_brush * (0.48 + 0.52 * t_norm.powf(0.75));
            let halo_radius = thickness + HALO_MARGIN;
            let radius = halo_radius.ceil() as i32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if f64::from(dx * dx + dy * dy) > halo_radius * halo_radius {
                        continue;
                    }
                    let nx = (sx + f64::from(dx)).round_ties_even() as i32;
                    let ny = (sy + f64::from(dy)).round_ties_even() as i32;
                    if nx >= 0 && ny >= 0 && nx < PIXELS_X as i32 && ny < PIXELS_Y as i32 {
                        halo[ny as usize][nx as usize] = true;
                    }
                }
            }
        }

        let text_width = (actual_width * TEXT_WIDTH_RATIO) as i32;
        let text_height = TEXT_HEIGHT_PX;
        let text_x0 = (offset_x + (actual_width - f64::from(text_width)) / 2.0) as i32;
        let text_y0 = (offset_y + (actual_height - f64::from(text_height)) / 2.0) as i32;

        Self {
            points,
            halo,
            text_x0,
            text_y0,
            text_width,
            text_height,
        }
    }

    /// Rasterises the first `progress` fraction of the curve. The path starts
    /// at the innermost 1x1 square, so increasing progress unwinds it outward.
    fn rasterize(&self, progress: f64) -> Vec<Vec<Option<f64>>> {
        let mut fg = vec![vec![None; PIXELS_X]; PIXELS_Y];
        let max_brush = THICKNESS * 1.35;

        for &(sx, sy, t_norm) in &self.points {
            if t_norm > progress {
                break;
            }
            let thickness = max_brush * (0.48 + 0.52 * t_norm.powf(0.75));
            let radius = thickness.ceil() as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let distance_sq = f64::from(dx * dx + dy * dy);
                    if distance_sq > thickness * thickness {
                        continue;
                    }
                    // Python's round() is half-to-even; match it for a faithful
                    // port of the reference rasteriser.
                    let nx = (sx + f64::from(dx)).round_ties_even() as i32;
                    let ny = (sy + f64::from(dy)).round_ties_even() as i32;
                    if nx < 0 || ny < 0 || nx >= PIXELS_X as i32 || ny >= PIXELS_Y as i32 {
                        continue;
                    }
                    let distance = distance_sq.sqrt() / thickness.max(0.001);
                    let value = (t_norm * 0.52 + distance.powf(1.5) * 0.48).min(1.0);
                    let (nx, ny) = (nx as usize, ny as usize);
                    if fg[ny][nx].is_none_or(|old| value < old) {
                        fg[ny][nx] = Some(value);
                    }
                }
            }
        }

        fg
    }

    /// Background word pixels, revealed from top to bottom. All letters share
    /// the same vertical threshold, so they appear simultaneously.
    fn word_grid(&self, reveal: f64) -> Vec<Vec<Option<f64>>> {
        let mut bg = vec![vec![None; PIXELS_X]; PIXELS_Y];
        let base_units = WORD.len() * GLYPH_WIDTH + WORD.len().saturating_sub(1) * GLYPH_GAP;
        let glyph_stride = GLYPH_WIDTH + GLYPH_GAP;
        let reveal = reveal.clamp(0.0, 1.0);

        for py in self.text_y0..self.text_y0 + self.text_height {
            if !(0..PIXELS_Y as i32).contains(&py) {
                continue;
            }
            let t_y = f64::from(py - self.text_y0) / f64::from(self.text_height.max(2) - 1);
            if t_y > reveal {
                continue;
            }
            let gy = ((t_y * GLYPH_HEIGHT as f64) as usize).min(GLYPH_HEIGHT - 1);

            for px in self.text_x0..self.text_x0 + self.text_width {
                if !(0..PIXELS_X as i32).contains(&px) {
                    continue;
                }
                let t_x = f64::from(px - self.text_x0) / f64::from(self.text_width.max(1));
                let unit = (t_x * base_units as f64) as usize;
                let char_index = unit / glyph_stride;
                let glyph_x = unit % glyph_stride;
                if char_index < WORD.len() && glyph_x < GLYPH_WIDTH {
                    let rows = glyph(WORD.as_bytes()[char_index]);
                    if rows[gy].as_bytes()[glyph_x] == b'1' {
                        bg[py as usize][px as usize] = Some(t_y);
                    }
                }
            }
        }

        bg
    }

    /// Packs one canvas row. Spiral dots suppress the word in the same cell;
    /// the word is also hidden inside the curve halo.
    fn composite_row(
        &self,
        fg: &[Vec<Option<f64>>],
        word: &[Vec<Option<f64>>],
        row: usize,
    ) -> Vec<LogoCell> {
        let mut cells = Vec::with_capacity(COLS);
        let mut last_ink = 0;
        for column in 0..COLS {
            let mut fg_mask = 0u8;
            let mut bg_mask = 0u8;
            let mut fg_value = f64::INFINITY;
            let mut bg_sum = 0.0;
            let mut bg_count = 0usize;

            for dy in 0..4 {
                for dx in 0..2 {
                    let py = row * 4 + dy;
                    let px = column * 2 + dx;
                    if let Some(value) = fg[py][px] {
                        fg_mask |= DOT_MAP[dy][dx];
                        fg_value = fg_value.min(value);
                    } else if let Some(value) = word[py][px] {
                        if !self.halo[py][px] {
                            bg_mask |= DOT_MAP[dy][dx];
                            bg_sum += value;
                            bg_count += 1;
                        }
                    }
                }
            }

            if fg_mask != 0 {
                let glyph =
                    char::from_u32(0x2800 + u32::from(fg_mask)).expect("braille code point");
                cells.push(Some((glyph, spiral_color(fg_value))));
                last_ink = cells.len();
            } else if bg_mask != 0 {
                let glyph =
                    char::from_u32(0x2800 + u32::from(bg_mask)).expect("braille code point");
                cells.push(Some((glyph, word_color(bg_sum / bg_count as f64))));
                last_ink = cells.len();
            } else {
                cells.push(None);
            }
        }
        cells.truncate(last_ink);
        cells
    }

    /// All canvas rows at `progress`, with empty rows kept as empty vectors.
    fn cells_at(&self, progress: f64) -> Vec<Vec<LogoCell>> {
        let fg = self.rasterize(progress);
        let word = self.word_grid(progress);
        (0..ROWS)
            .map(|row| self.composite_row(&fg, &word, row))
            .collect()
    }
}

/// Formats one Braille row without trailing whitespace.
fn row_line(cells: &[LogoCell], color: bool) -> String {
    if cells.is_empty() {
        return String::new();
    }
    let mut line = String::from(INDENT);
    for cell in cells {
        match cell {
            None => line.push(' '),
            Some((glyph, rgb)) if color => {
                line.push_str(&rgb_sgr(*rgb));
                line.push(*glyph);
                line.push_str("\x1b[0m");
            }
            Some((glyph, _)) => line.push(*glyph),
        }
    }
    line
}

/// Renders the logo at a growth fraction and returns its lines.
///
/// `progress` drives both animations: the spiral unwinds from the inside out
/// and the word appears from top to bottom.
pub fn lines_at(color: bool, progress: f64) -> Vec<String> {
    let progress = if progress.is_nan() {
        0.0
    } else {
        progress.clamp(0.0, 1.0)
    };
    let geometry = SpiralGeometry::new();
    let cells = geometry.cells_at(progress);
    cells
        .iter()
        .filter(|row| !row.is_empty())
        .map(|row| row_line(row, color))
        .collect()
}

/// Joins lines into a block ending with exactly one newline.
fn finish(lines: Vec<String>) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// Renders the logo at a growth fraction as a block ending with one newline.
pub fn render_at(color: bool, progress: f64) -> String {
    finish(lines_at(color, progress))
}

/// Renders the complete logo as a block ending with exactly one newline.
pub fn render(color: bool) -> String {
    render_at(color, 1.0)
}

/// Restores the terminal cursor when dropped, including during a panic.
struct CursorGuard;

impl Drop for CursorGuard {
    fn drop(&mut self) {
        let mut stderr = std::io::stderr();
        let _ = stderr.write_all(b"\x1b[?25h");
        let _ = stderr.flush();
    }
}

/// Animates the logo in place on stderr.
pub fn print_animated(color: bool) {
    let geometry = SpiralGeometry::new();
    let full = geometry.cells_at(1.0);
    let slots: Vec<usize> = full
        .iter()
        .enumerate()
        .filter(|(_, row)| !row.is_empty())
        .map(|(row, _)| row)
        .collect();
    let frame_height = slots.len();

    let mut stderr = std::io::stderr();
    let _cursor = CursorGuard;
    let _ = stderr.write_all(b"\x1b[?25l");
    let _ = stderr.flush();

    for frame in 0..ANIMATION_FRAMES {
        let progress = frame as f64 / (ANIMATION_FRAMES - 1) as f64;
        let cells = geometry.cells_at(progress);
        let lines: Vec<String> = slots
            .iter()
            .map(|&row| row_line(&cells[row], color))
            .collect();
        let text = finish(lines);
        let _ = stderr.write_all(text.as_bytes());

        if frame + 1 < ANIMATION_FRAMES {
            let _ = write!(stderr, "\x1b[{frame_height}A");
            let _ = stderr.flush();
            std::thread::sleep(std::time::Duration::from_millis(ANIMATION_DELAY_MS));
        } else {
            let _ = stderr.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn braille_count(text: &str) -> usize {
        text.chars()
            .filter(|ch| ('\u{2800}'..='\u{28FF}').contains(ch))
            .count()
    }

    #[test]
    fn logo_has_braille_and_no_printed_title() {
        let text = render(false);
        assert!(braille_count(&text) > 0, "logo must use Braille patterns");
        assert!(!text.contains("FIBONACCI"), "word is drawn as artwork");
        assert!(!text.contains('\x1b'));
        assert!(text.ends_with('\n'));
        assert!(!text.ends_with("\n\n"));
    }

    #[test]
    fn logo_fits_the_canvas_and_has_no_trailing_spaces() {
        let text = render(false);
        let lines: Vec<&str> = text.lines().collect();
        assert!(!lines.is_empty());
        assert!(lines.len() <= ROWS);
        for line in lines {
            assert!(line.chars().count() <= INDENT.len() + COLS, "{line:?}");
            assert!(!line.ends_with(' '), "{line:?}");
        }
        assert!(!text.contains('\u{2800}'), "empty cells use ASCII spaces");
    }

    #[test]
    fn logo_gradient_uses_the_reference_stops() {
        assert_eq!(spiral_color(0.0), (255, 255, 225));
        assert_eq!(spiral_color(0.22), (255, 215, 0));
        assert_eq!(spiral_color(0.50), (255, 145, 0));
        assert_eq!(spiral_color(0.78), (205, 75, 0));
        assert_eq!(spiral_color(1.0), (120, 35, 5));
        assert_eq!(word_color(0.0), (105, 140, 185));
        assert_eq!(word_color(0.5), (60, 82, 118));
        assert_eq!(word_color(1.0), (32, 44, 68));
    }

    #[test]
    fn animation_grows_from_inside_and_reveals_the_word_top_down() {
        let start = render_at_progress(0.0);
        let middle = render_at_progress(0.5);
        let full = render(false);
        assert!(
            braille_count(&start) > 0,
            "inner arc and word top are missing"
        );
        assert!(braille_count(&start) < braille_count(&middle));
        assert!(braille_count(&middle) < braille_count(&full));

        // At 25% only the upper quarter of the word is visible; at 100% the
        // whole word is. The word cells are the non-spiral background, so a
        // lower total count at 25% is enough to prove the top-down reveal.
        assert!(braille_count(&render_at_progress(0.25)) < braille_count(&full));
    }

    fn render_at_progress(progress: f64) -> String {
        finish(lines_at(false, progress))
    }

    #[test]
    fn word_reveal_is_top_down_and_starts_every_letter_together() {
        let geometry = SpiralGeometry::new();
        let half = geometry.word_grid(0.5);
        let full = geometry.word_grid(1.0);
        let extent = |grid: &[Vec<Option<f64>>]| {
            let mut min_x = PIXELS_X;
            let mut max_x = 0;
            let mut min_y = PIXELS_Y;
            let mut max_y = 0;
            for (y, row) in grid.iter().enumerate() {
                for (x, value) in row.iter().enumerate() {
                    if value.is_some() {
                        min_x = min_x.min(x);
                        max_x = max_x.max(x);
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
                    }
                }
            }
            (min_x, max_x, min_y, max_y)
        };
        let (half_min_x, half_max_x, _, half_max_y) = extent(&half);
        let (full_min_x, full_max_x, _, _) = extent(&full);
        assert_eq!(
            (half_min_x, half_max_x),
            (full_min_x, full_max_x),
            "all letters must start together"
        );
        let half_bottom = geometry.text_y0 + geometry.text_height / 2;
        assert!(
            half_max_y as i32 <= half_bottom,
            "half reveal must stay in the top half"
        );
    }

    #[test]
    fn colored_logo_uses_24_bit_rgb_and_resets() {
        let text = render(true);
        assert!(text.contains("\x1b[38;2;"));
        assert!(text.contains("\x1b[0m"));
        assert!(text.matches("\x1b[38;2;").count() > COLS);
    }

    #[test]
    fn animation_frames_have_a_stable_height() {
        let geometry = SpiralGeometry::new();
        let full = geometry.cells_at(1.0);
        let slots: Vec<usize> = full
            .iter()
            .enumerate()
            .filter(|(_, row)| !row.is_empty())
            .map(|(row, _)| row)
            .collect();
        let expected = slots.len();
        for step in 0..=10 {
            let progress = step as f64 / 10.0;
            let cells = geometry.cells_at(progress);
            let lines: Vec<String> = slots
                .iter()
                .map(|&row| row_line(&cells[row], false))
                .collect();
            assert_eq!(lines.len(), expected, "progress {progress}");
        }
    }
}
