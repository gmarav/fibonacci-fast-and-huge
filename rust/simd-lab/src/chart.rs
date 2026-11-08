//! Horizontal Braille bar chart for the measured Fibonacci methods.
//!
//! The chart is deliberately self-contained: it uses only the standard library,
//! keeps every line within 80 visible columns, and treats colour as optional.
//! Every measurement gets one row:
//!
//! ```text
//! MEASUREMENTS · LOWER IS BETTER
//!   direct   ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿  0.024 ms
//!   matrix   ⣿⣿⣿⣿⣿⣿⣿⣿⣿             0.012 ms
//! ```
//!
//! A full Braille cell (`⣿`) is one bar unit; an odd half unit is drawn with
//! the left-half cell (`⡇`). Bars grow from left to right, so the right-half
//! mirror (`⢸`) is never needed.
//!
//! The colour gradient runs along each bar, not across the series: the first
//! filled cell is white (value zero / bar origin) and the last is deep purple,
//! with lavender and violet in between. A shorter bar simply has fewer steps.
//!
//! [`render`] returns a block that ends with exactly one newline. [`print`]
//! writes that block to stderr so machine-readable stdout is left untouched.

use std::fmt::Write as _;

/// One measurement shown by the chart.
#[derive(Clone, Debug)]
pub struct Bar {
    /// Method name, for example `direct`.
    pub label: String,
    /// Median duration in nanoseconds. Used for scaling the bar length.
    pub value_ns: f64,
}

/// Chart title, including the middle dot used by the design.
const TITLE: &str = "MEASUREMENTS · LOWER IS BETTER";
/// Left margin shared by the title and every row.
const INDENT: &str = "  ";
/// Minimum label column width.
const LABEL_MIN: usize = 8;
/// Maximum label column width. Must fit `num-bigint doubling` (19 chars) and
/// `num-bigint matrix` without elision.
const LABEL_MAX: usize = 20;
/// Spaces between the label column and the bar.
const LABEL_GAP: usize = 1;
/// Spaces between the bar and the right-aligned value.
const VALUE_GAP: usize = 2;
/// Minimum width of the value column (`0.000 ms`).
const VALUE_MIN_WIDTH: usize = 8;
/// Maximum visible line width.
const MAX_WIDTH: usize = 80;
/// Preferred maximum bar width, leaving room for wider labels and values.
const MAX_BAR_WIDTH: usize = 48;
/// Smallest bar width still worth drawing.
const MIN_BAR_WIDTH: usize = 4;
/// Horizontal subpixels (half cells) in one Braille cell.
const HALF_CELLS_PER_CELL: usize = 2;
/// A completely filled Braille cell.
const FULL_CELL: char = '⣿';
/// A cell whose left half is filled, used for a fractional end.
const HALF_CELL: char = '⡇';
/// ANSI SGR reset.
const RESET: &str = "\x1b[0m";

/// Bar origin: white, representing zero.
const WHITE: (u8, u8, u8) = (255, 255, 255);
/// Lower third of a bar: light lavender.
const LAVENDER: (u8, u8, u8) = (216, 191, 255);
/// Middle of a bar: violet.
const VIOLET: (u8, u8, u8) = (155, 89, 182);
/// Bar end: deep purple.
const DEEP_PURPLE: (u8, u8, u8) = (75, 0, 130);

/// Renders the chart for `bars`.
///
/// When `color` is `false` the result contains no ESC bytes. The returned
/// string always ends with exactly one newline, including for empty input.
pub fn render(bars: &[Bar], color: bool) -> String {
    let layout = Layout::new(bars);
    let mut out = String::new();

    out.push_str(INDENT);
    out.push_str(TITLE);
    out.push('\n');

    for index in 0..layout.count() {
        out.push_str(&layout.row_line(index, color));
        out.push('\n');
    }

    out
}

/// Writes the chart to stderr.
///
/// Stdout is intentionally not used: the CLI keeps its machine-readable output
/// separate from this visual report.
pub fn print(bars: &[Bar], color: bool) {
    eprint!("{}", render(bars, color));
}

/// Pre-computed label, bar and value geometry.
struct Layout {
    /// Padded label for each row.
    labels: Vec<String>,
    /// Right-aligned millisecond value for each row.
    values: Vec<String>,
    /// Bar length for each row, in half cells.
    units: Vec<usize>,
    /// Width of the label column, always within `LABEL_MIN..=LABEL_MAX`.
    label_width: usize,
    /// Width of the right-aligned value column.
    value_width: usize,
    /// Width of the bar field in full Braille cells.
    bar_width: usize,
}

impl Layout {
    fn new(bars: &[Bar]) -> Self {
        let label_width = bars
            .iter()
            .map(|bar| bar.label.chars().count())
            .max()
            .unwrap_or(0)
            .clamp(LABEL_MIN, LABEL_MAX);

        let raw_values: Vec<String> = bars.iter().map(|bar| format_value(bar.value_ns)).collect();

        // Reserve a usable bar even when labels and values are unusually wide.
        let max_value_width = MAX_WIDTH.saturating_sub(
            INDENT.chars().count() + label_width + LABEL_GAP + VALUE_GAP + MIN_BAR_WIDTH,
        );
        let value_width = raw_values
            .iter()
            .map(|value| value.chars().count())
            .max()
            .unwrap_or(0)
            .max(VALUE_MIN_WIDTH)
            .min(max_value_width)
            .max(1);

        let bar_room = MAX_WIDTH.saturating_sub(
            INDENT.chars().count() + label_width + LABEL_GAP + VALUE_GAP + value_width,
        );
        // `value_width` was capped above, so `bar_room` is at least
        // `MIN_BAR_WIDTH`; the floor makes that invariant explicit.
        let bar_width = bar_room.min(MAX_BAR_WIDTH).max(MIN_BAR_WIDTH);

        let scaled: Vec<f64> = bars.iter().map(|bar| finite_value(bar.value_ns)).collect();
        let max_value = scaled.iter().copied().fold(0.0, f64::max);
        let total_units = bar_width * HALF_CELLS_PER_CELL;

        let mut labels = Vec::with_capacity(bars.len());
        let mut values = Vec::with_capacity(bars.len());
        let mut units = Vec::with_capacity(bars.len());

        for (bar, (value, raw_value)) in bars
            .iter()
            .zip(scaled.iter().copied().zip(raw_values.iter()))
        {
            labels.push(fit_label(&bar.label, label_width));
            values.push(fit_value(raw_value, value_width));

            let unit = if max_value > 0.0 && value > 0.0 {
                ((value / max_value) * total_units as f64)
                    .round()
                    .clamp(0.0, total_units as f64) as usize
            } else {
                0
            };
            units.push(unit);
        }

        Self {
            labels,
            values,
            units,
            label_width,
            value_width,
            bar_width,
        }
    }

    fn count(&self) -> usize {
        self.labels.len()
    }

    fn row_line(&self, index: usize, color: bool) -> String {
        let mut line = String::with_capacity(
            INDENT.chars().count()
                + self.label_width
                + LABEL_GAP
                + self.bar_width
                + VALUE_GAP
                + self.value_width
                + self.bar_width * 20,
        );

        line.push_str(INDENT);
        line.push_str(&self.labels[index]);
        line.push_str(&" ".repeat(LABEL_GAP));
        self.push_bar(&mut line, index, color);
        line.push_str(&" ".repeat(VALUE_GAP));
        line.push_str(&self.values[index]);
        line
    }

    /// Appends exactly `bar_width` visible cells: filled units, then padding.
    fn push_bar(&self, out: &mut String, index: usize, color: bool) {
        let units = self.units[index];
        let full = units / HALF_CELLS_PER_CELL;
        let half = units % HALF_CELLS_PER_CELL;
        let drawn = full + half;

        for cell in 0..full {
            push_cell(out, FULL_CELL, bar_gradient(cell, drawn), color);
        }
        if half == 1 {
            push_cell(out, HALF_CELL, bar_gradient(full, drawn), color);
        }

        out.push_str(&" ".repeat(self.bar_width - drawn));
    }
}

/// Formats a nanosecond value as milliseconds with three decimals.
fn format_value(value_ns: f64) -> String {
    format!("{:.3} ms", value_ns / 1_000_000.0)
}

/// Normalises a timing for geometry. Non-finite and negative values count as 0.
fn finite_value(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Interpolates white -> lavender -> violet -> deep purple along a bar.
fn gradient(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    if t <= 1.0 / 3.0 {
        lerp(WHITE, LAVENDER, t * 3.0)
    } else if t <= 2.0 / 3.0 {
        lerp(LAVENDER, VIOLET, (t - 1.0 / 3.0) * 3.0)
    } else {
        lerp(VIOLET, DEEP_PURPLE, (t - 2.0 / 3.0) * 3.0)
    }
}

/// Colour for one filled cell, spread across that bar's own visible length.
///
/// The first filled cell is white (zero/origin) and the last is deep purple.
/// A bar with a single filled cell uses the white endpoint.
fn bar_gradient(cell: usize, filled: usize) -> (u8, u8, u8) {
    let last = filled.saturating_sub(1);
    let t = if last == 0 {
        0.0
    } else {
        cell.min(last) as f64 / last as f64
    };
    gradient(t)
}

fn lerp(from: (u8, u8, u8), to: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let channel = |a: u8, b: u8| {
        (a as f64 + (b as f64 - a as f64) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    (
        channel(from.0, to.0),
        channel(from.1, to.1),
        channel(from.2, to.2),
    )
}

/// Appends one coloured or plain Braille cell.
fn push_cell(out: &mut String, glyph: char, rgb: (u8, u8, u8), color: bool) {
    if color {
        // Writing into a `String` cannot fail; ignore the `fmt::Result`.
        let _ = write!(
            out,
            "\x1b[38;2;{};{};{}m{glyph}{RESET}",
            rgb.0, rgb.1, rgb.2
        );
    } else {
        out.push(glyph);
    }
}

/// Truncates to `width` characters, marking elision with an ellipsis.
fn truncate_chars(text: &str, width: usize) -> String {
    let length = text.chars().count();
    if length <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }

    let mut out: String = text.chars().take(width - 1).collect();
    out.push('…');
    out
}

/// Truncates from the left so the most significant suffix survives.
fn truncate_left(text: &str, width: usize) -> String {
    let length = text.chars().count();
    if length <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }

    let mut out = String::with_capacity(width);
    out.push('…');
    out.extend(text.chars().skip(length - (width - 1)));
    out
}

/// Left-aligns and pads `text` to exactly `width` visible characters.
fn fit_label(text: &str, width: usize) -> String {
    let text = truncate_chars(text, width);
    let mut out = String::with_capacity(width);
    out.push_str(&text);
    out.push_str(&" ".repeat(width.saturating_sub(text.chars().count())));
    out
}

/// Right-aligns and pads `text` to exactly `width` visible characters.
fn fit_value(text: &str, width: usize) -> String {
    let text = truncate_left(text, width);
    let mut out = String::with_capacity(width);
    out.push_str(&" ".repeat(width.saturating_sub(text.chars().count())));
    out.push_str(&text);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Bar> {
        vec![
            Bar {
                label: "direct".to_owned(),
                value_ns: 24_000.0,
            },
            Bar {
                label: "matrix".to_owned(),
                value_ns: 12_000.0,
            },
            Bar {
                label: "doubling".to_owned(),
                value_ns: 4_000.0,
            },
        ]
    }

    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if next.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                out.push(ch);
            }
        }

        out
    }

    fn visible_width(line: &str) -> usize {
        strip_ansi(line).chars().count()
    }

    fn rows(output: &str) -> Vec<&str> {
        output.lines().skip(1).collect()
    }

    fn full_cells(line: &str) -> usize {
        line.chars().filter(|ch| *ch == FULL_CELL).count()
    }

    /// Collects the 24-bit colour of every painted Braille cell in a row.
    fn colored_cells(line: &str) -> Vec<(u8, u8, u8)> {
        let prefix = "\x1b[38;2;";
        let mut colors = Vec::new();
        let mut rest = line;

        while let Some(start) = rest.find(prefix) {
            let after_prefix = &rest[start + prefix.len()..];
            let Some(end) = after_prefix.find('m') else {
                break;
            };
            let channels: Vec<u8> = after_prefix[..end]
                .split(';')
                .map(|channel| channel.parse().unwrap())
                .collect();
            assert_eq!(channels.len(), 3);
            colors.push((channels[0], channels[1], channels[2]));
            rest = &after_prefix[end + 1..];
        }

        colors
    }

    #[test]
    fn plain_output_has_no_escapes() {
        let output = render(&sample(), false);
        assert!(!output.contains('\x1b'));
        assert!(!output.contains("[38;2;"));
        assert!(!output.contains(RESET));
    }

    #[test]
    fn color_output_uses_24_bit_ansi_and_reset() {
        let output = render(&sample(), true);
        assert!(output.contains("\x1b[38;2;"));
        assert!(output.contains(RESET));
        assert!(output.contains(FULL_CELL));

        // Every bar starts at white (zero) and ends deep purple.
        assert!(output.contains("\x1b[38;2;255;255;255m"));
        assert!(output.contains("\x1b[38;2;75;0;130m"));
    }

    #[test]
    fn visible_lines_fit_eighty_columns() {
        for color in [false, true] {
            let output = render(&sample(), color);
            for line in output.lines() {
                assert!(
                    visible_width(line) <= MAX_WIDTH,
                    "line is {} columns: {line:?}",
                    visible_width(line)
                );
            }
        }
    }

    #[test]
    fn long_labels_and_huge_values_still_fit() {
        let bars = vec![Bar {
            label: "a".repeat(200),
            value_ns: f64::MAX,
        }];

        for color in [false, true] {
            let output = render(&bars, color);
            for line in output.lines() {
                assert!(
                    visible_width(line) <= MAX_WIDTH,
                    "line is {} columns: {line:?}",
                    visible_width(line)
                );
            }
        }
    }

    #[test]
    fn largest_bar_reaches_configured_width_and_smaller_bars_are_short() {
        let layout = Layout::new(&sample());
        assert!((MIN_BAR_WIDTH..=MAX_BAR_WIDTH).contains(&layout.bar_width));
        assert_eq!(layout.units[0], layout.bar_width * HALF_CELLS_PER_CELL);

        let output = render(&sample(), false);
        let rows = rows(&output);
        assert_eq!(full_cells(rows[0]), layout.bar_width);
        assert!(full_cells(rows[1]) < full_cells(rows[0]));
        assert!(full_cells(rows[2]) < full_cells(rows[1]));

        // Half of the maximum value gives about half of the bar field.
        let half = vec![
            Bar {
                label: "half".to_owned(),
                value_ns: 50_000.0,
            },
            Bar {
                label: "full".to_owned(),
                value_ns: 100_000.0,
            },
        ];
        let layout = Layout::new(&half);
        assert_eq!(layout.units[0], layout.bar_width);
        assert_eq!(layout.units[1], layout.bar_width * HALF_CELLS_PER_CELL);
    }

    #[test]
    fn fractional_end_uses_a_half_cell() {
        let probe = vec![Bar {
            label: "max".to_owned(),
            value_ns: 100.0,
        }];
        let bar_width = Layout::new(&probe).bar_width;

        // A value of one half cell relative to the maximum rounds to exactly
        // one unit, so the row must end with the left-half glyph.
        let tiny = 100.0 / (bar_width * HALF_CELLS_PER_CELL) as f64;
        let bars = vec![
            Bar {
                label: "tiny".to_owned(),
                value_ns: tiny,
            },
            Bar {
                label: "max".to_owned(),
                value_ns: 100.0,
            },
        ];

        let layout = Layout::new(&bars);
        assert_eq!(layout.units[0], 1);
        assert_eq!(layout.units[0] % HALF_CELLS_PER_CELL, 1);

        let output = render(&bars, false);
        assert!(output.contains(HALF_CELL));
        assert_eq!(HALF_CELL, '⡇');
    }

    #[test]
    fn braille_output_only_uses_full_and_half_cells() {
        let output = render(&sample(), false);
        let braille: Vec<char> = output
            .chars()
            .filter(|ch| ('\u{2800}'..='\u{28ff}').contains(ch))
            .collect();
        assert!(!braille.is_empty());
        assert!(
            braille
                .iter()
                .all(|ch| matches!(*ch, FULL_CELL | HALF_CELL))
        );
    }

    #[test]
    fn labels_are_padded_to_a_column_between_eight_and_twelve() {
        let layout = Layout::new(&sample());
        assert!((LABEL_MIN..=LABEL_MAX).contains(&layout.label_width));
        assert_eq!(layout.label_width, LABEL_MIN);
        assert_eq!(layout.labels[0], "direct  ");

        let long = vec![Bar {
            label: "x".repeat(40),
            value_ns: 1_000.0,
        }];
        let long_layout = Layout::new(&long);
        assert_eq!(long_layout.label_width, LABEL_MAX);
        assert_eq!(long_layout.labels[0].chars().count(), LABEL_MAX);
        assert!(long_layout.labels[0].ends_with('…'));

        let output = render(&sample(), false);
        for (index, line) in rows(&output).iter().enumerate() {
            let after_indent = line.strip_prefix(INDENT).unwrap();
            let label: String = after_indent.chars().take(layout.label_width).collect();
            assert_eq!(label.chars().count(), layout.label_width);
            assert_eq!(label.trim_end(), sample()[index].label);
        }
    }

    #[test]
    fn millisecond_values_are_right_aligned() {
        let layout = Layout::new(&sample());
        assert_eq!(layout.value_width, 8);

        let output = render(&sample(), false);
        for (index, line) in rows(&output).iter().enumerate() {
            assert!(line.ends_with(&layout.values[index]));
            assert_eq!(layout.values[index].chars().count(), layout.value_width);
        }

        assert!(output.contains("0.024 ms"));
        assert!(output.contains("0.012 ms"));
        assert!(output.contains("0.004 ms"));
    }

    #[test]
    fn empty_zero_equal_and_non_finite_values_do_not_panic() {
        let empty = render(&[], false);
        assert_eq!(empty, format!("{INDENT}{TITLE}\n"));
        assert!(!empty.ends_with("\n\n"));

        let zero = vec![Bar {
            label: "zero".to_owned(),
            value_ns: 0.0,
        }];
        let output = render(&zero, false);
        assert!(output.contains("zero"));
        assert!(output.contains("0.000 ms"));
        assert!(!output.contains(FULL_CELL));
        assert!(!output.contains(HALF_CELL));

        let zeros = vec![
            Bar {
                label: "a".to_owned(),
                value_ns: 0.0,
            },
            Bar {
                label: "b".to_owned(),
                value_ns: 0.0,
            },
        ];
        let output = render(&zeros, false);
        assert!(!output.contains(FULL_CELL));
        assert!(!output.contains(HALF_CELL));

        let equal = vec![
            Bar {
                label: "a".to_owned(),
                value_ns: 7_000.0,
            },
            Bar {
                label: "b".to_owned(),
                value_ns: 7_000.0,
            },
        ];
        let layout = Layout::new(&equal);
        assert!(
            layout
                .units
                .iter()
                .all(|units| *units == layout.bar_width * HALF_CELLS_PER_CELL)
        );
        let output = render(&equal, false);
        assert_eq!(full_cells(output.lines().nth(1).unwrap()), layout.bar_width);
        assert_eq!(full_cells(output.lines().nth(2).unwrap()), layout.bar_width);

        let non_finite = vec![
            Bar {
                label: "nan".to_owned(),
                value_ns: f64::NAN,
            },
            Bar {
                label: "inf".to_owned(),
                value_ns: f64::INFINITY,
            },
            Bar {
                label: "neg".to_owned(),
                value_ns: -5.0,
            },
        ];
        let output = render(&non_finite, false);
        assert!(output.contains("NaN ms"));
        assert!(output.contains("inf ms"));
        assert!(!output.contains(FULL_CELL));
    }

    #[test]
    fn gradient_runs_from_white_through_purple_to_deep_purple() {
        assert_eq!(gradient(0.0), WHITE);
        assert_eq!(gradient(1.0 / 3.0), LAVENDER);
        assert_eq!(gradient(2.0 / 3.0), VIOLET);
        assert_eq!(gradient(1.0), DEEP_PURPLE);
        assert_eq!(gradient(-1.0), WHITE);
        assert_eq!(gradient(2.0), DEEP_PURPLE);

        assert_eq!(bar_gradient(0, 1), WHITE);
        assert_eq!(bar_gradient(0, 3), WHITE);
        assert_eq!(bar_gradient(2, 3), DEEP_PURPLE);
        assert_eq!(bar_gradient(0, 10), WHITE);
        assert_eq!(bar_gradient(9, 10), DEEP_PURPLE);
    }

    #[test]
    fn each_bar_runs_white_to_dark_along_its_own_length() {
        let output = render(&sample(), true);

        for line in rows(&output) {
            let colors = colored_cells(line);
            assert!(colors.len() > 1, "bar has no gradient: {line:?}");
            assert_eq!(colors.first(), Some(&WHITE), "start: {line:?}");
            assert_eq!(colors.last(), Some(&DEEP_PURPLE), "end: {line:?}");
        }
    }

    #[test]
    fn different_length_bars_share_the_same_gradient_endpoints() {
        let output = render(&sample(), true);
        let rows = rows(&output);
        let direct = colored_cells(rows[0]);
        let matrix = colored_cells(rows[1]);
        let doubling = colored_cells(rows[2]);

        assert!(direct.len() > matrix.len());
        assert!(matrix.len() > doubling.len());

        for colors in [&direct, &matrix, &doubling] {
            assert_eq!(colors.first(), Some(&WHITE));
            assert_eq!(colors.last(), Some(&DEEP_PURPLE));
        }
    }

    #[test]
    fn no_row_is_painted_with_a_single_series_color() {
        let output = render(&sample(), true);

        for line in rows(&output) {
            let colors = colored_cells(line);
            assert!(colors.len() > 1, "bar is too short to gradient: {line:?}");
            assert!(
                colors.iter().any(|color| *color != colors[0]),
                "row uses one solid colour: {line:?}"
            );
            assert_ne!(colors.first(), colors.last());
        }
    }

    #[test]
    fn title_is_present_and_output_has_exactly_one_trailing_newline() {
        let output = render(&sample(), false);
        assert!(output.contains(TITLE));
        assert!(output.ends_with('\n'));
        assert!(!output.ends_with("\n\n"));
        assert_eq!(output.matches('\n').count(), sample().len() + 1);
    }
}
