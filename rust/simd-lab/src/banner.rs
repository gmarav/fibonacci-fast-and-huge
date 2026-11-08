//! ANSI/Unicode startup banner for the terminal Fibonacci experiments.
//!
//! The banner is intentionally self-contained: it uses only the standard
//! library, every rendered line stays within 80 visible columns, and
//! [`render`] always terminates the block with a newline. [`print`] writes to
//! stderr so that machine-readable output on stdout remains untouched.
//!
//! Two compositions are available:
//! - [`BannerStyle::Formula`] shows block lettering, the Binet identity and a
//!   schematic golden rectangle tiled with Fibonacci squares;
//! - [`BannerStyle::Spiral`] draws the classical Fibonacci golden-rectangle
//!   spiral with Unicode Braille patterns and a large `FIBONACCI` background
//!   word. The artwork lives in [`crate::logo`].
//!
//! [`render_at`] exposes the spiral at a growth fraction, and
//! [`print_animated`] replays those frames in place on stderr.

use crate::logo;

/// Visual composition used by [`render`] and [`print`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BannerStyle {
    /// Block title, Binet formula and a golden-rectangle schematic.
    Formula,
    /// Classical Fibonacci spiral with a background word.
    Spiral,
}

/// ANSI SGR foreground for cyan/teal.
const CYAN: &str = "\x1b[38;5;45m";
/// ANSI SGR foreground for dim grey.
const DIM: &str = "\x1b[38;5;244m";
/// Gold with bold weighting, used by the formula title.
const GOLD_BOLD: &str = "\x1b[1m\x1b[38;5;214m";
/// ANSI SGR reset.
const RESET: &str = "\x1b[0m";

/// Left margin shared by both compositions.
const INDENT: &str = "   ";

/// Width of the block-letter title and its rule.
const TITLE_WIDTH: usize = 65;
/// Width of the text column next to the formula motif.
const LEFT_WIDTH: usize = 28;
/// Gap between the formula text and its motif.
const BODY_GAP: &str = "    ";

const FORMULA_TITLE: [&str; 6] = [
    "███████╗██╗██████╗  ██████╗ ███╗   ██╗ █████╗  ██████╗ ██████╗██╗",
    "██╔════╝██║██╔══██╗██╔═══██╗████╗  ██║██╔══██╗██╔════╝██╔════╝██║",
    "█████╗  ██║██████╔╝██║   ██║██╔██╗ ██║███████║██║     ██║     ██║",
    "██╔══╝  ██║██╔══██╗██║   ██║██║╚██╗██║██╔══██║██║     ██║     ██║",
    "██║     ██║██████╔╝╚██████╔╝██║ ╚████║██║  ██║╚██████╗╚██████╗██║",
    "╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝╚═╝",
];

const FORMULA_TAGLINE: &str = "EXACT FIBONACCI · NO FLOATING POINT";

/// Left-hand formula column and the golden-rectangle schematic beside it.
const FORMULA_BODY: [(&str, &str); 11] = [
    (
        "φⁿ = F(n)·φ + F(n−1)",
        "┌────────────────────────┬────────┐",
    ),
    (
        "F(n) = (φⁿ − ψⁿ) / √5",
        "│                        │        │",
    ),
    ("", "│                        │        │"),
    (
        "0 · 1 · 1 · 2 · 3 · 5 · 8",
        "│           21           │   8    │",
    ),
    (
        "13 · 21 · 34 · 55 · 89 · 144",
        "│                        │        │",
    ),
    ("", "├────────────────────────┼───┬────┤"),
    ("φ = (1 + √5) / 2", "│                        │ 5 │ 3  │"),
    (
        "  ≈ 1.6180339887498948",
        "│           13           │   ├─┬──┤",
    ),
    ("", "│                        │   │2│1 │"),
    ("F(n+1) / F(n) → φ", "│                        │   │ │1 │"),
    ("", "└────────────────────────┴───┴─┴──┘"),
];

/// Returns the default banner composition.
pub fn default_style() -> BannerStyle {
    BannerStyle::Spiral
}

/// Parses a style name case-insensitively.
///
/// `formula` and `spiral` are accepted. `portrait` is kept as a
/// backward-compatible alias for the spiral composition; any other input
/// yields `None`.
pub fn parse_style(value: &str) -> Option<BannerStyle> {
    if value.eq_ignore_ascii_case("formula") {
        Some(BannerStyle::Formula)
    } else if value.eq_ignore_ascii_case("spiral") || value.eq_ignore_ascii_case("portrait") {
        Some(BannerStyle::Spiral)
    } else {
        None
    }
}

/// Wraps `text` in an SGR prefix and a reset when `color` is enabled.
///
/// Empty text is never painted, so blank lines stay free of escape
/// sequences even in color mode.
fn paint(text: &str, sgr: &str, color: bool) -> String {
    if color && !text.is_empty() {
        format!("{sgr}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Pads `text` on the left so it is approximately centred in `width`.
///
/// The right-hand padding is omitted to keep rendered lines free of trailing
/// whitespace.
fn centered(text: &str, width: usize) -> String {
    let pad = width.saturating_sub(text.chars().count()) / 2;
    format!("{}{text}", " ".repeat(pad))
}

fn formula_lines(color: bool) -> Vec<String> {
    let mut lines = Vec::with_capacity(6 + 3 + FORMULA_BODY.len());

    for art in FORMULA_TITLE {
        lines.push(format!("{INDENT}{}", paint(art, GOLD_BOLD, color)));
    }
    lines.push(format!(
        "{INDENT}{}",
        paint(&"═".repeat(TITLE_WIDTH), DIM, color)
    ));
    lines.push(format!(
        "{INDENT}{}",
        paint(&centered(FORMULA_TAGLINE, TITLE_WIDTH), DIM, color)
    ));
    lines.push(String::new());

    for (left, right) in FORMULA_BODY {
        let mut line = String::from(INDENT);
        if left.is_empty() {
            line.push_str(&" ".repeat(LEFT_WIDTH));
        } else {
            let mut cell = String::from(left);
            let pad = LEFT_WIDTH.saturating_sub(left.chars().count());
            cell.push_str(&" ".repeat(pad));
            line.push_str(&paint(&cell, CYAN, color));
        }
        line.push_str(BODY_GAP);
        line.push_str(&paint(right, DIM, color));
        lines.push(line);
    }

    lines
}

/// Joins banner lines into a block that ends with exactly one newline.
fn finish_block(lines: Vec<String>) -> String {
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// Renders the banner at a given spiral growth fraction.
///
/// `progress` is clamped to `0.0..=1.0` and ignored by
/// [`BannerStyle::Formula`]. For the spiral, `0.0` draws the first inner arc
/// and the top row of the background word; `1.0` is identical to [`render`].
///
/// When `color` is false the result contains no ESC bytes. When it is true,
/// each coloured segment ends with `\x1b[0m`.
pub fn render_at(style: BannerStyle, color: bool, progress: f64) -> String {
    match style {
        BannerStyle::Formula => finish_block(formula_lines(color)),
        BannerStyle::Spiral => logo::render_at(color, progress),
    }
}

/// Renders the banner as a self-contained block ending in a newline.
///
/// When `color` is false the result contains no ESC bytes. When it is true,
/// each coloured segment ends with `\x1b[0m`.
pub fn render(style: BannerStyle, color: bool) -> String {
    match style {
        BannerStyle::Formula => finish_block(formula_lines(color)),
        BannerStyle::Spiral => logo::render(color),
    }
}

/// Prints the banner to stderr, keeping stdout clean for data output.
pub fn print(style: BannerStyle, color: bool) {
    eprint!("{}", render(style, color));
}

/// Prints the banner to stderr, optionally animating the spiral in place.
///
/// With `animate == false` this behaves exactly like [`print`]. Animation is
/// only meaningful for [`BannerStyle::Spiral`]; the formula composition is
/// static. Cursor control is written to stderr only, is always followed by a
/// cursor restore, and never touches stdout.
pub fn print_animated(style: BannerStyle, color: bool, animate: bool) {
    if !animate || style != BannerStyle::Spiral {
        print(style, color);
        return;
    }
    logo::print_animated(color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Removes CSI sequences so tests can measure visible width.
    fn strip_ansi(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch.is_ascii_alphabetic() {
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

    const STYLES: [BannerStyle; 2] = [BannerStyle::Formula, BannerStyle::Spiral];

    #[test]
    fn default_is_spiral() {
        assert_eq!(default_style(), BannerStyle::Spiral);
    }

    #[test]
    fn plain_output_has_no_escape_bytes() {
        for style in STYLES {
            let text = render(style, false);
            assert!(
                !text.contains('\x1b'),
                "{style:?} emitted ESC with color off"
            );
        }
    }

    #[test]
    fn colored_output_has_ansi_and_resets() {
        for style in STYLES {
            let text = render(style, true);
            assert!(text.contains("\x1b["), "{style:?} has no ANSI prefix");
            assert!(text.contains(RESET), "{style:?} never resets");
            assert!(text.ends_with('\n'), "{style:?} must end with a newline");
        }
    }

    #[test]
    fn every_line_fits_in_eighty_columns() {
        for style in STYLES {
            for color in [false, true] {
                let text = render(style, color);
                assert!(text.ends_with('\n'));
                assert!(!text.ends_with("\n\n"));
                for (index, line) in text.lines().enumerate() {
                    let width = visible_width(line);
                    assert!(
                        width <= 80,
                        "{style:?} color={color} line {} is {width} columns",
                        index + 1
                    );
                    assert!(
                        !strip_ansi(line).ends_with(' '),
                        "{style:?} color={color} line {} has trailing spaces",
                        index + 1
                    );
                }
            }
        }
    }

    #[test]
    fn render_ends_with_exactly_one_newline() {
        for style in STYLES {
            let text = render(style, false);
            assert!(text.ends_with('\n'));
            assert!(!text.ends_with("\n\n"));
            assert!(!text.trim_end_matches('\n').ends_with('\n'));
        }
    }

    #[test]
    fn formula_mentions_the_binet_identity() {
        let text = render(BannerStyle::Formula, false);
        assert!(text.contains("F(n)"));
        assert!(text.contains("F(n−1)"));
        assert!(text.contains('φ'));
        assert!(text.contains("FIBONACCI"));
    }

    #[test]
    fn spiral_differs_from_formula() {
        assert_ne!(
            render(BannerStyle::Formula, false),
            render(BannerStyle::Spiral, false)
        );
        assert_ne!(
            render(BannerStyle::Formula, true),
            render(BannerStyle::Spiral, true)
        );
    }

    #[test]
    fn render_at_final_frame_matches_render_for_both_colors() {
        for color in [false, true] {
            assert_eq!(
                render_at(BannerStyle::Spiral, color, 1.0),
                render(BannerStyle::Spiral, color)
            );
            assert_eq!(
                render_at(BannerStyle::Formula, color, 1.0),
                render(BannerStyle::Formula, color)
            );
        }
    }

    #[test]
    fn render_at_plain_output_has_no_escape_bytes() {
        for style in STYLES {
            for progress in [0.0, 0.5, 1.0] {
                let text = render_at(style, false, progress);
                assert!(
                    !text.contains('\x1b'),
                    "{style:?} progress={progress} emitted ESC with color off"
                );
                assert!(text.ends_with('\n'));
            }
        }
    }

    #[test]
    fn render_at_colored_output_has_ansi_and_resets() {
        for style in STYLES {
            for progress in [0.0, 0.5, 1.0] {
                let text = render_at(style, true, progress);
                assert!(text.contains("\x1b["), "{style:?} progress={progress}");
                assert!(text.contains(RESET), "{style:?} progress={progress}");
            }
        }
    }

    #[test]
    fn render_at_frames_stay_within_the_canvas() {
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for color in [false, true] {
                let text = render_at(BannerStyle::Spiral, color, progress);
                for (index, line) in text.lines().enumerate() {
                    assert!(
                        visible_width(line) <= 80,
                        "progress={progress} color={color} line {} is too wide",
                        index + 1
                    );
                    assert!(
                        !strip_ansi(line).ends_with(' '),
                        "progress={progress} color={color} line {} has trailing spaces",
                        index + 1
                    );
                }
            }
        }
    }

    #[test]
    fn formula_ignores_progress() {
        let plain = render(BannerStyle::Formula, false);
        let colored = render(BannerStyle::Formula, true);
        for progress in [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0] {
            assert_eq!(
                render_at(BannerStyle::Formula, false, progress),
                plain,
                "plain progress {progress}"
            );
            assert_eq!(
                render_at(BannerStyle::Formula, true, progress),
                colored,
                "colored progress {progress}"
            );
        }
    }

    #[test]
    fn spiral_progress_is_clamped_and_nan_means_origin() {
        let origin = render_at(BannerStyle::Spiral, false, 0.0);
        let full = render(BannerStyle::Spiral, false);
        assert_eq!(render_at(BannerStyle::Spiral, false, -1.0), origin);
        assert_eq!(render_at(BannerStyle::Spiral, false, f64::NAN), origin);
        assert_eq!(render_at(BannerStyle::Spiral, false, 2.0), full);
    }

    #[test]
    fn parse_style_accepts_spiral_formula_and_alias() {
        for name in ["formula", "FORMULA", "Formula"] {
            assert_eq!(parse_style(name), Some(BannerStyle::Formula));
        }
        for name in ["spiral", "SPIRAL", "Spiral"] {
            assert_eq!(parse_style(name), Some(BannerStyle::Spiral));
        }
        for name in ["portrait", "PORTRAIT", "Portrait"] {
            assert_eq!(parse_style(name), Some(BannerStyle::Spiral));
        }
    }

    #[test]
    fn parse_style_rejects_unknown_names() {
        for name in ["", "none", "fibonacci", "formula ", " spiral", "?"] {
            assert_eq!(parse_style(name), None, "unexpected match for {name:?}");
        }
    }
}
