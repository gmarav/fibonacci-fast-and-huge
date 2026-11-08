//! Live single-index progress line.
//!
//! The renderer is a pure function over a [`LineState`] snapshot, which keeps
//! it testable without a terminal. [`Progress`] owns a small background thread
//! that snapshots the shared state roughly every 100 ms and redraws one line on
//! stderr. Measurement code only calls [`Progress::begin`],
//! [`Progress::iteration`], and [`Progress::finish`]; the line never needs to
//! know which backend is running.
//!
//! The bar reflects completed methods only. The current method is represented
//! by a spinner plus elapsed time, so a single long iteration still looks
//! alive. ETA comes from the average method and per-iteration times of methods
//! that already finished. Everything is kept on one compact line, for example:
//!
//! ```text
//! [█████░░░░░░░] 4/10 left=6 CPU/GMP doubling iter 1/1 12s eta 48s ⠋
//! ```

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// ANSI palette shared with the rest of the CLI.
const CYAN: &str = "\x1b[38;5;45m";
const GREEN: &str = "\x1b[38;5;46m";
const GOLD: &str = "\x1b[38;5;214m";
const WHITE: &str = "\x1b[38;5;255m";
const DIM: &str = "\x1b[38;5;244m";
const RESET: &str = "\x1b[0m";

/// Clear the whole current line and return to column zero.
const CLEAR_LINE: &str = "\x1b[2K\r";

/// Inner width of the textual bar, excluding the brackets.
const BAR_CELLS: usize = 12;

/// Never draw a wider line than this. The compact field set normally stays
/// below 80 columns; the per-iteration detail gets the remaining room.
const MAX_LINE_WIDTH: usize = 120;

/// Only redraw once per second: frequent terminal updates look like flicker.
const RENDER_INTERVAL: Duration = Duration::from_millis(1000);
const WAKE_INTERVAL: Duration = Duration::from_millis(50);

/// Braille spinner frames; each is one terminal column wide.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A live progress line for the single-index benchmark path.
///
/// Construction is cheap when `enabled` is false: no thread is spawned and
/// every method becomes a no-op. The worker is joined by [`Progress::stop`] or
/// by `Drop`.
pub struct Progress {
    inner: Option<Arc<Shared>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct Shared {
    state: Mutex<State>,
    /// Serialises stderr writes between the worker and `finish`, so a stale
    /// frame can never land after the line was cleared.
    output: Mutex<()>,
    stop: AtomicBool,
    rendered: AtomicBool,
    color: bool,
}

struct State {
    total: usize,
    completed: usize,
    label: String,
    iterations: usize,
    /// 1-based index of the iteration currently running.
    iteration: usize,
    method_started: Instant,
    completed_method_ns: u128,
    completed_iterations: u128,
    completed_methods: u128,
    /// Optional detail for the per-iteration line (sample time, VRAM, ...).
    detail: String,
    /// Engine progress file (`iteration total detail`), when one is watched.
    progress_path: Option<PathBuf>,
    active: bool,
    frame: usize,
}

impl State {
    fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            label: String::new(),
            iterations: 1,
            iteration: 1,
            method_started: Instant::now(),
            completed_method_ns: 0,
            completed_iterations: 0,
            completed_methods: 0,
            detail: String::new(),
            progress_path: None,
            active: false,
            frame: 0,
        }
    }

    /// Average wall time of a finished method.
    fn average_method_ns(&self) -> Option<u128> {
        (self.completed_methods > 0).then(|| self.completed_method_ns / self.completed_methods)
    }

    /// Average wall time of one finished iteration across all methods.
    fn average_iteration_ns(&self) -> Option<u128> {
        (self.completed_iterations > 0)
            .then(|| self.completed_method_ns / self.completed_iterations)
    }
}

impl Progress {
    /// Create a progress line for `total_methods` measurements.
    ///
    /// When disabled, or when there is nothing to measure, no thread starts.
    pub fn new(total_methods: usize, enabled: bool, color: bool) -> Self {
        if !enabled || total_methods == 0 {
            return Self {
                inner: None,
                thread: Mutex::new(None),
            };
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State::new(total_methods)),
            output: Mutex::new(()),
            stop: AtomicBool::new(false),
            rendered: AtomicBool::new(false),
            color,
        });
        let worker_shared = Arc::clone(&shared);
        match thread::Builder::new()
            .name("fib-progress".into())
            .spawn(move || worker(worker_shared))
        {
            Ok(handle) => Self {
                inner: Some(shared),
                thread: Mutex::new(Some(handle)),
            },
            // A failed background spawn must never break the benchmark.
            Err(_) => Self {
                inner: None,
                thread: Mutex::new(None),
            },
        }
    }

    /// Mark the start of a measured method. `iterations` is the number of timed
    /// samples the method will run; the label is truncated by the renderer when
    /// it does not fit.
    pub fn begin(&self, label: &str, iterations: usize) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = lock(&inner.state);
        state.label = label.to_string();
        state.iterations = iterations.max(1);
        state.iteration = 1;
        state.detail.clear();
        state.progress_path = None;
        state.method_started = Instant::now();
        state.active = true;
    }

    /// Report the 1-based iteration that just started. Optional, but lets the
    /// line show `iter 2/3` for methods whose loop exposes its progress.
    pub fn iteration(&self, current: usize) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = lock(&inner.state);
        if state.active {
            state.iteration = current.clamp(1, state.iterations.max(1));
        }
    }

    /// Watch an engine-written progress file (`iteration total [detail]`).
    /// The background worker reads it once per second and redraws the
    /// per-iteration line. A process that is still initializing leaves the
    /// previous values in place.
    pub fn watch_file(&self, path: &Path) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = lock(&inner.state);
        if state.active {
            state.progress_path = Some(path.to_path_buf());
        }
    }

    /// Stop watching the engine progress file.
    pub fn unwatch(&self) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = lock(&inner.state);
        state.progress_path = None;
    }

    /// Set the short detail shown on the per-iteration line.
    pub fn detail(&self, text: &str) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = lock(&inner.state);
        if state.active {
            state.detail = text.to_string();
        }
    }

    /// Mark the current method as finished. `method_ns` is its median or total
    /// wall time and feeds the ETA average of later methods.
    pub fn finish(&self, method_ns: u128) {
        let Some(inner) = &self.inner else {
            return;
        };
        {
            let mut state = lock(&inner.state);
            state.completed = (state.completed + 1).min(state.total);
            state.completed_method_ns = state.completed_method_ns.saturating_add(method_ns);
            state.completed_iterations = state
                .completed_iterations
                .saturating_add(state.iterations as u128);
            state.completed_methods = state.completed_methods.saturating_add(1);
            state.active = false;
            state.progress_path = None;
            state.detail.clear();
        }
        // Leave the finished frame in place; the next `begin` overwrites both
        // rows. `stop` clears the block before the results table.
    }

    /// Stop the worker, wait for it, and clear the line.
    ///
    /// Safe to call more than once: the line is cleared only on the call that
    /// joins the thread, so `Drop` after an explicit `stop` is silent.
    pub fn stop(&self) {
        if let Some(inner) = &self.inner {
            inner.stop.store(true, Ordering::Release);
        }
        let handle = self.thread.lock().ok().and_then(|mut slot| slot.take());
        if let Some(handle) = handle {
            let _ = handle.join();
            if let Some(inner) = &self.inner {
                if inner.rendered.load(Ordering::Acquire) {
                    let mut stderr = std::io::stderr().lock();
                    // Clear both rows and return to the first row so the
                    // results table can overwrite the live block.
                    let _ = stderr.write_all(b"\r\x1b[2K\n\r\x1b[2K\r\x1b[1A");
                    let _ = stderr.flush();
                }
            }
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.stop();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Background loop: redraw at most once per [`RENDER_INTERVAL`].
fn worker(shared: Arc<Shared>) {
    let mut last_render = Instant::now();
    loop {
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        thread::sleep(WAKE_INTERVAL);
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        if last_render.elapsed() < RENDER_INTERVAL {
            continue;
        }
        last_render = Instant::now();

        let (overall, iteration_line) = {
            let mut state = lock(&shared.state);
            if !state.active {
                continue;
            }
            if let Some(path) = state.progress_path.clone() {
                if let Ok(text) = fs::read_to_string(&path) {
                    apply_progress_file(&mut state, &text);
                }
            }
            let frame = state.frame;
            state.frame = state.frame.wrapping_add(1);
            render_state(&state, shared.color, MAX_LINE_WIDTH, frame)
        };

        let _guard = lock(&shared.output);
        // `finish` or a synchronous detail frame may have changed the line
        // while this frame was being built; never draw a stale frame.
        if !lock(&shared.state).active {
            continue;
        }
        let mut stderr = std::io::stderr().lock();
        // Two-row live block; leave the cursor on the first row afterwards.
        let _ = stderr.write_all(CLEAR_LINE.as_bytes());
        let _ = stderr.write_all(overall.as_bytes());
        let _ = stderr.write_all(b"\n\r\x1b[2K");
        let _ = stderr.write_all(iteration_line.as_bytes());
        let _ = stderr.write_all(b"\r\x1b[1A");
        let _ = stderr.flush();
        shared.rendered.store(true, Ordering::Release);
    }
}

/// Snapshot consumed by the pure renderer.
struct LineState<'a> {
    completed: usize,
    total: usize,
    label: &'a str,
    iteration: usize,
    iterations: usize,
    elapsed: Duration,
    average_method_ns: Option<u128>,
    average_iteration_ns: Option<u128>,
    detail: &'a str,
    spinner: usize,
}

/// Parse `iteration total [detail...]` written by an engine.
fn apply_progress_file(state: &mut State, text: &str) {
    let mut fields = text.split_whitespace();
    let Some(iteration) = fields.next().and_then(|value| value.parse::<usize>().ok()) else {
        return;
    };
    let Some(total) = fields.next().and_then(|value| value.parse::<usize>().ok()) else {
        return;
    };
    if total == 0 {
        return;
    }
    state.iterations = total;
    state.iteration = iteration.clamp(1, total);
    state.detail = fields.collect::<Vec<_>>().join(" ");
}

fn render_state(state: &State, color: bool, width: usize, spinner: usize) -> (String, String) {
    let line = LineState {
        completed: state.completed,
        total: state.total,
        label: &state.label,
        iteration: state.iteration,
        iterations: state.iterations,
        elapsed: state.method_started.elapsed(),
        average_method_ns: state.average_method_ns(),
        average_iteration_ns: state.average_iteration_ns(),
        detail: &state.detail,
        spinner,
    };
    (
        render_overall(&line, color, width),
        render_iteration_line(&line, color, width),
    )
}

struct Segment {
    text: String,
    color: Option<&'static str>,
}

impl Segment {
    fn new(text: impl Into<String>, color: Option<&'static str>) -> Self {
        Self {
            text: text.into(),
            color,
        }
    }

    fn plain(text: impl Into<String>) -> Self {
        Self::new(text, None)
    }
}

/// Render one progress line. Visible width never exceeds `width`; ANSI escapes
/// do not count towards it. The result is always a single line.
fn render_overall(state: &LineState<'_>, color: bool, width: usize) -> String {
    let (filled, empty) = bar_counts(state.completed, state.total);
    let left = state.total.saturating_sub(state.completed);
    let counts = format!("{}/{} left={}", state.completed, state.total, left);
    let elapsed = format_duration_ns(state.elapsed.as_nanos());
    let rough = state.average_method_ns.is_none();
    // A single-iteration method has no completed iteration to estimate from
    // and no future work inside the method; omit the field instead of
    // permanently showing "estimating".
    let eta = match eta_ns(state) {
        Some(ns) => Some(format!(
            "eta{}{}",
            if rough { "~" } else { " " },
            format_eta_ns(ns)
        )),
        None if state.iterations <= 1 => None,
        None => Some("eta=estimating".to_string()),
    };

    let mut segments = Vec::with_capacity(12);
    segments.push(Segment::new("[", Some(WHITE)));
    segments.push(Segment::new("█".repeat(filled), Some(GREEN)));
    segments.push(Segment::new("░".repeat(empty), Some(DIM)));
    segments.push(Segment::new("]", Some(WHITE)));
    segments.push(Segment::plain(" "));
    segments.push(Segment::new(counts, Some(WHITE)));
    segments.push(Segment::plain(" "));
    let label_index = segments.len();
    segments.push(Segment::new(String::new(), Some(GOLD)));
    segments.push(Segment::new(format!(" {elapsed}"), Some(WHITE)));
    if let Some(eta) = eta {
        segments.push(Segment::new(format!(" {eta}"), Some(GOLD)));
    }
    segments.push(Segment::new(
        format!(" {}", SPINNER[state.spinner % SPINNER.len()]),
        Some(CYAN),
    ));

    // Give the label whatever room the fixed segments leave, then truncate.
    let fixed: usize = segments
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != label_index)
        .map(|(_, segment)| visible_width(&segment.text))
        .sum();
    let label_budget = width.saturating_sub(fixed);
    segments[label_index].text = truncate_label(state.label.trim(), label_budget);

    render_segments(&segments, color, width)
}

fn render_iteration_line(state: &LineState<'_>, color: bool, width: usize) -> String {
    let (filled, empty) = bar_counts(state.iteration, state.iterations);
    let mut segments = Vec::with_capacity(8);
    segments.push(Segment::plain("  iter "));
    segments.push(Segment::new("[", Some(WHITE)));
    segments.push(Segment::new("█".repeat(filled), Some(GREEN)));
    segments.push(Segment::new("░".repeat(empty), Some(DIM)));
    segments.push(Segment::new("]", Some(WHITE)));
    segments.push(Segment::new(
        format!(" {}/{}", state.iteration, state.iterations),
        Some(WHITE),
    ));
    if !state.detail.is_empty() {
        segments.push(Segment::new(format!(" · {}", state.detail), Some(DIM)));
    }
    render_segments(&segments, color, width)
}

fn render_segments(segments: &[Segment], color: bool, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    for segment in segments {
        if segment.text.is_empty() {
            continue;
        }
        let remaining = width.saturating_sub(used);
        if remaining == 0 {
            break;
        }
        let length = visible_width(&segment.text);
        let (text, truncated) = if length > remaining {
            (
                segment.text.chars().take(remaining).collect::<String>(),
                true,
            )
        } else {
            (segment.text.clone(), false)
        };
        used += visible_width(&text);
        if color {
            if let Some(style) = segment.color {
                out.push_str(style);
                out.push_str(&text);
                out.push_str(RESET);
            } else {
                out.push_str(&text);
            }
        } else {
            out.push_str(&text);
        }
        if truncated {
            break;
        }
    }
    if color && !out.is_empty() {
        out.push_str(RESET);
    }
    out
}

fn bar_counts(completed: usize, total: usize) -> (usize, usize) {
    if total == 0 {
        return (0, BAR_CELLS);
    }
    let completed = completed.min(total);
    // Round to the nearest cell so 50% is exactly half of the bar.
    let filled = ((completed * BAR_CELLS + total / 2) / total).min(BAR_CELLS);
    (filled, BAR_CELLS - filled)
}

/// ETA for the work that is still ahead.
///
/// The current method is not part of `remaining_methods`; its unfinished
/// iterations are added through `average_iteration_ns`.
fn eta_ns(state: &LineState<'_>) -> Option<u128> {
    let remaining_methods = state
        .total
        .saturating_sub(state.completed.saturating_add(1));
    // The current iteration is still running, so it counts as remaining even
    // when it is the last one and `completed` has not advanced yet.
    let remaining_iterations = state.iterations.saturating_sub(state.iteration) + 1;

    // Estimate one iteration from the completed iterations of the current
    // method. Before the first iteration finishes there is no measurement yet,
    // so the ETA stays at "estimating" instead of reporting the elapsed time.
    let live_per_iteration =
        (state.iteration > 1).then(|| state.elapsed.as_nanos() / (state.iteration - 1) as u128);
    let per_iteration = match state.average_iteration_ns {
        Some(average) => {
            let stale = state
                .average_method_ns
                .is_some_and(|method| state.elapsed.as_nanos() > method.saturating_mul(2));
            if stale {
                Some(live_per_iteration.map_or(average, |live| average.max(live)))
            } else {
                Some(average)
            }
        }
        None => live_per_iteration,
    };

    let current = per_iteration.map(|per| per.saturating_mul(remaining_iterations as u128));
    let future = state
        .average_method_ns
        .map(|average| average.saturating_mul(remaining_methods as u128));
    match (current, future) {
        (Some(current), Some(future)) => Some(current.saturating_add(future)),
        (Some(current), None) => Some(current),
        (None, Some(future)) => Some(future),
        (None, None) => None,
    }
}

fn truncate_label(label: &str, budget: usize) -> String {
    let length = visible_width(label);
    if length <= budget {
        return label.to_string();
    }
    if budget == 0 {
        return String::new();
    }
    if budget <= 3 {
        return label.chars().take(budget).collect();
    }
    let mut truncated: String = label.chars().take(budget - 1).collect();
    truncated.push('…');
    truncated
}

fn visible_width(text: &str) -> usize {
    text.chars().count()
}

/// Compact duration used by the progress line: `1.2s`, `1m 05s`, `1h 02m`.
fn format_duration_ns(ns: u128) -> String {
    format_seconds(ns / 1_000_000_000)
}

/// ETA never rounds down to zero while work is still running.
fn format_eta_ns(ns: u128) -> String {
    format_seconds(ns.div_ceil(1_000_000_000))
}

fn format_seconds(seconds: u128) -> String {
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3_600 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {:02}m", seconds / 3_600, (seconds % 3_600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(completed: usize, total: usize) -> LineState<'static> {
        LineState {
            completed,
            total,
            label: "CPU/SIMD direct n=1000000",
            iteration: 1,
            iterations: 1,
            elapsed: Duration::from_millis(1_250),
            average_method_ns: None,
            average_iteration_ns: None,
            detail: "",
            spinner: 0,
        }
    }

    fn visible_len(text: &str) -> usize {
        let mut length = 0;
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                // Skip a CSI sequence such as `\x1b[38;5;45m`.
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for ch in chars.by_ref() {
                        if ch.is_ascii_alphabetic() {
                            break;
                        }
                    }
                }
            } else {
                length += 1;
            }
        }
        length
    }

    #[test]
    fn bar_is_empty_half_and_full() {
        let empty = render_overall(&state(0, 4), false, MAX_LINE_WIDTH);
        assert_eq!(empty.matches('█').count(), 0);
        assert_eq!(empty.matches('░').count(), BAR_CELLS);

        let half = render_overall(&state(2, 4), false, MAX_LINE_WIDTH);
        assert_eq!(half.matches('█').count(), BAR_CELLS / 2);
        assert_eq!(half.matches('░').count(), BAR_CELLS / 2);

        let full = render_overall(&state(4, 4), false, MAX_LINE_WIDTH);
        assert_eq!(full.matches('█').count(), BAR_CELLS);
        assert_eq!(full.matches('░').count(), 0);
    }

    #[test]
    fn color_switches_ansi_off_and_on() {
        let plain = render_overall(&state(1, 3), false, MAX_LINE_WIDTH);
        assert!(!plain.contains('\x1b'));
        assert!(plain.contains("1/3 left=2"));
        assert!(plain.contains("CPU/SIMD direct"));

        let colored = render_overall(&state(1, 3), true, MAX_LINE_WIDTH);
        assert!(colored.contains(GREEN));
        assert!(colored.contains(DIM));
        assert!(colored.contains(GOLD));
        assert!(colored.contains(CYAN));
        assert!(colored.contains(RESET));
    }

    #[test]
    fn line_is_single_line_english_and_bounded() {
        let mut line_state = state(1, 4);
        line_state.iteration = 2;
        line_state.iterations = 3;
        line_state.average_method_ns = Some(4_000_000_000);
        line_state.average_iteration_ns = Some(1_000_000_000);
        for color in [false, true] {
            for width in [20, 40, 60, 80, 100] {
                let overall = render_overall(&line_state, color, width);
                let iterations = render_iteration_line(&line_state, color, width);
                for line in [&overall, &iterations] {
                    assert!(!line.contains('\n'), "line must not wrap: {line:?}");
                    assert!(!line.contains('\r'), "line must not contain CR: {line:?}");
                    assert!(
                        !line
                            .chars()
                            .any(|ch| ('\u{0400}'..='\u{04ff}').contains(&ch)),
                        "line must be English only: {line:?}"
                    );
                    assert!(
                        visible_len(line) <= width,
                        "width {width}: {line:?} is {} columns",
                        visible_len(line)
                    );
                }
                if width >= 40 {
                    assert!(overall.contains("left="));
                    assert!(iterations.contains("iter"));
                }
            }
        }
    }

    #[test]
    fn label_is_truncated_before_the_rest_of_the_line() {
        let mut state = state(1, 3);
        state.label = "CPU/num-bigint matrix with a very long descriptive label";
        let line = render_overall(&state, false, 60);
        assert!(visible_len(&line) <= 60);
        assert!(line.contains('…'));
        assert!(!line.contains("eta"));
    }

    #[test]
    fn iteration_counter_and_eta_are_rendered() {
        let mut line_state = state(1, 4);
        line_state.iteration = 2;
        line_state.iterations = 3;
        line_state.average_method_ns = Some(4_000_000_000);
        line_state.average_iteration_ns = Some(1_000_000_000);
        let line = render_overall(&line_state, false, MAX_LINE_WIDTH);
        let iter = render_iteration_line(&line_state, false, MAX_LINE_WIDTH);
        assert!(iter.contains("iter"));
        assert!(iter.contains("2/3"));
        assert!(line.contains("1/4 left=3"));
        assert!(line.contains("eta 10s"));
        assert!(!line.contains("estimating"));

        let first = render_overall(&state(0, 4), false, MAX_LINE_WIDTH);
        let first_iter = render_iteration_line(&state(0, 4), false, MAX_LINE_WIDTH);
        assert!(first_iter.contains("1/1"));
        assert!(first.contains("0/4 left=4"));
        assert!(!first.contains("eta"));
    }

    #[test]
    fn eta_stays_estimating_until_the_first_iteration_finishes() {
        let mut line_state = state(0, 1);
        line_state.iteration = 1;
        line_state.iterations = 3;
        line_state.elapsed = Duration::from_secs(10);
        let line = render_overall(&line_state, false, MAX_LINE_WIDTH);
        assert!(line.contains("eta=estimating"));
        assert!(!line.contains("eta 10s"));
    }

    #[test]
    fn eta_reports_remaining_time_from_completed_iterations() {
        let mut line_state = state(0, 1);
        line_state.iteration = 3;
        line_state.iterations = 3;
        line_state.elapsed = Duration::from_secs(40);
        let line = render_overall(&line_state, false, MAX_LINE_WIDTH);
        assert!(line.contains("eta~20s"));
    }

    #[test]
    fn preferred_shape_is_compact() {
        let mut line_state = state(4, 10);
        line_state.label = "CPU/GMP doubling";
        line_state.iteration = 1;
        line_state.iterations = 1;
        line_state.elapsed = Duration::from_millis(12_400);
        line_state.average_method_ns = Some(9_640_000_000);
        line_state.average_iteration_ns = Some(9_640_000_000);
        let line = render_overall(&line_state, false, 80);
        let iter = render_iteration_line(&line_state, false, 80);
        let bar = format!("{}{}", "█".repeat(5), "░".repeat(BAR_CELLS - 5));
        assert_eq!(
            line,
            format!("[{bar}] 4/10 left=6 CPU/GMP doubling 12s eta 58s ⠋")
        );
        let iter_bar = "█".repeat(BAR_CELLS);
        assert_eq!(iter, format!("  iter [{iter_bar}] 1/1"));
        assert!(visible_len(&line) <= 80);
    }

    #[test]
    fn durations_are_compact() {
        assert_eq!(format_duration_ns(0), "0s");
        assert_eq!(format_duration_ns(1_250_000_000), "1s");
        assert_eq!(format_duration_ns(65_000_000_000), "1m 05s");
        assert_eq!(format_duration_ns(3_725_000_000_000), "1h 02m");
    }

    #[test]
    fn disabled_progress_never_draws_or_spawns() {
        let progress = Progress::new(4, false, true);
        progress.begin("CPU/SIMD direct n=10", 1);
        progress.iteration(1);
        progress.finish(1);
        progress.stop();

        let empty = Progress::new(0, true, true);
        empty.stop();
    }
}
