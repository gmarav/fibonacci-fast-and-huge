//! Shared row model, results table, and timing formatting for the benchmark UI.
//!
//! The table is printed to stderr so that `--json` keeps stdout machine
//! readable. Colour is optional and limited to the same cyan/gold/white palette
//! used by the hardware report.

use std::fmt::Write as _;

/// ANSI palette shared with the rest of the CLI.
const CYAN: &str = "\x1b[38;5;45m";
const GOLD: &str = "\x1b[38;5;214m";
const WHITE: &str = "\x1b[38;5;255m";
const HEADER: &str = "\x1b[1m\x1b[38;5;45m";
const RESET: &str = "\x1b[0m";
/// Spaces between table columns.
const GAP: &str = "  ";

fn paint(text: &str, color: &str, enabled: bool) -> String {
    if enabled && !text.is_empty() {
        format!("{color}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Which implementation produced a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The SIMD CPU algorithms in this crate.
    Cpu,
    /// The original `fib-rust` num-bigint CPU algorithms.
    NumBigint,
    /// The `fib-rust` GMP CPU algorithms.
    Gmp,
    /// The CUDA `resident-27` fast-doubling benchmark.
    Gpu,
    /// The process-backed `cuda-ntt/block-27` engine.
    GpuBlock27,
    /// The process-backed `cuda-ntt/block-28` engine.
    GpuBlock28,
    /// The process-backed `cuda-ntt/block-29` engine.
    GpuBlock29,
}

impl Backend {
    /// Header-style label used by the results table.
    pub fn table_label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU/SIMD",
            Backend::NumBigint => "CPU/num-bigint",
            Backend::Gmp => "CPU/GMP",
            Backend::Gpu => "GPU/CUDA resident",
            Backend::GpuBlock27 => "GPU/CUDA block-27",
            Backend::GpuBlock28 => "GPU/CUDA block-28",
            Backend::GpuBlock29 => "GPU/CUDA block-29",
        }
    }

    /// Short prefix used by live measurement, output, and chart labels.
    pub fn live_label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::NumBigint => "num-bigint",
            Backend::Gmp => "gmp",
            Backend::Gpu => "GPU resident",
            Backend::GpuBlock27 => "GPU block-27",
            Backend::GpuBlock28 => "GPU block-28",
            Backend::GpuBlock29 => "GPU block-29",
        }
    }

    /// Family label used by the chart, where the text column is narrow.
    pub fn chart_label(self) -> &'static str {
        match self {
            Backend::Cpu => "SIMD",
            Backend::NumBigint => "num-bigint",
            Backend::Gmp => "GMP",
            Backend::Gpu => "GPU-r27",
            Backend::GpuBlock27 => "GPU-b27",
            Backend::GpuBlock28 => "GPU-b28",
            Backend::GpuBlock29 => "GPU-b29",
        }
    }

    /// Lowercase identifier used in the JSON summary.
    pub fn json_label(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::NumBigint => "num-bigint",
            Backend::Gmp => "gmp",
            Backend::Gpu => "gpu",
            Backend::GpuBlock27 => "gpu-block27",
            Backend::GpuBlock28 => "gpu-block28",
            Backend::GpuBlock29 => "gpu-block29",
        }
    }

    /// Whether this row is one of the CUDA engines.
    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            Backend::Gpu | Backend::GpuBlock27 | Backend::GpuBlock28 | Backend::GpuBlock29
        )
    }

    /// Whether this row comes from a CPU implementation.
    pub fn is_cpu(self) -> bool {
        matches!(self, Backend::Cpu | Backend::NumBigint | Backend::Gmp)
    }
}

/// One completed measurement, ready for the table and chart.
#[derive(Clone, Debug)]
pub struct Row {
    pub backend: Backend,
    pub method: &'static str,
    /// Worker threads used for this row; the GPU pipeline always reports 1.
    pub workers: usize,
    /// Bit length of `F(n)`.
    pub bits: u64,
    /// Median wall time in nanoseconds.
    pub median_ns: u128,
    /// Fibonacci index, so the table shows which number in the sequence was
    /// measured rather than only its bit length.
    pub index: Option<u64>,
    /// `None` when no independent reference ran; `Some` for GMP cross-family
    /// checks and GPU oracle checks.
    pub exact: Option<bool>,
}

impl Row {
    pub fn new(
        backend: Backend,
        method: &'static str,
        workers: usize,
        bits: u64,
        median_ns: u128,
    ) -> Self {
        Self {
            backend,
            method,
            workers,
            bits,
            median_ns,
            index: None,
            exact: None,
        }
    }
}

/// Format a nanosecond median as milliseconds with three decimals.
pub fn format_median(ns: u128) -> String {
    let ms = ns as f64 / 1_000_000.0;
    format!("{ms:.3} ms")
}

/// Render the aligned results table, ending with exactly one newline.
///
/// The `INDEX` column appears only when at least one row carries an index; that
/// is the minute-mode staircase, where rows from different steps would
/// otherwise be indistinguishable. The base columns are always
/// `BACKEND METHOD WORKERS BITS MEDIAN`.
pub fn render_table(rows: &[Row], color: bool) -> String {
    let show_index = rows.iter().any(|row| row.index.is_some());

    let mut headers: Vec<&str> = Vec::with_capacity(6);
    if show_index {
        headers.push("INDEX");
    }
    headers.extend(["BACKEND", "METHOD", "WORKERS", "BITS", "MEDIAN"]);

    let mut cells: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut line = Vec::with_capacity(headers.len());
        if show_index {
            line.push(row.index.map_or_else(String::new, |n| n.to_string()));
        }
        line.push(row.backend.table_label().to_string());
        line.push(row.method.to_string());
        line.push(row.workers.to_string());
        line.push(row.bits.to_string());
        line.push(format_median(row.median_ns));
        cells.push(line);
    }

    let mut widths: Vec<usize> = headers
        .iter()
        .map(|header| header.chars().count())
        .collect();
    for line in &cells {
        for (index, cell) in line.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let header_cells: Vec<String> = headers.iter().map(|header| header.to_string()).collect();
    let mut out = String::new();
    push_table_line(&mut out, &header_cells, &widths, &headers, color, true);
    for line in &cells {
        push_table_line(&mut out, line, &widths, &headers, color, false);
    }
    out
}

/// Writes the table to stderr.
pub fn print_table(rows: &[Row], color: bool) {
    eprint!("{}", render_table(rows, color));
}

/// Fixed widths for the streaming minute-mode table. They are wide enough for
/// the largest indices and millisecond values this CLI produces, so rows can be
/// emitted as soon as each step completes without a second table pass.
const STREAM_WIDTHS: [usize; 6] = [10, 17, 9, 7, 12, 12];
const STREAM_HEADERS: [&str; 6] = ["INDEX", "BACKEND", "METHOD", "WORKERS", "BITS", "MEDIAN"];

/// Print the header of the streaming minute-mode table.
pub fn stream_header(color: bool) {
    let cells: Vec<String> = STREAM_HEADERS.iter().map(|h| h.to_string()).collect();
    let mut out = String::new();
    push_table_line(
        &mut out,
        &cells,
        &STREAM_WIDTHS,
        &STREAM_HEADERS,
        color,
        true,
    );
    eprint!("{out}");
}

/// Print one completed minute-mode row immediately.
pub fn stream_row(row: &Row, color: bool) {
    let cells = vec![
        row.index.map_or_else(String::new, |n| n.to_string()),
        row.backend.table_label().to_string(),
        row.method.to_string(),
        row.workers.to_string(),
        row.bits.to_string(),
        format_median(row.median_ns),
    ];
    let mut out = String::new();
    push_table_line(
        &mut out,
        &cells,
        &STREAM_WIDTHS,
        &STREAM_HEADERS,
        color,
        false,
    );
    eprint!("{out}");
}

/// Render one table line. `names` identifies the column for alignment and
/// colour selection, while `cells` carries the actual text.
fn push_table_line(
    out: &mut String,
    cells: &[String],
    widths: &[usize],
    names: &[&str],
    color: bool,
    header: bool,
) {
    for (index, cell) in cells.iter().enumerate() {
        if index != 0 {
            out.push_str(GAP);
        }
        let name = names[index];
        let left = matches!(name, "BACKEND" | "METHOD");
        let padded = align(cell, widths[index], left);
        let tone = if header {
            HEADER
        } else {
            match name {
                "BACKEND" => CYAN,
                "METHOD" => GOLD,
                _ => WHITE,
            }
        };
        // The padding is deliberately painted as well; it keeps the visible
        // columns aligned while making the whole field one colour run.
        out.push_str(&paint(&padded, tone, color));
    }
    out.push('\n');
}

/// Left- or right-align `text` to exactly `width` visible characters.
fn align(text: &str, width: usize, left: bool) -> String {
    let text = truncate(text, width);
    let padding = width.saturating_sub(text.chars().count());
    let mut out = String::with_capacity(width);
    if left {
        out.push_str(&text);
        out.push_str(&" ".repeat(padding));
    } else {
        out.push_str(&" ".repeat(padding));
        out.push_str(&text);
    }
    out
}

/// Truncate to `width` characters, marking elision with an ellipsis. Table
/// fields never contain control characters, so this stays single-line.
fn truncate(text: &str, width: usize) -> String {
    let length = text.chars().count();
    if length <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_string();
    }
    let mut out: String = text.chars().take(width - 1).collect();
    out.push('…');
    out
}

/// GPU counters that accompany a measurement in the JSON summary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuJsonStats {
    pub gpu_ms: f64,
    pub host_ms: f64,
    pub steps: u64,
    pub transforms: u64,
    pub launches: u64,
}

/// Extra fields added by the process-backed block engines.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuBlockJson {
    pub engine: Option<String>,
    pub gpu_stream_ms: Option<f64>,
    pub call_wall_ms: Option<f64>,
    pub total_wall_ms: Option<f64>,
    pub process_wall_ms: Option<f64>,
    pub peak_vram_mib: Option<f64>,
    pub host_peak_mib: Option<f64>,
    pub output_sha256: Option<String>,
    pub output_bytes: Option<u64>,
    pub output_kept: Option<bool>,
    pub output_path: Option<String>,
    pub reference: Option<String>,
}

/// Append a JSON measurement object using the shared shape. Kept here so the
/// CLI has one implementation for normal and minute mode.
#[allow(clippy::too_many_arguments)]
pub fn push_measurement_json(
    out: &mut String,
    backend: Backend,
    method: &str,
    bits: u64,
    samples_ns: &[u128],
    median_ns: u128,
    exact: Option<bool>,
    gpu_stats: Option<GpuJsonStats>,
    block: Option<&GpuBlockJson>,
) {
    let _ = write!(
        out,
        "{{\"backend\":\"{}\",\"method\":\"{}\",\"bits\":{},\"compute_ns\":[{}],\"median_ns\":{}",
        backend.json_label(),
        method,
        bits,
        samples_ns
            .iter()
            .map(u128::to_string)
            .collect::<Vec<_>>()
            .join(","),
        median_ns
    );
    if let Some(stats) = gpu_stats {
        let _ = write!(
            out,
            ",\"gpu_ms\":{},\"host_ms\":{},\"steps\":{},\"transforms\":{},\"launches\":{}",
            json_number(stats.gpu_ms),
            json_number(stats.host_ms),
            stats.steps,
            stats.transforms,
            stats.launches
        );
    }
    if let Some(block) = block {
        if let Some(engine) = &block.engine {
            let _ = write!(out, ",\"engine\":\"{}\"", json_escape(engine));
        }
        if let Some(value) = block.gpu_stream_ms {
            let _ = write!(out, ",\"gpu_stream_ms\":{}", json_number(value));
        }
        if let Some(value) = block.call_wall_ms {
            let _ = write!(out, ",\"call_wall_ms\":{}", json_number(value));
        }
        if let Some(value) = block.total_wall_ms {
            let _ = write!(out, ",\"total_wall_ms\":{}", json_number(value));
        }
        if let Some(value) = block.process_wall_ms {
            let _ = write!(out, ",\"process_wall_ms\":{}", json_number(value));
        }
        if let Some(value) = block.peak_vram_mib {
            let _ = write!(out, ",\"peak_vram_mib\":{}", json_number(value));
        }
        if let Some(value) = block.host_peak_mib {
            let _ = write!(out, ",\"host_peak_mib\":{}", json_number(value));
        }
        if let Some(hash) = &block.output_sha256 {
            let _ = write!(out, ",\"output_sha256\":\"{}\"", json_escape(hash));
        }
        if let Some(value) = block.output_bytes {
            let _ = write!(out, ",\"output_bytes\":{value}");
        }
        if let Some(value) = block.output_kept {
            let _ = write!(out, ",\"output_kept\":{value}");
        }
        if let Some(path) = &block.output_path {
            let _ = write!(out, ",\"output_path\":\"{}\"", json_escape(path));
        }
        if let Some(reference) = &block.reference {
            let _ = write!(out, ",\"reference\":\"{}\"", json_escape(reference));
        }
    }
    let _ = write!(
        out,
        ",\"exact\":{}}}",
        exact.map_or_else(|| "null".to_string(), |value| value.to_string())
    );
}

/// JSON number or `null` for a non-finite diagnostic value.
fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

/// Escape a string for a JSON string literal.
fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{:04x}", c as u32);
            }
            c => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rows() -> Vec<Row> {
        vec![
            Row {
                backend: Backend::Cpu,
                method: "direct",
                workers: 24,
                bits: 694_241,
                median_ns: 14_957_000,
                index: None,
                exact: None,
            },
            Row {
                backend: Backend::Cpu,
                method: "matrix",
                workers: 24,
                bits: 694_241,
                median_ns: 30_794_000,
                index: None,
                exact: None,
            },
            Row {
                backend: Backend::Cpu,
                method: "doubling",
                workers: 24,
                bits: 694_241,
                median_ns: 9_587_000,
                index: None,
                exact: None,
            },
            Row {
                backend: Backend::NumBigint,
                method: "direct",
                workers: 24,
                bits: 694_241,
                median_ns: 8_500_000,
                index: None,
                exact: None,
            },
            Row {
                backend: Backend::Gmp,
                method: "direct",
                workers: 24,
                bits: 694_241,
                median_ns: 4_250_000,
                index: None,
                exact: None,
            },
            Row {
                backend: Backend::Gpu,
                method: "doubling",
                workers: 1,
                bits: 727_965,
                median_ns: 860_000,
                index: None,
                exact: Some(true),
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

    #[test]
    fn table_has_expected_columns_and_rows() {
        let output = render_table(&sample_rows(), false);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 7);

        assert!(lines[0].starts_with("BACKEND"));
        for column in ["BACKEND", "METHOD", "WORKERS", "BITS", "MEDIAN"] {
            assert!(
                lines[0].contains(column),
                "missing {column}: {:?}",
                lines[0]
            );
        }
        assert!(!lines[0].contains("INDEX"));

        assert!(lines[1].starts_with("CPU/SIMD"));
        assert!(lines[1].contains("direct"));
        assert!(lines[1].ends_with("14.957 ms"));

        assert!(lines[4].starts_with("CPU/num-bigint"));
        assert!(lines[4].contains("direct"));
        assert!(lines[4].ends_with("8.500 ms"));

        assert!(lines[5].starts_with("CPU/GMP"));
        assert!(lines[5].contains("direct"));
        assert!(lines[5].ends_with("4.250 ms"));

        assert!(lines[6].starts_with("GPU/CUDA"));
        assert!(lines[6].contains("doubling"));
        assert!(lines[6].ends_with("0.860 ms"));
    }

    #[test]
    fn table_columns_stay_aligned() {
        let output = render_table(&sample_rows(), false);
        let widths: Vec<usize> = output.lines().map(|line| line.chars().count()).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));

        // Split on whitespace and check the fixed columns of every row. The
        // resident label contains one space, and the median contributes two
        // tokens (`14.957` and `ms`).
        for line in output.lines().skip(1) {
            let mut fields: Vec<&str> = line.split_whitespace().collect();
            if fields[0] == "GPU/CUDA" {
                fields.remove(1);
                fields[0] = "GPU/CUDA";
            }
            assert_eq!(fields.len(), 6, "row: {line:?}");
            assert!(
                matches!(
                    fields[0],
                    "CPU/SIMD" | "CPU/num-bigint" | "CPU/GMP" | "GPU/CUDA"
                ),
                "row: {line:?}"
            );
            assert!(fields[4].parse::<f64>().is_ok(), "row: {line:?}");
            assert_eq!(fields[5], "ms");
        }
    }

    #[test]
    fn colour_uses_the_shared_palette_and_can_be_disabled() {
        let plain = render_table(&sample_rows(), false);
        assert!(!plain.contains('\x1b'));

        let colored = render_table(&sample_rows(), true);
        assert!(colored.contains("\x1b[1m\x1b[38;5;45m"));
        assert!(colored.contains(CYAN));
        assert!(colored.contains(GOLD));
        assert!(colored.contains(WHITE));
        assert!(colored.contains(RESET));
    }

    #[test]
    fn minute_rows_add_an_index_column() {
        let rows = vec![
            Row {
                index: Some(1_000_000),
                ..Row::new(Backend::Cpu, "doubling", 1, 694_241, 1_000_000)
            },
            Row {
                index: Some(2_000_000),
                ..Row::new(Backend::Gpu, "doubling", 1, 1_388_484, 2_000_000)
            },
        ];
        let output = render_table(&rows, false);
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines[0].trim_start().starts_with("INDEX"));
        assert!(lines[0].contains("BACKEND"));
        assert!(lines[1].contains("1000000"));
        assert!(lines[2].contains("2000000"));

        let widths: Vec<usize> = lines.iter().map(|line| line.chars().count()).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn empty_table_prints_only_the_header() {
        let output = render_table(&[], false);
        assert_eq!(output.lines().count(), 1);
        assert!(output.ends_with('\n'));
        assert!(output.starts_with("BACKEND"));
    }

    #[test]
    fn median_formatting_rounds_to_three_decimals() {
        assert_eq!(format_median(14_957_000), "14.957 ms");
        assert_eq!(format_median(24_000), "0.024 ms");
        assert_eq!(format_median(860_000), "0.860 ms");
        assert_eq!(format_median(0), "0.000 ms");
        assert_eq!(format_median(1_234_567_890_000), "1234567.890 ms");
        // Very large values lose integer precision through f64 but must stay
        // printable and finite.
        assert!(format_median(u128::MAX).ends_with(" ms"));
    }

    #[test]
    fn long_and_huge_values_do_not_break_layout() {
        let row = Row {
            backend: Backend::Gpu,
            method: "a-very-long-method-name-that-does-not-exist",
            workers: usize::MAX,
            bits: u64::MAX,
            median_ns: u128::MAX,
            index: Some(u64::MAX),
            exact: Some(false),
        };
        let output = render_table(&[row], false);
        assert!(!output.contains("\n\n"));
        assert_eq!(output.lines().count(), 2);
        // Values remain on their original line even when the table is wide.
        assert!(output.lines().nth(1).unwrap().contains("ms"));
    }

    #[test]
    fn measurement_json_is_single_object() {
        let mut out = String::new();
        push_measurement_json(
            &mut out,
            Backend::Gpu,
            "doubling",
            727_965,
            &[],
            860_000,
            Some(true),
            None,
            None,
        );
        assert_eq!(
            out,
            "{\"backend\":\"gpu\",\"method\":\"doubling\",\"bits\":727965,\"compute_ns\":[],\"median_ns\":860000,\"exact\":true}"
        );

        let mut cpu = String::new();
        push_measurement_json(
            &mut cpu,
            Backend::Cpu,
            "direct",
            694_241,
            &[15_000_000, 14_957_000],
            14_957_000,
            None,
            None,
            None,
        );
        assert!(cpu.contains("\"compute_ns\":[15000000,14957000]"));
        assert!(cpu.ends_with("\"exact\":null}"));
    }

    #[test]
    fn gpu_json_contains_engine_counters() {
        let mut out = String::new();
        push_measurement_json(
            &mut out,
            Backend::Gpu,
            "doubling",
            727_965,
            &[900_000],
            900_000,
            Some(true),
            Some(GpuJsonStats {
                gpu_ms: 0.806,
                host_ms: 1.501,
                steps: 21,
                transforms: 315,
                launches: 222,
            }),
            None,
        );
        assert!(out.contains("\"gpu_ms\":0.806"));
        assert!(out.contains("\"host_ms\":1.501"));
        assert!(out.contains("\"steps\":21"));
        assert!(out.contains("\"transforms\":315"));
        assert!(out.contains("\"launches\":222"));
        assert!(out.ends_with("\"exact\":true}"));

        let mut non_finite = String::new();
        push_measurement_json(
            &mut non_finite,
            Backend::Gpu,
            "doubling",
            1,
            &[],
            1,
            None,
            Some(GpuJsonStats {
                gpu_ms: f64::NAN,
                ..GpuJsonStats::default()
            }),
            None,
        );
        assert!(non_finite.contains("\"gpu_ms\":null"));
    }

    #[test]
    fn backend_labels_are_stable() {
        assert_eq!(Backend::Cpu.table_label(), "CPU/SIMD");
        assert_eq!(Backend::NumBigint.table_label(), "CPU/num-bigint");
        assert_eq!(Backend::Gmp.table_label(), "CPU/GMP");
        assert_eq!(Backend::Gpu.table_label(), "GPU/CUDA resident");
        assert_eq!(Backend::GpuBlock27.table_label(), "GPU/CUDA block-27");
        assert_eq!(Backend::GpuBlock28.table_label(), "GPU/CUDA block-28");
        assert_eq!(Backend::GpuBlock29.table_label(), "GPU/CUDA block-29");
        assert_eq!(Backend::Cpu.live_label(), "CPU");
        assert_eq!(Backend::NumBigint.live_label(), "num-bigint");
        assert_eq!(Backend::Gmp.live_label(), "gmp");
        assert_eq!(Backend::Gpu.live_label(), "GPU resident");
        assert_eq!(Backend::GpuBlock27.live_label(), "GPU block-27");
        assert_eq!(Backend::GpuBlock28.live_label(), "GPU block-28");
        assert_eq!(Backend::GpuBlock29.live_label(), "GPU block-29");
        assert_eq!(Backend::Cpu.chart_label(), "SIMD");
        assert_eq!(Backend::NumBigint.chart_label(), "num-bigint");
        assert_eq!(Backend::Gmp.chart_label(), "GMP");
        assert_eq!(Backend::Gpu.chart_label(), "GPU-r27");
        assert_eq!(Backend::GpuBlock27.chart_label(), "GPU-b27");
        assert_eq!(Backend::GpuBlock28.chart_label(), "GPU-b28");
        assert_eq!(Backend::GpuBlock29.chart_label(), "GPU-b29");
        assert_eq!(Backend::Cpu.json_label(), "cpu");
        assert_eq!(Backend::NumBigint.json_label(), "num-bigint");
        assert_eq!(Backend::Gmp.json_label(), "gmp");
        assert_eq!(Backend::Gpu.json_label(), "gpu");
        assert_eq!(Backend::GpuBlock27.json_label(), "gpu-block27");
        assert_eq!(Backend::GpuBlock28.json_label(), "gpu-block28");
        assert_eq!(Backend::GpuBlock29.json_label(), "gpu-block29");
        assert!(Backend::Cpu.is_cpu());
        assert!(Backend::NumBigint.is_cpu());
        assert!(Backend::Gmp.is_cpu());
        assert!(!Backend::Gpu.is_cpu());
        assert!(Backend::Gpu.is_gpu());
        assert!(Backend::GpuBlock27.is_gpu());
        assert!(Backend::GpuBlock28.is_gpu());
        assert!(Backend::GpuBlock29.is_gpu());
        // Keep the visible-width helper honest for future palette changes.
        assert_eq!(strip_ansi("\x1b[38;5;45mCPU\x1b[0m"), "CPU");
    }

    #[test]
    fn streaming_backend_column_fits_every_gpu_label() {
        // `GPU/CUDA resident` is the widest backend label; the streamed table
        // must not elide it.
        assert_eq!(STREAM_WIDTHS[1], "GPU/CUDA resident".chars().count());
        let row = Row::new(Backend::NumBigint, "doubling", 24, 694_241, 1_000_000);
        // The row renderer is private, so exercise the same width directly.
        assert_eq!(
            align(Backend::NumBigint.table_label(), STREAM_WIDTHS[1], true),
            format!("{:<width$}", "CPU/num-bigint", width = STREAM_WIDTHS[1])
        );
        assert_eq!(row.backend.table_label(), "CPU/num-bigint");
        assert_eq!(
            align(Backend::Gpu.table_label(), STREAM_WIDTHS[1], true),
            "GPU/CUDA resident"
        );
        assert_eq!(
            align(Backend::GpuBlock27.table_label(), STREAM_WIDTHS[1], true),
            "GPU/CUDA block-27"
        );
        assert_eq!(
            align(Backend::GpuBlock28.table_label(), STREAM_WIDTHS[1], true),
            "GPU/CUDA block-28"
        );
        assert_eq!(
            align(Backend::GpuBlock29.table_label(), STREAM_WIDTHS[1], true),
            "GPU/CUDA block-29"
        );
    }
}
