//! GPU capability detection for the SIMD lab.
//!
//! The module shells out to `nvidia-smi` (and optionally `nvcc`) without using
//! a shell, so it is safe on machines with no NVIDIA GPU or no toolkit. Every
//! failure path degrades to an unavailable [`GpuInfo`] with a short note; no
//! operation in this module panics because of a missing tool or unexpected
//! output.

use std::process::Command;

/// SNMP-query fields; kept separate so tests and callers see the exact contract.
const SMI_QUERY: &str = "--query-gpu=name,memory.total,driver_version,compute_cap";
const SMI_FORMAT: &str = "--format=csv,noheader";

/// Maximum number of characters retained from a tool diagnostic.
const NOTE_LIMIT: usize = 160;

const CYAN: &str = "\x1b[38;5;45m";
const GOLD: &str = "\x1b[38;5;214m";
const WHITE: &str = "\x1b[38;5;255m";
const DIM: &str = "\x1b[38;5;244m";
const RESET: &str = "\x1b[0m";

fn paint(text: &str, color: &str, enabled: bool) -> String {
    if enabled && !text.is_empty() {
        format!("{color}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Static description of the detected GPU.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub available: bool,
    pub name: String,
    pub memory_mib: u64,
    pub driver: String,
    pub compute_capability: String,
    pub cuda_version: String,
    pub note: String,
}

/// Detect the first GPU reported by `nvidia-smi` and the local CUDA toolkit.
///
/// This function never panics. If `nvidia-smi` cannot be started, exits with a
/// non-zero status, returns no data, or returns output that does not match the
/// requested CSV shape, the result has `available = false`, `name = "none"`,
/// `memory_mib = 0`, and a human-readable `note`.
pub fn detect() -> GpuInfo {
    let mut info = GpuInfo {
        available: false,
        name: "none".to_string(),
        memory_mib: 0,
        driver: String::new(),
        compute_capability: String::new(),
        cuda_version: String::new(),
        note: String::new(),
    };

    // The toolkit version is independent of GPU availability, so query it even
    // when nvidia-smi fails (a CPU-only CUDA install is still useful context).
    info.cuda_version = query_cuda_version().unwrap_or_default();

    match query_nvidia_smi() {
        Ok(stdout) => match parse_smi_csv(&stdout) {
            Some(gpu) => {
                info.available = true;
                info.name = gpu.name;
                info.memory_mib = gpu.memory_mib;
                info.driver = gpu.driver;
                info.compute_capability = gpu.compute_capability;
                if info.cuda_version.is_empty() {
                    info.note = "nvidia-smi detected the GPU; nvcc not found, CUDA version unknown"
                        .to_string();
                }
            }
            None => {
                info.note = "nvidia-smi output did not contain a parseable GPU line".to_string();
            }
        },
        Err(error) => {
            info.note = error;
        }
    }

    info
}

/// Compact, human-readable block suitable for stderr.
pub fn format_human(info: &GpuInfo) -> String {
    format_human_colored(info, false)
}

/// Colored variant of [`format_human`].
pub fn format_human_colored(info: &GpuInfo, color: bool) -> String {
    let driver = or_unknown(&info.driver);
    let compute = or_unknown(&info.compute_capability);
    let cuda = or_unknown(&info.cuda_version);

    let mut text = format!(
        "{} {}\n{} {} MiB\n{} {}  {} {}  {} {}\n",
        paint("GPU:", CYAN, color),
        paint(&info.name, GOLD, color),
        paint("GPU memory:", CYAN, color),
        paint(&info.memory_mib.to_string(), WHITE, color),
        paint("Driver:", CYAN, color),
        paint(driver, WHITE, color),
        paint("Compute capability:", CYAN, color),
        paint(compute, WHITE, color),
        paint("CUDA:", CYAN, color),
        paint(cuda, GOLD, color)
    );
    if !info.note.is_empty() {
        text.push_str(&format!(
            "{} {}\n",
            paint("Note:", DIM, color),
            paint(&info.note, DIM, color)
        ));
    }
    text
}

/// One-line JSON object with the same fields as [`GpuInfo`].
pub fn format_json(info: &GpuInfo) -> String {
    format!(
        "{{\"available\":{},\"name\":\"{}\",\"memory_mib\":{},\"driver\":\"{}\",\"compute_capability\":\"{}\",\"cuda_version\":\"{}\",\"note\":\"{}\"}}",
        info.available,
        json_escape(&info.name),
        info.memory_mib,
        json_escape(&info.driver),
        json_escape(&info.compute_capability),
        json_escape(&info.cuda_version),
        json_escape(&info.note)
    )
}

fn or_unknown(value: &str) -> &str {
    if value.is_empty() { "unknown" } else { value }
}

/// Build a command without a console window on Windows. Uses std only.
fn command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Run `nvidia-smi` and return its stdout, or a concise diagnostic.
fn query_nvidia_smi() -> Result<String, String> {
    let output = command("nvidia-smi")
        .args([SMI_QUERY, SMI_FORMAT])
        .output()
        .map_err(|error| format!("nvidia-smi could not be started: {error}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = first_line(&stderr);
        return Err(if detail.is_empty() {
            format!("nvidia-smi failed: {}", output.status)
        } else {
            format!("nvidia-smi failed ({}): {detail}", output.status)
        });
    }
    if stdout.trim().is_empty() {
        return Err("nvidia-smi returned no GPU data".to_string());
    }
    Ok(stdout)
}

/// Run `nvcc --version` and extract `release X.Y`.
fn query_cuda_version() -> Option<String> {
    let output = command("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(version) = parse_nvcc_release(&stdout) {
        return Some(version);
    }
    // Some builds print the version banner on stderr.
    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_nvcc_release(&stderr)
}

/// Parse the first parseable GPU line of `nvidia-smi` CSV output.
fn parse_smi_csv(text: &str) -> Option<GpuInfo> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .and_then(parse_smi_line)
}

/// Parse one CSV line:
/// `name, memory.total, driver_version, compute_cap`.
fn parse_smi_line(line: &str) -> Option<GpuInfo> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    // The requested query always yields four fields. Requiring all of them
    // rejects unrelated stdout such as "No devices were found".
    if fields.len() < 4 {
        return None;
    }

    let name = clean_field(fields[0]);
    if name.is_empty() || name.eq_ignore_ascii_case("none") {
        return None;
    }

    Some(GpuInfo {
        available: true,
        name,
        memory_mib: parse_memory_mib(fields[1]).unwrap_or(0),
        driver: clean_field(fields[2]),
        compute_capability: clean_field(fields[3]),
        cuda_version: String::new(),
        note: String::new(),
    })
}

/// Extract the integer MiB value from a field such as `24463 MiB`.
fn parse_memory_mib(field: &str) -> Option<u64> {
    let token = field.split_whitespace().next()?;
    if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    token.parse().ok()
}

/// Trim a CSV field and normalize the driver's "not available" marker to empty.
fn clean_field(field: &str) -> String {
    let field = field.trim();
    if field.is_empty() || field.eq_ignore_ascii_case("[n/a]") || field.eq_ignore_ascii_case("n/a")
    {
        String::new()
    } else {
        field.to_string()
    }
}

/// Find `release X.Y` in `nvcc --version` output.
fn parse_nvcc_release(text: &str) -> Option<String> {
    const MARKER: &str = "release ";
    for line in text.lines() {
        let Some(position) = line.find(MARKER) else {
            continue;
        };
        let rest = &line[position + MARKER.len()..];
        let token = rest
            .split(|c: char| c == ',' || c.is_whitespace())
            .next()
            .unwrap_or("");
        if !token.is_empty()
            && token.contains('.')
            && token.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        {
            return Some(token.to_string());
        }
    }
    None
}

/// First non-empty line, trimmed and length-bounded for use in a note.
fn first_line(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    if line.chars().count() <= NOTE_LIMIT {
        line.to_string()
    } else {
        let mut shortened: String = line.chars().take(NOTE_LIMIT).collect();
        shortened.push_str("...");
        shortened
    }
}

/// Escape a string for inclusion in a JSON string literal.
fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{0008}' => escaped.push_str("\\b"),
            '\u{000c}' => escaped.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                escaped.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SMI: &str = "NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB, 610.47, 12.0\n";

    const SAMPLE_NVCC: &str = "nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_13:58:20_Pacific_Daylight_Time_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
";

    #[test]
    fn parses_sample_nvidia_smi_line() {
        let gpu = parse_smi_csv(SAMPLE_SMI).expect("sample must parse");
        assert!(gpu.available);
        assert_eq!(gpu.name, "NVIDIA GeForce RTX 5090 Laptop GPU");
        assert_eq!(gpu.memory_mib, 24_463);
        assert_eq!(gpu.driver, "610.47");
        assert_eq!(gpu.compute_capability, "12.0");
    }

    #[test]
    fn parses_first_gpu_of_multiple_lines() {
        let text = "GPU A, 1024 MiB, 1.0, 8.6\nGPU B, 2048 MiB, 2.0, 9.0\n";
        let gpu = parse_smi_csv(text).expect("first line must parse");
        assert_eq!(gpu.name, "GPU A");
        assert_eq!(gpu.memory_mib, 1024);
    }

    #[test]
    fn trims_irregular_spacing() {
        let gpu = parse_smi_line("  NVIDIA   Test GPU ,  8192 MiB ,  555.1 ,  8.9 ").unwrap();
        assert_eq!(gpu.name, "NVIDIA   Test GPU");
        assert_eq!(gpu.memory_mib, 8192);
        assert_eq!(gpu.driver, "555.1");
        assert_eq!(gpu.compute_capability, "8.9");
    }

    #[test]
    fn memory_without_unit_and_na_values() {
        assert_eq!(parse_memory_mib("24463 MiB"), Some(24_463));
        assert_eq!(parse_memory_mib("24463"), Some(24_463));
        assert_eq!(parse_memory_mib("[N/A]"), None);
        assert_eq!(parse_memory_mib(""), None);

        let gpu = parse_smi_line("GPU X, [N/A], [N/A], [N/A]").unwrap();
        assert_eq!(gpu.memory_mib, 0);
        assert!(gpu.driver.is_empty());
        assert!(gpu.compute_capability.is_empty());
    }

    #[test]
    fn rejects_non_gpu_output() {
        assert!(parse_smi_csv("").is_none());
        assert!(parse_smi_csv("No devices were found\n").is_none());
        assert!(parse_smi_line("NVIDIA GPU, 1024 MiB").is_none());
    }

    #[test]
    fn parses_nvcc_release() {
        assert_eq!(parse_nvcc_release(SAMPLE_NVCC).as_deref(), Some("13.0"));
        assert_eq!(
            parse_nvcc_release("Cuda compilation tools, release 12.4, V12.4.131").as_deref(),
            Some("12.4")
        );
        assert!(parse_nvcc_release("command not found").is_none());
    }

    #[test]
    fn human_format_matches_contract_shape() {
        let info = GpuInfo {
            available: true,
            name: "NVIDIA GeForce RTX 5090 Laptop GPU".into(),
            memory_mib: 24_463,
            driver: "32.0.16.1047".into(),
            compute_capability: "12.0".into(),
            cuda_version: "13.0".into(),
            note: String::new(),
        };
        assert_eq!(
            format_human(&info),
            "GPU: NVIDIA GeForce RTX 5090 Laptop GPU\n\
             GPU memory: 24463 MiB\n\
             Driver: 32.0.16.1047  Compute capability: 12.0  CUDA: 13.0\n"
        );
    }

    #[test]
    fn unavailable_format_is_informative() {
        let info = GpuInfo {
            available: false,
            name: "none".into(),
            memory_mib: 0,
            driver: String::new(),
            compute_capability: String::new(),
            cuda_version: "13.0".into(),
            note: "nvidia-smi could not be started: not found".into(),
        };
        let human = format_human(&info);
        assert!(human.contains("GPU: none"));
        assert!(human.contains("Driver: unknown  Compute capability: unknown  CUDA: 13.0"));
        assert!(human.contains("Note: nvidia-smi could not be started"));
    }

    #[test]
    fn json_is_single_line_and_escaped() {
        let info = GpuInfo {
            available: true,
            name: "GPU \"quoted\" \\ slash".into(),
            memory_mib: 24_463,
            driver: "1.0".into(),
            compute_capability: "12.0".into(),
            cuda_version: "13.0".into(),
            note: "line1\nline2\ttab".into(),
        };
        let json = format_json(&info);
        assert!(!json.contains('\n'));
        assert!(json.starts_with("{\"available\":true,"));
        assert!(json.ends_with('}'));
        assert!(json.contains(r#""name":"GPU \"quoted\" \\ slash""#));
        assert!(json.contains(r#""note":"line1\nline2\ttab""#));
    }

    #[test]
    fn detect_never_panics_and_respects_fallback_invariants() {
        let info = detect();
        if !info.available {
            assert_eq!(info.name, "none");
            assert_eq!(info.memory_mib, 0);
            assert!(!info.note.is_empty());
        }
        assert!(!format_json(&info).is_empty());
    }
}
