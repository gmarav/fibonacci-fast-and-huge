//! CPU and SIMD capability detection for the startup hardware report.
//!
//! The module uses only the standard library. On Windows the CPU brand and the
//! physical core count come from a single short PowerShell/CIM query; if that
//! query is unavailable the module falls back to architecture-level
//! information instead of failing.

/// Feature names reported by [`detect`], in display order.
///
/// The x86 detection path must spell the names as literals for
/// `is_x86_feature_detected!`, so on that target the table is only referenced
/// by tests and by the non-x86 fallback.
#[allow(dead_code)]
const FEATURE_NAMES: [&str; 7] = [
    "sse2",
    "avx2",
    "avx512f",
    "avx512ifma",
    "avxvnni",
    "bmi2",
    "adx",
];

/// ANSI colors used by the colored hardware block.
const CYAN: &str = "\x1b[38;5;45m";
const GOLD: &str = "\x1b[38;5;214m";
const WHITE: &str = "\x1b[38;5;255m";
const GREEN: &str = "\x1b[38;5;114m";
const RED: &str = "\x1b[38;5;203m";
const RESET: &str = "\x1b[0m";

fn paint(text: &str, color: &str, enabled: bool) -> String {
    if enabled && !text.is_empty() {
        format!("{color}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Detected CPU capabilities.
#[derive(Clone, Debug)]
pub struct CpuInfo {
    pub brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub architecture: &'static str,
    pub features: Vec<(&'static str, bool)>,
    pub best_simd: &'static str,
}

/// Intermediate result of the platform-specific lookup.
struct PlatformCpu {
    brand: String,
    physical_cores: Option<usize>,
}

/// Detect the CPU model, core counts, SIMD features, and the kernel that
/// [`crate::uint`] will select for this process.
pub fn detect() -> CpuInfo {
    // `available_parallelism` respects process affinity and cgroup limits, so it
    // can be lower than the machine's logical core count in restricted setups.
    let logical_cores = std::thread::available_parallelism().map_or(1, usize::from);

    let platform = platform_cpu();
    let brand = platform
        .as_ref()
        .map(|p| p.brand.trim())
        .filter(|brand| !brand.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("unknown ({})", std::env::consts::ARCH));

    // Best effort: when the platform lookup cannot determine physical cores
    // (missing PowerShell/CIM, unusual VM, or non-Windows host), fall back to
    // the logical count. This over-reports on SMT machines but never fails.
    let physical_cores = platform
        .and_then(|p| p.physical_cores)
        .filter(|&count| count > 0)
        .unwrap_or(logical_cores);

    let features = detect_features();
    let best_simd = select_best_simd(&features);

    CpuInfo {
        brand,
        physical_cores,
        logical_cores,
        architecture: std::env::consts::ARCH,
        features,
        best_simd,
    }
}

/// Render a compact block suitable for stderr (one line per item).
pub fn format_human(info: &CpuInfo) -> String {
    format_human_colored(info, false)
}

/// Render the same block with optional ANSI colors. The SIMD features share one
/// compact line; the selected kernel is marked in place.
pub fn format_human_colored(info: &CpuInfo, color: bool) -> String {
    let mut out = String::with_capacity(420);
    out.push_str(&format!(
        "{} {}\n",
        paint("CPU:", CYAN, color),
        paint(&info.brand, GOLD, color)
    ));
    out.push_str(&format!(
        "{} {} physical / {} logical\n",
        paint("Cores:", CYAN, color),
        paint(&info.physical_cores.to_string(), WHITE, color),
        paint(&info.logical_cores.to_string(), WHITE, color)
    ));
    out.push_str(&format!("{} ", paint("SIMD features:", CYAN, color)));
    for (index, (name, present)) in info.features.iter().enumerate() {
        if index > 0 {
            out.push_str("  ");
        }
        let mark = if *present { "✓" } else { "✗" };
        let mark_color = if *present { GREEN } else { RED };
        out.push_str(&paint(mark, mark_color, color));
        out.push(' ');
        out.push_str(&paint(feature_label(name), WHITE, color));
        if name == &info.best_simd {
            out.push_str(&paint(" (selected)", GOLD, color));
        }
    }
    if info.best_simd == "scalar32" {
        out.push_str(&format!("  {}", paint("· scalar (selected)", GOLD, color)));
    }
    out.push('\n');
    out
}

/// Render the same information as a single valid JSON object without a
/// trailing newline.
pub fn format_json(info: &CpuInfo) -> String {
    let mut out = String::with_capacity(320);
    out.push_str("{\"brand\":\"");
    out.push_str(&escape_json(&info.brand));
    out.push_str("\",\"physical_cores\":");
    out.push_str(&info.physical_cores.to_string());
    out.push_str(",\"logical_cores\":");
    out.push_str(&info.logical_cores.to_string());
    out.push_str(",\"architecture\":\"");
    out.push_str(&escape_json(info.architecture));
    out.push_str("\",\"features\":{");
    for (index, (name, present)) in info.features.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&escape_json(name));
        out.push_str("\":");
        out.push_str(if *present { "true" } else { "false" });
    }
    out.push_str("},\"best_simd\":\"");
    out.push_str(&escape_json(info.best_simd));
    out.push_str("\"}");
    out
}

/// Human-readable label for a detected feature name.
fn feature_label(name: &str) -> &str {
    match name {
        "sse2" => "SSE2",
        "avx2" => "AVX2",
        "avx512f" => "AVX-512F",
        "avx512ifma" => "IFMA",
        "avxvnni" => "VNNI",
        "bmi2" => "BMI2",
        "adx" => "ADX",
        other => other,
    }
}

/// Quoted, backslash, and control-character escaping sufficient for JSON
/// string values.
fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                escaped.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => escaped.push(c),
        }
    }
    escaped
}

/// SIMD feature table. On x86 targets every entry is queried through the same
/// runtime detection used by [`crate::uint`]; everywhere else all entries are
/// `false` so the report stays valid on non-x86 hosts.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_features() -> Vec<(&'static str, bool)> {
    vec![
        ("sse2", is_x86_feature_detected!("sse2")),
        ("avx2", is_x86_feature_detected!("avx2")),
        ("avx512f", is_x86_feature_detected!("avx512f")),
        ("avx512ifma", is_x86_feature_detected!("avx512ifma")),
        ("avxvnni", is_x86_feature_detected!("avxvnni")),
        ("bmi2", is_x86_feature_detected!("bmi2")),
        ("adx", is_x86_feature_detected!("adx")),
    ]
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_features() -> Vec<(&'static str, bool)> {
    FEATURE_NAMES.iter().map(|&name| (name, false)).collect()
}

/// Pick the best kernel name using the same priority as
/// [`crate::uint::selected_kernel_name`]: `avx512f > avx2 > sse2 > scalar32`.
///
/// `uint` only dispatches SIMD kernels on `x86_64`, so other targets report the
/// generic `scalar` kernel name required by the hardware-info contract.
fn select_best_simd(features: &[(&'static str, bool)]) -> &'static str {
    if !cfg!(target_arch = "x86_64") {
        return "scalar";
    }
    let enabled = |name: &str| {
        features
            .iter()
            .any(|&(feature, present)| feature == name && present)
    };
    if enabled("avx512f") {
        "avx512f"
    } else if enabled("avx2") {
        "avx2"
    } else if enabled("sse2") {
        "sse2"
    } else {
        "scalar32"
    }
}

// ---------------------------------------------------------------------------
// Platform lookup
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn platform_cpu() -> Option<PlatformCpu> {
    // A single CIM query returns both the brand and the sum of `NumberOfCores`
    // across sockets. `Get-CimInstance` is preferred over the removed `wmic`
    // tool. PowerShell startup costs a few hundred milliseconds; `run_capture`
    // bounds the call so a wedged shell cannot hang the experiment.
    const SCRIPT: &str = concat!(
        "$ps = @(Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop); ",
        "if ($ps.Count -eq 0) { exit 1 }; ",
        "$name = if ($ps[0].Name) { $ps[0].Name.Trim() } else { '' }; ",
        "$cores = ($ps | Measure-Object -Property NumberOfCores -Sum).Sum; ",
        "Write-Output ($name + '|' + $cores)"
    );
    let stdout = run_capture(
        "powershell",
        &["-NoProfile", "-NonInteractive", "-Command", SCRIPT],
    )?;
    parse_platform_line(&stdout)
}

#[cfg(target_os = "linux")]
fn platform_cpu() -> Option<PlatformCpu> {
    use std::collections::HashSet;

    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut brand: Option<String> = None;
    let mut cores: HashSet<(u32, u32)> = HashSet::new();
    let (mut physical_id, mut core_id) = (None::<u32>, None::<u32>);

    // `/proc/cpuinfo` separates processor blocks with blank lines; the extra
    // empty line flushes the last block.
    for line in text.lines().chain(std::iter::once("")) {
        match line.split_once(':') {
            Some((key, value)) => match key.trim() {
                "model name" if brand.is_none() && !value.trim().is_empty() => {
                    brand = Some(value.trim().to_string());
                }
                "physical id" => physical_id = value.trim().parse().ok(),
                "core id" => core_id = value.trim().parse().ok(),
                _ => {}
            },
            None => {
                if let Some(core) = core_id.take() {
                    cores.insert((physical_id.take().unwrap_or(0), core));
                }
                physical_id = None;
                core_id = None;
            }
        }
    }

    Some(PlatformCpu {
        brand: brand?,
        physical_cores: (!cores.is_empty()).then_some(cores.len()),
    })
}

#[cfg(not(any(windows, target_os = "linux")))]
fn platform_cpu() -> Option<PlatformCpu> {
    None
}

/// Parse `<brand>|<physical cores>` produced by the platform query.
fn parse_platform_line(stdout: &str) -> Option<PlatformCpu> {
    let line = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    let (name, cores) = line.rsplit_once('|')?;
    let brand = name.trim().trim_matches('"').trim();
    if brand.is_empty() {
        return None;
    }
    let physical_cores = cores.trim().parse::<usize>().ok().filter(|&n| n > 0);
    Some(PlatformCpu {
        brand: brand.to_string(),
        physical_cores,
    })
}

/// Run `program` and capture stdout, giving up after a short timeout. Returns
/// `None` on spawn failure, timeout, or a non-zero exit status.
#[cfg(windows)]
fn run_capture(program: &str, args: &[&str]) -> Option<String> {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const TIMEOUT: Duration = Duration::from_secs(3);
    /// `CREATE_NO_WINDOW`: avoid flashing a console window.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);

    let mut child = command.spawn().ok()?;
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => return None,
        }
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CpuInfo {
        CpuInfo {
            brand: "Acme \"Quoted\" \\ CPU".to_string(),
            physical_cores: 4,
            logical_cores: 8,
            architecture: "x86_64",
            features: FEATURE_NAMES
                .iter()
                .map(|&name| (name, name == "sse2" || name == "avx2"))
                .collect(),
            best_simd: "avx2",
        }
    }

    /// Minimal structural check: balanced braces/brackets, terminated strings,
    /// no raw control characters, and no trailing commas.
    fn assert_basic_json(text: &str) {
        assert!(text.starts_with('{'), "JSON object must start with '{{'");
        assert!(text.ends_with('}'), "JSON object must end with '}}'");
        assert!(!text.contains(",}"), "trailing comma before '}}'");
        assert!(!text.contains(",]"), "trailing comma before ']'");

        let mut in_string = false;
        let mut escaped = false;
        let mut depth = 0i32;
        for c in text.chars() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                } else {
                    assert!((c as u32) >= 0x20, "raw control character in JSON string");
                }
            } else {
                match c {
                    '"' => in_string = true,
                    '{' | '[' => depth += 1,
                    '}' | ']' => {
                        depth -= 1;
                        assert!(depth >= 0, "unbalanced closing bracket");
                    }
                    _ => {}
                }
            }
        }
        assert!(!in_string, "unterminated JSON string");
        assert_eq!(depth, 0, "unbalanced brackets");
    }

    #[test]
    fn json_is_structurally_valid_and_escaped() {
        let json = format_json(&sample());
        assert_basic_json(&json);
        assert!(json.contains(r#""brand":"Acme \"Quoted\" \\ CPU""#));
        assert!(json.contains(r#""physical_cores":4"#));
        assert!(json.contains(r#""logical_cores":8"#));
        assert!(json.contains(r#""architecture":"x86_64""#));
        assert!(json.contains(r#""best_simd":"avx2""#));
        assert!(json.contains(r#""features":{"sse2":true,"avx2":true,"avx512f":false"#));
        assert!(json.contains(r#""adx":false}"#));
    }

    #[test]
    fn json_from_detection_is_valid() {
        assert_basic_json(&format_json(&detect()));
    }

    #[test]
    fn feature_names_are_stable() {
        let detected = detect();
        let names: Vec<&str> = detected.features.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, FEATURE_NAMES);
    }

    #[test]
    fn best_simd_uses_known_kernel_name() {
        let info = detect();
        let allowed: &[&str] = if cfg!(target_arch = "x86_64") {
            &["scalar32", "sse2", "avx2", "avx512f"]
        } else {
            &["scalar"]
        };
        assert!(allowed.contains(&info.best_simd), "{}", info.best_simd);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn best_simd_matches_dispatcher_priority() {
        let with = |enabled: &[&str]| -> Vec<(&'static str, bool)> {
            FEATURE_NAMES
                .iter()
                .map(|&name| (name, enabled.contains(&name)))
                .collect()
        };
        assert_eq!(
            select_best_simd(&with(&["sse2", "avx2", "avx512f"])),
            "avx512f"
        );
        assert_eq!(select_best_simd(&with(&["sse2", "avx2"])), "avx2");
        assert_eq!(select_best_simd(&with(&["sse2"])), "sse2");
        assert_eq!(select_best_simd(&with(&[])), "scalar32");
        assert_eq!(detect().best_simd, crate::uint::selected_kernel_name());
    }

    #[test]
    fn human_format_keeps_simd_features_on_one_line() {
        let human = format_human(&sample());
        let lines: Vec<&str> = human.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "CPU: Acme \"Quoted\" \\ CPU");
        assert_eq!(lines[1], "Cores: 4 physical / 8 logical");
        assert_eq!(
            lines[2],
            "SIMD features: ✓ SSE2  ✓ AVX2 (selected)  ✗ AVX-512F  ✗ IFMA  ✗ VNNI  ✗ BMI2  ✗ ADX"
        );
        assert!(human.ends_with('\n'));
    }

    #[test]
    fn colored_human_format_uses_ansi_and_checks() {
        let colored = format_human_colored(&sample(), true);
        assert!(colored.contains("\x1b[38;5;114m✓"));
        assert!(colored.contains("\x1b[38;5;203m✗"));
        assert!(colored.contains("(selected)"));
        assert!(colored.contains("\x1b[0m"));
        assert!(!format_human(&sample()).contains('\x1b'));
    }

    #[test]
    fn detection_has_sane_defaults() {
        let info = detect();
        assert!(!info.brand.is_empty());
        assert!(info.logical_cores >= 1);
        assert!(info.physical_cores >= 1);
        assert_eq!(info.architecture, std::env::consts::ARCH);
        assert_eq!(info.features.len(), FEATURE_NAMES.len());
    }
}
