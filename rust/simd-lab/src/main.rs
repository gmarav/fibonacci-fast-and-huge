#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::useless_format
)]

//! CLI for the SIMD-backed exact Fibonacci experiment.
//!
//! This binary intentionally mirrors the main project's method selection while
//! remaining independent of its backend adapters. The SIMD kernels are the CPU
//! subject of the experiment; every CUDA engine is an external executable that
//! runs its own iteration loop and reports timings through a JSON stats file.

use fib_rust::Method;
use fib_rust::backend::Natural;
use fib_simd_lab::banner::{self, BannerStyle};
use fib_simd_lab::chart;
use fib_simd_lab::cpu;
use fib_simd_lab::direct::direct;
use fib_simd_lab::doubling::doubling;
use fib_simd_lab::gpu;
use fib_simd_lab::gpu_block::{self, BlockEngine, BlockVariant};
use fib_simd_lab::limits::{self, AutoChoice, EngineFit, GpuEngine, Resources};
use fib_simd_lab::matrix::matrix;
use fib_simd_lab::native_runner::{self, GmpRunner, NumBigintRunner};
use fib_simd_lab::progress::Progress;
use fib_simd_lab::report::{self, Backend, GpuBlockJson, Row};
use fib_simd_lab::uint::{SimdUint, selected_kernel_name};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const HELP: &str = "Fibonacci: fast and huge.

Usage: fib-simd-lab [flags] n
       fib-simd-lab --show-limits
       fib-simd-lab limits

Exact Fibonacci with F(0)=0 and F(1)=1. The CLI runs three independent CPU
families alongside optional CUDA backends:
  CPU/SIMD        project-local SIMD-backed unsigned integers
  CPU/num-bigint  the original fib-rust pure-Rust num-bigint implementation
  CPU/GMP         the fib-rust rug::Integer implementation on GMP
  GPU/CUDA resident   cuda-ntt/resident-27 process-backed 2^27 engine
  GPU/CUDA block-27   cuda-ntt/block-27 process-backed block engine
  GPU/CUDA block-28   cuda-ntt/block-28 process-backed block engine
  GPU/CUDA block-29   cuda-ntt/block-29 process-backed block engine

Flags must precede n. In minute mode n is optional and defaults to 1,000,000.
  --method direct|matrix|doubling|compare   default: doubling
  --backend VALUE                 default: cpu; auto above 64e6
      cpu             CPU/SIMD only
      num-bigint      CPU/num-bigint only (native is an alias)
      gmp             CPU/GMP only
      gpu             GPU/CUDA resident 2^27 only
      gpu-block27     GPU/CUDA block-27 only
      gpu-block28     GPU/CUDA block-28 only
      gpu-block29     GPU/CUDA block-29 only
      gpu-block       all three block engines, one table row each, run sequentially
      gpu-all         all four GPU engines: resident + block27/28/29
      gmp-gpu         GMP + resident GPU
      both            CPU/SIMD + resident GPU
      all             CPU/SIMD + num-bigint + GMP + resident GPU
      auto            pick the fastest exact CUDA engine that fits (see below)
  --gpu-resident-bin PATH        resident-27 engine executable
  --gpu-block27-bin PATH         block-27 block27_fib.exe (env FIB_GPU_BLOCK27)
  --gpu-block28-bin PATH         block-28 block28_fib.exe (env FIB_GPU_BLOCK28)
  --gpu-block29-bin PATH         block-29 block29_fib.exe (env FIB_GPU_BLOCK29)
  --gpu-out PATH                 retain the full block-engine output at PATH
  --keep-gpu-output              keep the temporary output instead of deleting it
  --gpu-reference PATH          compare block output with this binary file
  --gpu-sha256 HEX              compare block output SHA-256 with this value
  --force-memory                run selected GPU backends even if preflight fails
  --force-index                 override a hard/planner index limit
  --show-limits                 print the GPU limit/memory table and exit
  --minute                      phased staircase benchmark
  --minute-seconds N            budget per phase (default: 60)
  --iterations N                timed samples per method (default: 1)
  --workers N                   0 selects available CPUs (default: 0)
  --banner=formula|spiral|portrait|none     startup banner (default: spiral)
  --no-banner                   skip the startup banner
  --no-color                    disable ANSI colors in the banner
  --no-progress                 disable the live progress line
  --no-animate                  disable banner animation
  --no-print                    skip decimal output
  --no-chart                    skip the measurement chart
  --force-simd                  run slow SIMD methods at large n
  --json                        print a JSON summary
  --no-hwinfo                   skip the CPU/GPU hardware report
  --help                        show this help

Backend limits (verified by the project):
  resident   max exact n = 6,186,556,317; n = 6,186,556,318 is rejected.
             Device model: 108 B/element * 2^27 + 1.30 GiB context = 14.80 GiB.
  block-27   verified n = 61,865,563,170; no planner cap. Device is VRAM-bound:
             12.31 GiB fixed + 0.75 GiB per 2^26-limb operand block. On a 24 GB
             card the practical maximum is roughly 80-87e9 (k=13-14).
  block-28   exact 1e11 anchor (verified), measured max n = 184,000,000,000;
             planner max n = 184,070,026,875 (15 x 2^27 limbs per operand).
             185e9 needs 16 blocks and is rejected by the engine preflight.
             Device model 16.31 GiB at 1e11, 22.31 GiB at 184e9; host model
             25.37 GiB at 1e11, 42.49 GiB at 184e9.
  block-29   planner max n = 73,628,010,691 (3 x 2^28 limbs per operand).
             Device model 22.30 GiB at the target; host model 27.08 GiB per
             product.

Startup preflight: before starting a CUDA engine, allocating device memory, or
starting a block engine, the CLI prints a compact GPU limit block and a
recommendation line. It estimates required device/host memory, compares it with
free VRAM (nvidia-smi) and available host RAM (GlobalMemoryStatusEx), keeps a
0.5 GiB safety margin, and skips an infeasible engine with an English note.
--force-memory overrides the memory check; --force-index overrides a hard index
cap. --backend auto prefers resident up to 6,186,556,317, then block-27 while its
VRAM preflight fits, then block-28 above block-27's practical maximum up to
184,070,026,875, and block-29 only as a fallback if block-27/block-28 do not fit
and n is within 73,628,010,691. It rejects n above 184,070,026,875 and never
silently selects an over-limit engine.

--method applies to all three CPU families. Both fib-rust backends use
reuse=true and split-multiplication=true, matching the reference CLI, and run
in private Rayon pools with the same worker cap. All GPU backends implement fast
doubling only, so GPU runs require --method doubling or compare. Block engines
are process-backed: the CLI parses their stdout, hashes the output file, compares
the full value with a CPU doubling reference at n <= 6,186,556,317, and reports
the SHA-256 above that. A huge result is never marked exact unless it was
actually compared. Minute mode supports the resident GPU but not the block
engines.
";

/// Below this index the main Rust implementation runs sequentially; mirrored
/// here so small indices do not pay Rayon overhead.
const SMALL_INDEX_WORKERS: u64 = 95_000;

/// Above this index SIMD direct/matrix are skipped by default: they are orders
/// of magnitude slower than num-bigint/GMP at multi-million-bit sizes.
const SIMD_SKIP_HEAVY_INDEX: u64 = 8_000_000;
/// Above this index all SIMD methods are skipped by default.
const SIMD_SKIP_ALL_INDEX: u64 = 64_000_000;

/// Default staircase start when `--minute` is used without an index.
const MINUTE_START: u64 = 1_000_000;

#[derive(Debug)]
struct Options {
    method: String,
    backend: String,
    backend_explicit: bool,
    gpu_resident_bin: Option<String>,
    gpu_block27_bin: Option<String>,
    gpu_block28_bin: Option<String>,
    gpu_block29_bin: Option<String>,
    gpu_out: Option<String>,
    keep_gpu_output: bool,
    gpu_reference: Option<String>,
    gpu_sha256: Option<String>,
    force_memory: bool,
    force_index: bool,
    show_limits: bool,
    iterations: usize,
    workers: usize,
    minute: bool,
    minute_seconds: u64,
    banner: Option<BannerStyle>,
    no_color: bool,
    no_progress: bool,
    force_simd: bool,
    no_animate: bool,
    no_print: bool,
    no_chart: bool,
    json: bool,
    no_hwinfo: bool,
    index: u64,
}

/// Which implementation families a `--backend` value selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Selection {
    simd: bool,
    num_bigint: bool,
    gmp: bool,
    /// Resident CUDA `resident-27` process-backed engine.
    gpu: bool,
    /// Process-backed block-27 engine.
    block27: bool,
    /// Process-backed block-28 engine.
    block28: bool,
    /// Process-backed block-29 engine.
    block29: bool,
    /// Resolve the CUDA engine from the index and the memory preflight.
    auto: bool,
}

impl Selection {
    fn any_gpu(self) -> bool {
        self.gpu || self.block27 || self.block28 || self.block29 || self.auto
    }

    fn engines(self) -> Vec<GpuEngine> {
        let mut engines = Vec::new();
        if self.gpu {
            engines.push(GpuEngine::Resident);
        }
        if self.block27 {
            engines.push(GpuEngine::Block27);
        }
        if self.block28 {
            engines.push(GpuEngine::Block28);
        }
        if self.block29 {
            engines.push(GpuEngine::Block29);
        }
        engines
    }
}

#[allow(clippy::too_many_arguments)]
fn selection(
    simd: bool,
    num_bigint: bool,
    gmp: bool,
    gpu: bool,
    block27: bool,
    block28: bool,
    block29: bool,
    auto: bool,
) -> Selection {
    Selection {
        simd,
        num_bigint,
        gmp,
        gpu,
        block27,
        block28,
        block29,
        auto,
    }
}

/// Map a `--backend` value to the families it runs.
fn select_backend(backend: &str) -> Option<Selection> {
    match backend {
        "cpu" => Some(selection(
            true, false, false, false, false, false, false, false,
        )),
        // `native` is the earlier name of the num-bigint backend.
        "num-bigint" | "native" => Some(selection(
            false, true, false, false, false, false, false, false,
        )),
        "gmp" => Some(selection(
            false, false, true, false, false, false, false, false,
        )),
        "gpu" => Some(selection(
            false, false, false, true, false, false, false, false,
        )),
        "gpu-block27" | "gpu-block-27" => Some(selection(
            false, false, false, false, true, false, false, false,
        )),
        "gpu-block28" | "gpu-block-28" => Some(selection(
            false, false, false, false, false, true, false, false,
        )),
        "gpu-block29" | "gpu-block-29" => Some(selection(
            false, false, false, false, false, false, true, false,
        )),
        "gpu-block" | "gpu-blocks" => Some(selection(
            false, false, false, false, true, true, true, false,
        )),
        "gpu-all" | "gpu-everything" | "all-gpu" => Some(selection(
            false, false, false, true, true, true, true, false,
        )),
        "auto" | "recommend" => Some(selection(
            false, false, false, false, false, false, false, true,
        )),
        // GMP plus GPU keeps the CPU reference while dropping SIMD and
        // num-bigint, which are the slowest rows at large indices.
        "gmp-gpu" | "gmp+gpu" => Some(selection(
            false, false, true, true, false, false, false, false,
        )),
        "both" => Some(selection(
            true, false, false, true, false, false, false, false,
        )),
        "all" => Some(selection(
            true, true, true, true, false, false, false, false,
        )),
        _ => None,
    }
}

fn backend_values() -> &'static str {
    "cpu, num-bigint, gmp, gpu, gpu-block27, gpu-block28, gpu-block29, gpu-block, gpu-all, gmp-gpu, both, all, or auto"
}

fn parse(args: &[String]) -> Result<Option<Options>, String> {
    let mut options = Options {
        method: "doubling".into(),
        backend: "cpu".into(),
        backend_explicit: false,
        gpu_resident_bin: None,
        gpu_block27_bin: None,
        gpu_block28_bin: None,
        gpu_block29_bin: None,
        gpu_out: None,
        keep_gpu_output: false,
        gpu_reference: None,
        gpu_sha256: None,
        force_memory: false,
        force_index: false,
        show_limits: false,
        iterations: 1,
        workers: 0,
        minute: false,
        minute_seconds: 60,
        banner: Some(banner::default_style()),
        no_color: false,
        no_progress: false,
        force_simd: false,
        no_animate: false,
        no_print: false,
        no_chart: false,
        json: false,
        no_hwinfo: false,
        index: 0,
    };
    let mut minute_seconds_set = false;
    let mut index = 0;
    let mut number: Option<&str> = None;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with('-') {
            if number.is_some() || index != args.len() - 1 {
                return Err("provide exactly one index after all flags".into());
            }
            number = Some(arg);
            index += 1;
            continue;
        }
        let stripped = arg.trim_start_matches('-');
        let (name, inline) = stripped
            .split_once('=')
            .map_or((stripped, None), |(a, b)| (a, Some(b)));
        if name == "help" || name == "h" {
            return Ok(None);
        }
        if number.is_some() {
            return Err("all flags must precede the index".into());
        }
        match name {
            "no-print" | "no-chart" | "json" | "no-hwinfo" | "no-banner" | "no-color"
            | "no-progress" | "force-simd" | "no-animate" | "minute" | "keep-gpu-output"
            | "force-memory" | "force-index" | "show-limits" => {
                let value = inline
                    .unwrap_or("true")
                    .parse::<bool>()
                    .map_err(|_| format!("invalid boolean for --{name}"))?;
                match name {
                    "no-print" => options.no_print = value,
                    "no-chart" => options.no_chart = value,
                    "json" => options.json = value,
                    "no-hwinfo" => options.no_hwinfo = value,
                    "no-banner" => {
                        options.banner = if value {
                            None
                        } else {
                            Some(banner::default_style())
                        };
                    }
                    "no-color" => options.no_color = value,
                    "no-progress" => options.no_progress = value,
                    "force-simd" => options.force_simd = value,
                    "no-animate" => options.no_animate = value,
                    "minute" => options.minute = value,
                    "keep-gpu-output" => options.keep_gpu_output = value,
                    "force-memory" => options.force_memory = value,
                    "force-index" => options.force_index = value,
                    "show-limits" => options.show_limits = value,
                    _ => unreachable!(),
                }
            }
            "method" | "iterations" | "workers" | "banner" | "backend" | "gpu-resident-bin"
            | "gpu-bin" | "minute-seconds" | "gpu-block27-bin" | "gpu-block28-bin"
            | "gpu-block29-bin" | "gpu-out" | "gpu-reference" | "gpu-sha256" => {
                let value = match inline {
                    Some(value) => value,
                    None => {
                        index += 1;
                        args.get(index)
                            .ok_or_else(|| format!("missing value for --{name}"))?
                    }
                };
                match name {
                    "method" => options.method = value.into(),
                    "backend" => {
                        options.backend = value.into();
                        options.backend_explicit = true;
                    }
                    // `--gpu-bin` is a compatibility alias from the earlier
                    // subprocess design; the path must now be the engine exe.
                    "gpu-resident-bin" | "gpu-bin" => options.gpu_resident_bin = Some(value.into()),
                    "gpu-block27-bin" => options.gpu_block27_bin = Some(value.into()),
                    "gpu-block28-bin" => options.gpu_block28_bin = Some(value.into()),
                    "gpu-block29-bin" => options.gpu_block29_bin = Some(value.into()),
                    "gpu-out" => {
                        if value.is_empty() {
                            return Err("--gpu-out path must not be empty".into());
                        }
                        options.gpu_out = Some(value.into());
                    }
                    "gpu-reference" => {
                        if value.is_empty() {
                            return Err("--gpu-reference path must not be empty".into());
                        }
                        options.gpu_reference = Some(value.into());
                    }
                    "gpu-sha256" => {
                        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
                        {
                            return Err(
                                "--gpu-sha256 must be exactly 64 hexadecimal characters".into()
                            );
                        }
                        options.gpu_sha256 = Some(value.to_ascii_uppercase());
                    }
                    "iterations" => {
                        options.iterations =
                            value.parse().map_err(|_| "iterations must be positive")?
                    }
                    "workers" => {
                        options.workers =
                            value.parse().map_err(|_| "workers must be nonnegative")?
                    }
                    "minute-seconds" => {
                        options.minute_seconds = value
                            .parse()
                            .map_err(|_| "minute-seconds must be nonnegative")?;
                        minute_seconds_set = true;
                    }
                    "banner" => {
                        // `none` mirrors `--no-banner`; every other value must be
                        // one of the styles known to the banner module. `spiral`
                        // is the default, and `portrait` is a legacy alias for it.
                        options.banner = if value.eq_ignore_ascii_case("none") {
                            None
                        } else {
                            Some(banner::parse_style(value).ok_or_else(|| {
                                format!(
                                    "unknown banner style {value:?}; use formula, spiral, portrait, or none"
                                )
                            })?)
                        };
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(format!("unknown flag --{name}")),
        }
        index += 1;
    }
    if options.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    if number == Some("limits") {
        options.show_limits = true;
        number = None;
    }
    if !["direct", "matrix", "doubling", "compare"].contains(&options.method.as_str()) {
        return Err("unknown method; use direct, matrix, doubling, or compare".into());
    }
    let selection = select_backend(&options.backend)
        .ok_or_else(|| format!("unknown backend; use {}", backend_values()))?;
    if !options.show_limits
        && selection.any_gpu()
        && !["doubling", "compare"].contains(&options.method.as_str())
    {
        return Err(format!(
            "--backend {} includes CUDA fast doubling only; use --method doubling or --method compare",
            options.backend
        ));
    }
    if (selection.block27 || selection.block28 || selection.block29) && options.minute {
        return Err(
            "block engines are single-index only; --minute supports the resident GPU, not gpu-block27, gpu-block28, or gpu-block29"
                .into(),
        );
    }
    if selection.auto && options.minute {
        return Err(
            "--backend auto is single-index only; use --backend gpu for a minute staircase".into(),
        );
    }
    if (options.gpu_out.is_some()
        || options.keep_gpu_output
        || options.gpu_reference.is_some()
        || options.gpu_sha256.is_some())
        && !(selection.gpu
            || selection.block27
            || selection.block28
            || selection.block29
            || selection.auto)
    {
        return Err(
            "--gpu-out, --keep-gpu-output, --gpu-reference, and --gpu-sha256 require a CUDA engine backend (gpu, gpu-block27, gpu-block28, gpu-block29, gpu-block, or auto)"
                .into(),
        );
    }
    if minute_seconds_set && !options.minute {
        return Err("--minute-seconds requires --minute".into());
    }
    options.index = match number {
        Some(number) => {
            if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
                return Err("index must contain only decimal ASCII digits".into());
            }
            number
                .parse()
                .map_err(|_| "index exceeds the uint64 range")?
        }
        None if options.show_limits => 0,
        None if options.minute => MINUTE_START,
        None => return Err("provide a nonnegative decimal index".into()),
    };
    Ok(Some(options))
}

fn method_name(method: &str) -> &'static str {
    match method {
        "direct" => "direct",
        "matrix" => "matrix",
        _ => "doubling",
    }
}

fn run(method: &str, n: u64) -> SimdUint {
    match method {
        "direct" => direct(n),
        "matrix" => matrix(n),
        _ => doubling(n),
    }
}

fn median(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Worker count with the project's small-index rule.
fn workers_for(index: u64, requested: usize, available: usize) -> usize {
    let requested = if requested == 0 {
        available
    } else {
        requested.min(available).max(1)
    };
    if index < SMALL_INDEX_WORKERS {
        1
    } else {
        requested
    }
}

/// ANSI color is opt-in: it needs a terminal on stderr and no explicit opt-out.
fn color_enabled(no_color: bool) -> bool {
    if no_color {
        return false;
    }
    // A non-empty NO_COLOR (https://no-color.org) overrides terminal detection.
    if std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()) {
        return false;
    }
    std::io::stderr().is_terminal()
}

/// ANSI colors for the metadata lines.
const CYAN: &str = "\x1b[38;5;45m";
const GOLD: &str = "\x1b[38;5;214m";
const WHITE: &str = "\x1b[38;5;255m";
const DIM: &str = "\x1b[38;5;244m";
const RESET: &str = "\x1b[0m";

/// Metadata prefix for the SIMD kernel line, shared with the tests.
const SIMD_LABEL: &str = "SIMD:";

fn paint(text: &str, color: &str, enabled: bool) -> String {
    if enabled && !text.is_empty() {
        format!("{color}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// Keep stderr and stdout in the order the user sees them. The banner animation
/// already flushes each frame; these explicit flushes cover the hardware block,
/// per-method measurements, the results table, the chart, and the final result.
fn flush_output() {
    let _ = std::io::stderr().flush();
    let _ = std::io::stdout().flush();
}

/// Shared run configuration handed to the single-index and minute paths.
struct Context<'a> {
    options: &'a Options,
    color: bool,
    workers: usize,
    /// SIMD CPU methods to run; empty when that family is not selected.
    simd_methods: &'a [&'a str],
    /// num-bigint CPU methods to run; empty when not selected.
    num_bigint_methods: &'a [Method],
    /// GMP CPU methods to run; empty when not selected.
    gmp_methods: &'a [Method],
    /// num-bigint adapter, present exactly when that family is selected.
    num_bigint: Option<&'a NumBigintRunner>,
    /// GMP adapter, present exactly when that family is selected.
    gmp: Option<&'a GmpRunner>,
    /// CUDA engine processes selected after the preflight.
    block_engines: &'a [BlockEngine],
    /// Precomputed GPU preflight object for `--json`.
    preflight_json: Option<&'a str>,
}

impl Context<'_> {
    fn run_blocks(&self) -> bool {
        !self.block_engines.is_empty()
    }
}

/// Exact CPU value, kept in the representation of its family.
#[derive(Clone, Debug)]
enum CpuValue {
    Simd(SimdUint),
    NumBigint(num_bigint::BigUint),
    Gmp(rug::Integer),
}

impl CpuValue {
    /// Canonical big-endian bytes, empty for zero.
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            CpuValue::Simd(value) => value.to_bytes_be(),
            CpuValue::NumBigint(value) => value.bytes(),
            CpuValue::Gmp(value) => value.bytes(),
        }
    }

    /// Convert through canonical big-endian bytes for GPU references and
    /// cross-family equality.
    fn to_simd(&self) -> SimdUint {
        match self {
            CpuValue::Simd(value) => value.clone(),
            CpuValue::NumBigint(value) => native_runner::to_simd(value),
            CpuValue::Gmp(value) => native_runner::to_simd(value),
        }
    }

    fn to_decimal(&self) -> String {
        match self {
            CpuValue::Simd(value) => value.to_decimal(),
            CpuValue::NumBigint(value) => value.decimal(),
            CpuValue::Gmp(value) => value.decimal(),
        }
    }
}

/// One completed measurement, including the CPU value when one exists.
struct Measurement {
    backend: Backend,
    method: &'static str,
    workers: usize,
    bits: u64,
    samples: Vec<u128>,
    median_ns: u128,
    index: u64,
    exact: Option<bool>,
    /// Present only for CPU rows; used for equality checks and decimal output.
    value: Option<CpuValue>,
    /// Present only for CUDA engine-process rows.
    block: Option<BlockMeasurement>,
}

/// Process-backed block-engine details retained for the report and JSON.
struct BlockMeasurement {
    variant: BlockVariant,
    /// Engine stream/device time when printed.
    gpu_stream_ms: Option<f64>,
    /// Engine call wall time when printed.
    call_wall_ms: Option<f64>,
    /// Engine total wall time including its output write, when printed.
    total_wall_ms: Option<f64>,
    /// Wall time of the child process measured by this CLI.
    process_wall_ms: f64,
    /// Engine-reported peak VRAM in MiB.
    peak_vram_mib: Option<f64>,
    /// Engine-reported peak host working set in MiB (block-28 only).
    host_peak_mib: Option<f64>,
    output_sha256: String,
    output_bytes: u64,
    output_kept: bool,
    output_path: Option<String>,
    /// Description of the comparison source, if any.
    reference: Option<String>,
}

impl BlockMeasurement {
    fn json(&self) -> GpuBlockJson {
        GpuBlockJson {
            engine: Some(self.variant.engine_name().to_string()),
            gpu_stream_ms: self.gpu_stream_ms,
            call_wall_ms: self.call_wall_ms,
            total_wall_ms: self.total_wall_ms,
            process_wall_ms: Some(self.process_wall_ms),
            peak_vram_mib: self.peak_vram_mib,
            host_peak_mib: self.host_peak_mib,
            output_sha256: Some(self.output_sha256.clone()),
            output_bytes: Some(self.output_bytes),
            output_kept: Some(self.output_kept),
            output_path: self.output_path.clone(),
            reference: self.reference.clone(),
        }
    }
}

impl Measurement {
    fn table_row(&self, index: Option<u64>) -> Row {
        Row {
            backend: self.backend,
            method: self.method,
            workers: self.workers,
            bits: self.bits,
            median_ns: self.median_ns,
            index,
            exact: self.exact,
        }
    }

    /// Label that keeps the CPU family visible when several values are printed.
    fn value_label(&self) -> String {
        format!("{} {}", self.backend.live_label(), self.method)
    }
}

fn simd_measurement(
    method: &str,
    n: u64,
    iterations: usize,
    workers: usize,
    progress: Option<&Progress>,
) -> Measurement {
    let iterations = iterations.max(1);
    let mut samples = Vec::with_capacity(iterations);
    let mut value = SimdUint::zero();
    for iteration in 0..iterations {
        if let Some(progress) = progress {
            progress.iteration(iteration + 1);
        }
        drop(std::mem::take(&mut value));
        let start = Instant::now();
        value = run(method, n);
        samples.push(start.elapsed().as_nanos());
    }
    let bits = value.bits();
    let median_ns = median(samples.clone());
    Measurement {
        backend: Backend::Cpu,
        method: method_name(method),
        workers,
        bits,
        samples,
        median_ns,
        index: n,
        exact: None,
        value: Some(CpuValue::Simd(value)),
        block: None,
    }
}

fn num_bigint_measurement(
    runner: &NumBigintRunner,
    method: Method,
    n: u64,
    iterations: usize,
    progress: Option<&Progress>,
) -> Measurement {
    let iterations = iterations.max(1);
    let mut samples = Vec::with_capacity(iterations);
    // Keep the previous result alive no longer than the original loop did: it
    // is dropped before the next timed call rather than during it.
    let mut value: Option<num_bigint::BigUint> = None;
    for iteration in 0..iterations {
        if let Some(progress) = progress {
            progress.iteration(iteration + 1);
        }
        drop(value.take());
        let measured = runner.measure(method, n, 1);
        samples.extend(measured.samples);
        value = Some(measured.value);
    }
    let value = value.expect("at least one timed iteration");
    let bits = value.bits();
    let median_ns = median(samples.clone());
    Measurement {
        backend: Backend::NumBigint,
        method: method.name(),
        workers: runner.workers(),
        bits,
        samples,
        median_ns,
        index: n,
        exact: None,
        value: Some(CpuValue::NumBigint(value)),
        block: None,
    }
}

fn gmp_measurement(
    runner: &GmpRunner,
    method: Method,
    n: u64,
    iterations: usize,
    progress: Option<&Progress>,
) -> Measurement {
    let iterations = iterations.max(1);
    let mut samples = Vec::with_capacity(iterations);
    let mut value: Option<rug::Integer> = None;
    for iteration in 0..iterations {
        if let Some(progress) = progress {
            progress.iteration(iteration + 1);
        }
        drop(value.take());
        let measured = runner.measure(method, n, 1);
        samples.extend(measured.samples);
        value = Some(measured.value);
    }
    let value = value.expect("at least one timed iteration");
    let bits = value.bits();
    let median_ns = median(samples.clone());
    Measurement {
        backend: Backend::Gmp,
        method: method.name(),
        workers: runner.workers(),
        bits,
        samples,
        median_ns,
        index: n,
        exact: None,
        value: Some(CpuValue::Gmp(value)),
        block: None,
    }
}

fn block_backend(variant: BlockVariant) -> Backend {
    match variant {
        BlockVariant::Resident => Backend::Gpu,
        BlockVariant::Block27 => Backend::GpuBlock27,
        BlockVariant::Block28 => Backend::GpuBlock28,
        BlockVariant::Block29 => Backend::GpuBlock29,
    }
}

/// Destination for one block-engine output.
///
/// A single selected engine uses `--gpu-out` verbatim. When both engines run,
/// the path is treated as a directory if it exists, otherwise a
/// `-block-27`/`-block-29` suffix is appended so the files cannot collide.
fn block_output_path(
    options: &Options,
    variant: BlockVariant,
    n: u64,
    multiple: bool,
    iteration: usize,
) -> Option<PathBuf> {
    let base = Path::new(options.gpu_out.as_ref()?);
    let directory_hint = options
        .gpu_out
        .as_deref()
        .is_some_and(|text| text.ends_with('/') || text.ends_with('\\'));
    let mut path = if base.is_dir() || directory_hint {
        base.join(format!("gpu-F-{n}-{}.bin", variant.engine_name()))
    } else if multiple {
        let mut text = base.as_os_str().to_os_string();
        text.push(format!("-{}", variant.engine_name()));
        PathBuf::from(text)
    } else {
        base.to_path_buf()
    };
    if iteration > 0 {
        let mut text = path.into_os_string();
        text.push(format!("-run{}", iteration + 1));
        path = PathBuf::from(text);
    }
    Some(path)
}

/// Compute a CPU reference for the block engines at indices where GMP can hold
/// the value. Prefer an already measured CPU doubling row; otherwise run GMP
/// doubling once.
fn block_cpu_reference(
    measurements: &[Measurement],
    n: u64,
    workers: usize,
) -> Option<(Vec<u8>, String)> {
    if let Some(value) = cpu_doubling_reference(measurements) {
        return Some((value.to_bytes_be(), "CPU doubling".into()));
    }
    if n <= limits::RESIDENT_MAX_INDEX {
        let runner = GmpRunner::new(workers).ok()?;
        let measured = runner.measure(Method::Doubling, n, 1);
        Some((measured.value.bytes(), "CPU/GMP doubling".into()))
    } else {
        None
    }
}

/// Run one process-backed block engine and compare/fingerprint its output.
#[allow(clippy::too_many_arguments)]
fn measure_block(
    engine: &mut BlockEngine,
    variant: BlockVariant,
    n: u64,
    options: &Options,
    iterations: usize,
    multiple: bool,
    reference_be: Option<&[u8]>,
    reference_label: Option<&str>,
    expected_sha256: Option<&str>,
    reference_file: Option<&Path>,
    progress: &Progress,
) -> Result<Measurement, gpu_block::BlockError> {
    let iterations = iterations.max(1);
    let output_path = block_output_path(options, variant, n, multiple, 0);
    let progress_path =
        std::env::temp_dir().join(format!("fib-progress-{}-{}.txt", std::process::id(), n));
    let _ = std::fs::remove_file(&progress_path);
    progress.watch_file(&progress_path);
    let run = engine.run(
        n,
        output_path.as_deref(),
        options.keep_gpu_output,
        iterations,
        Some(&progress_path),
    );
    progress.unwatch();
    let _ = std::fs::remove_file(&progress_path);
    let run = run?;

    let mut exact: Option<bool> = None;
    let mut references = Vec::new();
    if let Some(reference) = reference_be {
        let matched = gpu_block::compare_output_to_be_bytes(&run.output_path, reference)?;
        exact = Some(exact.unwrap_or(true) && matched);
        references.push(reference_label.unwrap_or("CPU reference").to_string());
    }
    if let Some(expected) = expected_sha256 {
        let matched = expected.eq_ignore_ascii_case(&run.output_sha256);
        exact = Some(exact.unwrap_or(true) && matched);
        references.push(format!("--gpu-sha256 {expected}"));
    }
    if let Some(reference) = reference_file {
        let matched = gpu_block::compare_output_files(&run.output_path, reference)?;
        exact = Some(exact.unwrap_or(true) && matched);
        references.push(format!("--gpu-reference {}", reference.display()));
    }

    if !run.kept {
        let _ = std::fs::remove_file(&run.output_path);
    }
    if exact == Some(false) {
        return Err(if expected_sha256.is_some() {
            gpu_block::BlockError::ExpectedHashMismatch {
                expected: expected_sha256.unwrap_or_default().to_string(),
                computed: run.output_sha256,
            }
        } else if reference_be.is_some() {
            gpu_block::BlockError::CpuReferenceMismatch {
                engine: variant.engine_name().to_string(),
                n,
            }
        } else {
            gpu_block::BlockError::ReferenceMismatch {
                reference: reference_file
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("<supplied reference>")),
                output: run.output_path.clone(),
            }
        });
    }

    let mut samples: Vec<u128> = run
        .stats
        .samples_ms
        .iter()
        .map(|ms| (ms * 1_000_000.0).round() as u128)
        .collect();
    if samples.is_empty() {
        samples.push((run.process_wall_ms * 1_000_000.0).round() as u128);
    }
    let median_ns = median(samples.clone());
    let block = BlockMeasurement {
        variant,
        gpu_stream_ms: run.parsed.gpu_stream_ms,
        call_wall_ms: run.parsed.call_wall_ms,
        total_wall_ms: run.parsed.total_wall_ms,
        process_wall_ms: run.process_wall_ms,
        peak_vram_mib: run.parsed.peak_vram_mib,
        host_peak_mib: run.parsed.host_peak_mib,
        output_sha256: run.output_sha256.clone(),
        output_bytes: run.output_bytes,
        output_kept: run.kept,
        output_path: run.kept.then(|| run.output_path.display().to_string()),
        reference: (!references.is_empty()).then(|| references.join(" + ")),
    };
    Ok(Measurement {
        backend: block_backend(variant),
        method: "doubling",
        workers: 1,
        bits: run.parsed.bits,
        samples,
        median_ns,
        index: n,
        exact,
        value: None,
        block: Some(block),
    })
}

fn file_name(path: Option<&Path>) -> String {
    path.and_then(Path::file_name).map_or_else(
        || "<not discovered>".to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// One compact line for a process-backed block engine.
fn print_block_block(engine: &BlockEngine, variant: BlockVariant, color: bool) {
    eprintln!(
        "{} {}",
        paint(&format!("GPU/CUDA {}:", variant.engine_name()), CYAN, color),
        paint(&file_name(engine.binary()), WHITE, color)
    );
}

/// Shortens a SHA-256 for a one-line summary.
fn short_sha256(sha: &str) -> String {
    if sha.len() <= 20 {
        sha.to_string()
    } else {
        format!("{}…{}", &sha[..12], &sha[sha.len() - 8..])
    }
}

/// CPU doubling value for the same index, used as the GPU reference.
///
/// Prefer the SIMD doubling row, then num-bigint, then GMP. Values from the
/// reference backends are converted through canonical big-endian bytes.
fn cpu_doubling_reference(measurements: &[Measurement]) -> Option<SimdUint> {
    let find = |backend: Backend| {
        measurements
            .iter()
            .find(|measurement| measurement.backend == backend && measurement.method == "doubling")
            .and_then(|measurement| measurement.value.as_ref())
    };
    find(Backend::Cpu)
        .or_else(|| find(Backend::NumBigint))
        .or_else(|| find(Backend::Gmp))
        .map(CpuValue::to_simd)
}

/// Build table rows. Every row carries its Fibonacci index so the table shows
/// which number in the sequence was measured, not only its bit length.
fn table_rows(measurements: &[&Measurement]) -> Vec<Row> {
    measurements
        .iter()
        .map(|measurement| measurement.table_row(Some(measurement.index)))
        .collect()
}

/// Build the full, unelided chart label.
fn chart_bar_label(backend: Backend, method: &str) -> String {
    format!("{} {}", backend.chart_label(), method)
}

fn chart_bars<'a>(measurements: impl IntoIterator<Item = &'a Measurement>) -> Vec<chart::Bar> {
    measurements
        .into_iter()
        .map(|measurement| chart::Bar {
            label: chart_bar_label(measurement.backend, measurement.method),
            value_ns: measurement.median_ns as f64,
        })
        .collect()
}

/// True when every CPU value in `measurements` has identical limbs.
fn values_equal(measurements: &[&Measurement]) -> bool {
    let mut values = measurements
        .iter()
        .filter_map(|measurement| measurement.value.as_ref());
    let Some(first) = values.next() else {
        return true;
    };
    let first = first.to_bytes();
    values.all(|value| value.to_bytes() == first)
}

/// CPU families agree and no GPU row mismatched the CPU reference. An
/// unvalidated GPU row (`None`) does not make the step unequal.
fn step_equal(step: &[Measurement]) -> bool {
    let cpu: Vec<&Measurement> = step
        .iter()
        .filter(|measurement| measurement.backend.is_cpu())
        .collect();
    let gpu_ok = step
        .iter()
        .filter(|measurement| measurement.backend.is_gpu())
        .all(|measurement| measurement.exact != Some(false));
    values_equal(&cpu) && gpu_ok
}

/// Cross-family check for the staircase: every CPU value measured at the same
/// index must have identical canonical bytes, and no GPU row may have
/// mismatched its reference.
fn minute_equal(steps: &[(u64, Vec<Measurement>)]) -> bool {
    let mut by_index: HashMap<u64, Vec<&CpuValue>> = HashMap::new();
    let mut gpu_ok = true;
    for (index, step) in steps {
        for measurement in step {
            if measurement.backend.is_cpu() {
                if let Some(value) = measurement.value.as_ref() {
                    by_index.entry(*index).or_default().push(value);
                }
            } else if measurement.backend.is_gpu() && measurement.exact == Some(false) {
                gpu_ok = false;
            }
        }
    }
    gpu_ok
        && by_index.values().all(|values| {
            let mut bytes = values.iter().map(|value| value.to_bytes());
            let Some(first) = bytes.next() else {
                return true;
            };
            bytes.all(|value| value == first)
        })
}

/// Mark GMP rows with their cross-family exactness result for the JSON report.
///
/// The reference is the num-bigint value for the same method when that family
/// ran, otherwise the SIMD value. GMP-only runs leave `exact` as `None`.
fn annotate_gmp_exactness(measurements: &mut [Measurement]) {
    let mut references: HashMap<&'static str, Vec<u8>> = HashMap::new();
    for method in ["direct", "matrix", "doubling"] {
        let found = measurements
            .iter()
            .find(|measurement| {
                measurement.backend == Backend::NumBigint && measurement.method == method
            })
            .or_else(|| {
                measurements.iter().find(|measurement| {
                    measurement.backend == Backend::Cpu && measurement.method == method
                })
            })
            .and_then(|measurement| measurement.value.as_ref());
        if let Some(value) = found {
            references.insert(method, value.to_bytes());
        }
    }
    for measurement in measurements
        .iter_mut()
        .filter(|measurement| measurement.backend == Backend::Gmp)
    {
        if let Some(reference) = references.get(measurement.method) {
            measurement.exact = Some(
                measurement
                    .value
                    .as_ref()
                    .is_some_and(|value| value.to_bytes() == *reference),
            );
        }
    }
}

/// Per-index cross-family exactness for GMP rows in the staircase table.
fn annotate_gmp_exactness_steps(steps: &mut [(u64, Vec<Measurement>)]) {
    let mut references: HashMap<(u64, &'static str), Vec<u8>> = HashMap::new();
    // Prefer num-bigint as the GMP reference; fall back to SIMD when that
    // family did not run at this index.
    for (index, step) in steps.iter() {
        for measurement in step {
            if measurement.backend == Backend::NumBigint {
                if let Some(value) = measurement.value.as_ref() {
                    references
                        .entry((*index, measurement.method))
                        .or_insert_with(|| value.to_bytes());
                }
            }
        }
    }
    for (index, step) in steps.iter() {
        for measurement in step {
            if measurement.backend == Backend::Cpu {
                if let Some(value) = measurement.value.as_ref() {
                    references
                        .entry((*index, measurement.method))
                        .or_insert_with(|| value.to_bytes());
                }
            }
        }
    }
    for (index, step) in steps.iter_mut() {
        for measurement in step
            .iter_mut()
            .filter(|measurement| measurement.backend == Backend::Gmp)
        {
            if let Some(reference) = references.get(&(*index, measurement.method)) {
                measurement.exact = Some(
                    measurement
                        .value
                        .as_ref()
                        .is_some_and(|value| value.to_bytes() == *reference),
                );
            }
        }
    }
}

fn push_measurement_json(out: &mut String, measurement: &Measurement) {
    let block_json = measurement.block.as_ref().map(BlockMeasurement::json);
    report::push_measurement_json(
        out,
        measurement.backend,
        measurement.method,
        measurement.bits,
        &measurement.samples,
        measurement.median_ns,
        measurement.exact,
        None,
        block_json.as_ref(),
    );
}

fn print_json_single(
    measurements: &[&Measurement],
    index: u64,
    workers: usize,
    equal: bool,
    preflight: Option<&str>,
) {
    let mut out = String::with_capacity(256);
    let _ = write!(
        out,
        "{{\"kernel\":\"{}\",\"index\":{},\"workers\":{},\"equal\":{}",
        selected_kernel_name(),
        index,
        workers,
        equal
    );
    if let Some(preflight) = preflight {
        let _ = write!(out, ",\"gpu_preflight\":{preflight}");
    }
    out.push_str(",\"measurements\":[");
    for (position, measurement) in measurements.iter().enumerate() {
        if position != 0 {
            out.push(',');
        }
        push_measurement_json(&mut out, measurement);
    }
    out.push_str("]}");
    println!("{out}");
}

fn print_json_minute(
    steps: &[(u64, Vec<Measurement>)],
    seconds: u64,
    equal: bool,
    preflight: Option<&str>,
) {
    let mut out = String::with_capacity(512);
    let _ = write!(
        out,
        "{{\"kernel\":\"{}\",\"minute\":true,\"minute_seconds\":{},\"equal\":{}",
        selected_kernel_name(),
        seconds,
        equal
    );
    if let Some(preflight) = preflight {
        let _ = write!(out, ",\"gpu_preflight\":{preflight}");
    }
    out.push_str(",\"steps\":[");
    for (step_index, (index, step)) in steps.iter().enumerate() {
        if step_index != 0 {
            out.push(',');
        }
        let _ = write!(out, "{{\"index\":{index},\"measurements\":[");
        for (position, measurement) in step.iter().enumerate() {
            if position != 0 {
                out.push(',');
            }
            push_measurement_json(&mut out, measurement);
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    println!("{out}");
}

fn note_unvalidated_gpu(measurements: &[&Measurement], color: bool) {
    if measurements
        .iter()
        .any(|measurement| measurement.backend.is_gpu() && measurement.exact.is_none())
    {
        eprintln!(
            "{} one or more GPU results were not independently validated; compare with --backend both, --gpu-sha256, or --gpu-reference",
            paint("note:", DIM, color)
        );
    }
}

/// Single-index measurement path (`--minute` is off).
fn run_single(context: &Context<'_>) {
    let options = context.options;
    let total_methods = context.simd_methods.len()
        + context
            .num_bigint
            .map_or(0, |_| context.num_bigint_methods.len())
        + context.gmp.map_or(0, |_| context.gmp_methods.len())
        + context.block_engines.len();
    if total_methods == 0 {
        eprintln!(
            "Error: no selected backend can run n={}; use --backend auto for the fastest fitting CUDA engine, or --force-simd for CPU/SIMD",
            options.index
        );
        std::process::exit(2);
    }
    // Progress is opt-out and only for an interactive stderr. `--json` keeps
    // stdout clean and `--minute` streams its own table, so both disable it.
    let progress_enabled =
        !options.no_progress && !options.json && !options.minute && std::io::stderr().is_terminal();
    if !options.json {
        eprintln!(
            "{} {}",
            paint("target:", CYAN, context.color),
            paint(&format!("n={}", options.index), GOLD, context.color)
        );
        flush_output();
    }
    let progress = Progress::new(total_methods, progress_enabled, context.color);
    let mut measurements: Vec<Measurement> = Vec::new();

    for method in context.simd_methods {
        progress.begin(&format!("CPU/SIMD {method}"), options.iterations);
        let measurement = simd_measurement(
            method,
            options.index,
            options.iterations,
            context.workers,
            Some(&progress),
        );
        progress.finish(measurement.median_ns);
        measurements.push(measurement);
    }
    if let Some(runner) = context.num_bigint {
        for &method in context.num_bigint_methods {
            progress.begin(
                &format!("CPU/num-bigint {}", method.name()),
                options.iterations,
            );
            let measurement = num_bigint_measurement(
                runner,
                method,
                options.index,
                options.iterations,
                Some(&progress),
            );
            progress.finish(measurement.median_ns);
            measurements.push(measurement);
        }
    }
    if let Some(runner) = context.gmp {
        for &method in context.gmp_methods {
            progress.begin(&format!("CPU/GMP {}", method.name()), options.iterations);
            let measurement = gmp_measurement(
                runner,
                method,
                options.index,
                options.iterations,
                Some(&progress),
            );
            progress.finish(measurement.median_ns);
            measurements.push(measurement);
        }
    }

    if context.run_blocks() {
        let multiple = context.block_engines.len() > 1;
        let reference = block_cpu_reference(&measurements, options.index, context.workers);
        let reference_be = reference.as_ref().map(|(bytes, _)| bytes.as_slice());
        let reference_label = reference.as_ref().map(|(_, label)| label.as_str());
        let expected_sha = options.gpu_sha256.as_deref();
        let reference_file = options.gpu_reference.as_deref().map(Path::new);
        let mut engines = context.block_engines.to_vec();
        for engine in &mut engines {
            let variant = engine.variant();
            if !progress_enabled {
                print_block_block(engine, variant, context.color);
            }
            progress.begin(
                &format!("GPU/CUDA {} doubling", variant.engine_name()),
                options.iterations.max(1),
            );
            match measure_block(
                engine,
                variant,
                options.index,
                options,
                options.iterations,
                multiple,
                reference_be,
                reference_label,
                expected_sha,
                reference_file,
                &progress,
            ) {
                Ok(measurement) => {
                    progress.finish(measurement.median_ns);
                    if let Some(block) = &measurement.block {
                        let stream = block
                            .gpu_stream_ms
                            .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.1} ms"));
                        let call = block
                            .call_wall_ms
                            .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.1} ms"));
                        let total = block
                            .total_wall_ms
                            .map_or_else(|| "n/a".to_string(), |ms| format!("{ms:.1} ms"));
                        let peak = block
                            .peak_vram_mib
                            .map_or_else(|| "n/a".to_string(), |mib| format!("{mib:.1} MiB"));
                        let host_peak = block
                            .host_peak_mib
                            .map_or_else(|| "n/a".to_string(), |mib| format!("{mib:.1} MiB"));
                        if !progress_enabled {
                            eprintln!(
                                "{} stream {} | call {} | total {} | peak VRAM {} | peak host {} | output {} bytes | sha256 {}",
                                paint(
                                    &format!("GPU/CUDA {}:", variant.engine_name()),
                                    CYAN,
                                    context.color
                                ),
                                paint(&stream, GOLD, context.color),
                                paint(&call, WHITE, context.color),
                                paint(&total, WHITE, context.color),
                                paint(&peak, WHITE, context.color),
                                paint(&host_peak, WHITE, context.color),
                                paint(&block.output_bytes.to_string(), WHITE, context.color),
                                paint(&block.output_sha256, WHITE, context.color)
                            );
                        }
                        if block.output_kept && !progress_enabled {
                            eprintln!(
                                "{} full output retained at {} ({} bytes)",
                                paint("GPU output:", DIM, context.color),
                                block.output_path.as_deref().unwrap_or("<unknown>"),
                                block.output_bytes
                            );
                        }
                    }
                    measurements.push(measurement);
                }
                Err(error) => {
                    progress.stop();
                    eprintln!("Error: {error}");
                    flush_output();
                    std::process::exit(3);
                }
            }
        }
    }

    annotate_gmp_exactness(&mut measurements);
    // The live line must disappear before the results table owns the terminal.
    progress.stop();

    let refs: Vec<&Measurement> = measurements.iter().collect();
    let rows = table_rows(&refs);
    if !rows.is_empty() {
        report::print_table(&rows, context.color);
        flush_output();
    }

    // A GPU row that disagreed with the CPU reference is reported after the row
    // and the table have been printed.
    if let Some(measurement) = measurements
        .iter()
        .find(|measurement| measurement.backend.is_gpu() && measurement.exact == Some(false))
    {
        eprintln!(
            "Error: GPU result for n={} did not match the supplied reference",
            measurement.index
        );
        flush_output();
        std::process::exit(3);
    }
    note_unvalidated_gpu(&refs, context.color);
    if progress_enabled {
        print_gpu_details(&measurements, context.color);
    }

    let equal = step_equal(&measurements);
    if !equal {
        eprintln!("Error: method results differ");
        flush_output();
        std::process::exit(3);
    }

    if options.json {
        print_json_single(
            &refs,
            options.index,
            context.workers,
            equal,
            context.preflight_json,
        );
    }

    if !options.no_chart && !measurements.is_empty() {
        chart::print(&chart_bars(measurements.iter()), context.color);
        flush_output();
    }

    // The GPU-only path keeps the value as device limbs, so there is nothing to
    // print. CPU rows keep the original labelled/single decimal output.
    if !options.no_print && !options.json {
        let values: Vec<(String, &CpuValue)> = measurements
            .iter()
            .filter_map(|measurement| {
                measurement
                    .value
                    .as_ref()
                    .map(|value| (measurement.value_label(), value))
            })
            .collect();
        if values.len() > 1 {
            for (label, value) in values {
                println!("{label}: {}", value.to_decimal());
            }
        } else if let Some((_, value)) = values.first() {
            println!("{}", value.to_decimal());
        }
    }
    flush_output();
}

/// Stable per-engine details for an interactive run. The live progress line is
/// transient, so VRAM and the output hash are printed once, after all
/// measurements, instead of flashing inside the bar.
fn print_gpu_details(measurements: &[Measurement], color: bool) {
    let rows: Vec<(&BlockMeasurement, u128)> = measurements
        .iter()
        .filter_map(|measurement| {
            measurement
                .block
                .as_ref()
                .map(|block| (block, measurement.median_ns))
        })
        .collect();
    if rows.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}", paint("GPU details", CYAN, color));
    eprintln!(
        "{:<20} {:>12} {:>12}  SHA-256",
        "BACKEND", "MEDIAN", "PEAK VRAM"
    );
    for (block, median_ns) in rows {
        let peak = block
            .peak_vram_mib
            .map_or_else(|| "-".to_string(), |mib| format!("{mib:.1} MiB"));
        eprintln!(
            "{:<20} {:>12} {:>12}  {}",
            format!("GPU/CUDA {}", block.variant.engine_name()),
            report::format_median(median_ns),
            peak,
            short_sha256(&block.output_sha256)
        );
    }
}

/// Staircase measurement path (`--minute`).
///
/// Each selected CPU method gets its own time budget and staircase, in order.
/// The GPU phase runs last, after all CPU phases, so CPU and GPU measurements
/// are never interleaved.
fn run_minute(context: &Context<'_>) {
    let options = context.options;
    let budget = Duration::from_secs(options.minute_seconds);
    let mut steps: Vec<(u64, Vec<Measurement>)> = Vec::new();
    let mut cpu_doubling: HashMap<u64, SimdUint> = HashMap::new();
    let mut failure: Option<String> = None;

    // The minute table is streamed: a row appears as soon as its step finishes,
    // and there is no second duplicate table at the end.
    if !options.json {
        report::stream_header(context.color);
        flush_output();
    }

    // All SIMD CPU phases first, in method order.
    for method in context.simd_methods {
        let started = Instant::now();
        let mut index = options.index;
        loop {
            let measurement = simd_measurement(method, index, 1, context.workers, None);
            if *method == "doubling" {
                if let Some(value) = measurement.value.as_ref() {
                    cpu_doubling.insert(index, value.to_simd());
                }
            }
            if !options.json {
                report::stream_row(&measurement.table_row(Some(index)), context.color);
                flush_output();
            }
            steps.push((index, vec![measurement]));

            if started.elapsed() >= budget {
                break;
            }
            match index.checked_mul(2) {
                Some(next) if next > index => index = next,
                _ => break,
            }
        }
    }

    // Then all num-bigint CPU phases, in method order, with no interleaving.
    if let Some(runner) = context.num_bigint {
        for &method in context.num_bigint_methods {
            let started = Instant::now();
            let mut index = options.index;
            loop {
                let measurement = num_bigint_measurement(runner, method, index, 1, None);
                if method == Method::Doubling {
                    if let Some(value) = measurement.value.as_ref() {
                        // Keep the SIMD doubling value when both families ran.
                        cpu_doubling.entry(index).or_insert_with(|| value.to_simd());
                    }
                }
                if !options.json {
                    report::stream_row(&measurement.table_row(Some(index)), context.color);
                    flush_output();
                }
                steps.push((index, vec![measurement]));

                if started.elapsed() >= budget {
                    break;
                }
                match index.checked_mul(2) {
                    Some(next) if next > index => index = next,
                    _ => break,
                }
            }
        }
    }

    // Then all GMP CPU phases, in method order, with no interleaving.
    if let Some(runner) = context.gmp {
        for &method in context.gmp_methods {
            let started = Instant::now();
            let mut index = options.index;
            loop {
                let measurement = gmp_measurement(runner, method, index, 1, None);
                if method == Method::Doubling {
                    if let Some(value) = measurement.value.as_ref() {
                        // Keep the SIMD or num-bigint doubling value first.
                        cpu_doubling.entry(index).or_insert_with(|| value.to_simd());
                    }
                }
                if !options.json {
                    report::stream_row(&measurement.table_row(Some(index)), context.color);
                    flush_output();
                }
                steps.push((index, vec![measurement]));

                if started.elapsed() >= budget {
                    break;
                }
                match index.checked_mul(2) {
                    Some(next) if next > index => index = next,
                    _ => break,
                }
            }
        }
    }

    if let Some(mut resident) = context
        .block_engines
        .iter()
        .find(|engine| engine.variant() == BlockVariant::Resident)
        .cloned()
    {
        let started = Instant::now();
        let mut index = options.index;
        let progress = Progress::new(0, false, false);
        loop {
            let reference = cpu_doubling.get(&index).map(SimdUint::to_bytes_be);
            match measure_block(
                &mut resident,
                BlockVariant::Resident,
                index,
                options,
                1,
                false,
                reference.as_deref(),
                Some("CPU doubling"),
                None,
                None,
                &progress,
            ) {
                Ok(measurement) => {
                    if !options.json {
                        report::stream_row(&measurement.table_row(Some(index)), context.color);
                        flush_output();
                    }
                    if measurement.exact == Some(false) {
                        failure = Some(format!(
                            "GPU result for n={index} did not match the CPU doubling reference"
                        ));
                    }
                    steps.push((index, vec![measurement]));
                }
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }

            if started.elapsed() >= budget {
                break;
            }
            match index.checked_mul(2) {
                Some(next) if next > index => index = next,
                _ => break,
            }
        }
    }

    annotate_gmp_exactness_steps(&mut steps);

    let measurements: Vec<&Measurement> = steps.iter().flat_map(|(_, step)| step.iter()).collect();

    if let Some(error) = failure {
        eprintln!("Error: {error}");
        flush_output();
        std::process::exit(3);
    }
    note_unvalidated_gpu(&measurements, context.color);

    let equal = minute_equal(&steps);
    if !equal {
        eprintln!("Error: method results differ");
        flush_output();
        std::process::exit(3);
    }

    if options.json {
        print_json_minute(
            &steps,
            options.minute_seconds,
            equal,
            context.preflight_json,
        );
    }

    if !options.no_chart && !measurements.is_empty() {
        chart::print(&chart_bars(measurements), context.color);
        flush_output();
    }
    // Minute mode intentionally omits decimal output; the staircase can reach
    // millions of digits and the table plus chart are the benchmark result.
    flush_output();
}

/// Outcome of the startup GPU preflight.
#[derive(Debug)]
struct GpuResolution {
    /// Selection after `--backend auto` resolution.
    selection: Selection,
    /// Automatic choice, when `--backend auto` was used.
    auto: Option<AutoChoice>,
    /// One fit record per selected engine, in selection order.
    fits: Vec<EngineFit>,
    /// Engines that will actually run.
    run: Vec<GpuEngine>,
    /// Engines skipped by the index or memory check.
    skipped: Vec<EngineFit>,
    /// Resources used for the check.
    resources: Resources,
}

/// Resolve `--backend auto` and apply the index/memory preflight without
/// starting a CUDA engine or creating a CUDA context.
fn resolve_gpu(
    options: &Options,
    requested: Selection,
    resources: Resources,
) -> Result<GpuResolution, String> {
    let (selection, auto) = if requested.auto {
        let choice = limits::choose_auto(options.index, &resources, options.force_memory)
            .ok_or_else(|| {
                format!(
                    "no GPU backend is available for n={} (resident max {}, block-28 planner max {}, block-29 planner max {})",
                    options.index,
                    limits::format_u64(limits::RESIDENT_MAX_INDEX),
                    limits::format_u64(limits::BLOCK28_MAX_INDEX),
                    limits::format_u64(limits::BLOCK29_MAX_INDEX)
                )
            })?;
        if !choice.fit.fits && !options.force_memory {
            return Err(choice.fit.recommendation());
        }
        let mut selection = selection(false, false, false, false, false, false, false, false);
        match choice.engine {
            GpuEngine::Resident => selection.gpu = true,
            GpuEngine::Block27 => selection.block27 = true,
            GpuEngine::Block28 => selection.block28 = true,
            GpuEngine::Block29 => selection.block29 = true,
        }
        (selection, Some(choice))
    } else {
        (requested, None)
    };

    let mut fits = Vec::new();
    let mut run = Vec::new();
    let mut skipped = Vec::new();
    for engine in selection.engines() {
        let fit = limits::check(engine, options.index, &resources, options.force_memory);
        let index_ok = fit.index_allowed || options.force_index;
        let memory_ok = options.force_memory || (fit.fits_device && fit.fits_host);
        if index_ok && memory_ok {
            run.push(engine);
        } else {
            skipped.push(fit.clone());
        }
        fits.push(fit);
    }
    Ok(GpuResolution {
        selection,
        auto,
        fits,
        run,
        skipped,
        resources,
    })
}

fn json_escape_option(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn fit_json(fit: &EngineFit) -> String {
    format!(
        "{{\"engine\":\"{}\",\"label\":\"{}\",\"n\":{},\"verified_max\":{},\"hard_max\":{},\"required_device_bytes\":{},\"required_host_bytes\":{},\"available_device_bytes\":{},\"available_host_bytes\":{},\"index_allowed\":{},\"fits_device\":{},\"fits_host\":{},\"fits\":{},\"over_verified\":{},\"skip_reason\":{},\"note\":{}}}",
        fit.engine.backend_flag(),
        fit.engine.label(),
        fit.n,
        fit.engine.verified_max(),
        fit.engine
            .hard_max()
            .map_or_else(|| "null".to_string(), |max| max.to_string()),
        fit.required_device_bytes,
        fit.required_host_bytes,
        fit.available_device_bytes
            .map_or_else(|| "null".to_string(), |bytes| bytes.to_string()),
        fit.available_host_bytes
            .map_or_else(|| "null".to_string(), |bytes| bytes.to_string()),
        fit.index_allowed,
        fit.fits_device,
        fit.fits_host,
        fit.fits,
        fit.over_verified,
        fit.skip_reason.as_ref().map_or_else(
            || "null".to_string(),
            |value| format!("\"{}\"", json_escape_option(value))
        ),
        fit.note.as_ref().map_or_else(
            || "null".to_string(),
            |value| format!("\"{}\"", json_escape_option(value))
        )
    )
}

fn preflight_json(resolved: &GpuResolution, backend: &str) -> String {
    let auto = resolved.auto.as_ref().map_or_else(
        || "null".to_string(),
        |choice| {
            format!(
                "{{\"selected\":\"{}\",\"reason\":\"{}\"}}",
                choice.engine.backend_flag(),
                json_escape_option(&choice.reason)
            )
        },
    );
    let selected = resolved
        .run
        .iter()
        .map(|engine| format!("\"{}\"", engine.backend_flag()))
        .collect::<Vec<_>>()
        .join(",");
    let skipped = resolved
        .skipped
        .iter()
        .map(fit_json)
        .collect::<Vec<_>>()
        .join(",");
    let engines = resolved
        .fits
        .iter()
        .map(fit_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"requested_backend\":\"{}\",\"auto\":{},\"selected\":[{}],\"skipped\":[{}],\"engines\":[{}],\"resources\":{{\"device_total_bytes\":{},\"device_free_bytes\":{},\"host_total_bytes\":{},\"host_available_bytes\":{}}}}}",
        json_escape_option(backend),
        auto,
        selected,
        skipped,
        engines,
        resolved
            .resources
            .device_total_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resolved
            .resources
            .device_free_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resolved
            .resources
            .host_total_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resolved
            .resources
            .host_available_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
    )
}

fn print_preflight(options: &Options, color: bool, resolved: &GpuResolution) {
    if !resolved.selection.any_gpu() {
        return;
    }
    eprint!(
        "{}",
        limits::render_preflight(
            &resolved.fits,
            &resolved.resources,
            options.index,
            options.force_index,
            options.force_memory
        )
    );
    if let Some(auto) = &resolved.auto {
        eprintln!(
            "{} {}",
            paint("recommendation:", CYAN, color),
            paint(&auto.reason, GOLD, color)
        );
    }
    if resolved
        .fits
        .iter()
        .any(|fit| fit.over_verified && resolved.run.contains(&fit.engine))
    {
        eprintln!(
            "{} n is above the verified maximum for one or more engines; results stay unverified unless --gpu-sha256 or --gpu-reference is supplied",
            paint("note:", DIM, color)
        );
    }
    for fit in &resolved.fits {
        if !resolved.run.contains(&fit.engine) {
            let flag = if !fit.index_allowed && !options.force_index {
                "--force-index"
            } else {
                "--force-memory"
            };
            eprintln!(
                "{} {}; use {} to override",
                paint("skipped:", DIM, color),
                paint(&fit.recommendation(), GOLD, color),
                flag
            );
        } else if !fit.index_allowed || !fit.fits {
            let flag = if !fit.index_allowed {
                "--force-index"
            } else {
                "--force-memory"
            };
            eprintln!(
                "{} {} overrides the preflight for {} at n={}; the engine may fail",
                paint("note:", DIM, color),
                flag,
                fit.engine.label(),
                fit.n
            );
        }
    }
    flush_output();
}

fn main() {
    let mut options = match parse(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(None) => {
            print!("{HELP}");
            return;
        }
        Ok(Some(options)) => options,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(2);
        }
    };

    if options.show_limits {
        let resources = limits::detect_resources();
        if options.json {
            println!("{}", limits::render_limits_json(&resources));
        } else {
            print!("{}", limits::render_limits(&resources));
        }
        return;
    }

    // Without an explicit --backend, a large index would silently skip every
    // CPU/SIMD method. Switch to the fitting CUDA engine automatically.
    if !options.backend_explicit && options.index >= SIMD_SKIP_ALL_INDEX {
        options.backend = "auto".into();
        eprintln!(
            "note: no --backend given; using auto for n={} (CPU/SIMD is skipped above {})",
            limits::format_u64(options.index),
            limits::format_u64(SIMD_SKIP_ALL_INDEX)
        );
    }

    let requested_selection = select_backend(&options.backend).expect("validated --backend");
    let any_gpu_requested = requested_selection.any_gpu();
    let resources = if any_gpu_requested {
        limits::detect_resources()
    } else {
        Resources::default()
    };
    let resolved = if any_gpu_requested {
        match resolve_gpu(&options, requested_selection, resources) {
            Ok(resolved) => resolved,
            Err(error) => {
                eprintln!("recommendation: {error}");
                std::process::exit(2);
            }
        }
    } else {
        GpuResolution {
            selection: requested_selection,
            auto: None,
            fits: Vec::new(),
            run: Vec::new(),
            skipped: Vec::new(),
            resources,
        }
    };
    let selection = resolved.selection;
    let gpu_selected = selection.gpu || selection.block27 || selection.block28 || selection.block29;

    // The preflight runs before starting any engine. A skipped resident must
    // not run behind the back of `gpu-all`.
    let mut block_engines = Vec::new();
    for engine in &resolved.run {
        let (variant, explicit) = match engine {
            GpuEngine::Resident => (BlockVariant::Resident, options.gpu_resident_bin.as_deref()),
            GpuEngine::Block27 => (BlockVariant::Block27, options.gpu_block27_bin.as_deref()),
            GpuEngine::Block28 => (BlockVariant::Block28, options.gpu_block28_bin.as_deref()),
            GpuEngine::Block29 => (BlockVariant::Block29, options.gpu_block29_bin.as_deref()),
        };
        let mut adapter = BlockEngine::new(variant, explicit);
        if let Err(error) = adapter.discover() {
            eprintln!("Error: {error}");
            std::process::exit(2);
        }
        block_engines.push(adapter);
    }

    let mut simd_methods: Vec<&str> = if !selection.simd {
        Vec::new()
    } else if options.method == "compare" {
        vec!["direct", "matrix", "doubling"]
    } else {
        vec![options.method.as_str()]
    };
    let mut simd_skip_note: Option<String> = None;
    let skip_slow_simd = selection.simd && !options.force_simd && !options.minute;
    if skip_slow_simd {
        if options.index >= SIMD_SKIP_ALL_INDEX {
            if !simd_methods.is_empty() {
                simd_skip_note = Some(format!(
                    "skipping all CPU/SIMD methods for n={} (slow; use --force-simd or --backend auto)",
                    options.index
                ));
                simd_methods.clear();
            }
        } else if options.index >= SIMD_SKIP_HEAVY_INDEX {
            let before = simd_methods.len();
            simd_methods.retain(|method| *method == "doubling");
            if simd_methods.len() != before {
                simd_skip_note = Some(format!(
                    "skipping CPU/SIMD direct, matrix for n={} (slow; use --force-simd)",
                    options.index
                ));
            }
        }
    }
    let reference_methods: Vec<Method> = if options.method == "compare" {
        Method::ALL.to_vec()
    } else {
        vec![
            options
                .method
                .parse::<Method>()
                .expect("validated --method"),
        ]
    };
    let num_bigint_methods: Vec<Method> = if selection.num_bigint {
        reference_methods.clone()
    } else {
        Vec::new()
    };
    let gmp_methods: Vec<Method> = if selection.gmp {
        reference_methods
    } else {
        Vec::new()
    };

    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let workers = workers_for(options.index, options.workers, available);

    // The banner goes to stderr before any report or calculation, so it never
    // pollutes `--json` output on stdout.
    let color = color_enabled(options.no_color);
    if let Some(style) = options.banner {
        // Animate only when explicitly allowed and stderr is an interactive
        // terminal; redirected output gets the plain, single-shot banner.
        banner::print_animated(
            style,
            color,
            !options.no_animate && std::io::stderr().is_terminal(),
        );
        // Keep the hardware report visually separate from the artwork.
        eprintln!();
        flush_output();
    }

    // Report hardware once, on stderr, before the thread pool and any work. The
    // GPU report doubles as the source of the device name for the GPU block.
    let gpu_info = if gpu_selected || !options.no_hwinfo {
        Some(gpu::detect())
    } else {
        None
    };
    let kernel = selected_kernel_name();
    if !options.no_hwinfo {
        let cpu = cpu::detect();
        eprintln!("{}", cpu::format_human_colored(&cpu, color));
        if let Some(info) = &gpu_info {
            eprintln!("{}", gpu::format_human_colored(info, color));
        }
    }
    print_preflight(&options, color, &resolved);
    if gpu_selected
        && resolved.run.is_empty()
        && !selection.simd
        && !selection.num_bigint
        && !selection.gmp
    {
        eprintln!(
            "Error: no selected GPU backend can run n={}; see the preflight notes above",
            options.index
        );
        std::process::exit(2);
    }
    // Each fib-rust adapter owns a separate Rayon pool. Build them here, after
    // the banner and hardware report, and before the SIMD global pool exists.
    let num_bigint_runner = if selection.num_bigint {
        match NumBigintRunner::new(workers) {
            Ok(runner) => Some(runner),
            Err(error) => {
                eprintln!("Error: {error}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    let gmp_runner = if selection.gmp {
        match GmpRunner::new(workers) {
            Ok(runner) => Some(runner),
            Err(error) => {
                eprintln!("Error: {error}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    eprintln!(
        "{} kernel={} workers={}",
        paint(SIMD_LABEL, CYAN, color),
        paint(kernel, GOLD, color),
        paint(&workers.to_string(), GOLD, color)
    );
    if let Some(note) = &simd_skip_note {
        eprintln!("{} {}", paint("note:", DIM, color), paint(note, DIM, color));
    }
    if let Some(runner) = &num_bigint_runner {
        eprintln!(
            "{} backend={} workers={} reuse={} split-mul={}",
            paint("num-bigint:", CYAN, color),
            paint(runner.backend_name(), GOLD, color),
            paint(&runner.workers().to_string(), GOLD, color),
            paint(&runner.reuse().to_string(), GOLD, color),
            paint(&runner.split_multiplication().to_string(), GOLD, color)
        );
    }
    if let Some(runner) = &gmp_runner {
        eprintln!(
            "{} backend={} workers={} reuse={} split-mul={}",
            paint("GMP:", CYAN, color),
            paint(runner.backend_name(), GOLD, color),
            paint(&runner.workers().to_string(), GOLD, color),
            paint(&runner.reuse().to_string(), GOLD, color),
            paint(&runner.split_multiplication().to_string(), GOLD, color)
        );
    }

    let gpu_preflight_json = any_gpu_requested.then(|| preflight_json(&resolved, &options.backend));
    let context = Context {
        options: &options,
        color,
        workers,
        simd_methods: &simd_methods,
        num_bigint_methods: &num_bigint_methods,
        gmp_methods: &gmp_methods,
        num_bigint: num_bigint_runner.as_ref(),
        gmp: gmp_runner.as_ref(),
        block_engines: &block_engines,
        preflight_json: gpu_preflight_json.as_deref(),
    };
    flush_output();
    flush_output();

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global();

    if options.minute {
        run_minute(&context);
    } else {
        run_single(&context);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_args(values: &[&str]) -> Result<Option<Options>, String> {
        parse(
            &values
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn backend_selection_matrix() {
        let cpu = selection(true, false, false, false, false, false, false, false);
        let num_bigint = selection(false, true, false, false, false, false, false, false);
        let gmp = selection(false, false, true, false, false, false, false, false);
        let gpu = selection(false, false, false, true, false, false, false, false);
        let gmp_gpu = selection(false, false, true, true, false, false, false, false);
        let both = selection(true, false, false, true, false, false, false, false);
        let all = selection(true, true, true, true, false, false, false, false);
        assert_eq!(select_backend("cpu"), Some(cpu));
        assert_eq!(select_backend("num-bigint"), Some(num_bigint));
        // `native` remains a backward-compatible alias for num-bigint.
        assert_eq!(select_backend("native"), Some(num_bigint));
        assert_eq!(select_backend("gmp"), Some(gmp));
        assert_eq!(select_backend("gpu"), Some(gpu));
        assert_eq!(select_backend("gmp-gpu"), Some(gmp_gpu));
        // `gmp+gpu` is accepted as a readable alias.
        assert_eq!(select_backend("gmp+gpu"), Some(gmp_gpu));
        assert_eq!(select_backend("both"), Some(both));
        assert_eq!(select_backend("all"), Some(all));
        assert_eq!(
            select_backend("gpu-block27"),
            Some(selection(
                false, false, false, false, true, false, false, false
            ))
        );
        assert_eq!(
            select_backend("gpu-block28"),
            Some(selection(
                false, false, false, false, false, true, false, false
            ))
        );
        assert_eq!(
            select_backend("gpu-block29"),
            Some(selection(
                false, false, false, false, false, false, true, false
            ))
        );
        assert_eq!(
            select_backend("gpu-block"),
            Some(selection(
                false, false, false, false, true, true, true, false
            ))
        );
        assert_eq!(
            select_backend("gpu-all"),
            Some(selection(
                false, false, false, true, true, true, true, false
            ))
        );
        assert_eq!(
            select_backend("auto"),
            Some(selection(
                false, false, false, false, false, false, false, true
            ))
        );
        assert_eq!(select_backend("simd"), None);
    }

    #[test]
    fn cpu_backends_accept_all_cpu_methods() {
        for backend in ["num-bigint", "native", "gmp"] {
            for method in ["direct", "matrix", "doubling", "compare"] {
                let options = parse_args(&["--backend", backend, "--method", method, "100"])
                    .unwrap()
                    .unwrap();
                assert_eq!(options.backend, backend);
                assert_eq!(options.method, method);
            }
            // CPU-only runs may combine with the staircase.
            assert!(parse_args(&["--backend", backend, "--minute"]).is_ok());
        }
    }

    #[test]
    fn gpu_backends_still_require_doubling_or_compare() {
        for backend in [
            "gpu",
            "gpu-block27",
            "gpu-block28",
            "gpu-block29",
            "gpu-block",
            "gmp-gpu",
            "gmp+gpu",
            "both",
            "all",
        ] {
            for method in ["direct", "matrix"] {
                assert!(
                    parse_args(&["--backend", backend, "--method", method, "100"]).is_err(),
                    "{backend} {method} should be rejected"
                );
            }
            for method in ["doubling", "compare"] {
                assert!(
                    parse_args(&["--backend", backend, "--method", method, "100"]).is_ok(),
                    "{backend} {method} should be accepted"
                );
            }
        }
    }

    #[test]
    fn backend_error_lists_every_value() {
        let error = parse_args(&["--backend", "simd", "100"]).unwrap_err();
        assert!(error.contains("num-bigint"));
        assert!(error.contains("gmp"));
        assert!(error.contains("gmp-gpu"));
        assert!(error.contains("gpu-all"));
        assert!(error.contains("all"));
    }

    #[test]
    fn simd_metadata_prefix_is_upper_case() {
        assert_eq!(SIMD_LABEL, "SIMD:");
        assert!(HELP.contains("CPU/SIMD"));
        assert!(HELP.contains("CPU/num-bigint"));
        assert!(HELP.contains("CPU/GMP"));
        assert!(HELP.contains("block-28"));
        assert!(HELP.contains("gpu-block28"));
        assert!(HELP.contains("FIB_GPU_BLOCK28"));
        assert!(HELP.contains("184,070,026,875"));
        assert!(HELP.contains("reuse=true"));
        assert!(HELP.contains("split-multiplication=true"));
    }

    #[test]
    fn progress_flag_is_optional_and_documented() {
        assert!(HELP.contains("--no-progress"));
        let default = parse_args(&["100"]).unwrap().unwrap();
        assert!(!default.no_progress);
        let disabled = parse_args(&["--no-progress", "100"]).unwrap().unwrap();
        assert!(disabled.no_progress);
        let explicit = parse_args(&["--no-progress=false", "100"])
            .unwrap()
            .unwrap();
        assert!(!explicit.no_progress);
    }

    #[test]
    fn chart_labels_fit_and_stay_distinct() {
        assert_eq!(chart_bar_label(Backend::Gmp, "direct"), "GMP direct");
        assert_eq!(chart_bar_label(Backend::Gmp, "matrix"), "GMP matrix");
        assert_eq!(chart_bar_label(Backend::Gmp, "doubling"), "GMP doubling");
        assert_eq!(
            chart_bar_label(Backend::Gpu, "doubling"),
            "GPU-r27 doubling"
        );
        assert_eq!(
            chart_bar_label(Backend::GpuBlock27, "doubling"),
            "GPU-b27 doubling"
        );
        assert_eq!(
            chart_bar_label(Backend::GpuBlock28, "doubling"),
            "GPU-b28 doubling"
        );
        assert_eq!(
            chart_bar_label(Backend::GpuBlock29, "doubling"),
            "GPU-b29 doubling"
        );
        assert_eq!(chart_bar_label(Backend::Cpu, "doubling"), "SIMD doubling");
        assert_eq!(
            chart_bar_label(Backend::NumBigint, "direct"),
            "num-bigint direct"
        );
        assert_eq!(
            chart_bar_label(Backend::NumBigint, "matrix"),
            "num-bigint matrix"
        );
        assert_eq!(
            chart_bar_label(Backend::NumBigint, "doubling"),
            "num-bigint doubling"
        );
        for backend in [
            Backend::Cpu,
            Backend::NumBigint,
            Backend::Gmp,
            Backend::Gpu,
            Backend::GpuBlock27,
            Backend::GpuBlock28,
            Backend::GpuBlock29,
        ] {
            for method in ["direct", "matrix", "doubling"] {
                let label = chart_bar_label(backend, method);
                assert!(label.chars().count() <= 20, "{label:?} is too wide");
            }
        }
    }

    fn test_measurement(backend: Backend, method: &'static str, value: CpuValue) -> Measurement {
        Measurement {
            backend,
            method,
            workers: 1,
            bits: 0,
            samples: Vec::new(),
            median_ns: 0,
            index: 1,
            exact: None,
            value: Some(value),
            block: None,
        }
    }

    #[test]
    fn gmp_exactness_is_annotated_against_num_bigint() {
        let mut measurements = vec![
            test_measurement(
                Backend::NumBigint,
                "direct",
                CpuValue::NumBigint(num_bigint::BigUint::from(42u64)),
            ),
            test_measurement(
                Backend::NumBigint,
                "matrix",
                CpuValue::NumBigint(num_bigint::BigUint::from(41u64)),
            ),
            test_measurement(
                Backend::Gmp,
                "direct",
                CpuValue::Gmp(rug::Integer::from(42u64)),
            ),
            test_measurement(
                Backend::Gmp,
                "matrix",
                CpuValue::Gmp(rug::Integer::from(43u64)),
            ),
            test_measurement(
                Backend::Gmp,
                "doubling",
                CpuValue::Gmp(rug::Integer::from(42u64)),
            ),
        ];
        annotate_gmp_exactness(&mut measurements);
        assert_eq!(measurements[2].exact, Some(true));
        assert_eq!(measurements[3].exact, Some(false));
        // No num-bigint doubling reference ran, so nothing was claimed.
        assert_eq!(measurements[4].exact, None);
    }

    fn parsed_options(args: &[&str]) -> Options {
        parse_args(args).unwrap().unwrap()
    }

    fn machine() -> Resources {
        Resources::fixture(23.89, 23.15, 63.4, 49.0)
    }

    #[test]
    fn auto_resolution_follows_the_documented_ranking() {
        let cases = [
            (1_000_000u64, true, false, false, false),
            (6_186_556_317, true, false, false, false),
            (61_865_563_170, false, true, false, false),
            (70_000_000_000, false, true, false, false),
            (73_628_010_691, false, true, false, false),
            (73_628_010_692, false, true, false, false),
            (86_600_000_000, false, false, true, false),
            (100_000_000_000, false, false, true, false),
            (184_000_000_000, false, false, true, false),
        ];
        for (index, resident, block27, block28, block29) in cases {
            let index_text = index.to_string();
            let options = parsed_options(&["--backend", "auto", index_text.as_str()]);
            let requested = select_backend("auto").unwrap();
            let resolved = resolve_gpu(&options, requested, machine()).unwrap();
            assert_eq!(resolved.selection.gpu, resident, "n={index}");
            assert_eq!(resolved.selection.block27, block27, "n={index}");
            assert_eq!(resolved.selection.block28, block28, "n={index}");
            assert_eq!(resolved.selection.block29, block29, "n={index}");
            assert_eq!(resolved.run.len(), 1, "n={index}");
        }
    }

    #[test]
    fn auto_rejects_indices_above_the_block28_planner_maximum() {
        for index in [184_100_000_000u64, 185_000_000_000, 200_000_000_000] {
            let index_text = index.to_string();
            let options = parsed_options(&["--backend", "auto", index_text.as_str()]);
            let requested = select_backend("auto").unwrap();
            let error = resolve_gpu(&options, requested, machine()).unwrap_err();
            assert!(error.contains("184,070,026,875"), "{index}: {error}");
        }
    }

    #[test]
    fn preflight_skips_and_force_flags_override() {
        let tight = Resources::fixture(23.89, 18.20, 63.4, 49.0);
        let options = parsed_options(&["--backend", "gpu-block29", "73628010691"]);
        let requested = select_backend("gpu-block29").unwrap();
        let resolved = resolve_gpu(&options, requested, tight).unwrap();
        assert!(resolved.run.is_empty());
        assert_eq!(resolved.skipped.len(), 1);
        assert!(
            resolved.skipped[0]
                .skip_reason
                .as_deref()
                .unwrap()
                .contains("device")
        );

        let forced = parsed_options(&["--backend", "gpu-block29", "--force-memory", "73628010691"]);
        let resolved = resolve_gpu(&forced, requested, tight).unwrap();
        assert_eq!(resolved.run, vec![GpuEngine::Block29]);

        let over = parsed_options(&["--backend", "gpu", "6186556318"]);
        let requested = select_backend("gpu").unwrap();
        let resolved = resolve_gpu(&over, requested, machine()).unwrap();
        assert!(resolved.run.is_empty());
        let forced_index = parsed_options(&["--backend", "gpu", "--force-index", "6186556318"]);
        let resolved = resolve_gpu(&forced_index, requested, machine()).unwrap();
        assert_eq!(resolved.run, vec![GpuEngine::Resident]);
    }

    #[test]
    fn gpu_flags_are_validated_and_parsed() {
        let options = parsed_options(&[
            "--backend",
            "gpu-block",
            "--gpu-out",
            "D:/ref/out.bin",
            "--keep-gpu-output",
            "--gpu-sha256",
            "4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580",
            "--gpu-reference",
            "D:/ref/gmp.bin",
            "--gpu-block27-bin",
            "b27.exe",
            "--gpu-block28-bin",
            "b28.exe",
            "--gpu-block29-bin",
            "b29.exe",
            "--force-memory",
            "--force-index",
            "100",
        ]);
        assert_eq!(options.backend, "gpu-block");
        assert_eq!(options.gpu_out.as_deref(), Some("D:/ref/out.bin"));
        assert!(options.keep_gpu_output);
        assert!(options.force_memory);
        assert!(options.force_index);
        assert!(options.gpu_sha256.is_some());
        assert!(options.gpu_reference.is_some());
        assert_eq!(options.gpu_block28_bin.as_deref(), Some("b28.exe"));

        assert!(parse_args(&["--backend", "gpu", "--gpu-out", "x", "100"]).is_ok());
        assert!(parse_args(&["--backend", "cpu", "--gpu-out", "x", "100"]).is_err());
        assert!(parse_args(&["--backend", "gpu-block27", "--gpu-sha256", "abc", "100"]).is_err());
        assert!(parse_args(&["--backend", "gpu-block28", "--gpu-sha256", "abc", "100"]).is_err());
        assert!(parse_args(&["--backend", "gpu-block27", "--minute"]).is_err());
        assert!(parse_args(&["--backend", "gpu-block28", "--minute"]).is_err());
        assert!(parse_args(&["--backend", "auto", "--minute"]).is_err());
    }

    #[test]
    fn show_limits_accepts_the_command_or_flag_without_an_index() {
        assert!(parse_args(&["--show-limits"]).unwrap().unwrap().show_limits);
        assert!(parse_args(&["limits"]).unwrap().unwrap().show_limits);
        assert!(
            parse_args(&["--json", "limits"])
                .unwrap()
                .unwrap()
                .show_limits
        );
        assert!(parse_args(&[]).is_err());
    }

    #[test]
    fn block_output_paths_never_collide() {
        let mut options = parsed_options(&["--backend", "gpu-block", "100"]);
        options.gpu_out = Some("D:/ref/out.bin".into());
        assert_eq!(
            block_output_path(&options, BlockVariant::Block27, 100, true, 0)
                .unwrap()
                .to_string_lossy(),
            "D:/ref/out.bin-block-27"
        );
        assert_eq!(
            block_output_path(&options, BlockVariant::Block28, 100, true, 0)
                .unwrap()
                .to_string_lossy(),
            "D:/ref/out.bin-block-28"
        );
        assert_eq!(
            block_output_path(&options, BlockVariant::Block29, 100, true, 1)
                .unwrap()
                .to_string_lossy(),
            "D:/ref/out.bin-block-29-run2"
        );
        options.gpu_out = Some("D:/ref/".into());
        assert_eq!(
            block_output_path(&options, BlockVariant::Block27, 100, true, 0)
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "gpu-F-100-block-27.bin"
        );
    }

    #[test]
    fn preflight_runs_before_binary_discovery() {
        // `resolve_gpu` never constructs a BlockEngine, loads a CUDA library, or starts a
        // process. A nonexistent binary therefore does not matter until after
        // the fit check, which is where main() discovers it.
        let options = parsed_options(&[
            "--backend",
            "gpu-block27",
            "--gpu-block27-bin",
            "Z:/definitely-missing/block27_fib.exe",
            "1000000",
        ]);
        let requested = select_backend("gpu-block27").unwrap();
        let resolved = resolve_gpu(&options, requested, machine()).unwrap();
        assert_eq!(resolved.run, vec![GpuEngine::Block27]);
        assert!(resolved.skipped.is_empty());
    }

    #[test]
    fn preflight_json_names_selected_and_skipped_engines() {
        let options = parsed_options(&["--backend", "gpu-block29", "73628010691"]);
        let requested = select_backend("gpu-block29").unwrap();
        let resolved = resolve_gpu(&options, requested, machine()).unwrap();
        let json = preflight_json(&resolved, "gpu-block29");
        assert!(json.contains("\"requested_backend\":\"gpu-block29\""));
        assert!(json.contains("\"selected\":[\"gpu-block29\"]"));
        assert!(json.contains("\"skipped\":[]"));
        assert!(json.contains("\"required_device_bytes\":23944697107"));
        assert!(json.contains("\"hard_max\":73628010691"));

        let tight = Resources::fixture(23.89, 18.20, 63.4, 49.0);
        let resolved = resolve_gpu(&options, requested, tight).unwrap();
        let json = preflight_json(&resolved, "gpu-block29");
        assert!(json.contains("\"selected\":[]"));
        assert!(json.contains("\"skipped\":[{"));
        assert!(json.contains("needs 22.30 GiB device"));

        let options = parsed_options(&["--backend", "gpu-block28", "184000000000"]);
        let requested = select_backend("gpu-block28").unwrap();
        let resolved = resolve_gpu(&options, requested, machine()).unwrap();
        let json = preflight_json(&resolved, "gpu-block28");
        assert!(json.contains("\"requested_backend\":\"gpu-block28\""));
        assert!(json.contains("\"selected\":[\"gpu-block28\"]"));
        assert!(json.contains("\"skipped\":[]"));
        assert!(
            json.contains("\"required_device_bytes\":23955549661"),
            "{json}"
        );
        assert!(
            json.contains("\"required_host_bytes\":45618709172"),
            "{json}"
        );
        assert!(json.contains("\"hard_max\":184070026875"));
        assert!(json.contains("\"verified_max\":100000000000"));
    }
}
