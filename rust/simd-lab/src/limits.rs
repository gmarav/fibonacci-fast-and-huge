//! GPU backend limits, memory models, resource detection, and preflight.
//!
//! The four CUDA variants have different index ceilings and very different
//! memory footprints. This module keeps the models in one place so the CLI can
//! answer three questions before it launches anything:
//!
//! 1. Is the requested index within the engine's hard/planner limit?
//! 2. Does the estimated device VRAM fit in the currently free VRAM?
//! 3. Does the estimated host RAM fit in the currently available RAM?
//!
//! No function in this module loads a CUDA library or allocates device memory.
//! `detect_resources` only shells out to `nvidia-smi` and calls
//! `GlobalMemoryStatusEx`; the actual engines are started later, one at a time.

use std::fmt::Write as _;
use std::path::PathBuf;

/// Resident 2^27 / 3x32 engine limit (verified and hard).
pub const RESIDENT_MAX_INDEX: u64 = 6_186_556_317;
/// Highest block-27 index verified by a full independent comparison (GMP).
pub const BLOCK27_VERIFIED_MAX_INDEX: u64 = 61_865_563_170;
/// Highest block-27 index measured successfully on the 24 GB reference card.
/// This is a hash-only limit point (k=14), not an exactness anchor; the
/// practical maximum depends on free VRAM.
pub const BLOCK27_MEASURED_MAX_INDEX: u64 = 86_600_000_000;
/// Verified maximum for the block-28 engine: the exact 1e11 anchor.
pub const BLOCK28_VERIFIED_MAX_INDEX: u64 = 100_000_000_000;
/// Highest block-28 index measured successfully (hash-only limit point).
pub const BLOCK28_MEASURED_MAX_INDEX: u64 = 184_000_000_000;
/// Planner maximum for the block-28 engine: 15 blocks of 2^27 limbs per operand.
pub const BLOCK28_MAX_INDEX: u64 = 184_070_026_875;
/// SHA-256 of the exact GMP-verified F(100,000,000,000) block-28 output.
pub const BLOCK28_KNOWN_1E11_SHA256: &str =
    "9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0";
/// Verified maximum for the block-29 engine (the published target).
pub const BLOCK29_VERIFIED_MAX_INDEX: u64 = 61_865_563_170;
/// Planner maximum for the block-29 engine: 3 blocks of 2^28 limbs per operand.
pub const BLOCK29_MAX_INDEX: u64 = 73_628_010_691;

/// log2(phi) with enough digits to size transforms. Used only for memory
/// models, never to calculate Fibonacci results.
pub const LOG2_PHI: f64 = 0.694_241_913_630_617_4;

pub const GIB: u64 = 1024 * 1024 * 1024;
pub const MIB: u64 = 1024 * 1024;

/// Safety margin subtracted from free device memory before a fit is accepted.
pub const DEVICE_SAFETY_BYTES: u64 = 512 * MIB;
/// Safety margin subtracted from available host memory.
pub const HOST_SAFETY_BYTES: u64 = 512 * MIB;

const RESIDENT_BYTES_PER_ELEMENT: u64 = 108;
const RESIDENT_CONTEXT_BYTES: u64 = 1_395_864_371; // ~1.30 GiB
const BLOCK27_B: u64 = 1 << 26;
const BLOCK27_N: u64 = 1 << 27;
const BLOCK27_CONTEXT_BYTES: u64 = 1_406_601_789; // ~1.31 GiB measured
const BLOCK28_BLOCK_B: u64 = 1 << 27;
const BLOCK28_N: u64 = 1 << 28;
const BLOCK28_CONTEXT_BYTES: u64 = 1_406_601_789; // ~1.31 GiB measured
const BLOCK28_TWIDDLES: u64 = 2 * (2 * 29 * 8 + 2 * 29 * 256 * 8 + 2 * 4096 * 8);
const BLOCK29_BLOCK_B: u64 = 1 << 28;
const BLOCK29_N: u64 = 1 << 29;
const BLOCK29_CONTEXT_BYTES: u64 = 1_395_864_371; // ~1.30 GiB measured
const BLOCK29_TWIDDLES: u64 = 2 * 30 * 8 + 2 * 30 * 256 * 8 + 4 * 4096 * 8;

/// The four CUDA families the CLI knows about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuEngine {
    Resident,
    Block27,
    Block28,
    Block29,
}

impl GpuEngine {
    /// All engines in preference order for automatic selection.
    pub const ALL: [GpuEngine; 4] = [
        GpuEngine::Resident,
        GpuEngine::Block27,
        GpuEngine::Block28,
        GpuEngine::Block29,
    ];

    /// `--backend` value selecting this engine alone.
    pub fn backend_flag(self) -> &'static str {
        match self {
            GpuEngine::Resident => "gpu",
            GpuEngine::Block27 => "gpu-block27",
            GpuEngine::Block28 => "gpu-block28",
            GpuEngine::Block29 => "gpu-block29",
        }
    }

    /// Table/display label.
    pub fn label(self) -> &'static str {
        match self {
            GpuEngine::Resident => "GPU/CUDA resident",
            GpuEngine::Block27 => "GPU/CUDA block-27",
            GpuEngine::Block28 => "GPU/CUDA block-28",
            GpuEngine::Block29 => "GPU/CUDA block-29",
        }
    }

    /// Short label used by recommendations.
    pub fn short(self) -> &'static str {
        match self {
            GpuEngine::Resident => "resident",
            GpuEngine::Block27 => "block-27",
            GpuEngine::Block28 => "block-28",
            GpuEngine::Block29 => "block-29",
        }
    }

    /// Hard index ceiling, when one exists.
    pub fn hard_max(self) -> Option<u64> {
        match self {
            GpuEngine::Resident => Some(RESIDENT_MAX_INDEX),
            GpuEngine::Block27 => None,
            GpuEngine::Block28 => Some(BLOCK28_MAX_INDEX),
            GpuEngine::Block29 => Some(BLOCK29_MAX_INDEX),
        }
    }

    /// Highest index verified by a full independent comparison.
    pub fn verified_max(self) -> u64 {
        match self {
            GpuEngine::Resident => RESIDENT_MAX_INDEX,
            GpuEngine::Block27 => BLOCK27_VERIFIED_MAX_INDEX,
            GpuEngine::Block28 => BLOCK28_VERIFIED_MAX_INDEX,
            GpuEngine::Block29 => BLOCK29_VERIFIED_MAX_INDEX,
        }
    }

    /// Why this engine stops where it does.
    pub fn limiter(self) -> &'static str {
        match self {
            GpuEngine::Resident => {
                "2^27 transform / 32-bit limb-count cap (6,186,556,318 is rejected)"
            }
            GpuEngine::Block27 => {
                "device VRAM: 0.75 GiB per extra 2^26-limb operand block; exact 61,865,563,170; 86.6e9 measured (hash-only)"
            }
            GpuEngine::Block28 => {
                "planner: k<=15 x 2^27 limbs (max 184,070,026,875; measured 184e9; 185e9 needs k=16)"
            }
            GpuEngine::Block29 => "planner: at most 3 x 2^28 limbs per operand",
        }
    }

    /// Measured device memory at the verified maximum, in bytes.
    pub fn measured_device_at_verified(self) -> Option<u64> {
        match self {
            // 15,157.6 MiB measured process VRAM at cap 2^27.
            GpuEngine::Resident => Some(15_157 * MIB + (6 * MIB / 10)),
            // 20,289.6 MiB measured peak.
            GpuEngine::Block27 => Some(20_289 * MIB + (6 * MIB / 10)),
            // 16,673.6 MiB measured peak at the 1e11 exactness anchor.
            GpuEngine::Block28 => Some(16_673 * MIB + (6 * MIB / 10)),
            // 22,817.6 MiB measured peak.
            GpuEngine::Block29 => Some(22_817 * MIB + (6 * MIB / 10)),
        }
    }
}

/// A conservative upper bound on the limb count of the largest doubling
/// operand, matching `cuda-ntt/block-29/plan_model.h`.
pub fn block29_operand_limb_bound(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let half = n / 2;
    let bits = (half / 10) * 7 + ((half % 10) * 7) / 10 + 2;
    bits / 32 + 2
}

/// Operand limbs for the resident and block-27 models. Uses the true
/// `log2(phi)` so the 2^27 resident cap is not over-estimated above its
/// validated maximum.
pub fn operand_limb_bound(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let bits = (n as f64 * 0.5 * LOG2_PHI).ceil();
    (bits / 32.0).ceil() as u64 + 2
}

fn next_power_of_two(value: u64) -> u64 {
    let mut result = 1u64;
    while result < value {
        match result.checked_mul(2) {
            Some(next) => result = next,
            None => return u64::MAX,
        }
    }
    result
}

/// Number of 2^26-limb operand blocks used by block-27.
pub fn block27_blocks(n: u64) -> u64 {
    let limbs = operand_limb_bound(n);
    if limbs == 0 {
        return 1;
    }
    limbs.div_ceil(BLOCK27_B).max(1)
}

/// Required device bytes for block-27 at `n`.
///
/// Fixed NTT sets, twiddles, accumulator, limb/hi buffers, and measured context
/// overhead are ~12.31 GiB; each operand block adds 0.75 GiB for `a`, `b`, `t`.
pub fn block27_device_bytes(n: u64) -> u64 {
    let blocks = block27_blocks(n);
    let ntt = 12 * (BLOCK27_N + 8) * 4;
    let twiddles = 6 * (BLOCK27_N - 1) * 4;
    let acc = (BLOCK27_N + 8) * 8;
    let limb_hi = 2 * (BLOCK27_N + 8) * 4;
    let operands = 3 * blocks * BLOCK27_B * 4;
    ntt + twiddles + acc + limb_hi + operands + BLOCK27_CONTEXT_BYTES
}

/// Required host bytes for block-27 at `n`.
///
/// Three host operand buffers, the `c` and optional `d` results, the 512 MiB
/// pinned staging buffer, and a 1 GiB allocator/copy margin.
pub fn block27_host_bytes(n: u64) -> u64 {
    let blocks = block27_blocks(n);
    let limbs = operand_limb_bound(n);
    3 * blocks * BLOCK27_B * 4 + 2 * 4 * limbs + 2 * BLOCK27_B * 4 + GIB
}

/// Required transform length for the resident engine.
pub fn resident_elements(n: u64) -> u64 {
    let limbs = operand_limb_bound(n);
    if limbs == 0 {
        return 1 << 10;
    }
    next_power_of_two(limbs.saturating_mul(2)).min(1 << 27)
}

/// Required device bytes for the resident engine.
pub fn resident_device_bytes(n: u64) -> u64 {
    RESIDENT_BYTES_PER_ELEMENT * resident_elements(n) + RESIDENT_CONTEXT_BYTES
}

/// Required host bytes for the resident engine (result plus a small margin).
pub fn resident_host_bytes(n: u64) -> u64 {
    4 * operand_limb_bound(n) + 256 * MIB
}

/// One block-28 plan, mirroring `cuda-ntt/block-28/plan_model.h`.
///
/// The engine always uses a 2^28 transform with 2^27-limb blocks once the
/// operands no longer fit a single transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block28Plan {
    pub lg: u32,
    pub b: u64,
    pub n: u64,
    pub block_mode: bool,
    pub na: u64,
    pub nb: u64,
    pub k: u64,
    pub need: u64,
}

/// Plan one block-28 product for operands of `la` and `lb` limbs.
pub fn block28_plan(la: u64, lb: u64) -> Block28Plan {
    let need = la.saturating_add(lb);
    if need == 0 {
        return Block28Plan {
            lg: 1,
            b: 1,
            n: 2,
            block_mode: false,
            na: 0,
            nb: 0,
            k: 0,
            need: 0,
        };
    }
    let (lg, b, n, block_mode) = if need <= (1u64 << 28) {
        let lg = ceil_log2(need).max(1);
        (lg, 1u64 << (lg - 1), 1u64 << lg, false)
    } else {
        (28, BLOCK28_BLOCK_B, BLOCK28_N, true)
    };
    let na = la.div_ceil(b);
    let nb = lb.div_ceil(b);
    let k = na.saturating_add(nb).saturating_sub(1);
    Block28Plan {
        lg,
        b,
        n,
        block_mode,
        na,
        nb,
        k,
        need,
    }
}

/// Number of staging blocks per operand for block-28 at `n`.
///
/// The planner's documented k=15 boundary sizes both operands by the same
/// conservative bound; the second operand's extra limb is handled by the
/// measured headroom used by the block-28 planner.
pub fn block28_blocks(n: u64) -> u64 {
    let limbs = block29_operand_limb_bound(n);
    let plan = block28_plan(limbs, limbs);
    plan.na.max(plan.nb).max(1)
}

/// Whether the block-28 device schedule supports this plan.
pub fn block28_plan_supported(plan: &Block28Plan) -> bool {
    plan.na <= 15 && plan.nb <= 15
}

/// Planner maximum computed from the same integer model as `plan_model.h`.
pub fn block28_planner_max() -> u64 {
    let mut low = 0u64;
    let mut high = 200_000_000_000u64;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let limbs = block29_operand_limb_bound(middle);
        let plan = block28_plan(limbs, limbs);
        if block28_plan_supported(&plan) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// Required device bytes for block-28 at `n` (model, no allocation).
///
/// Option A keeps three 2^28-entry u64 transform arrays, two staging buffers of
/// `k` 2^27-limb blocks, twiddles, and a measured ~1.31 GiB CUDA context.
pub fn block28_device_bytes(n: u64) -> u64 {
    let limbs = block29_operand_limb_bound(n);
    let plan = block28_plan(limbs, limbs);
    let blocks = plan.na.max(plan.nb).max(1);
    let arrays = 3 * plan.n * 8;
    let staging = 2 * blocks * plan.b * 4;
    arrays + staging + BLOCK28_TWIDDLES + BLOCK28_CONTEXT_BYTES
}

/// Required host bytes for block-28 at `n`, one product at a time.
///
/// Mirrors `plan_model::host_total`: `a`, `b`, `2b-a`, the product being built,
/// the four download windows per block, the carry scratch, and the output.
pub fn block28_host_bytes(n: u64) -> u64 {
    let limbs = block29_operand_limb_bound(n);
    let plan = block28_plan(limbs, limbs.saturating_add(1));
    let operands = (limbs + limbs.saturating_add(1) + limbs.saturating_add(1)) * 4;
    let download = 4 * plan.b * 8;
    let segwork = plan.b * 8;
    let output = plan.need * 4;
    operands + download + segwork + output
}

/// One block-29 plan, mirroring `plan_model.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block29Plan {
    pub lg: u32,
    pub b: u64,
    pub n: u64,
    pub block_mode: bool,
    pub na: u64,
    pub nb: u64,
    pub k: u64,
    pub need: u64,
}

fn ceil_log2(value: u64) -> u32 {
    let mut lg = 0;
    let mut v = 1u64;
    while v < value {
        v <<= 1;
        lg += 1;
    }
    lg
}

/// Plan one block-29 product for operands of `la` and `lb` limbs.
pub fn block29_plan(la: u64, lb: u64) -> Block29Plan {
    let need = la.saturating_add(lb);
    if need == 0 {
        return Block29Plan {
            lg: 1,
            b: 1,
            n: 2,
            block_mode: false,
            na: 0,
            nb: 0,
            k: 0,
            need: 0,
        };
    }
    let (lg, b, n, block_mode) = if need <= (1u64 << 29) {
        let lg = ceil_log2(need).max(1);
        (lg, 1u64 << (lg - 1), 1u64 << lg, false)
    } else {
        (29, BLOCK29_BLOCK_B, BLOCK29_N, true)
    };
    let na = la.div_ceil(b);
    let nb = lb.div_ceil(b);
    let k = na.saturating_add(nb).saturating_sub(1);
    Block29Plan {
        lg,
        b,
        n,
        block_mode,
        na,
        nb,
        k,
        need,
    }
}

/// Whether the block-29 device schedule supports this plan.
pub fn block29_plan_supported(plan: &Block29Plan) -> bool {
    plan.na <= 3 && plan.nb <= 3 && plan.na + plan.nb <= 6
}

/// Planner maximum computed from the same integer model as `plan_model.h`.
pub fn block29_planner_max() -> u64 {
    // The constant is derived from the planner formula; keep the computation
    // here so a future change to the formula is visible.
    let mut low = 0u64;
    let mut high = 80_000_000_000u64;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let limbs = block29_operand_limb_bound(middle);
        let plan = block29_plan(limbs, limbs);
        if block29_plan_supported(&plan) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// Required device bytes for block-29 at `n` (model, no allocation).
pub fn block29_device_bytes(n: u64) -> u64 {
    let limbs = block29_operand_limb_bound(n);
    let plan = block29_plan(limbs, limbs);
    let stage_cap = (plan.n / 2).min(1 << 28);
    5 * plan.n * 8 + stage_cap * 4 + BLOCK29_TWIDDLES + BLOCK29_CONTEXT_BYTES
}

/// Required host bytes for block-29 at `n`, one product at a time.
pub fn block29_host_bytes(n: u64) -> u64 {
    let limbs = block29_operand_limb_bound(n);
    let plan = block29_plan(limbs, limbs);
    let operands = plan.need * 4;
    let seg0 = (plan.k + 1) * plan.b * 6;
    let working = plan.b * 8 + plan.n * 8 + plan.b * 8;
    let output = plan.need * 4;
    operands + seg0 + working + output
}

/// Model device and host bytes for one engine/index.
pub fn estimate(engine: GpuEngine, n: u64) -> (u64, u64) {
    match engine {
        GpuEngine::Resident => (resident_device_bytes(n), resident_host_bytes(n)),
        GpuEngine::Block27 => (block27_device_bytes(n), block27_host_bytes(n)),
        GpuEngine::Block28 => (block28_device_bytes(n), block28_host_bytes(n)),
        GpuEngine::Block29 => (block29_device_bytes(n), block29_host_bytes(n)),
    }
}

/// Measured host memory at the verified maximum, when a measurement exists.
pub fn measured_host_at_verified(engine: GpuEngine) -> Option<&'static str> {
    match engine {
        // The resident result is ~0.54 GiB; 0.75 GiB is a model value.
        GpuEngine::Resident => None,
        // The block-27 host peak during a step was reported as 14-19 GB.
        GpuEngine::Block27 => Some("14-19 GB measured range; 13.0 GiB model"),
        // 1e11: 25,941 MiB (25.33 GiB) measured; 25.37 GiB model.
        GpuEngine::Block28 => Some("25.33 GiB measured at 1e11; 25.37 GiB model"),
        GpuEngine::Block29 => Some("27.08 GiB model (true peak can reach ~37 GiB)"),
    }
}

/// Current machine resources, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub device_total_bytes: Option<u64>,
    pub device_free_bytes: Option<u64>,
    pub host_total_bytes: Option<u64>,
    pub host_available_bytes: Option<u64>,
}

impl Resources {
    /// A deterministic fixture used by tests and by `--show-limits` when no
    /// GPU is present.
    pub fn fixture(
        device_total_gib: f64,
        device_free_gib: f64,
        host_total_gib: f64,
        host_free_gib: f64,
    ) -> Self {
        Self {
            device_total_bytes: Some((device_total_gib * GIB as f64) as u64),
            device_free_bytes: Some((device_free_gib * GIB as f64) as u64),
            host_total_bytes: Some((host_total_gib * GIB as f64) as u64),
            host_available_bytes: Some((host_free_gib * GIB as f64) as u64),
        }
    }
}

/// Result of checking one engine at one index.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineFit {
    pub engine: GpuEngine,
    pub n: u64,
    pub required_device_bytes: u64,
    pub required_host_bytes: u64,
    pub available_device_bytes: Option<u64>,
    pub available_host_bytes: Option<u64>,
    pub index_allowed: bool,
    pub fits_device: bool,
    pub fits_host: bool,
    pub fits: bool,
    pub over_verified: bool,
    pub skip_reason: Option<String>,
    pub note: Option<String>,
}

impl EngineFit {
    /// Compact English recommendation line for the startup preflight.
    pub fn recommendation(&self) -> String {
        if !self.index_allowed {
            let max = self.engine.hard_max().unwrap_or(u64::MAX);
            let alternative = match self.engine {
                GpuEngine::Resident => {
                    "use --backend gpu-block27 (no planner cap), gpu-block28, or gpu-block29"
                }
                GpuEngine::Block27 => "reduce n",
                GpuEngine::Block28 => {
                    "reduce n to at most 184,070,026,875 (185e9 needs k=16 and fails the device preflight)"
                }
                GpuEngine::Block29 => {
                    "use --backend gpu-block27 (no planner cap) or gpu-block28 (max 184,070,026,875)"
                }
            };
            return format!(
                "{}: n={} exceeds {} max {}; {}",
                self.engine.label(),
                self.n,
                self.engine.short(),
                format_u64(max),
                alternative
            );
        }
        if self.fits {
            return format!(
                "{} fits (device {}, host {})",
                self.engine.label(),
                memory_pair(self.required_device_bytes, self.available_device_bytes),
                memory_pair(self.required_host_bytes, self.available_host_bytes)
            );
        }
        if !self.fits_device {
            format!(
                "{} needs {} device plus the 0.50 GiB safety margin but only {} is free; close other GPU processes, reduce n, or use --force-memory",
                self.engine.label(),
                gib(self.required_device_bytes),
                self.available_device_bytes
                    .map_or_else(|| "an unknown amount".to_string(), gib)
            )
        } else {
            format!(
                "{} needs {} host plus the 0.50 GiB safety margin but only {} is available; close other applications, reduce n, or use --force-memory",
                self.engine.label(),
                gib(self.required_host_bytes),
                self.available_host_bytes
                    .map_or_else(|| "an unknown amount".to_string(), gib)
            )
        }
    }
}

/// Check one engine/index against the current resources. Pure: no allocation,
/// no CUDA, no process launch.
pub fn check(engine: GpuEngine, n: u64, resources: &Resources, force_memory: bool) -> EngineFit {
    let (required_device_bytes, required_host_bytes) = estimate(engine, n);
    let index_allowed = engine.hard_max().is_none_or(|max| n <= max);
    let fits_device = resources
        .device_free_bytes
        .is_none_or(|free| required_device_bytes.saturating_add(DEVICE_SAFETY_BYTES) <= free);
    let fits_host = resources
        .host_available_bytes
        .is_none_or(|available| required_host_bytes.saturating_add(HOST_SAFETY_BYTES) <= available);
    let fits = index_allowed && (force_memory || (fits_device && fits_host));
    let over_verified = index_allowed && n > engine.verified_max();

    let skip_reason = if !index_allowed {
        Some(format!(
            "n={} exceeds the {} maximum {}",
            n,
            engine.short(),
            format_u64(engine.hard_max().unwrap_or(u64::MAX))
        ))
    } else if force_memory {
        None
    } else if !fits_device {
        Some(format!(
            "needs {} device plus 0.50 GiB safety but only {} is free",
            gib(required_device_bytes),
            resources
                .device_free_bytes
                .map_or_else(|| "unknown".to_string(), gib)
        ))
    } else if !fits_host {
        Some(format!(
            "needs {} host plus 0.50 GiB safety but only {} is available",
            gib(required_host_bytes),
            resources
                .host_available_bytes
                .map_or_else(|| "unknown".to_string(), gib)
        ))
    } else {
        None
    };

    let note = over_verified.then(|| {
        format!(
            "n={} is above the verified {} maximum {}; the result is unverified unless --gpu-sha256 or --gpu-reference is supplied",
            n,
            engine.short(),
            format_u64(engine.verified_max())
        )
    });

    EngineFit {
        engine,
        n,
        required_device_bytes,
        required_host_bytes,
        available_device_bytes: resources.device_free_bytes,
        available_host_bytes: resources.host_available_bytes,
        index_allowed,
        fits_device,
        fits_host,
        fits,
        over_verified,
        skip_reason,
        note,
    }
}

/// Automatic choice: the fastest exact engine that fits and is allowed.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoChoice {
    pub engine: GpuEngine,
    pub fit: EngineFit,
    pub reason: String,
}

/// Candidate order for `--backend auto`.
///
/// - resident up to 6,186,556,317;
/// - block-27 above that whenever the VRAM preflight fits;
/// - block-28 above block-27's practical maximum, up to 184,070,026,875;
/// - block-29 only when block-27 and block-28 do not fit and n <= 73,628,010,691;
/// - above 184,070,026,875 no engine is eligible, so `auto` rejects the index.
pub fn auto_candidates(n: u64) -> Vec<GpuEngine> {
    if n <= RESIDENT_MAX_INDEX {
        vec![
            GpuEngine::Resident,
            GpuEngine::Block27,
            GpuEngine::Block28,
            GpuEngine::Block29,
        ]
    } else if n <= BLOCK29_MAX_INDEX {
        vec![GpuEngine::Block27, GpuEngine::Block28, GpuEngine::Block29]
    } else if n <= BLOCK28_MAX_INDEX {
        vec![GpuEngine::Block27, GpuEngine::Block28]
    } else {
        Vec::new()
    }
}

/// Choose one engine for `--backend auto`.
pub fn choose_auto(n: u64, resources: &Resources, force_memory: bool) -> Option<AutoChoice> {
    let mut first: Option<EngineFit> = None;
    for engine in auto_candidates(n) {
        let fit = check(engine, n, resources, force_memory);
        if first.is_none() {
            first = Some(fit.clone());
        }
        if fit.index_allowed && fit.fits {
            let mut reason = format!(
                "{} selected as the fastest {}backend for n={}",
                engine.label(),
                if n <= engine.verified_max() {
                    "verified "
                } else {
                    "available "
                },
                n
            );
            if fit.over_verified {
                reason.push_str(&format!(
                    "; note: n is above the verified maximum {}",
                    format_u64(engine.verified_max())
                ));
            }
            if let Some(note) = &fit.note {
                reason.push_str(&format!(" ({note})"));
            }
            return Some(AutoChoice {
                engine,
                fit,
                reason,
            });
        }
    }
    // No candidate fits. Return the first candidate so the caller can print
    // its concrete recommendation instead of a generic failure.
    first.map(|fit| {
        let reason = fit.recommendation();
        AutoChoice {
            engine: fit.engine,
            fit,
            reason,
        }
    })
}

/// A row of the `--show-limits` table.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitsRow {
    pub engine: GpuEngine,
    pub verified_max: u64,
    pub hard_max: Option<u64>,
    /// Model practical maximum for block-27 given current device memory.
    pub practical_max: Option<u64>,
    pub verified_device_bytes: u64,
    pub verified_host_bytes: u64,
    pub hard_device_bytes: Option<u64>,
    pub hard_host_bytes: Option<u64>,
    pub verified_device_source: &'static str,
    pub verified_host_source: &'static str,
    pub fits_verified: bool,
    pub fits_hard: bool,
}

/// Build the limits table for the current resources.
///
/// `hard_max` for block-27 is `None` because the engine has no planner cap.
/// `practical_max` is a VRAM-bound model derived from currently free device
/// memory; it is not a verified index.
pub fn limits_rows(resources: &Resources) -> Vec<LimitsRow> {
    let mut rows = Vec::new();
    for engine in GpuEngine::ALL {
        let verified_device_bytes = engine
            .measured_device_at_verified()
            .unwrap_or_else(|| estimate(engine, engine.verified_max()).0);
        let verified_host_bytes = estimate(engine, engine.verified_max()).1;
        let (verified_device_source, verified_host_source) = match engine {
            GpuEngine::Resident => ("measured", "model"),
            GpuEngine::Block27 => ("measured", "model"),
            GpuEngine::Block28 => ("measured", "model"),
            GpuEngine::Block29 => ("measured", "model"),
        };
        let practical_max =
            (engine == GpuEngine::Block27).then(|| block27_practical_max(resources));
        let hard_max = engine.hard_max();
        let hard_basis = hard_max.or(practical_max);
        let (hard_device_bytes, hard_host_bytes) = hard_basis
            .map(|max| {
                let (device, host) = estimate(engine, max);
                (Some(device), Some(host))
            })
            .unwrap_or((None, None));
        let fits_verified = check(engine, engine.verified_max(), resources, false).fits;
        let fits_hard = hard_basis.is_some_and(|max| check(engine, max, resources, false).fits);
        rows.push(LimitsRow {
            engine,
            verified_max: engine.verified_max(),
            hard_max,
            practical_max,
            verified_device_bytes,
            verified_host_bytes,
            hard_device_bytes,
            hard_host_bytes,
            verified_device_source,
            verified_host_source,
            fits_verified,
            fits_hard,
        });
    }
    rows
}

/// Model the largest block-27 index that fits in currently free device memory.
///
/// The device model is `fixed (including context) + 0.75 GiB * blocks`. The host
/// model is checked as a second bound. This is a projection, not a verified
/// index; the CLI labels it accordingly.
pub fn block27_practical_max(resources: &Resources) -> u64 {
    let Some(free) = resources.device_free_bytes else {
        return BLOCK27_VERIFIED_MAX_INDEX;
    };
    let budget = free.saturating_sub(DEVICE_SAFETY_BYTES);
    let fixed = block27_device_bytes(1) - 3 * BLOCK27_B * 4;
    let per_block = 3 * BLOCK27_B * 4;
    let blocks = budget.saturating_sub(fixed) / per_block;
    if blocks == 0 {
        return 0;
    }
    let device_max = invert_block27_blocks(blocks);
    let host_max = resources
        .host_available_bytes
        .map(block27_host_blocks_bound)
        .unwrap_or(u64::MAX);
    device_max.min(host_max)
}

fn block27_host_blocks_bound(available: u64) -> u64 {
    let budget = available.saturating_sub(HOST_SAFETY_BYTES);
    if budget <= GIB {
        return 0;
    }
    // host = 5 * blocks * B * 4 + 2 * B * 4 + 1 GiB
    let per_block = 5 * BLOCK27_B * 4;
    let fixed = 2 * BLOCK27_B * 4 + GIB;
    let blocks = budget.saturating_sub(fixed) / per_block;
    invert_block27_blocks(blocks.max(1))
}

fn invert_block27_blocks(blocks: u64) -> u64 {
    // Largest n with block27_blocks(n) <= blocks. Binary search over a generous
    // upper bound; the function is monotonic.
    let mut low = 0u64;
    let mut high = (blocks.max(1) as f64 * 12_000_000_000.0).ceil() as u64;
    high = high.max(BLOCK27_VERIFIED_MAX_INDEX).max(1);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if block27_blocks(middle) <= blocks.max(1) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// Compact startup table for the requested index.
///
/// One row per selected engine, then notes only for engines that are skipped or
/// forced. Resource totals appear once in the header.
pub fn render_preflight(
    fits: &[EngineFit],
    resources: &Resources,
    n: u64,
    force_index: bool,
    force_memory: bool,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "GPU preflight n={} | device {} free of {} | host {} available of {}",
        format_u64(n),
        resources
            .device_free_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .device_total_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .host_available_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .host_total_bytes
            .map_or_else(|| "unknown".to_string(), gib)
    );
    let _ = writeln!(
        out,
        "{:<18}  {:>26}  {:>26}  {:>17}  STATUS",
        "BACKEND", "DEVICE NEEDED / FREE", "HOST NEEDED / FREE", "MAX INDEX"
    );
    for fit in fits {
        let device = memory_pair_table(fit.required_device_bytes, fit.available_device_bytes);
        let host = memory_pair_table(fit.required_host_bytes, fit.available_host_bytes);
        let max = fit.engine.hard_max().map_or_else(
            || {
                format!(
                    "none (~{:.1}e9)",
                    block27_practical_max(resources) as f64 / 1e9
                )
            },
            format_u64,
        );
        let _ = writeln!(
            out,
            "{:<18}  {:>26}  {:>26}  {:>17}  {}",
            fit.engine.label(),
            device,
            host,
            max,
            preflight_status(fit, force_index, force_memory)
        );
    }
    out
}

/// Status cell for the startup preflight table. The check/cross matches the
/// SIMD feature report; the word only explains a non-plain result.
fn preflight_status(fit: &EngineFit, force_index: bool, force_memory: bool) -> &'static str {
    if !fit.index_allowed {
        if force_index {
            "✓ forced"
        } else {
            "✗ index"
        }
    } else if !fit.fits {
        if force_memory {
            "✓ forced"
        } else if !fit.fits_device {
            "✗ vram"
        } else {
            "✗ host"
        }
    } else if fit.over_verified {
        "✓ unverified"
    } else {
        "✓"
    }
}

/// Render the `--show-limits` tables and resource summary.
pub fn render_limits(resources: &Resources) -> String {
    let rows = limits_rows(resources);
    let mut out = String::new();
    let _ = writeln!(out, "GPU backend limits");
    let _ = writeln!(
        out,
        "resources: device {} free of {}; host {} available of {}",
        resources
            .device_free_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .device_total_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .host_available_bytes
            .map_or_else(|| "unknown".to_string(), gib),
        resources
            .host_total_bytes
            .map_or_else(|| "unknown".to_string(), gib)
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "at the verified maximum");
    let _ = writeln!(
        out,
        "{:<18} {:>16} {:>14} {:>14} {:>6}",
        "BACKEND", "VERIFIED MAX", "DEVICE", "HOST", "FITS"
    );
    for row in &rows {
        let _ = writeln!(
            out,
            "{:<18} {:>16} {:>14} {:>14} {:>6}",
            row.engine.label(),
            format_u64(row.verified_max),
            gib(row.verified_device_bytes),
            gib(row.verified_host_bytes),
            fit_mark(row.fits_verified)
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "at the hard/planner maximum");
    let _ = writeln!(
        out,
        "{:<18} {:>28} {:>14} {:>14} {:>6}",
        "BACKEND", "MAX INDEX", "DEVICE", "HOST", "FITS"
    );
    for row in &rows {
        let max = match (row.engine, row.hard_max, row.practical_max) {
            (GpuEngine::Block27, _, Some(practical)) => {
                format!("none (VRAM ~{})", format_u64(practical))
            }
            (_, Some(max), _) => format_u64(max),
            _ => "none".to_string(),
        };
        let device = row.hard_device_bytes.map_or_else(|| "-".to_string(), gib);
        let host = row.hard_host_bytes.map_or_else(|| "-".to_string(), gib);
        let _ = writeln!(
            out,
            "{:<18} {:>28} {:>14} {:>14} {:>6}",
            row.engine.label(),
            max,
            device,
            host,
            fit_mark(row.fits_hard)
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "memory: verified device values are measured peaks; every host value and all hard-maximum values are model estimates"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "limiter");
    for row in &rows {
        let short = match row.engine {
            GpuEngine::Resident => {
                "2^27 transform / 32-bit limb-count cap; 6,186,556,318 is rejected"
            }
            GpuEngine::Block27 => "no planner cap; VRAM-bound at about 86.6e9 on this machine",
            GpuEngine::Block28 => {
                "k<=15 x 2^27 limbs; 185e9 needs k=16 and fails the engine preflight"
            }
            GpuEngine::Block29 => "at most 3 x 2^28 limbs per operand",
        };
        let _ = writeln!(out, "  {:<18} {}", row.engine.label(), short);
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "measured speed (device / call / total)");
    let _ = writeln!(
        out,
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>14}",
        "BACKEND", "N", "DEVICE", "CALL", "TOTAL", "PEAK VRAM"
    );
    for (backend, n, device, call, total, peak) in [
        (
            "GPU/CUDA resident",
            "6.186e9",
            "1.712 s",
            "7.319 s",
            "-",
            "-",
        ),
        (
            "GPU/CUDA block-27",
            "61.865e9",
            "68.11 s",
            "90.58 s",
            "114.49 s",
            "-",
        ),
        (
            "GPU/CUDA block-28",
            "61.865e9",
            "66.16 s",
            "138.13 s",
            "144.78 s",
            "-",
        ),
        (
            "GPU/CUDA block-28",
            "100e9",
            "169.07 s",
            "272.27 s",
            "281.46 s",
            "16,673.6 MiB",
        ),
        (
            "GPU/CUDA block-29",
            "61.865e9",
            "72.63 s",
            "172.74 s",
            "177.12 s",
            "-",
        ),
    ] {
        let _ = writeln!(
            out,
            "{backend:<18} {n:>10} {device:>10} {call:>10} {total:>10} {peak:>14}"
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "measured limit points (not independently reproduced)");
    let _ = writeln!(
        out,
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>14}",
        "BACKEND", "N", "DEVICE", "CALL", "TOTAL", "PEAK VRAM"
    );
    for (backend, n, device, call, total, peak) in [
        (
            "GPU/CUDA block-27",
            "80e9",
            "112.18 s",
            "145.92 s",
            "195.49 s",
            "22,593.6 MiB",
        ),
        (
            "GPU/CUDA block-27",
            "86.6e9",
            "121.28 s",
            "156.76 s",
            "205.56 s",
            "23,361.6 MiB",
        ),
        (
            "GPU/CUDA block-28",
            "134.5e9",
            "210.46 s",
            "346.09 s",
            "361.33 s",
            "18,721.6 MiB",
        ),
        (
            "GPU/CUDA block-28",
            "150e9",
            "275.55 s",
            "417.05 s",
            "431.02 s",
            "20,769.6 MiB",
        ),
        (
            "GPU/CUDA block-28",
            "164.5e9",
            "298.74 s",
            "458.17 s",
            "474.42 s",
            "21,793.6 MiB",
        ),
        (
            "GPU/CUDA block-28",
            "175e9",
            "338.52 s",
            "494.38 s",
            "510.76 s",
            "22,817.6 MiB",
        ),
        (
            "GPU/CUDA block-28",
            "184e9",
            "335.92 s",
            "496.77 s",
            "515.03 s",
            "22,817.6 MiB",
        ),
        (
            "GPU/CUDA block-29",
            "73.628e9",
            "82.18 s",
            "268.37 s",
            "276.79 s",
            "22,817.6 MiB",
        ),
    ] {
        let _ = writeln!(
            out,
            "{backend:<18} {n:>10} {device:>10} {call:>10} {total:>10} {peak:>14}"
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "notes: block-28 n=100,000,000,000 is GMP-verified (SHA-256 {}); 184e9 is the measured maximum (hash only, not marked exact); 185e9 needs k=16 and was rejected by the engine preflight",
        BLOCK28_KNOWN_1E11_SHA256
    );
    let _ = writeln!(
        out,
        "notes: block-27 123.731e9 (2x target) needs k=20 and about 27.3 GiB device; it does not fit a 24 GB card and was not run"
    );
    out
}

/// JSON form of the limits table, used by `--show-limits --json`.
pub fn render_limits_json(resources: &Resources) -> String {
    let rows = limits_rows(resources);
    let rows = rows
        .iter()
        .map(|row| {
            format!(
                "{{\"backend\":\"{}\",\"label\":\"{}\",\"verified_max\":{},\"hard_max\":{},\"practical_max\":{},\"device_at_verified_bytes\":{},\"host_at_verified_bytes\":{},\"device_source\":\"{}\",\"host_source\":\"{}\",\"limiter\":\"{}\",\"fits_verified\":{},\"fits_hard\":{}}}",
                row.engine.backend_flag(),
                row.engine.label(),
                row.verified_max,
                row.hard_max
                    .map_or_else(|| "null".to_string(), |v| v.to_string()),
                row.practical_max
                    .map_or_else(|| "null".to_string(), |v| v.to_string()),
                row.verified_device_bytes,
                row.verified_host_bytes,
                row.verified_device_source,
                row.verified_host_source,
                row.engine.limiter().replace('"', "\\\""),
                row.fits_verified,
                row.fits_hard
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"resources\":{{\"device_total_bytes\":{},\"device_free_bytes\":{},\"host_total_bytes\":{},\"host_available_bytes\":{}}},\"rows\":[{}]}}",
        resources
            .device_total_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resources
            .device_free_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resources
            .host_total_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        resources
            .host_available_bytes
            .map_or_else(|| "null".to_string(), |v| v.to_string()),
        rows
    )
}

/// Detect current resources without allocating GPU memory.
pub fn detect_resources() -> Resources {
    let (device_total_bytes, device_free_bytes) = detect_device_memory();
    let (host_total_bytes, host_available_bytes) = detect_host_memory();
    Resources {
        device_total_bytes,
        device_free_bytes,
        host_total_bytes,
        host_available_bytes,
    }
}

/// Query the first GPU with `nvidia-smi`. No CUDA context is created.
pub fn detect_device_memory() -> (Option<u64>, Option<u64>) {
    let mut command = std::process::Command::new("nvidia-smi");
    command.args([
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let Ok(output) = command.output() else {
        return (None, None);
    };
    if !output.status.success() {
        return (None, None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 2 {
            continue;
        }
        let Ok(total_mib) = fields[0].parse::<u64>() else {
            continue;
        };
        let Ok(free_mib) = fields[1].parse::<u64>() else {
            continue;
        };
        return (Some(total_mib * MIB), Some(free_mib * MIB));
    }
    (None, None)
}

/// Available host RAM. Uses `GlobalMemoryStatusEx` on Windows.
pub fn detect_host_memory() -> (Option<u64>, Option<u64>) {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the structure is correctly sized and initialized, and the
        // Win32 call only writes into it.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 {
            return (Some(status.total_phys), Some(status.avail_phys));
        }
        (None, None)
    }
    #[cfg(not(windows))]
    {
        (None, None)
    }
}

/// Format bytes as GiB with two decimals.
pub fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// Format a byte count as MiB with one decimal.
pub fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / MIB as f64)
}

/// Format a u64 with thousands separators.
pub fn format_u64(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in digits.bytes().enumerate() {
        if index != 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(byte as char);
    }
    out
}

fn memory_pair(required: u64, available: Option<u64>) -> String {
    available.map_or_else(
        || format!("{} required / free unknown", gib(required)),
        |bytes| format!("{} / {} free", gib(required), gib(bytes)),
    )
}

/// Table cell variant: the header already says "free", so rows stay compact.
fn memory_pair_table(required: u64, available: Option<u64>) -> String {
    available.map_or_else(
        || format!("{} / unknown", gib(required)),
        |bytes| format!("{} / {}", gib(required), gib(bytes)),
    )
}

fn fit_mark(value: bool) -> &'static str {
    if value { "✓" } else { "✗" }
}

/// Suggested binary flags for the user's selected backend.
pub fn binary_override(engine: GpuEngine) -> Option<&'static str> {
    match engine {
        GpuEngine::Resident => Some("--gpu-resident-bin"),
        GpuEngine::Block27 => Some("--gpu-block27-bin"),
        GpuEngine::Block28 => Some("--gpu-block28-bin"),
        GpuEngine::Block29 => Some("--gpu-block29-bin"),
    }
}

/// Default repository-relative executable paths, for diagnostics.
pub fn default_binary(engine: GpuEngine) -> Option<PathBuf> {
    match engine {
        GpuEngine::Resident => Some(PathBuf::from(
            "cuda-ntt/resident-27/build/resident27_fib.exe",
        )),
        GpuEngine::Block27 => Some(PathBuf::from("cuda-ntt/block-27/build/block27_fib.exe")),
        GpuEngine::Block28 => Some(PathBuf::from("cuda-ntt/block-28/build/block28_fib.exe")),
        GpuEngine::Block29 => Some(PathBuf::from("cuda-ntt/block-29/build/block29_fib.exe")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> Resources {
        Resources::fixture(23.89, 23.15, 63.4, 49.0)
    }

    #[test]
    fn planner_maximum_matches_the_documented_value() {
        assert_eq!(block29_planner_max(), BLOCK29_MAX_INDEX);
        assert!(block29_plan_supported(&block29_plan(
            block29_operand_limb_bound(BLOCK29_MAX_INDEX),
            block29_operand_limb_bound(BLOCK29_MAX_INDEX)
        )));
        assert!(!block29_plan_supported(&block29_plan(
            block29_operand_limb_bound(BLOCK29_MAX_INDEX + 1),
            block29_operand_limb_bound(BLOCK29_MAX_INDEX + 1)
        )));
    }

    #[test]
    fn block28_planner_maximum_matches_the_documented_value() {
        assert_eq!(block28_planner_max(), BLOCK28_MAX_INDEX);
        assert_eq!(block28_blocks(BLOCK28_MAX_INDEX), 15);
        assert!(block28_plan_supported(&block28_plan(
            block29_operand_limb_bound(BLOCK28_MAX_INDEX),
            block29_operand_limb_bound(BLOCK28_MAX_INDEX)
        )));
        assert!(!block28_plan_supported(&block28_plan(
            block29_operand_limb_bound(BLOCK28_MAX_INDEX + 1),
            block29_operand_limb_bound(BLOCK28_MAX_INDEX + 1)
        )));
    }

    #[test]
    fn models_match_the_published_memory_table() {
        // block-29 at the target: 22.30 GiB device, 27.08 GiB host.
        assert_eq!(
            block29_device_bytes(BLOCK29_VERIFIED_MAX_INDEX),
            23_944_697_107
        );
        let host = block29_host_bytes(BLOCK29_VERIFIED_MAX_INDEX);
        assert!((gib(host) == "27.08 GiB"), "{host} -> {}", gib(host));
        // block-27 at the target: k=10, 19.81 GiB device.
        assert_eq!(block27_blocks(BLOCK27_VERIFIED_MAX_INDEX), 10);
        let device = block27_device_bytes(BLOCK27_VERIFIED_MAX_INDEX);
        assert_eq!(gib(device), "19.81 GiB");
        // Resident at the cap: 108 B * 2^27 + 1.30 GiB.
        assert_eq!(resident_elements(RESIDENT_MAX_INDEX), 1 << 27);
        let resident = resident_device_bytes(RESIDENT_MAX_INDEX);
        assert_eq!(gib(resident), "14.80 GiB");
        // block-28 at the 1e11 exactness anchor: 9 staging blocks, 16.31 GiB
        // device and 25.37 GiB host (25.33 GiB measured).
        assert_eq!(block28_blocks(BLOCK28_VERIFIED_MAX_INDEX), 9);
        assert_eq!(
            gib(block28_device_bytes(BLOCK28_VERIFIED_MAX_INDEX)),
            "16.31 GiB"
        );
        assert_eq!(
            gib(block28_host_bytes(BLOCK28_VERIFIED_MAX_INDEX)),
            "25.37 GiB"
        );
        // At the measured maximum (k=15) the device model is 22.31 GiB and the
        // host model is 42.49 GiB (42.30 GiB measured).
        assert_eq!(block28_blocks(BLOCK28_MEASURED_MAX_INDEX), 15);
        assert_eq!(
            gib(block28_device_bytes(BLOCK28_MEASURED_MAX_INDEX)),
            "22.31 GiB"
        );
        assert_eq!(
            gib(block28_host_bytes(BLOCK28_MEASURED_MAX_INDEX)),
            "42.49 GiB"
        );
        // k=16 for 185e9 would need 23.31 GiB and is rejected by the engine.
        assert_eq!(block28_blocks(185_000_000_000), 16);
        assert_eq!(gib(block28_device_bytes(185_000_000_000)), "23.31 GiB");
    }

    #[test]
    fn block27_device_limit_is_vram_bound() {
        // k=12 at the block-29 planner maximum, k=13 at ~80e9, k=14 at ~86.6e9.
        assert_eq!(block27_blocks(BLOCK29_MAX_INDEX), 12);
        assert_eq!(block27_blocks(80_000_000_000), 13);
        assert_eq!(block27_blocks(BLOCK27_MEASURED_MAX_INDEX), 14);
        assert!(gib(block27_device_bytes(80_000_000_000)).starts_with("22.06"));
        assert!(gib(block27_device_bytes(BLOCK27_MEASURED_MAX_INDEX)).starts_with("22.81"));
        // 2x target needs 20 blocks and cannot fit a 24 GB card.
        assert_eq!(block27_blocks(123_731_126_340), 20);
        assert!(block27_device_bytes(123_731_126_340) > 24 * GIB);
    }

    #[test]
    fn max_index_rejection() {
        let resident_over = check(
            GpuEngine::Resident,
            RESIDENT_MAX_INDEX + 1,
            &machine(),
            false,
        );
        assert!(!resident_over.index_allowed);
        assert!(!resident_over.fits);
        assert!(
            resident_over
                .skip_reason
                .as_deref()
                .unwrap()
                .contains("exceeds")
        );

        let block29_over = check(GpuEngine::Block29, BLOCK29_MAX_INDEX + 1, &machine(), false);
        assert!(!block29_over.index_allowed);
        assert!(!block29_over.fits);

        let block28_at = check(GpuEngine::Block28, BLOCK28_MAX_INDEX, &machine(), false);
        assert!(block28_at.index_allowed);
        assert!(block28_at.fits);
        let block28_over = check(GpuEngine::Block28, BLOCK28_MAX_INDEX + 1, &machine(), false);
        assert!(!block28_over.index_allowed);
        assert!(!block28_over.fits);
        assert!(
            block28_over.recommendation().contains("184,070,026,875"),
            "{}",
            block28_over.recommendation()
        );

        let force_index = check(
            GpuEngine::Resident,
            RESIDENT_MAX_INDEX + 1,
            &machine(),
            true,
        );
        assert!(!force_index.index_allowed);
        assert!(
            !force_index.fits,
            "force-memory must not bypass an index cap"
        );
    }

    #[test]
    fn fits_and_does_not_fit_are_reported() {
        let fit = check(GpuEngine::Block29, BLOCK29_MAX_INDEX, &machine(), false);
        assert!(fit.fits);
        assert!(fit.recommendation().contains("fits"));

        let tight = Resources::fixture(23.89, 18.20, 63.4, 49.0);
        let no_fit = check(GpuEngine::Block29, BLOCK29_MAX_INDEX, &tight, false);
        assert!(!no_fit.fits_device);
        assert!(!no_fit.fits);
        assert!(no_fit.recommendation().contains("18.20 GiB"));
        assert!(no_fit.recommendation().contains("--force-memory"));

        let forced = check(GpuEngine::Block29, BLOCK29_MAX_INDEX, &tight, true);
        assert!(forced.fits);
    }

    #[test]
    fn auto_prefers_resident_then_block27_then_block28_then_block29() {
        assert_eq!(
            choose_auto(1_000_000, &machine(), false).unwrap().engine,
            GpuEngine::Resident
        );
        assert_eq!(
            choose_auto(RESIDENT_MAX_INDEX, &machine(), false)
                .unwrap()
                .engine,
            GpuEngine::Resident
        );
        assert_eq!(
            choose_auto(BLOCK27_VERIFIED_MAX_INDEX, &machine(), false)
                .unwrap()
                .engine,
            GpuEngine::Block27
        );
        assert_eq!(
            choose_auto(70_000_000_000, &machine(), false)
                .unwrap()
                .engine,
            GpuEngine::Block27
        );
        // Above block-29's planner cap block-27 is still the first choice while
        // its VRAM preflight fits.
        let above = choose_auto(BLOCK29_MAX_INDEX + 1, &machine(), false).unwrap();
        assert_eq!(above.engine, GpuEngine::Block27);
        assert!(above.reason.contains("block-27"));
        // block-27 no longer fits at 1e11/184e9; block-28 does.
        assert_eq!(
            choose_auto(BLOCK28_VERIFIED_MAX_INDEX, &machine(), false)
                .unwrap()
                .engine,
            GpuEngine::Block28
        );
        assert_eq!(
            choose_auto(BLOCK28_MEASURED_MAX_INDEX, &machine(), false)
                .unwrap()
                .engine,
            GpuEngine::Block28
        );
        // Above the block-28 planner boundary no engine is eligible.
        assert!(choose_auto(BLOCK28_MAX_INDEX + 1, &machine(), false).is_none());
        assert!(choose_auto(184_100_000_000, &machine(), false).is_none());
        assert!(choose_auto(185_000_000_000, &machine(), false).is_none());
    }

    #[test]
    fn auto_candidates_put_block28_before_block29() {
        // Block-29 is a fallback only: block-27 uses less VRAM at every index
        // block-29 supports, and block-28 covers the larger indices.
        assert_eq!(
            auto_candidates(73_000_000_000),
            vec![GpuEngine::Block27, GpuEngine::Block28, GpuEngine::Block29]
        );
        assert_eq!(
            auto_candidates(BLOCK29_MAX_INDEX + 1),
            vec![GpuEngine::Block27, GpuEngine::Block28]
        );
        assert_eq!(
            auto_candidates(BLOCK28_MAX_INDEX + 1),
            Vec::<GpuEngine>::new()
        );
        assert_eq!(
            auto_candidates(1_000_000),
            vec![
                GpuEngine::Resident,
                GpuEngine::Block27,
                GpuEngine::Block28,
                GpuEngine::Block29
            ]
        );
    }

    #[test]
    fn practical_block27_max_is_around_eighty_billion() {
        let max = block27_practical_max(&machine());
        assert!(
            (78_000_000_000..=82_000_000_000).contains(&max),
            "practical max {max}"
        );
        assert!(block27_blocks(max) <= 13);
    }

    #[test]
    fn limits_table_contains_all_four_engines_and_measurements() {
        let text = render_limits(&machine());
        for engine in GpuEngine::ALL {
            assert!(text.contains(engine.label()), "{text}");
        }
        assert!(text.contains("73,628,010,691"));
        assert!(text.contains("6,186,556,317"));
        assert!(text.contains("184,070,026,875"));
        assert!(text.contains("184e9"));
        assert!(text.contains("9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0"));
        assert!(text.contains("27.08 GiB"));
        assert!(text.contains("19.81 GiB"));
        assert!(text.contains("22.28 GiB"));
        assert!(text.contains("16.28 GiB"));
        assert!(text.contains("25.37 GiB"));
        assert!(text.contains("42.50 GiB"));
        assert!(text.contains("measured speed"));
        assert!(text.contains("86.6e9"));
        assert!(text.contains("2x target"));
        let json = render_limits_json(&machine());
        assert!(json.starts_with("{\"resources\":"));
        assert!(json.contains("\"backend\":\"gpu-block27\""));
        assert!(json.contains("\"backend\":\"gpu-block28\""));
        assert!(json.contains("\"practical_max\":"));
        let rows = limits_rows(&machine());
        let block27 = rows
            .iter()
            .find(|row| row.engine == GpuEngine::Block27)
            .unwrap();
        assert_eq!(block27.hard_max, None);
        assert!(block27.practical_max.is_some());
        let block28 = rows
            .iter()
            .find(|row| row.engine == GpuEngine::Block28)
            .unwrap();
        assert_eq!(block28.hard_max, Some(BLOCK28_MAX_INDEX));
        assert_eq!(block28.verified_max, BLOCK28_VERIFIED_MAX_INDEX);
        assert!(block28.fits_verified);
        assert!(block28.fits_hard);
    }

    #[test]
    fn demo_resource_detection_does_not_panic() {
        // The CI machine may not have nvidia-smi; the function must degrade to
        // `None` rather than fail.
        let resources = detect_resources();
        let _ = format!("{resources:?}");
        let _ = render_limits(&resources);
    }
}
