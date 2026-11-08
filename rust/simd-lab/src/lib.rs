// The release uses a newer clippy than the code's original lint baseline.
// These style lints are intentionally left to a focused refactor.
#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::useless_format
)]

//! Exact Fibonacci algorithms on a project-local SIMD-backed big unsigned integer.
//!
//! The algorithms are written in separate modules so they can be reviewed and
//! benchmarked independently:
//! - `uint`: exact limb arithmetic and runtime SIMD dispatch;
//! - `direct`: symbolic Binet evaluation in Z[phi];
//! - `matrix`: 2x2 binary matrix exponentiation;
//! - `doubling`: fast doubling;
//! - `cpu` and `gpu`: hardware capability reporting for the CLI;
//! - `banner`: the startup banner rendered before the hardware report;
//! - `logo`: the animated Fibonacci spiral and background word;
//! - `chart`: the Braille measurement chart printed after a run;
//! - `native_runner`: adapters for the original `fib-rust` num-bigint and GMP
//!   backends;
//! - `progress`: the live single-index progress line;
//! - `report`: the shared results table and timing formatting.

pub mod banner;
pub mod chart;
pub mod cpu;
pub mod direct;
pub mod doubling;
pub mod gpu;
pub mod gpu_block;
pub mod limits;
pub mod logo;
pub mod matrix;
pub mod native_runner;
pub mod progress;
pub mod report;
pub mod sha256;
pub mod uint;
