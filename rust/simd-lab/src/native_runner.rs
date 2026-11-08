//! Adapter for the original `fib-rust` CPU implementations.
//!
//! `fib-rust` is a sibling crate and the reference implementation for this
//! experiment: direct algebraic evaluation, binary matrix exponentiation, and
//! fast doubling, instantiated over both `num_bigint::BigUint` and
//! `rug::Integer` (GMP). It owns a private Rayon pool, so the CLI must call it
//! from the main thread, outside the SIMD crate's global pool. That keeps the
//! two pools from nesting.
//!
//! The defaults mirror the reference CLI: reusable workspaces (`reuse = true`)
//! and parallel Karatsuba splitting for large products
//! (`split_multiplication = true`).

use crate::uint::SimdUint;
use fib_rust::backend::Natural;
use fib_rust::{Calculator, Method};
use num_bigint::BigUint;
use rug::Integer;
use std::marker::PhantomData;
use std::time::Instant;

/// Reuse algorithm workspaces between bit steps, as in `fib-rust`.
pub const DEFAULT_REUSE: bool = true;
/// Enable the reference implementation's parallel Karatsuba splitting.
pub const DEFAULT_SPLIT_MULTIPLICATION: bool = true;

/// One timed `fib-rust` calculation.
#[derive(Clone, Debug)]
pub struct NativeMeasurement<N: Natural> {
    /// Algorithm that produced the value.
    pub method: Method,
    /// Exact result in the backend's integer type.
    pub value: N,
    /// Bit length of `value`.
    pub bits: u64,
    /// Wall-clock samples in nanoseconds, one per iteration.
    pub samples: Vec<u128>,
    /// Median sample in nanoseconds.
    pub median_ns: u128,
}

/// Private-pool wrapper around [`fib_rust::Calculator`] for one integer backend.
pub struct NativeRunner<N: Natural> {
    calculator: Calculator,
    reuse: bool,
    split_multiplication: bool,
    _value: PhantomData<N>,
}

impl<N: Natural> NativeRunner<N> {
    /// Create a runner with the reference defaults.
    ///
    /// `workers` follows the reference CLI convention: `0` selects all
    /// available CPUs, and the value is capped by the available parallelism.
    /// With one worker the reference implementation runs sequentially and does
    /// not build a pool.
    pub fn new(workers: usize) -> Result<Self, String> {
        let calculator = Calculator::new(workers, DEFAULT_SPLIT_MULTIPLICATION)?;
        Ok(Self {
            calculator,
            reuse: DEFAULT_REUSE,
            split_multiplication: DEFAULT_SPLIT_MULTIPLICATION,
            _value: PhantomData,
        })
    }

    /// Integer backend name reported by `fib-rust`.
    pub fn backend_name(&self) -> &'static str {
        N::NAME
    }

    /// Effective worker count of the private pool.
    pub fn workers(&self) -> usize {
        self.calculator.workers()
    }

    /// Whether workspaces are reused between bit steps (always `true` here).
    pub fn reuse(&self) -> bool {
        self.reuse
    }

    /// Whether large products use split multiplication (always `true` here).
    pub fn split_multiplication(&self) -> bool {
        self.split_multiplication
    }

    /// Run one algorithm `iterations` times and keep the last value.
    pub fn measure(&self, method: Method, n: u64, iterations: usize) -> NativeMeasurement<N> {
        let iterations = iterations.max(1);
        let mut samples = Vec::with_capacity(iterations);
        let mut value = N::default();
        for _ in 0..iterations {
            // Release the previous result before starting the next timed sample.
            drop(std::mem::take(&mut value));
            let start = Instant::now();
            value = self.calculator.calculate::<N>(method, n, self.reuse);
            samples.push(start.elapsed().as_nanos());
        }
        let bits = value.bits();
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let median_ns = sorted[sorted.len() / 2];
        NativeMeasurement {
            method,
            value,
            bits,
            samples,
            median_ns,
        }
    }
}

/// The original pure-Rust `num-bigint` backend.
pub type NumBigintRunner = NativeRunner<BigUint>;
/// The GMP-backed `rug::Integer` backend.
pub type GmpRunner = NativeRunner<Integer>;

/// Parse one of the reference method names (`direct`, `matrix`, `doubling`).
pub fn parse_method(name: &str) -> Option<Method> {
    name.parse::<Method>().ok()
}

/// Convert a reference value to the SIMD integer type through canonical
/// big-endian bytes. Used for GPU reference values and cross-family equality.
pub fn to_simd<N: Natural>(value: &N) -> SimdUint {
    SimdUint::from_bytes_be(&value.bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_reference_cli() {
        let num_bigint = NumBigintRunner::new(1).expect("num-bigint runner");
        assert!(num_bigint.reuse());
        assert!(num_bigint.split_multiplication());
        assert_eq!(num_bigint.workers(), 1);
        assert_eq!(num_bigint.backend_name(), "num-bigint");

        let gmp = GmpRunner::new(1).expect("gmp runner");
        assert!(gmp.reuse());
        assert!(gmp.split_multiplication());
        assert_eq!(gmp.workers(), 1);
        assert_eq!(gmp.backend_name(), "gmp");
    }

    #[test]
    fn num_bigint_methods_match_a_known_value_and_each_other() {
        let runner = NumBigintRunner::new(1).expect("runner");

        let direct = runner.measure(Method::Direct, 100, 1);
        assert_eq!(direct.value.to_str_radix(10), "354224848179261915075");

        // Compare all three algorithms against the independent SIMD doubling
        // implementation at n = 1000.
        let reference = crate::doubling::doubling(1000);
        let reference_bits = reference.bits();
        let reference = reference.to_bytes_be();
        for method in Method::ALL {
            let measurement = runner.measure(method, 1000, 1);
            assert_eq!(measurement.bits, reference_bits);
            assert_eq!(
                to_simd(&measurement.value).to_bytes_be(),
                reference,
                "method {}",
                method.name()
            );
        }
    }

    #[test]
    fn gmp_matches_num_bigint_at_100_and_1000() {
        let num_bigint = NumBigintRunner::new(1).expect("num-bigint runner");
        let gmp = GmpRunner::new(1).expect("gmp runner");
        for n in [100, 1000] {
            for method in Method::ALL {
                let expected = num_bigint.measure(method, n, 1);
                let actual = gmp.measure(method, n, 1);
                assert_eq!(actual.bits, expected.bits, "n={n} method={}", method.name());
                assert_eq!(
                    to_simd(&actual.value).to_bytes_be(),
                    to_simd(&expected.value).to_bytes_be(),
                    "n={n} method={}",
                    method.name()
                );
            }
        }
    }

    #[test]
    fn gmp_matches_num_bigint_at_100000() {
        let num_bigint = NumBigintRunner::new(1).expect("num-bigint runner");
        let gmp = GmpRunner::new(1).expect("gmp runner");
        for method in Method::ALL {
            let expected = num_bigint.measure(method, 100_000, 1);
            let actual = gmp.measure(method, 100_000, 1);
            assert_eq!(actual.bits, expected.bits);
            assert_eq!(
                to_simd(&actual.value).to_bytes_be(),
                to_simd(&expected.value).to_bytes_be(),
                "method={}",
                method.name()
            );
        }
    }

    #[test]
    fn method_names_parse() {
        for name in ["direct", "matrix", "doubling"] {
            assert_eq!(parse_method(name).map(Method::name), Some(name));
        }
        assert_eq!(parse_method("compare"), None);
    }
}
