//! Fast doubling for Fibonacci numbers on `SimdUint`.
//!
//! Uses the identities
//!
//! ```text
//! F(2k)   = F(k) * (2 * F(k + 1) - F(k))
//! F(2k + 1) = F(k)^2 + F(k + 1)^2
//! ```
//!
//! Starting from `(a, b) = (F(0), F(1)) = (0, 1)`, one step per bit of `n`
//! from the most significant to the least significant bit produces
//! `(F(k), F(k + 1))` for the prefix `k` read so far.

use crate::uint::SimdUint;

/// Returns the exact Fibonacci number `F(n)` with `F(0) = 0` and `F(1) = 1`.
///
/// The result is computed only with exact integer operations from
/// [`SimdUint`]; no floating-point arithmetic is involved.
pub fn doubling(n: u64) -> SimdUint {
    // Invariant: `a = F(k)` and `b = F(k + 1)` for the prefix `k` processed so far.
    let mut a = SimdUint::zero();
    let mut b = SimdUint::one();

    // For n = 0 the empty range leaves (a, b) = (0, 1), so F(0) = 0 is returned.
    for bit in (0..(64 - n.leading_zeros())).rev() {
        // t = 2 * F(k + 1) - F(k), which is non-negative because
        // 2 * F(k + 1) >= F(k) for every k.
        let t = b.add(&b).sub(&a);

        // Candidates for F(2k) and F(2k + 1).
        let (mut c, (aa, bb)) = if rayon::current_num_threads() > 1 {
            rayon::join(
                || a.mul_par(&t),
                || rayon::join(|| a.mul_par(&a), || b.mul_par(&b)),
            )
        } else {
            (a.mul(&t), (a.mul(&a), b.mul(&b)))
        };
        let mut d = aa.add(&bb);

        if (n >> bit) & 1 == 0 {
            // Prefix doubles: (F(2k), F(2k + 1)) = (c, d).
            std::mem::swap(&mut a, &mut c);
            std::mem::swap(&mut b, &mut d);
        } else {
            // Prefix doubles and increments: (F(2k + 1), F(2k + 2)) = (d, c + d).
            // Compute c + d before moving d into `a`.
            let mut next_b = c.add(&d);
            std::mem::swap(&mut a, &mut d);
            std::mem::swap(&mut b, &mut next_b);
        }
    }

    a
}

#[cfg(test)]
mod tests {
    use super::doubling;
    use num_bigint::BigUint;

    /// Independent oracle: fast doubling on `num_bigint::BigUint`.
    fn fib_biguint(n: u64) -> BigUint {
        let mut a = BigUint::from(0u32);
        let mut b = BigUint::from(1u32);

        for bit in (0..(64 - n.leading_zeros())).rev() {
            let t = &b + &b - &a;
            let c = &a * &t;
            let d = &a * &a + &b * &b;

            if (n >> bit) & 1 == 0 {
                a = c;
                b = d;
            } else {
                let next_b = &c + &d;
                a = d;
                b = next_b;
            }
        }

        a
    }

    fn assert_fib(n: u64) {
        let expected = fib_biguint(n);
        assert_eq!(
            doubling(n).to_bytes_be(),
            canonical_bytes(&expected),
            "F({n}) mismatch"
        );
    }

    /// `num-bigint` encodes zero as `[0]`; `SimdUint` uses an empty slice.
    fn canonical_bytes(value: &BigUint) -> Vec<u8> {
        if value == &BigUint::from(0u32) {
            Vec::new()
        } else {
            value.to_bytes_be()
        }
    }

    #[test]
    fn matches_sequential_reference_up_to_2000() {
        // Sequential BigUint reference: advance (F(n), F(n + 1)) one step at a time.
        let mut current = BigUint::from(0u32);
        let mut next = BigUint::from(1u32);

        for n in 0..=2000u64 {
            assert_eq!(
                doubling(n).to_bytes_be(),
                canonical_bytes(&current),
                "F({n}) mismatch against sequential reference"
            );

            let following = &current + &next;
            current = std::mem::replace(&mut next, following);
        }
    }

    #[test]
    fn known_fibonacci_numbers() {
        assert_eq!(doubling(100).to_decimal(), "354224848179261915075");
    }

    #[test]
    fn matches_biguint_doubling_for_larger_n() {
        for n in [10_000u64, 100_000] {
            assert_fib(n);
        }
    }

    #[test]
    fn matches_biguint_doubling_for_one_million() {
        // This test is intentionally separate because it is slow, especially
        // in an unoptimized build.
        assert_fib(1_000_000);
    }

    #[test]
    fn boundary_values_and_powers_of_two() {
        for n in [0u64, 1, 2, 3] {
            assert_fib(n);
        }

        // Around powers of two: 2^k - 1, 2^k, 2^k + 1.
        for exponent in 0..=20u32 {
            let power = 1u64 << exponent;
            assert_fib(power - 1);
            assert_fib(power);
            assert_fib(power + 1);
        }
    }
}
