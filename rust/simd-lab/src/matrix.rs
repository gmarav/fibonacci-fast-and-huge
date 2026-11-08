//! Fibonacci numbers by binary exponentiation of the 2x2 matrix
//! `M = [[1, 1], [1, 0]]`.
//!
//! With `F(0) = 0` and `F(1) = 1`, the top-right element of `M^n` is `F(n)`.
//! Matrices are stored in row-major order as `[m00, m01, m10, m11]`.

use crate::uint::SimdUint;

/// Multiply two 2x2 matrices over `SimdUint`.
///
/// All eight products are computed independently and only then added in
/// pairs, matching the main Rust implementation. Squaring uses the same code
/// path as any other product so benchmarks compare identical algorithms.
fn mat_mul(left: &[SimdUint; 4], right: &[SimdUint; 4]) -> [SimdUint; 4] {
    let (p00, p01, p02, p03, p04, p05, p06, p07) = if rayon::current_num_threads() > 1 {
        let ((p00, p01), (p02, p03)) = rayon::join(
            || rayon::join(|| left[0].mul_par(&right[0]), || left[1].mul_par(&right[2])),
            || rayon::join(|| left[0].mul_par(&right[1]), || left[1].mul_par(&right[3])),
        );
        let ((p04, p05), (p06, p07)) = rayon::join(
            || rayon::join(|| left[2].mul_par(&right[0]), || left[3].mul_par(&right[2])),
            || rayon::join(|| left[2].mul_par(&right[1]), || left[3].mul_par(&right[3])),
        );
        (p00, p01, p02, p03, p04, p05, p06, p07)
    } else {
        (
            left[0].mul(&right[0]),
            left[1].mul(&right[2]),
            left[0].mul(&right[1]),
            left[1].mul(&right[3]),
            left[2].mul(&right[0]),
            left[3].mul(&right[2]),
            left[2].mul(&right[1]),
            left[3].mul(&right[3]),
        )
    };

    [p00.add(&p01), p02.add(&p03), p04.add(&p05), p06.add(&p07)]
}

/// Return the exact Fibonacci number `F(n)` for `n` using matrix
/// exponentiation. The index is scanned least-significant bit first.
pub fn matrix(mut n: u64) -> SimdUint {
    let mut result: [SimdUint; 4] = [
        SimdUint::one(),
        SimdUint::zero(),
        SimdUint::zero(),
        SimdUint::one(),
    ];
    let mut base: [SimdUint; 4] = [
        SimdUint::one(),
        SimdUint::one(),
        SimdUint::one(),
        SimdUint::zero(),
    ];

    while n != 0 {
        if n & 1 != 0 {
            let mut next = mat_mul(&result, &base);
            std::mem::swap(&mut result, &mut next);
        }
        n >>= 1;
        if n != 0 {
            let mut next = mat_mul(&base, &base);
            std::mem::swap(&mut base, &mut next);
        }
    }

    // `result` is `M^n` and its top-right element is `F(n)`.
    let [_, top_right, _, _] = result;
    top_right
}

#[cfg(test)]
mod tests {
    use super::matrix;
    use num_bigint::BigUint;

    /// Sequential `F(n)` using the BigUint oracle.
    fn big_sequential(n: u64) -> BigUint {
        let mut a = BigUint::from(0u32);
        let mut b = BigUint::from(1u32);
        for _ in 0..n {
            let next = &a + &b;
            a = std::mem::replace(&mut b, next);
        }
        a
    }

    /// Fast doubling with the BigUint oracle, used for indices that are too
    /// large for the sequential loop.
    fn big_doubling(n: u64) -> BigUint {
        let mut a = BigUint::from(0u32);
        let mut b = BigUint::from(1u32);
        if n == 0 {
            return a;
        }
        let bits = 64 - n.leading_zeros();
        for bit in (0..bits).rev() {
            let c = &a * ((&b << 1u32) - &a); // F(2k)
            let d = &a * &a + &b * &b; // F(2k + 1)
            if (n >> bit) & 1 == 1 {
                let next_b = &c + &d;
                a = d;
                b = next_b;
            } else {
                a = c;
                b = d;
            }
        }
        a
    }

    fn assert_matches_big(n: u64) {
        let expected = big_doubling(n);
        let actual = matrix(n);
        assert_eq!(
            actual.to_decimal(),
            expected.to_string(),
            "matrix F({n}) differs from the BigUint oracle"
        );
    }

    #[test]
    fn matches_sequential_reference_up_to_2000() {
        let mut a = BigUint::from(0u32);
        let mut b = BigUint::from(1u32);
        for n in 0..=2000u64 {
            let actual = matrix(n);
            assert_eq!(
                actual.to_decimal(),
                a.to_string(),
                "matrix F({n}) differs from sequential BigUint"
            );
            let next = &a + &b;
            a = std::mem::replace(&mut b, next);
        }
    }

    #[test]
    fn known_f100() {
        let actual = matrix(100).to_decimal();
        assert_eq!(actual, "354224848179261915075");
    }

    #[test]
    fn matches_big_doubling_for_large_indices() {
        for n in [10_000u64, 100_000, 1_000_000] {
            assert_matches_big(n);
        }
    }

    #[test]
    fn handles_small_indices_and_power_of_two_boundaries() {
        for n in [0u64, 1, 2, 3] {
            assert_matches_big(n);
        }
        for exponent in 1..=20u32 {
            let power = 1u64 << exponent;
            assert_matches_big(power - 1);
            assert_matches_big(power + 1);
        }
    }

    #[test]
    fn sequential_helper_agrees_with_doubling_helper() {
        // Keep the two independent oracles in sync on a small deterministic
        // range so the large-index tests rest on a checked reference.
        for n in 0..=200u64 {
            assert_eq!(big_sequential(n), big_doubling(n), "oracles differ at {n}");
        }
    }
}
