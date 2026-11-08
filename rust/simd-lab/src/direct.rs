//! Symbolic Binet evaluation in `Z[phi]`.
//!
//! `phi = (1 + sqrt(5)) / 2` satisfies `phi^2 = phi + 1`, and therefore
//!
//! ```text
//! phi^n = F(n) * phi + F(n - 1)
//! ```
//!
//! so the `phi` coefficient of `phi^n` is exactly `F(n)`. An element
//! `x + y * phi` is stored as `Pair { a: x, b: y }`; binary exponentiation of
//! `phi` using the golden-ratio relation yields `F(n)` with exact integer
//! operations only.

use crate::uint::SimdUint;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Pair {
    a: SimdUint,
    b: SimdUint,
}

impl Pair {
    /// Multiplicative identity `1 = 1 + 0 * phi`.
    fn one() -> Self {
        Self {
            a: SimdUint::one(),
            b: SimdUint::zero(),
        }
    }

    /// The golden ratio `phi = 0 + 1 * phi`.
    fn phi() -> Self {
        Self {
            a: SimdUint::zero(),
            b: SimdUint::one(),
        }
    }
}

/// Multiplies two elements of `Z[phi]` exactly.
///
/// Using `p = a1 * a2`, `q = b1 * b2` and
/// `r = (a1 + b1) * (a2 + b2)`:
///
/// ```text
/// (a1 + b1 * phi) * (a2 + b2 * phi)
///   = (a1 * a2 + b1 * b2) + (a1 * b2 + a2 * b1 + b1 * b2) * phi
/// ```
///
/// with `r = p + a1 * b2 + a2 * b1 + q`, hence the constant part is `p + q`
/// and the `phi` coefficient is `r - p`.
fn pair_mul(left: &Pair, right: &Pair) -> Pair {
    // Optional fast path: a square needs only one sum `a + b` instead of two.
    if std::ptr::eq(left, right) {
        return pair_square(left);
    }

    let (p, (q, r)) = if rayon::current_num_threads() > 1 {
        rayon::join(
            || left.a.mul_par(&right.a),
            || {
                rayon::join(
                    || left.b.mul_par(&right.b),
                    || left.a.add(&left.b).mul_par(&right.a.add(&right.b)),
                )
            },
        )
    } else {
        (
            left.a.mul(&right.a),
            (
                left.b.mul(&right.b),
                left.a.add(&left.b).mul(&right.a.add(&right.b)),
            ),
        )
    };

    let a = p.add(&q);
    // `r >= p` because `r = p + (a1 * b2 + a2 * b1) + q`.
    let b = r.sub(&p);
    Pair { a, b }
}

/// Squaring variant of [`pair_mul`], used when both factors alias.
fn pair_square(value: &Pair) -> Pair {
    let (p, (q, r)) = if rayon::current_num_threads() > 1 {
        rayon::join(
            || value.a.mul_par(&value.a),
            || {
                rayon::join(
                    || value.b.mul_par(&value.b),
                    || {
                        let sum = value.a.add(&value.b);
                        sum.mul_par(&sum)
                    },
                )
            },
        )
    } else {
        let sum = value.a.add(&value.b);
        (
            value.a.mul(&value.a),
            (value.b.mul(&value.b), sum.mul(&sum)),
        )
    };

    let a = p.add(&q);
    // For `value = x + y * phi`, `r = p + 2 * x * y + q`, so `r >= p`.
    let b = r.sub(&p);
    Pair { a, b }
}

/// Returns the exact Fibonacci number `F(n)` with `F(0) = 0` and `F(1) = 1`.
///
/// The exponentiation scans the bits of `n` from the least significant to the
/// most significant one, keeping `result = phi^k` for the bits consumed so far
/// and `base = phi^(2^i)`. No floating-point arithmetic is involved.
pub fn direct(n: u64) -> SimdUint {
    let mut result = Pair::one();
    let mut base = Pair::phi();
    let mut remaining = n;

    while remaining != 0 {
        if remaining & 1 == 1 {
            let mut product = pair_mul(&result, &base);
            std::mem::swap(&mut result, &mut product);
        }

        remaining >>= 1;
        if remaining != 0 {
            let mut squared = pair_mul(&base, &base);
            std::mem::swap(&mut base, &mut squared);
        }
    }

    result.b
}

#[cfg(test)]
mod tests {
    use super::{Pair, direct, pair_mul};
    use crate::uint::SimdUint;
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
            direct(n).to_bytes_be(),
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
    fn pair_arithmetic_uses_golden_ratio_relation() {
        // phi^2 = phi + 1.
        let phi = Pair::phi();
        let square = pair_mul(&phi, &phi);
        assert_eq!(
            square,
            Pair {
                a: SimdUint::one(),
                b: SimdUint::one()
            }
        );

        // The square fast path must agree with the general product.
        let other = Pair {
            a: SimdUint::from_u64(3),
            b: SimdUint::from_u64(2),
        };
        assert_eq!(pair_mul(&other, &other), pair_mul(&other, &other.clone()));

        // (3 + 2 * phi) * (1 + 4 * phi) = 11 + 22 * phi for phi^2 = phi + 1.
        let left = Pair {
            a: SimdUint::from_u64(3),
            b: SimdUint::from_u64(2),
        };
        let right = Pair {
            a: SimdUint::one(),
            b: SimdUint::from_u64(4),
        };
        let product = pair_mul(&left, &right);
        assert_eq!(product.a, SimdUint::from_u64(11));
        assert_eq!(product.b, SimdUint::from_u64(22));
    }

    #[test]
    fn matches_sequential_reference_up_to_2000() {
        // Sequential BigUint reference: advance (F(n), F(n + 1)) one step at a time.
        let mut current = BigUint::from(0u32);
        let mut next = BigUint::from(1u32);

        for n in 0..=2000u64 {
            assert_eq!(
                direct(n).to_bytes_be(),
                canonical_bytes(&current),
                "F({n}) mismatch against sequential reference"
            );

            let following = &current + &next;
            current = std::mem::replace(&mut next, following);
        }
    }

    #[test]
    fn known_fibonacci_numbers() {
        let expected = BigUint::parse_bytes(b"354224848179261915075", 10).unwrap();
        assert_eq!(direct(100).to_bytes_be(), expected.to_bytes_be());
        assert_eq!(direct(100).to_bytes_be(), fib_biguint(100).to_bytes_be());
    }

    #[test]
    fn matches_biguint_doubling_for_larger_n() {
        for n in [10_000u64, 100_000] {
            assert_fib(n);
        }
    }

    #[test]
    fn matches_biguint_doubling_for_one_million() {
        // Kept separate because it is slow, especially in an unoptimized build.
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
