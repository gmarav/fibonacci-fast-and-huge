//! Exact unsigned big integers on little-endian `u32` limbs.
//!
//! The value is stored as a `Vec<u32>` in little-endian order with no trailing
//! zero limbs. Zero is the empty vector. Arithmetic is exact: `add`, `sub`, and
//! `mul` never use floating point.
//!
//! Multiplication uses a runtime-selected schoolbook kernel (scalar, SSE2,
//! AVX2, or AVX-512F) and switches to recursive Karatsuba once both operands
//! are large enough; substantially unbalanced products stay on the schoolbook
//! path.

use std::cmp::Ordering;
use std::sync::OnceLock;

use num_bigint::BigUint;

/// Shorter-operand length, in limbs, at which `mul` switches from the selected
/// schoolbook kernel to recursive Karatsuba. 64 limbs = 2048 bits.
const KARATSUBA_THRESHOLD: usize = 64;

/// Operands whose lengths differ by at least this factor are treated as
/// substantially unbalanced. Karatsuba's split would leave the shorter high
/// half empty, so those products use the schoolbook kernel directly.
const KARATSUBA_MAX_RATIO: usize = 2;

/// Exact unsigned integer over little-endian 32-bit limbs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SimdUint {
    pub(crate) limbs: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Runtime kernel selection
// ---------------------------------------------------------------------------

#[allow(dead_code)] // Some variants are unavailable on non-x86-64 targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kernel {
    Scalar32,
    Sse2,
    Avx2,
    Avx512f,
}

impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Scalar32 => "scalar32",
            Kernel::Sse2 => "sse2",
            Kernel::Avx2 => "avx2",
            Kernel::Avx512f => "avx512f",
        }
    }
}

fn detect_kernel() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            Kernel::Avx512f
        } else if is_x86_feature_detected!("avx2") {
            Kernel::Avx2
        } else if is_x86_feature_detected!("sse2") {
            Kernel::Sse2
        } else {
            Kernel::Scalar32
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Kernel::Scalar32
    }
}

fn selected_kernel() -> Kernel {
    static SELECTED: OnceLock<Kernel> = OnceLock::new();
    *SELECTED.get_or_init(detect_kernel)
}

/// Name of the schoolbook kernel selected for this process.
///
/// One of `scalar32`, `sse2`, `avx2`, or `avx512f`.
pub fn selected_kernel_name() -> &'static str {
    selected_kernel().name()
}

// ---------------------------------------------------------------------------
// Public arithmetic API
// ---------------------------------------------------------------------------

// The API intentionally mirrors arithmetic operator names (`add`, `sub`,
// `mul`) rather than implementing `std::ops` traits.
#[allow(clippy::should_implement_trait)]
impl SimdUint {
    /// Zero, represented by an empty limb vector.
    pub fn zero() -> Self {
        Self::default()
    }

    /// One.
    pub fn one() -> Self {
        Self::from_u64(1)
    }

    /// Build from a machine word.
    pub fn from_u64(value: u64) -> Self {
        let mut limbs = Vec::with_capacity(2);
        if value != 0 {
            limbs.push(value as u32);
            let high = (value >> 32) as u32;
            if high != 0 {
                limbs.push(high);
            }
        }
        Self { limbs }
    }

    /// Whether this value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Number of significant bits; zero has zero bits.
    pub fn bits(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() as u64 - 1) * 32 + (32 - top.leading_zeros() as u64),
        }
    }

    /// Exact addition.
    pub fn add(&self, rhs: &Self) -> Self {
        let mut result = self.clone();
        result.add_assign(rhs);
        result
    }

    /// Exact subtraction. The caller must guarantee `self >= rhs`.
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut result = self.clone();
        result.sub_assign(rhs);
        result
    }

    /// Exact multiplication.
    pub fn mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        Self {
            limbs: mul_limbs(&self.limbs, &rhs.limbs),
        }
    }

    /// Exact multiplication with bounded Rayon parallelism.
    ///
    /// Uses the current Rayon pool size to bound the Karatsuba recursion depth.
    /// With one worker it delegates to the sequential [`Self::mul`] path.
    pub fn mul_par(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let depth = parallel_depth();
        if depth == 0 {
            return self.mul(rhs);
        }
        Self {
            limbs: mul_limbs_par(&self.limbs, &rhs.limbs, depth),
        }
    }

    /// In-place exact addition.
    pub fn add_assign(&mut self, rhs: &Self) {
        if rhs.is_zero() {
            return;
        }
        if self.limbs.len() < rhs.limbs.len() {
            self.limbs.resize(rhs.limbs.len(), 0);
        }
        let mut carry = 0u64;
        for (lhs_limb, &rhs_limb) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            let sum = *lhs_limb as u64 + rhs_limb as u64 + carry;
            *lhs_limb = sum as u32;
            carry = sum >> 32;
        }
        let mut index = rhs.limbs.len();
        while carry != 0 {
            if index == self.limbs.len() {
                self.limbs.push(carry as u32);
                break;
            }
            let sum = self.limbs[index] as u64 + carry;
            self.limbs[index] = sum as u32;
            carry = sum >> 32;
            index += 1;
        }
        debug_assert_ne!(self.limbs.last(), Some(&0), "non-canonical sum");
    }

    /// In-place exact subtraction. The caller must guarantee `self >= rhs`.
    pub fn sub_assign(&mut self, rhs: &Self) {
        debug_assert!(
            limbs_cmp(&self.limbs, &rhs.limbs) != Ordering::Less,
            "SimdUint::sub underflow: self < rhs"
        );
        if limbs_cmp(&self.limbs, &rhs.limbs) == Ordering::Less {
            panic!("SimdUint::sub underflow: self < rhs");
        }

        let mut borrow = 0i64;
        const BASE: i64 = 1 << 32;
        for (i, lhs_limb) in self.limbs.iter_mut().enumerate() {
            let lhs = *lhs_limb as i64;
            let rhs = rhs.limbs.get(i).copied().unwrap_or(0) as i64;
            let difference = lhs - rhs - borrow;
            if difference < 0 {
                *lhs_limb = (difference + BASE) as u32;
                borrow = 1;
            } else {
                *lhs_limb = difference as u32;
                borrow = 0;
            }
        }
        debug_assert_eq!(borrow, 0, "SimdUint::sub underflow: borrow escaped");
        trim_in_place(&mut self.limbs);
    }

    /// Canonical big-endian bytes: empty for zero, no leading zero bytes.
    pub fn to_bytes_be(&self) -> Vec<u8> {
        if self.is_zero() {
            return Vec::new();
        }
        let mut bytes = Vec::with_capacity(self.limbs.len() * 4);
        for &limb in self.limbs.iter().rev() {
            bytes.extend_from_slice(&limb.to_be_bytes());
        }
        let first = bytes
            .iter()
            .position(|&byte| byte != 0)
            .unwrap_or(bytes.len());
        if first > 0 {
            bytes.drain(..first);
        }
        bytes
    }

    /// Build from big-endian bytes; leading zero bytes are ignored.
    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        let mut limbs = Vec::with_capacity(bytes.len().div_ceil(4));
        for chunk in bytes.rchunks(4) {
            let mut limb = 0u32;
            for &byte in chunk {
                limb = (limb << 8) | byte as u32;
            }
            limbs.push(limb);
        }
        trim_in_place(&mut limbs);
        Self { limbs }
    }

    /// Exact decimal representation.
    pub fn to_decimal(&self) -> String {
        BigUint::from_slice(&self.limbs).to_str_radix(10)
    }
}

// ---------------------------------------------------------------------------
// Limb helpers
// ---------------------------------------------------------------------------

fn trim_in_place(limbs: &mut Vec<u32>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

fn trim(mut limbs: Vec<u32>) -> Vec<u32> {
    trim_in_place(&mut limbs);
    limbs
}

/// Compare two canonical limb slices.
fn limbs_cmp(left: &[u32], right: &[u32]) -> Ordering {
    if left.len() != right.len() {
        return left.len().cmp(&right.len());
    }
    for (&lhs, &rhs) in left.iter().rev().zip(right.iter().rev()) {
        match lhs.cmp(&rhs) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

/// Add two limb slices, returning a canonical result.
fn add_slices(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry = 0u64;
    for (i, &limb) in long.iter().enumerate() {
        let sum = limb as u64 + short.get(i).copied().unwrap_or(0) as u64 + carry;
        out.push(sum as u32);
        carry = sum >> 32;
    }
    if carry != 0 {
        out.push(carry as u32);
    }
    trim(out)
}

/// `target -= subtrahend`; the caller guarantees `target >= subtrahend`.
fn sub_assign_limbs(target: &mut Vec<u32>, subtrahend: &[u32]) {
    debug_assert!(
        limbs_cmp(target, subtrahend) != Ordering::Less,
        "Karatsuba intermediate underflow"
    );
    let mut borrow = 0i64;
    const BASE: i64 = 1 << 32;
    for (i, target_limb) in target.iter_mut().enumerate() {
        let lhs = *target_limb as i64;
        let rhs = subtrahend.get(i).copied().unwrap_or(0) as i64;
        let difference = lhs - rhs - borrow;
        if difference < 0 {
            *target_limb = (difference + BASE) as u32;
            borrow = 1;
        } else {
            *target_limb = difference as u32;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0, "Karatsuba intermediate underflow");
    trim_in_place(target);
}

/// Add `addend` into `out` starting at `offset`, propagating the carry.
///
/// This is the limb-offset add used by the Karatsuba combine; it never builds
/// a shifted copy of the addend.
fn add_shifted(out: &mut [u32], offset: usize, addend: &[u32]) {
    debug_assert!(offset + addend.len() <= out.len());
    let mut carry = 0u64;
    for (i, &limb) in addend.iter().enumerate() {
        let index = offset + i;
        let sum = out[index] as u64 + limb as u64 + carry;
        out[index] = sum as u32;
        carry = sum >> 32;
    }
    let mut index = offset + addend.len();
    while carry != 0 {
        debug_assert!(index < out.len(), "carry escaped the Karatsuba result");
        let sum = out[index] as u64 + carry;
        out[index] = sum as u32;
        carry = sum >> 32;
        index += 1;
    }
}

/// Add a carry into the tail of a schoolbook product row.
fn propagate_carry(out: &mut [u32], mut carry: u64) {
    let mut index = 0;
    while carry != 0 {
        debug_assert!(index < out.len(), "carry escaped the product buffer");
        let sum = out[index] as u64 + carry;
        out[index] = sum as u32;
        carry = sum >> 32;
        index += 1;
    }
}

// ---------------------------------------------------------------------------
// Multiplication dispatch and Karatsuba
// ---------------------------------------------------------------------------

fn mul_limbs(a: &[u32], b: &[u32]) -> Vec<u32> {
    debug_assert!(!a.is_empty() && !b.is_empty());
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };

    let unbalanced = long.len() >= short.len() * KARATSUBA_MAX_RATIO;
    if short.len() < KARATSUBA_THRESHOLD || unbalanced {
        return schoolbook(long, short);
    }
    karatsuba(long, short)
}

fn karatsuba(long: &[u32], short: &[u32]) -> Vec<u32> {
    debug_assert!(short.len() >= KARATSUBA_THRESHOLD);
    debug_assert!(long.len() < short.len() * KARATSUBA_MAX_RATIO);

    let split = long.len() / 2;
    debug_assert!(split > 0 && split < short.len());

    let (a0, a1) = long.split_at(split);
    let (b0, b1) = short.split_at(split);

    // z0 = a0*b0, z2 = a1*b1, z1 = (a0+a1)(b0+b1) - z0 - z2.
    let z0 = mul_limbs(a0, b0);
    let z2 = mul_limbs(a1, b1);
    let sum_a = add_slices(a0, a1);
    let sum_b = add_slices(b0, b1);
    let mut z1 = mul_limbs(&sum_a, &sum_b);
    sub_assign_limbs(&mut z1, &z0);
    sub_assign_limbs(&mut z1, &z2);

    // result = z0 + z1 * B^split + z2 * B^(2*split), added by limb offset.
    let mut out = vec![0u32; long.len() + short.len()];
    add_shifted(&mut out, 0, &z0);
    add_shifted(&mut out, split, &z1);
    add_shifted(&mut out, 2 * split, &z2);
    trim(out)
}

/// Bounded Rayon depth. The top-level multiplication gets a small number of
/// parallel levels; the remaining Karatsuba levels stay sequential so the
/// task count does not grow as `3^depth`.
fn parallel_depth() -> usize {
    let threads = rayon::current_num_threads();
    let mut reduced = threads;
    let mut depth = 0;
    while reduced >= 3 {
        reduced /= 3;
        depth += 1;
    }
    if depth == 0 && threads > 1 { 1 } else { depth }
}

fn mul_limbs_par(a: &[u32], b: &[u32], depth: usize) -> Vec<u32> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let unbalanced = long.len() >= short.len() * KARATSUBA_MAX_RATIO;
    if short.len() < KARATSUBA_THRESHOLD || unbalanced || depth == 0 {
        return mul_limbs(a, b);
    }
    karatsuba_par(long, short, depth)
}

fn karatsuba_par(long: &[u32], short: &[u32], depth: usize) -> Vec<u32> {
    debug_assert!(short.len() >= KARATSUBA_THRESHOLD);
    debug_assert!(long.len() < short.len() * KARATSUBA_MAX_RATIO);
    debug_assert!(depth > 0);

    let split = long.len() / 2;
    debug_assert!(split > 0 && split < short.len());

    let (a0, a1) = long.split_at(split);
    let (b0, b1) = short.split_at(split);
    let sum_a = add_slices(a0, a1);
    let sum_b = add_slices(b0, b1);

    // Three-way Karatsuba fan-out. The nested depth bound keeps the number of
    // Rayon tasks proportional to the worker count rather than 3^levels.
    let (z0, (z2, mut z1)) = rayon::join(
        || mul_limbs_par(a0, b0, depth - 1),
        || {
            rayon::join(
                || mul_limbs_par(a1, b1, depth - 1),
                || mul_limbs_par(&sum_a, &sum_b, depth - 1),
            )
        },
    );
    sub_assign_limbs(&mut z1, &z0);
    sub_assign_limbs(&mut z1, &z2);

    let mut out = vec![0u32; long.len() + short.len()];
    add_shifted(&mut out, 0, &z0);
    add_shifted(&mut out, split, &z1);
    add_shifted(&mut out, 2 * split, &z2);
    trim(out)
}

/// Schoolbook multiplication using the best kernel available at runtime.
fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
    match selected_kernel() {
        Kernel::Scalar32 => mul_scalar32(a, b),
        #[cfg(target_arch = "x86_64")]
        Kernel::Sse2 => {
            // SAFETY: the kernel is only selected after runtime CPU detection.
            unsafe { mul_sse2(a, b) }
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => {
            // SAFETY: the kernel is only selected after runtime CPU detection.
            unsafe { mul_avx2(a, b) }
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx512f => {
            // SAFETY: the kernel is only selected after runtime CPU detection.
            unsafe { mul_avx512f(a, b) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => mul_scalar32(a, b),
    }
}

// ---------------------------------------------------------------------------
// Scalar schoolbook kernel
// ---------------------------------------------------------------------------

fn mul_scalar32(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let product = ai as u64 * bj as u64 + out[i + j] as u64 + carry;
            out[i + j] = product as u32;
            carry = product >> 32;
        }
        propagate_carry(&mut out[i + b.len()..], carry);
    }
    trim(out)
}

// ---------------------------------------------------------------------------
// SIMD schoolbook kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2: two `u32` products per iteration widened to `u64` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn mul_sse2(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    // SAFETY: SSE2 is present on every x86-64 CPU; the loads stay in bounds
    // because the loop guard reserves two limbs and `out` has `a+b` limbs.
    unsafe {
        for (i, &ai) in a.iter().enumerate() {
            let va = _mm_set1_epi64x(ai as i64);
            let mut carry = 0u64;
            let mut j = 0usize;
            while j + 2 <= b.len() {
                let vb_bits = _mm_loadl_epi64(b.as_ptr().add(j) as *const __m128i);
                let vb = _mm_unpacklo_epi32(vb_bits, _mm_setzero_si128());
                let product = _mm_mul_epu32(va, vb);

                let vo_bits = _mm_loadl_epi64(out.as_ptr().add(i + j) as *const __m128i);
                let vo = _mm_unpacklo_epi32(vo_bits, _mm_setzero_si128());
                let sum = _mm_add_epi64(product, vo);

                let mut lanes = [0u64; 2];
                _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, sum);
                for (k, &lane) in lanes.iter().enumerate() {
                    let t = lane + carry;
                    out[i + j + k] = t as u32;
                    carry = t >> 32;
                }
                j += 2;
            }
            while j < b.len() {
                let product = ai as u64 * b[j] as u64 + out[i + j] as u64 + carry;
                out[i + j] = product as u32;
                carry = product >> 32;
                j += 1;
            }
            propagate_carry(&mut out[i + b.len()..], carry);
        }
    }
    trim(out)
}

/// AVX2: four `u32` products per iteration widened to `u64` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    // SAFETY: AVX2 was detected at runtime; the loads stay in bounds because
    // the loop guard reserves four limbs and `out` has `a+b` limbs.
    unsafe {
        for (i, &ai) in a.iter().enumerate() {
            let va = _mm256_set1_epi64x(ai as i64);
            let mut carry = 0u64;
            let mut j = 0usize;
            while j + 4 <= b.len() {
                let vb_bits = _mm_loadu_si128(b.as_ptr().add(j) as *const __m128i);
                let vb = _mm256_cvtepu32_epi64(vb_bits);
                let product = _mm256_mul_epu32(va, vb);

                let vo_bits = _mm_loadu_si128(out.as_ptr().add(i + j) as *const __m128i);
                let vo = _mm256_cvtepu32_epi64(vo_bits);
                let sum = _mm256_add_epi64(product, vo);

                let mut lanes = [0u64; 4];
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
                for (k, &lane) in lanes.iter().enumerate() {
                    let t = lane + carry;
                    out[i + j + k] = t as u32;
                    carry = t >> 32;
                }
                j += 4;
            }
            while j < b.len() {
                let product = ai as u64 * b[j] as u64 + out[i + j] as u64 + carry;
                out[i + j] = product as u32;
                carry = product >> 32;
                j += 1;
            }
            propagate_carry(&mut out[i + b.len()..], carry);
        }
    }
    trim(out)
}

/// AVX-512F: eight `u32` products per iteration widened to `u64` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn mul_avx512f(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    // SAFETY: AVX-512F was detected at runtime; the loads stay in bounds
    // because the loop guard reserves eight limbs and `out` has `a+b` limbs.
    unsafe {
        for (i, &ai) in a.iter().enumerate() {
            let va = _mm512_set1_epi64(ai as i64);
            let mut carry = 0u64;
            let mut j = 0usize;
            while j + 8 <= b.len() {
                let vb_bits = _mm256_loadu_si256(b.as_ptr().add(j) as *const __m256i);
                let vb = _mm512_cvtepu32_epi64(vb_bits);
                let product = _mm512_mul_epu32(va, vb);

                let vo_bits = _mm256_loadu_si256(out.as_ptr().add(i + j) as *const __m256i);
                let vo = _mm512_cvtepu32_epi64(vo_bits);
                let sum = _mm512_add_epi64(product, vo);

                let mut lanes = [0u64; 8];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, sum);
                for (k, &lane) in lanes.iter().enumerate() {
                    let t = lane + carry;
                    out[i + j + k] = t as u32;
                    carry = t >> 32;
                }
                j += 8;
            }
            while j < b.len() {
                let product = ai as u64 * b[j] as u64 + out[i + j] as u64 + carry;
                out[i + j] = product as u32;
                carry = product >> 32;
                j += 1;
            }
            propagate_carry(&mut out[i + b.len()..], carry);
        }
    }
    trim(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    /// Deterministic linear congruential generator with a mixing step.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed ^ 0x9e37_79b9_7f4a_7c15,
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut mixed = self.state;
            mixed ^= mixed >> 33;
            mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
            mixed ^= mixed >> 33;
            mixed
        }
    }

    /// Build an exact `bits`-bit random value (the top bit is forced to one).
    fn random_uint(rng: &mut Lcg, bits: usize) -> SimdUint {
        if bits == 0 {
            return SimdUint::zero();
        }
        let byte_len = bits.div_ceil(8);
        let mut bytes = vec![0u8; byte_len];
        for byte in &mut bytes {
            *byte = (rng.next_u64() >> 32) as u8;
        }
        let excess = byte_len * 8 - bits;
        bytes[0] &= 0xffu8 >> excess;
        bytes[0] |= 1u8 << (7 - excess);
        SimdUint::from_bytes_be(&bytes)
    }

    fn oracle(value: &SimdUint) -> BigUint {
        BigUint::from_slice(&value.limbs)
    }

    /// Compare against `num-bigint` and check the representation invariant.
    fn check(actual: &SimdUint, expected: &BigUint) {
        assert_eq!(oracle(actual), *expected);
        assert_eq!(actual.bits(), expected.bits());
        // num-bigint encodes zero as `[0]`; the API here requires empty bytes.
        let expected_bytes = if expected == &BigUint::from(0u32) {
            Vec::new()
        } else {
            expected.to_bytes_be()
        };
        assert_eq!(actual.to_bytes_be(), expected_bytes);
        assert_eq!(SimdUint::from_bytes_be(&actual.to_bytes_be()), *actual);
        assert_ne!(
            actual.limbs.last().copied().unwrap_or(1),
            0,
            "trailing zero limb"
        );
    }

    #[test]
    fn zero_and_one() {
        let zero = SimdUint::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.bits(), 0);
        assert!(zero.limbs.is_empty());
        assert_eq!(zero.to_bytes_be(), Vec::<u8>::new());
        assert_eq!(zero.to_decimal(), "0");
        check(&zero, &BigUint::from(0u32));

        let one = SimdUint::one();
        assert!(!one.is_zero());
        assert_eq!(one.bits(), 1);
        assert_eq!(one.to_bytes_be(), vec![1]);
        assert_eq!(one.to_decimal(), "1");
        check(&one, &BigUint::from(1u32));
    }

    #[test]
    fn from_u64_edges() {
        let values = [
            0u64,
            1,
            2,
            3,
            u32::MAX as u64,
            1u64 << 32,
            (1u64 << 32) + 1,
            u64::MAX,
        ];
        for value in values {
            let uint = SimdUint::from_u64(value);
            let expected = BigUint::from(value);
            check(&uint, &expected);
            assert_eq!(
                uint.bits(),
                if value == 0 {
                    0
                } else {
                    64 - value.leading_zeros() as u64
                }
            );
        }
    }

    #[test]
    fn bytes_are_canonical() {
        let mut rng = Lcg::new(0xb17e5);
        for _ in 0..40 {
            let bits = (rng.next_u64() % 2001) as usize;
            let value = random_uint(&mut rng, bits);
            let bytes = value.to_bytes_be();
            if !value.is_zero() {
                assert_ne!(bytes[0], 0, "leading zero byte");
                assert_eq!(bytes.len(), value.bits().div_ceil(8) as usize);
            } else {
                assert!(bytes.is_empty());
            }
            assert_eq!(SimdUint::from_bytes_be(&bytes), value);

            let padded = [&[0u8, 0, 0, 0][..], &bytes[..]].concat();
            assert_eq!(SimdUint::from_bytes_be(&padded), value);
        }
        assert_eq!(SimdUint::from_bytes_be(&[0, 0, 0]), SimdUint::zero());
        assert_eq!(SimdUint::from_bytes_be(&[0, 0, 7]), SimdUint::from_u64(7));
    }

    #[test]
    fn add_and_sub_match_oracle() {
        let mut rng = Lcg::new(0xadd5_u64);
        for bits in (0..=2000).step_by(29) {
            let a = random_uint(&mut rng, bits);
            let b_bits = (rng.next_u64() as usize) % (bits + 1);
            let b = random_uint(&mut rng, b_bits);

            let expected_sum = &oracle(&a) + &oracle(&b);
            let sum = a.add(&b);
            check(&sum, &expected_sum);
            assert_eq!(sum, b.add(&a));

            let mut in_place = a.clone();
            in_place.add_assign(&b);
            assert_eq!(in_place, sum);

            let (larger, smaller) = if oracle(&a) >= oracle(&b) {
                (&a, &b)
            } else {
                (&b, &a)
            };
            let expected_difference = &oracle(larger) - &oracle(smaller);
            let difference = larger.sub(smaller);
            check(&difference, &expected_difference);

            let mut in_place = larger.clone();
            in_place.sub_assign(smaller);
            assert_eq!(in_place, difference);

            assert_eq!(sum.sub(&a), b);
            assert_eq!(sum.sub(&b), a);
            assert!(a.sub(&a).is_zero());
        }
    }

    #[test]
    #[should_panic(expected = "underflow")]
    fn sub_underflow_panics() {
        let _ = SimdUint::from_u64(3).sub(&SimdUint::from_u64(5));
    }

    #[test]
    fn mul_matches_oracle_across_sizes() {
        let mut rng = Lcg::new(0x5eed_1234);
        // Lengths are in bits. The list crosses the 64-limb / 2048-bit
        // Karatsuba threshold and includes balanced, unbalanced, and very
        // unbalanced (schoolbook fallback) products.
        let cases = [
            (1usize, 1usize),
            (2, 3),
            (31, 32),
            (32, 33),
            (63, 64),
            (64, 64),
            (65, 65),
            (127, 128),
            (255, 256),
            (511, 512),
            (1023, 1024),
            (1535, 1536),
            (2047, 2047),
            (2048, 2048),
            (2049, 2049),
            (2048, 64),
            (2048, 1023),
            (2048, 2047),
            (3000, 2000),
            (4064, 2048),
            (4064, 2049),
            (4096, 2048),
            (4096, 2049),
            (4096, 3000),
            (4096, 4095),
            (8192, 104),
            (8192, 4096),
            (12000, 6000),
            (16384, 8192),
            (20000, 100),
            (20000, 19999),
            (20000, 20000),
            (100, 20000),
            (31, 65),
        ];

        for (a_bits, b_bits) in cases {
            let a = random_uint(&mut rng, a_bits);
            let b = random_uint(&mut rng, b_bits);
            let expected = &oracle(&a) * &oracle(&b);
            let product = a.mul(&b);
            check(&product, &expected);
            assert_eq!(product, b.mul(&a));
        }
    }

    #[test]
    fn square_and_identity_cases() {
        let zero = SimdUint::zero();
        let one = SimdUint::one();
        let mut rng = Lcg::new(0x51a3_9e77);

        for &bits in &[
            0usize, 1, 2, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 511, 512, 1024, 1536, 2047,
            2048, 2049, 3000, 4096, 8192, 12000, 20000,
        ] {
            let a = random_uint(&mut rng, bits);
            let expected_square = &oracle(&a) * &oracle(&a);
            check(&a.mul(&a), &expected_square);

            assert_eq!(a.mul(&one), a);
            assert_eq!(one.mul(&a), a);
            assert!(a.mul(&zero).is_zero());
            assert!(zero.mul(&a).is_zero());
            assert_eq!(a.add(&zero), a);
            assert_eq!(a.sub(&zero), a);
        }
    }

    #[test]
    fn decimal_is_exact() {
        assert_eq!(SimdUint::zero().to_decimal(), "0");
        assert_eq!(SimdUint::one().to_decimal(), "1");
        assert_eq!(
            SimdUint::from_u64(u64::MAX).to_decimal(),
            "18446744073709551615"
        );

        let mut rng = Lcg::new(0xdec1_5eed);
        for &bits in &[32usize, 64, 100, 1000, 2048, 5000] {
            let value = random_uint(&mut rng, bits);
            assert_eq!(
                value.to_decimal(),
                BigUint::from_slice(&value.limbs).to_str_radix(10)
            );
        }
    }

    #[test]
    fn selected_kernel_is_known() {
        let name = selected_kernel_name();
        assert!(
            ["scalar32", "sse2", "avx2", "avx512f"].contains(&name),
            "unexpected kernel name: {name}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_available_simd_kernel_matches_scalar() {
        let mut rng = Lcg::new(0x51d0_c0de);
        for &(a_bits, b_bits) in &[
            (1usize, 1usize),
            (7, 9),
            (33, 17),
            (64, 64),
            (65, 63),
            (100, 37),
            (257, 129),
        ] {
            let a = random_uint(&mut rng, a_bits);
            let b = random_uint(&mut rng, b_bits);
            let expected = mul_scalar32(&a.limbs, &b.limbs);
            assert_eq!(mul_limbs(&a.limbs, &b.limbs), expected);

            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 is available on this CPU and the kernel is
                // exercised on normalized operands.
                let got = unsafe { mul_sse2(&a.limbs, &b.limbs) };
                assert_eq!(got, expected, "sse2 mismatch");
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 is available on this CPU and the kernel is
                // exercised on normalized operands.
                let got = unsafe { mul_avx2(&a.limbs, &b.limbs) };
                assert_eq!(got, expected, "avx2 mismatch");
            }
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F is available on this CPU and the kernel is
                // exercised on normalized operands.
                let got = unsafe { mul_avx512f(&a.limbs, &b.limbs) };
                assert_eq!(got, expected, "avx512f mismatch");
            }
        }
    }
}
