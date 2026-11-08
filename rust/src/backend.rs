//! Exact nonnegative integer adapters. No floating-point result calculation.

use num_bigint::BigUint;
use std::fmt::Debug;

pub trait Natural: Clone + Default + Eq + Ord + Send + Sync + Debug {
    const NAME: &'static str;
    fn small(value: u64) -> Self;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn bits(&self) -> u64;
    fn negative(&self) -> bool {
        false
    }
    fn bytes(&self) -> Vec<u8>;
    fn decimal(&self) -> String;
    fn add_into(destination: &mut Self, left: &Self, right: &Self);
    fn sub_into(destination: &mut Self, left: &Self, right: &Self);
    fn mul_into(destination: &mut Self, left: &Self, right: &Self);
    fn add_assign(&mut self, right: &Self);
    fn sub_assign(&mut self, right: &Self);
    fn shift_left(&mut self, bits: usize);
    fn shift_right(&mut self, bits: usize);
}

impl Natural for BigUint {
    const NAME: &'static str = "num-bigint";
    fn small(value: u64) -> Self {
        value.into()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_be(bytes)
    }
    fn bits(&self) -> u64 {
        self.bits()
    }
    fn bytes(&self) -> Vec<u8> {
        if self.bits() == 0 {
            Vec::new()
        } else {
            self.to_bytes_be()
        }
    }
    fn decimal(&self) -> String {
        self.to_str_radix(10)
    }
    fn add_into(destination: &mut Self, left: &Self, right: &Self) {
        destination.clone_from(left);
        *destination += right;
    }
    fn sub_into(destination: &mut Self, left: &Self, right: &Self) {
        destination.clone_from(left);
        *destination -= right;
    }
    fn mul_into(destination: &mut Self, left: &Self, right: &Self) {
        // num-bigint does not expose a destination-based multiply API. Its
        // product allocation remains backend-owned even in reusable workspaces.
        *destination = left * right;
    }
    fn add_assign(&mut self, right: &Self) {
        *self += right;
    }
    fn sub_assign(&mut self, right: &Self) {
        *self -= right;
    }
    fn shift_left(&mut self, bits: usize) {
        *self <<= bits;
    }
    fn shift_right(&mut self, bits: usize) {
        *self >>= bits;
    }
}

#[cfg(feature = "gmp")]
impl Natural for rug::Integer {
    const NAME: &'static str = "gmp";
    fn small(value: u64) -> Self {
        value.into()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_digits(bytes, rug::integer::Order::MsfBe)
    }
    fn bits(&self) -> u64 {
        // Integer::significant_bits returns u32. Count borrowed limbs instead
        // so the minute experiment can exceed four billion bits without a copy.
        let limbs = self.as_limbs();
        match limbs.split_last() {
            None => 0,
            Some((top, prefix)) => {
                let width = (std::mem::size_of_val(top) * 8) as u64;
                prefix.len() as u64 * width + width - u64::from(top.leading_zeros())
            }
        }
    }
    fn negative(&self) -> bool {
        self < &0
    }
    fn bytes(&self) -> Vec<u8> {
        if self.as_limbs().is_empty() {
            Vec::new()
        } else {
            self.to_digits(rug::integer::Order::MsfBe)
        }
    }
    fn decimal(&self) -> String {
        self.to_string()
    }
    fn add_into(destination: &mut Self, left: &Self, right: &Self) {
        use rug::Assign;
        destination.assign(left + right);
    }
    fn sub_into(destination: &mut Self, left: &Self, right: &Self) {
        use rug::Assign;
        destination.assign(left - right);
    }
    fn mul_into(destination: &mut Self, left: &Self, right: &Self) {
        use rug::Assign;
        destination.assign(left * right);
    }
    fn add_assign(&mut self, right: &Self) {
        *self += right;
    }
    fn sub_assign(&mut self, right: &Self) {
        *self -= right;
    }
    fn shift_left(&mut self, bits: usize) {
        *self <<= bits;
    }
    fn shift_right(&mut self, bits: usize) {
        *self >>= bits;
    }
}
