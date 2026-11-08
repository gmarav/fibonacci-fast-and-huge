use fib_rust::{Calculator, LEAF_BITS, Method, backend::Natural};
use num_bigint::BigUint;

fn reference(n: u64) -> BigUint {
    let (mut a, mut b) = (BigUint::default(), BigUint::from(1_u8));
    for _ in 0..n {
        let next = &a + &b;
        a = b;
        b = next;
    }
    a
}

fn check_backend<N: Natural>() {
    let sequential = Calculator::new(1, false).unwrap();
    let (mut a, mut b) = (BigUint::default(), BigUint::from(1_u8));
    for n in 0..=2000 {
        for method in Method::ALL {
            for reuse in [false, true] {
                let actual: N = sequential.calculate(method, n, reuse);
                assert_eq!(
                    actual.bytes(),
                    a.bytes(),
                    "{} {method:?} n={n} reuse={reuse}",
                    N::NAME
                );
            }
        }
        let next = &a + &b;
        a = b;
        b = next;
    }
    for k in 0..=13 {
        let power = 1_u64 << k;
        for n in [power - 1, power, power + 1] {
            let expected = reference(n).bytes();
            for method in Method::ALL {
                assert_eq!(sequential.calculate::<N>(method, n, true).bytes(), expected);
            }
        }
    }
    for (n, digits) in [(0, 1), (100, 21), (1000, 209), (100_000, 20899)] {
        for method in Method::ALL {
            assert_eq!(
                sequential.calculate::<N>(method, n, true).decimal().len(),
                digits
            );
        }
    }
    let expected = reference(100).decimal();
    assert_eq!(expected, "354224848179261915075");
    for split in [false, true] {
        for workers in [1, 2, 4, 0] {
            let calculator = Calculator::new(workers, split).unwrap();
            let expected = sequential
                .calculate::<BigUint>(Method::Direct, 1_000_000, true)
                .bytes();
            for method in Method::ALL {
                for reuse in [false, true] {
                    assert_eq!(
                        calculator.calculate::<N>(method, 1_000_000, reuse).bytes(),
                        expected,
                        "{} {method:?} workers={workers} split={split} reuse={reuse}",
                        N::NAME
                    );
                }
            }
        }
    }
    let mut original: N = sequential.calculate(Method::Direct, 1000, true);
    let second: N = sequential.calculate(Method::Matrix, 1000, true);
    let saved = second.bytes();
    original.add_assign(&N::small(1));
    assert_eq!(second.bytes(), saved);
    assert_eq!(
        sequential
            .calculate::<N>(Method::Doubling, 1000, true)
            .bytes(),
        saved
    );
    assert!(N::small(0).bytes().is_empty());
    for bytes in [&[][..], &[1][..], &[0x80, 0, 0xff][..]] {
        assert_eq!(N::from_bytes(bytes).bytes(), bytes);
    }
}

fn check_multiplication<N: Natural>() {
    let calculator = Calculator::new(4, true).unwrap();
    let mut seed = 719_u64;
    for length in [
        (LEAF_BITS / 8 - 1) as usize,
        (LEAF_BITS / 8) as usize,
        (LEAF_BITS / 8 + 1) as usize,
        (LEAF_BITS / 4 + 7) as usize,
    ] {
        let mut left = vec![0xff; length];
        let mut right = vec![0; length];
        for byte in &mut right {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (seed >> 32) as u8;
        }
        left[0] |= 0x80;
        right[0] |= 0x80;
        let a = N::from_bytes(&left);
        let b = N::from_bytes(&right);
        let mut expected = N::default();
        N::mul_into(&mut expected, &a, &b);
        assert_eq!(calculator.multiply(&a, &b), expected);
        N::mul_into(&mut expected, &a, &a);
        assert_eq!(calculator.multiply(&a, &a), expected);
        assert_eq!(a.bytes(), left);
        assert_eq!(b.bytes(), right);
        assert_eq!(calculator.multiply(&a, &N::small(0)), N::small(0));
        assert_eq!(calculator.multiply(&a, &N::small(1)), a);
    }
}

#[test]
fn num_bigint_algorithms() {
    check_backend::<BigUint>();
}
#[test]
fn num_bigint_parallel_multiply() {
    check_multiplication::<BigUint>();
}

#[cfg(feature = "gmp")]
#[test]
fn gmp_algorithms() {
    check_backend::<rug::Integer>();
}
#[cfg(feature = "gmp")]
#[test]
fn gmp_parallel_multiply() {
    check_multiplication::<rug::Integer>();
    let calculator = Calculator::new(4, true).unwrap();
    let negative = rug::Integer::from(-7);
    let positive = rug::Integer::from(9);
    assert_eq!(calculator.multiply(&negative, &positive), -63);
}

#[test]
fn independent_concurrent_calculators() {
    std::thread::scope(|scope| {
        for _ in 0..3 {
            scope.spawn(|| {
                let calculator = Calculator::new(2, true).unwrap();
                assert_eq!(
                    calculator.calculate::<BigUint>(Method::Direct, 1_000_000, true),
                    calculator.calculate::<BigUint>(Method::Doubling, 1_000_000, false)
                );
            });
        }
    });
}

#[test]
fn worker_limit_is_capped() {
    let available = std::thread::available_parallelism().unwrap().get();
    assert_eq!(
        Calculator::new(usize::MAX, false).unwrap().workers(),
        available
    );
    assert_eq!(Calculator::new(1, false).unwrap().workers(), 1);
}
