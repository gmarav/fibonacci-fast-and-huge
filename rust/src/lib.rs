//! Exact Binet, matrix, and fast-doubling Fibonacci implementations.
//! The algorithms share backend adapters, not their mathematical recurrences.

pub mod backend;
use backend::Natural;
use rayon::prelude::*;
use std::{str::FromStr, thread};

pub const LEAF_BITS: u64 = 262_144;
pub const TASK_BITS: u64 = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Direct,
    Matrix,
    Doubling,
}

impl Method {
    pub const ALL: [Self; 3] = [Self::Direct, Self::Matrix, Self::Doubling];
    pub fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Matrix => "matrix",
            Self::Doubling => "doubling",
        }
    }
}

impl FromStr for Method {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "direct" => Ok(Self::Direct),
            "matrix" => Ok(Self::Matrix),
            "doubling" => Ok(Self::Doubling),
            _ => Err(format!(
                "unknown method {value:?}; use direct, matrix, doubling, or compare"
            )),
        }
    }
}

/// A private, reusable Rayon pool. It never initializes Rayon's global pool.
pub struct Calculator {
    workers: usize,
    split_multiplication: bool,
    pool: Option<rayon::ThreadPool>,
}

impl Calculator {
    pub fn new(requested: usize, split_multiplication: bool) -> Result<Self, String> {
        let available = thread::available_parallelism().map_or(1, usize::from);
        let workers = if requested == 0 {
            available
        } else {
            requested.min(available)
        }
        .max(1);
        let pool = if workers == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .map_err(|e| e.to_string())?,
            )
        };
        Ok(Self {
            workers,
            split_multiplication,
            pool,
        })
    }
    pub fn workers(&self) -> usize {
        self.workers
    }
    pub fn split_multiplication(&self) -> bool {
        self.split_multiplication
    }

    pub fn calculate<N: Natural>(&self, method: Method, n: u64, reuse: bool) -> N {
        let run = || match method {
            Method::Direct => self.direct(n, reuse),
            Method::Matrix => self.matrix(n, reuse),
            Method::Doubling => self.doubling(n, reuse),
        };
        match &self.pool {
            Some(pool) if n >= 262_144 => pool.install(run),
            _ => run(),
        }
    }

    fn budget(&self, bits: u64) -> usize {
        if bits < TASK_BITS { 1 } else { self.workers }
    }

    /// Compare this primitive with the backend's native multiplication in tests.
    pub fn multiply<N: Natural>(&self, left: &N, right: &N) -> N {
        let run = || {
            let mut value = N::default();
            self.mul_into(&mut value, left, right, self.workers);
            value
        };
        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }

    fn mul_into<N: Natural>(&self, out: &mut N, left: &N, right: &N, budget: usize) {
        let small = left.bits().min(right.bits());
        let large = left.bits().max(right.bits());
        if !self.split_multiplication
            || budget <= 1
            || small < LEAF_BITS
            || small < large / 2
            || left.negative()
            || right.negative()
        {
            N::mul_into(out, left, right);
            return;
        }
        let shift = usize::try_from(large / 2).expect("operand bit length fits usize");
        let mut lh = left.clone();
        lh.shift_right(shift);
        let mut ll = lh.clone();
        ll.shift_left(shift);
        let mut temporary = N::default();
        N::sub_into(&mut temporary, left, &ll);
        ll = temporary;
        let square = std::ptr::eq(left, right);
        let mut rh = N::default();
        let mut rl = N::default();
        if !square {
            rh.clone_from(right);
            rh.shift_right(shift);
            rl.clone_from(&rh);
            rl.shift_left(shift);
            let mut temporary = N::default();
            N::sub_into(&mut temporary, right, &rl);
            rl = temporary;
        }
        let mut sx = N::default();
        N::add_into(&mut sx, &ll, &lh);
        let mut sy = N::default();
        if !square {
            N::add_into(&mut sy, &rl, &rh);
        }
        let (rlo, rhi, rsum) = if square {
            (&ll, &lh, &sx)
        } else {
            (&rl, &rh, &sy)
        };
        let mut low = N::default();
        let mut high = N::default();
        let mut cross = N::default();
        triple(
            budget,
            |w| self.mul_into(&mut low, &ll, rlo, w),
            |w| self.mul_into(&mut high, &lh, rhi, w),
            |w| self.mul_into(&mut cross, &sx, rsum, w),
        );
        cross.sub_assign(&low);
        cross.sub_assign(&high);
        high.shift_left(2 * shift);
        cross.shift_left(shift);
        N::add_into(out, &high, &cross);
        out.add_assign(&low);
    }

    fn pair_into<N: Natural>(
        &self,
        out: &mut Pair<N>,
        left: &Pair<N>,
        right: &Pair<N>,
        scratch: &mut PairScratch<N>,
    ) {
        let PairScratch { p, q, r, sx, sy } = scratch;
        N::add_into(sx, &left.a, &left.b);
        let right_sum = if std::ptr::eq(left, right) {
            &*sx
        } else {
            N::add_into(sy, &right.a, &right.b);
            &*sy
        };
        let budget = self.budget(
            left.a
                .bits()
                .max(left.b.bits())
                .max(right.a.bits())
                .max(right.b.bits()),
        );
        triple(
            budget,
            |w| self.mul_into(p, &left.a, &right.a, w),
            |w| self.mul_into(q, &left.b, &right.b, w),
            |w| self.mul_into(r, sx, right_sum, w),
        );
        N::add_into(&mut out.a, p, q);
        N::sub_into(&mut out.b, r, p);
    }

    fn direct<N: Natural>(&self, mut n: u64, reuse: bool) -> N {
        let mut result = Pair {
            a: N::small(1),
            b: N::default(),
        };
        let mut base = Pair {
            a: N::default(),
            b: N::small(1),
        };
        let mut output = Pair::default();
        let mut scratch = PairScratch::default();
        while n != 0 {
            if n & 1 != 0 {
                if !reuse {
                    output = Pair::default();
                    scratch = PairScratch::default();
                }
                self.pair_into(&mut output, &result, &base, &mut scratch);
                std::mem::swap(&mut output, &mut result);
            }
            n >>= 1;
            if n != 0 {
                if !reuse {
                    output = Pair::default();
                    scratch = PairScratch::default();
                }
                self.pair_into(&mut output, &base, &base, &mut scratch);
                std::mem::swap(&mut output, &mut base);
            }
        }
        result.b
    }

    fn matrix_into<N: Natural>(
        &self,
        out: &mut [N; 4],
        left: &[N; 4],
        right: &[N; 4],
        products: &mut [N; 8],
    ) {
        let largest = left.iter().chain(right).map(N::bits).max().unwrap_or(0);
        let budget = self.budget(largest);
        let compute = |(index, product): (usize, &mut N)| {
            let row = index / 4;
            let column = (index / 2) % 2;
            let k = index % 2;
            let per_job = if budget < 8 {
                1
            } else {
                budget / 8 + usize::from(index < budget % 8)
            };
            self.mul_into(product, &left[row * 2 + k], &right[k * 2 + column], per_job);
        };
        if budget == 1 {
            products.iter_mut().enumerate().for_each(compute);
        } else {
            products.par_iter_mut().enumerate().for_each(compute);
        }
        for index in 0..4 {
            N::add_into(
                &mut out[index],
                &products[2 * index],
                &products[2 * index + 1],
            );
        }
    }

    fn matrix<N: Natural>(&self, mut n: u64, reuse: bool) -> N {
        let mut result = [N::small(1), N::default(), N::default(), N::small(1)];
        let mut base = [N::small(1), N::small(1), N::small(1), N::default()];
        let mut output = std::array::from_fn(|_| N::default());
        let mut products = std::array::from_fn(|_| N::default());
        while n != 0 {
            if n & 1 != 0 {
                if !reuse {
                    output = std::array::from_fn(|_| N::default());
                    products = std::array::from_fn(|_| N::default());
                }
                self.matrix_into(&mut output, &result, &base, &mut products);
                std::mem::swap(&mut output, &mut result);
            }
            n >>= 1;
            if n != 0 {
                if !reuse {
                    output = std::array::from_fn(|_| N::default());
                    products = std::array::from_fn(|_| N::default());
                }
                self.matrix_into(&mut output, &base, &base, &mut products);
                std::mem::swap(&mut output, &mut base);
            }
        }
        std::mem::take(&mut result[1])
    }

    fn doubling<N: Natural>(&self, n: u64, reuse: bool) -> N {
        let mut a = N::default();
        let mut b = N::small(1);
        let mut c = N::default();
        let mut d = N::default();
        let mut aa = N::default();
        let mut bb = N::default();
        let mut t = N::default();
        let mut next_b = N::default();
        for bit in (0..(64 - n.leading_zeros())).rev() {
            if !reuse {
                c = N::default();
                d = N::default();
                aa = N::default();
                bb = N::default();
                t = N::default();
                next_b = N::default();
            }
            N::add_into(&mut t, &b, &b);
            t.sub_assign(&a);
            triple(
                self.budget(b.bits()),
                |w| self.mul_into(&mut c, &a, &t, w),
                |w| self.mul_into(&mut aa, &a, &a, w),
                |w| self.mul_into(&mut bb, &b, &b, w),
            );
            N::add_into(&mut d, &aa, &bb);
            if n >> bit & 1 == 0 {
                std::mem::swap(&mut a, &mut c);
                std::mem::swap(&mut b, &mut d);
            } else {
                N::add_into(&mut next_b, &c, &d);
                std::mem::swap(&mut a, &mut d);
                std::mem::swap(&mut b, &mut next_b);
            }
        }
        a
    }
}

#[derive(Default)]
struct Pair<N: Natural> {
    a: N,
    b: N,
}
#[derive(Default)]
struct PairScratch<N: Natural> {
    p: N,
    q: N,
    r: N,
    sx: N,
    sy: N,
}

fn triple<A: Send, B: Send, C: Send>(
    workers: usize,
    a: impl FnOnce(usize) -> A + Send,
    b: impl FnOnce(usize) -> B + Send,
    c: impl FnOnce(usize) -> C + Send,
) -> (A, B, C) {
    if workers <= 1 {
        return (a(1), b(1), c(1));
    }
    if workers == 2 {
        let ((a, c), b) = rayon::join(|| (a(1), c(1)), || b(1));
        return (a, b, c);
    }
    let base = workers / 3;
    let remainder = workers % 3;
    let (a, (b, c)) = rayon::join(
        || a(base + usize::from(remainder > 0)),
        || rayon::join(|| b(base + usize::from(remainder > 1)), || c(base)),
    );
    (a, b, c)
}
