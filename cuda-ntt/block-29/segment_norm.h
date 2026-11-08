// Segment CRT + base-2^32 carry normalisation for the block engine.
//
// Pure host C++ (no CUDA, no __int128 when built with MSVC).  It is used by
// block29_fib.cpp and by segment_test.cpp, which checks it against a direct
// __int128 carry reference for random 96-bit coefficient sequences.
//
// Representation:
//   * a coefficient x_j of the exact product is x_j < P0*P1 < 2^96;
//   * the segment residue pair is (x_j mod P0, x_j mod P1);
//   * d_t = a_t + b_{t-1} + c_{t-2} is the standard 3-word (base 2^32)
//     spread of the coefficients, so d_t < 3*2^32 and the carry is 0..3.

#ifndef BLOCK_SEGMENT_NORM_H
#define BLOCK_SEGMENT_NORM_H

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace segnorm {

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint64_t u64;

static inline u64 mulmod(u64 a, u64 b, u64 p) {
#if defined(_MSC_VER) && defined(_M_X64)
    u64 hi;
    u64 lo = _umul128(a, b, &hi);
    u64 rem = 0;
    _udiv128(hi, lo, p, &rem);
    return rem;
#else
    return (u64)(((unsigned __int128)a * b) % p);
#endif
}

struct U96 {
    u64 lo;
    u32 hi;
};

// x = a0 + P0 * ((a1 - a0) * invP0 mod P1), with a0 < P0 < P1.
static inline U96 crt(u64 a0, u64 a1, u64 p0, u64 p1, u64 invP0) {
    u64 d = (a1 >= a0) ? (a1 - a0) : (a1 + p1 - a0);
    u64 t = mulmod(d, invP0, p1);
#if defined(_MSC_VER) && defined(_M_X64)
    u64 hi;
    u64 lo = _umul128(p0, t, &hi);
#else
    u64 hi = (u64)(((unsigned __int128)p0 * t) >> 64);
    u64 lo = p0 * t;
#endif
    u64 s = lo + a0;
    u64 carry = (s < lo) ? 1ull : 0ull;
    U96 r;
    r.lo = s;
    r.hi = (u32)(hi + carry);
    return r;
}

template <class F>
static inline void parallel_chunks(size_t n, int threads, F f) {
    if (n == 0) return;
    if (threads <= 1 || n < (1u << 16)) {
        f((size_t)0, n);
        return;
    }
    std::vector<std::thread> ts;
    size_t chunk = (n + (size_t)threads - 1) / (size_t)threads;
    for (int i = 0; i < threads; ++i) {
        size_t a = (size_t)i * chunk;
        size_t b = std::min(n, a + chunk);
        if (a < b) ts.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto& t : ts) t.join();
}

static inline void fold_add(u64* dl, const u64* fold, size_t n, u64 p,
                            int threads) {
    parallel_chunks(n, threads, [&](size_t a, size_t b) {
        for (size_t t = a; t < b; ++t) {
            u64 s = dl[t] + fold[t];
            if (s >= p) s -= p;
            dl[t] = s;
        }
    });
}

static inline void pack(u32* lo, u16* hi, const u64* seg, size_t n, u64 p,
                        int threads) {
    parallel_chunks(n, threads, [&](size_t a, size_t b) {
        for (size_t t = a; t < b; ++t) {
            u64 v = seg[t] % p;
            lo[t] = (u32)v;
            hi[t] = (u16)(v >> 32);
        }
    });
}

// d[t] = a_t + b_{t-1} + c_{t-2}.  The spills are the b/c words of the last
// two coefficients of the previous segment (zero for the first segment).
static inline void build_d(const u32* lo, const u16* hi, const u64* seg1,
                           size_t lim, u32 spill_b1, u32 spill_c1, u32 spill_c2,
                           u64* d, u64 p0, u64 p1, u64 invP0, int threads) {
    parallel_chunks(lim, threads, [&](size_t t0, size_t t1) {
        U96 xm2, xm1;
        auto real = [&](size_t j) {
            u64 a0 = (u64)lo[j] | ((u64)hi[j] << 32);
            return crt(a0, seg1[j], p0, p1, invP0);
        };
        auto virt1 = [&]() {
            U96 v;
            v.lo = (u64)spill_b1 << 32;
            v.hi = spill_c1;
            return v;
        };
        auto virt2 = [&]() {
            U96 v;
            v.lo = 0;
            v.hi = spill_c2;
            return v;
        };
        if (t0 == 0) {
            xm2 = virt2();
            xm1 = virt1();
        } else if (t0 == 1) {
            xm2 = virt1();
            xm1 = real(0);
        } else {
            xm2 = real(t0 - 2);
            xm1 = real(t0 - 1);
        }
        for (size_t t = t0; t < t1; ++t) {
            U96 xt = real(t);
            d[t] = (u64)(u32)xt.lo + (u64)(u32)(xm1.lo >> 32) + (u64)xm2.hi;
            xm2 = xm1;
            xm1 = xt;
        }
    });
}

// Parallel carry propagation.  d < 3*2^32 so the carry state is 0..3; each
// chunk is evaluated for all four possible carry-ins, then a small serial
// scan fixes the true carry-in and a second parallel pass writes the limbs.
static inline void carry_norm(const u64* d, size_t len, u32 carry_in, u32* out,
                              u32& carry_out, int threads) {
    if (len == 0) {
        carry_out = carry_in;
        return;
    }
    if (threads <= 1 || len < (1u << 20)) {
        u64 c = carry_in;
        for (size_t i = 0; i < len; ++i) {
            u64 s = d[i] + c;
            out[i] = (u32)s;
            c = s >> 32;
        }
        carry_out = (u32)c;
        return;
    }
    size_t nchunks = (size_t)threads * 4;
    size_t chunk = (len + nchunks - 1) / nchunks;
    nchunks = (len + chunk - 1) / chunk;
    std::vector<u32> cin(nchunks + 1, 0);
    std::vector<u32> cout4(nchunks * 4, 0);

    parallel_chunks(nchunks, threads, [&](size_t c0, size_t c1) {
        for (size_t ci = c0; ci < c1; ++ci) {
            size_t a = ci * chunk;
            size_t b = std::min(len, a + chunk);
            u32 k0 = 0, k1 = 1, k2 = 2, k3 = 3;
            for (size_t i = a; i < b; ++i) {
                u64 v = d[i];
                k0 = (u32)((v + k0) >> 32);
                k1 = (u32)((v + k1) >> 32);
                k2 = (u32)((v + k2) >> 32);
                k3 = (u32)((v + k3) >> 32);
            }
            cout4[ci * 4 + 0] = k0;
            cout4[ci * 4 + 1] = k1;
            cout4[ci * 4 + 2] = k2;
            cout4[ci * 4 + 3] = k3;
        }
    });
    cin[0] = carry_in;
    for (size_t ci = 0; ci < nchunks; ++ci)
        cin[ci + 1] = cout4[ci * 4 + cin[ci]];

    parallel_chunks(nchunks, threads, [&](size_t c0, size_t c1) {
        for (size_t ci = c0; ci < c1; ++ci) {
            size_t a = ci * chunk;
            size_t b = std::min(len, a + chunk);
            u64 c = cin[ci];
            for (size_t i = a; i < b; ++i) {
                u64 s = d[i] + c;
                out[i] = (u32)s;
                c = s >> 32;
            }
        }
    });
    carry_out = cin[nchunks];
}

// One output segment.  `full` means lim == B, so the trailing b/c words are
// saved for the next segment.  (The caller guarantees lim >= 2 for full
// segments; the driver uses B >= 2^11.)
static inline void process_segment(const u32* lo, const u16* hi,
                                   const u64* seg1, size_t lim, u64* d,
                                   u32* out, u32& carry, u32& sb1, u32& sc1,
                                   u32& sc2, bool full, u64 p0, u64 p1,
                                   u64 invP0, int threads) {
    build_d(lo, hi, seg1, lim, sb1, sc1, sc2, d, p0, p1, invP0, threads);
    carry_norm(d, lim, carry, out, carry, threads);
    if (full) {
        if (lim == 1) {
            u32 old_c1 = sc1;
            U96 x1 = crt((u64)lo[0] | ((u64)hi[0] << 32), seg1[0], p0, p1,
                         invP0);
            sb1 = (u32)(x1.lo >> 32);
            sc1 = x1.hi;
            sc2 = old_c1;
        } else {
            U96 x1 = crt((u64)lo[lim - 1] | ((u64)hi[lim - 1] << 32),
                         seg1[lim - 1], p0, p1, invP0);
            U96 x2 = crt((u64)lo[lim - 2] | ((u64)hi[lim - 2] << 32),
                         seg1[lim - 2], p0, p1, invP0);
            sb1 = (u32)(x1.lo >> 32);
            sc1 = x1.hi;
            sc2 = x2.hi;
        }
    }
}

}  // namespace segnorm

#endif  // BLOCK_SEGMENT_NORM_H
