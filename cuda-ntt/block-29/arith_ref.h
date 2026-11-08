// Host-side exact reference for the 2^29 two-prime 48-bit NTT engine.
//
// This header mirrors the arithmetic used by the CUDA kernels:
//   * Montgomery multiplication with R = 2^64 (mont48)
//   * DIF forward / DIT inverse NTT, twiddles in Montgomery form
//   * pointwise Montgomery product, final scale R * n^-1 (plain output)
//   * two-prime CRT and base-2^32 carry normalisation
//   * block convolution with zero padding and overlap-add
//
// Everything is integer arithmetic.  Host-only helpers (mulmod, powmod) use
// unsigned __int128 for the oracle; the mont48 body itself uses only 64-bit
// operations so it can be copied verbatim into device code.
//
// No CUDA headers are required here.

#ifndef BLOCK_ARITH_REF_H
#define BLOCK_ARITH_REF_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "block_constants.h"

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

namespace blk {

struct Mod {
    u64 p;
    u64 np;
    u64 R;
    u64 R2;
    u64 w;      // primitive 2^lg_max-th root
    u64 winv;
};

inline Mod mod0() { return Mod{P0, P0_NP, P0_R, P0_R2, P0_W, P0_W_INV}; }
inline Mod mod1() { return Mod{P1, P1_NP, P1_R, P1_R2, P1_W, P1_W_INV}; }

inline u64 mulmod(u64 a, u64 b, u64 p) {
    return (u64)((u128)a * b % p);
}

inline u64 powmod(u64 b, u64 e, u64 p) {
    u64 r = 1;
    b %= p;
    while (e) {
        if (e & 1) r = mulmod(r, b, p);
        b = mulmod(b, b, p);
        e >>= 1;
    }
    return r;
}

// Montgomery product a*b*R^-1 mod p, R = 2^64, p < 2^48.
// Identical body to the device __device__ __forceinline__ mont48.
inline u64 mont48(u64 a, u64 b, u64 p, u64 np) {
    u64 lo = a * b;
    // __umul64hi(a,b) on device == (a*b) >> 64.  Emulate with __int128 here.
    u64 hi = (u64)(((u128)a * b) >> 64);
    u64 m = lo * np;
    u64 mp_lo = m * p;
    u64 mp_hi = (u64)(((u128)m * p) >> 64);
    u64 s = lo + mp_lo;
    u64 carry = (s < lo) ? 1ull : 0ull;
    u64 u = hi + mp_hi + carry;
    return u >= p ? u - p : u;
}

inline u64 add48(u64 a, u64 b, u64 p) {
    u64 s = a + b;           // a,b < p < 2^48 -> s < 2^49, no u64 overflow
    if (s >= p) s -= p;
    return s;
}

inline u64 sub48(u64 a, u64 b, u64 p) {
    return a >= b ? a - b : a + (p - b);
}

// Montgomery-form conversion: x -> x*R mod p.
inline u64 to_mont(u64 x, const Mod& m) {
    return mont48(x % m.p, m.R2, m.p, m.np);
}

// Montgomery-form back to plain: X = x*R -> x.
inline u64 from_mont(u64 X, const Mod& m) {
    return mont48(X, 1, m.p, m.np);
}

// Twiddle in Montgomery form: w^e * R mod p.
inline u64 twiddle(const Mod& m, u64 w, u64 e) {
    return mulmod(powmod(w, e, m.p), m.R, m.p);
}

// ---------------------------------------------------------------------------
// NTT.  Stage order matches the device schedule:
//   forward: len = 2^lg, 2^(lg-1), ..., 2      (DIF, Gentleman-Sande output)
//   inverse: len = 2, 4, ..., 2^lg             (DIT)
// The final inverse scale is R * n^-1 in Montgomery form, which leaves the
// coefficients in plain (non-Montgomery) form.
// ---------------------------------------------------------------------------

inline void ntt_fwd(u64* a, int lg, const Mod& m) {
    const size_t n = (size_t)1 << lg;
    for (int s = lg; s >= 1; --s) {
        const size_t len = (size_t)1 << s;
        const size_t half = len >> 1;
        // m.w is a primitive 2^BLK_MAX_LOG2-th root.  A primitive 2^s-th
        // root is m.w^(2^(BLK_MAX_LOG2-s)), so the same w serves every
        // transform length (and the device can derive twiddles on the fly).
        const u64 root = powmod(m.w, 1ull << (BLK_MAX_LOG2 - s), m.p);
        const u64 step = mulmod(root, m.R, m.p);      // root * R
        for (size_t blk = 0; blk < n; blk += len) {
            u64 wj = m.R;
            for (size_t j = 0; j < half; ++j) {
                u64 u = a[blk + j];
                u64 v = a[blk + j + half];
                a[blk + j] = add48(u, v, m.p);
                a[blk + j + half] = mont48(sub48(u, v, m.p), wj, m.p, m.np);
                wj = mont48(wj, step, m.p, m.np);
            }
        }
    }
}

inline void ntt_inv(u64* a, int lg, const Mod& m) {
    const size_t n = (size_t)1 << lg;
    for (int s = 1; s <= lg; ++s) {
        const size_t len = (size_t)1 << s;
        const size_t half = len >> 1;
        const u64 root = powmod(m.winv, 1ull << (BLK_MAX_LOG2 - s), m.p);
        const u64 step = mulmod(root, m.R, m.p);
        for (size_t blk = 0; blk < n; blk += len) {
            u64 wj = m.R;
            for (size_t j = 0; j < half; ++j) {
                u64 u = a[blk + j];
                u64 v = mont48(a[blk + j + half], wj, m.p, m.np);
                a[blk + j] = add48(u, v, m.p);
                a[blk + j + half] = sub48(u, v, m.p);
                wj = mont48(wj, step, m.p, m.np);
            }
        }
    }
    const u64 ninv = powmod((u64)(n % m.p), m.p - 2, m.p);
    // Inverse structure output is N * conv * R (R-scaled); mont(X, n^-1)
    // gives N*conv*R * n^-1 * R^-1 = conv, i.e. plain residues.
    for (size_t i = 0; i < n; ++i) a[i] = mont48(a[i], ninv, m.p, m.np);
}

// Inverse without the final scale.  Used by the device convention, where the
// n^-1 factor is folded into the pointwise product: mont(mont(A,B), n^-1).
inline void ntt_inv_noscale(u64* a, int lg, const Mod& m) {
    const size_t n = (size_t)1 << lg;
    for (int s = 1; s <= lg; ++s) {
        const size_t len = (size_t)1 << s;
        const size_t half = len >> 1;
        const u64 root = powmod(m.winv, 1ull << (BLK_MAX_LOG2 - s), m.p);
        const u64 step = mulmod(root, m.R, m.p);
        for (size_t blk = 0; blk < n; blk += len) {
            u64 wj = m.R;
            for (size_t j = 0; j < half; ++j) {
                u64 u = a[blk + j];
                u64 v = mont48(a[blk + j + half], wj, m.p, m.np);
                a[blk + j] = add48(u, v, m.p);
                a[blk + j + half] = sub48(u, v, m.p);
                wj = mont48(wj, step, m.p, m.np);
            }
        }
    }
}

// One-block convolution of limb vectors whose padded length is at most n/2.
// Returns plain residues mod p of length n.
inline void conv_single(const std::vector<u32>& x, const std::vector<u32>& y,
                        int lg, const Mod& m, std::vector<u64>& out) {
    const size_t n = (size_t)1 << lg;
    out.assign(n, 0);
    std::vector<u64> b(n, 0);
    for (size_t i = 0; i < x.size(); ++i) out[i] = to_mont(x[i], m);
    for (size_t i = 0; i < y.size(); ++i) b[i] = to_mont(y[i], m);
    ntt_fwd(out.data(), lg, m);
    ntt_fwd(b.data(), lg, m);
    for (size_t i = 0; i < n; ++i) out[i] = mont48(out[i], b[i], m.p, m.np);
    ntt_inv(out.data(), lg, m);
}

// Device convention: the n^-1 factor is folded into the pointwise product and
// the inverse transform has no final scale.  Result is identical (plain
// convolution residues mod p).
inline void conv_single_scaled(const std::vector<u32>& x,
                               const std::vector<u32>& y, int lg,
                               const Mod& m, std::vector<u64>& out) {
    const size_t n = (size_t)1 << lg;
    out.assign(n, 0);
    std::vector<u64> b(n, 0);
    for (size_t i = 0; i < x.size(); ++i) out[i] = to_mont(x[i], m);
    for (size_t i = 0; i < y.size(); ++i) b[i] = to_mont(y[i], m);
    ntt_fwd(out.data(), lg, m);
    ntt_fwd(b.data(), lg, m);
    const u64 ninv = powmod((u64)(n % m.p), m.p - 2, m.p);
    for (size_t i = 0; i < n; ++i) {
        out[i] = mont48(mont48(out[i], b[i], m.p, m.np), ninv, m.p, m.np);
    }
    ntt_inv_noscale(out.data(), lg, m);
}

// ---------------------------------------------------------------------------
// CRT
// ---------------------------------------------------------------------------

inline u128 crt(u64 a0, u64 a1) {
    // a0 < P0 < P1, so no reduction of a0 is needed.
    const u64 d = (a1 + P1 - (a0 % P1)) % P1;
    const u64 t = mulmod(d, BLK_INV_P0_MOD_P1, P1);
    return (u128)a0 + (u128)P0 * t;
}

// ---------------------------------------------------------------------------
// Block convolution, exact, with overlap-add and carry normalisation.
//
// Block size is BL = 2^bl limbs; each transform is length 2*BL.  A block sum
// C_k holds sum_{i+j=k} A_i * B_j at positions k*BL + t, t in [0, 2*BL).
// The first half of C_k and the second half of C_{k-1} are the two
// contributions to output segment k; they are combined before CRT, which is
// valid because the exact coefficient is below the CRT product.
// ---------------------------------------------------------------------------

struct BlockPlan {
    int bl;                 // block log2 (limbs)
    size_t bl_limbs;        // 1 << bl
    int lg;                 // transform log2 = bl + 1
    size_t n;               // 1 << lg
};

inline BlockPlan make_block_plan(int bl) {
    BlockPlan p;
    p.bl = bl;
    p.bl_limbs = (size_t)1 << bl;
    p.lg = bl + 1;
    p.n = (size_t)1 << p.lg;
    return p;
}

// Compute the exact product of two limb vectors using block convolution.
inline std::vector<u32> conv_blocks(const std::vector<u32>& a,
                                    const std::vector<u32>& b,
                                    const BlockPlan& bp,
                                    const Mod& m0, const Mod& m1) {
    const size_t BL = bp.bl_limbs;
    const size_t N = bp.n;
    const size_t na = (a.size() + BL - 1) / BL;
    const size_t nb = (b.size() + BL - 1) / BL;
    const size_t K = na + nb - 1;

    // Forward transforms of every nonzero block, both primes.
    std::vector<std::vector<u64>> ta0(na), tb0(nb), ta1(na), tb1(nb);
    auto build = [&](const std::vector<u32>& src, size_t bi,
                     std::vector<u64>& dst, const Mod& m) {
        dst.assign(N, 0);
        const size_t lo = bi * BL;
        const size_t hi = std::min(lo + BL, src.size());
        for (size_t i = lo; i < hi; ++i) dst[i - lo] = to_mont(src[i], m);
        ntt_fwd(dst.data(), bp.lg, m);
    };
    for (size_t i = 0; i < na; ++i) {
        build(a, i, ta0[i], m0);
        build(a, i, ta1[i], m1);
    }
    for (size_t j = 0; j < nb; ++j) {
        build(b, j, tb0[j], m0);
        build(b, j, tb1[j], m1);
    }

    // Block sums, one k at a time.
    std::vector<u64> c0(N), c1(N);
    std::vector<u64> prev0(BL, 0), prev1(BL, 0);   // second half of k-1
    std::vector<u64> next0(BL, 0), next1(BL, 0);
    std::vector<u32> out;
    out.reserve((K + 1) * BL);
    u128 carry = 0;
    u64 work[2] = {0, 0};

    for (size_t k = 0; k < K; ++k) {
        std::fill(c0.begin(), c0.end(), 0);
        std::fill(c1.begin(), c1.end(), 0);
        const size_t ilo = (k >= nb - 1) ? k - (nb - 1) : 0;
        const size_t ihi = std::min(na - 1, k);
        for (size_t i = ilo; i <= ihi; ++i) {
            const size_t j = k - i;
            for (size_t t = 0; t < N; ++t) {
                u64 p0 = mont48(ta0[i][t], tb0[j][t], m0.p, m0.np);
                u64 p1 = mont48(ta1[i][t], tb1[j][t], m1.p, m1.np);
                c0[t] = add48(c0[t], p0, m0.p);
                c1[t] = add48(c1[t], p1, m1.p);
            }
        }
        ntt_inv(c0.data(), bp.lg, m0);
        ntt_inv(c1.data(), bp.lg, m1);

        // Output segment k: C_k[0..BL) + C_{k-1}[BL..2BL).
        for (size_t t = 0; t < BL; ++t) {
            u64 x0 = c0[t];
            u64 x1 = c1[t];
            if (k > 0) {
                x0 = add48(x0, prev0[t], m0.p);
                x1 = add48(x1, prev1[t], m1.p);
            }
            u128 X = crt(x0, x1);       // exact coefficient (< P)
            u128 total = X + carry;
            out.push_back((u32)(total & 0xFFFFFFFFu));
            carry = total >> 32;
        }
        // Save the second half for the next segment.
        for (size_t t = 0; t < BL; ++t) {
            next0[t] = c0[BL + t];
            next1[t] = c1[BL + t];
        }
        std::swap(prev0, next0);
        std::swap(prev1, next1);
        (void)work;
    }

    // Trailing segment K from the last block sum's second half.
    for (size_t t = 0; t < BL; ++t) {
        u128 X = crt(prev0[t], prev1[t]);
        u128 total = X + carry;
        out.push_back((u32)(total & 0xFFFFFFFFu));
        carry = total >> 32;
    }
    while (carry) {
        out.push_back((u32)(carry & 0xFFFFFFFFu));
        carry >>= 32;
    }

    // Trim to the exact product length and remove high zero limbs.
    size_t want = a.size() + b.size();
    if (want == 0) return {};
    if (out.size() > want) out.resize(want);
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

// Single-transform convolution with CRT and carry normalisation; used for
// operands that fit one transform (2 * max(len) <= 2^lg).
inline std::vector<u32> conv_single_crt(const std::vector<u32>& a,
                                        const std::vector<u32>& b, int lg,
                                        const Mod& m0, const Mod& m1) {
    std::vector<u64> c0, c1;
    conv_single(a, b, lg, m0, c0);
    conv_single(a, b, lg, m1, c1);
    std::vector<u32> out;
    out.reserve(a.size() + b.size());
    u128 carry = 0;
    for (size_t t = 0; t < (size_t)1 << lg; ++t) {
        u128 X = crt(c0[t], c1[t]);
        u128 total = X + carry;
        out.push_back((u32)(total & 0xFFFFFFFFu));
        carry = total >> 32;
    }
    while (carry) {
        out.push_back((u32)(carry & 0xFFFFFFFFu));
        carry >>= 32;
    }
    size_t want = a.size() + b.size();
    if (want == 0) return {};
    if (out.size() > want) out.resize(want);
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

// ---------------------------------------------------------------------------
// Exact big-integer helpers for the reference (base 2^32 limbs).
// ---------------------------------------------------------------------------

inline int cmp_limbs(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

inline std::vector<u32> add_limbs(const std::vector<u32>& a,
                                  const std::vector<u32>& b) {
    std::vector<u32> r(std::max(a.size(), b.size()) + 1, 0);
    u64 carry = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        u64 s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        r[i] = (u32)s;
        carry = s >> 32;
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

inline std::vector<u32> sub_limbs(const std::vector<u32>& a,
                                  const std::vector<u32>& b) {
    // requires a >= b
    std::vector<u32> r(a.size(), 0);
    u64 borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        u64 av = a[i];
        u64 bv = (i < b.size() ? b[i] : 0) + borrow;
        if (av < bv) {
            r[i] = (u32)(av + ((u64)1 << 32) - bv);
            borrow = 1;
        } else {
            r[i] = (u32)(av - bv);
            borrow = 0;
        }
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

inline std::vector<u32> shl1_limbs(const std::vector<u32>& a) {
    std::vector<u32> r(a.size() + 1, 0);
    u32 carry = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        u32 v = a[i];
        r[i] = (v << 1) | carry;
        carry = v >> 31;
    }
    r[a.size()] = carry;
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

inline std::vector<u32> mul_schoolbook(const std::vector<u32>& a,
                                       const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<u32> r(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) {
        u64 carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            u128 cur = (u128)a[i] * b[j] + r[i + j] + carry;
            r[i + j] = (u32)cur;
            carry = (u64)(cur >> 32);
        }
        size_t p = i + b.size();
        while (carry) {
            u128 cur = (u128)r[p] + carry;
            r[p] = (u32)cur;
            carry = (u64)(cur >> 32);
            ++p;
        }
    }
    while (!r.empty() && r.back() == 0) r.pop_back();
    return r;
}

}  // namespace blk

#endif  // BLOCK_ARITH_REF_H
