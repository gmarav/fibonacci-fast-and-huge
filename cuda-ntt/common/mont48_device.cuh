// Shared 48-bit Montgomery arithmetic for the two-prime CUDA engines.
//
// Device-only inline primitives.  The layout and operation order are identical
// in block-28 and block-29; keeping one definition prevents the two copies
// from drifting apart.  Hot NTT kernels remain engine-specific.
#pragma once

#include <cstdint>

struct ModArgs {
    uint64_t p, np, R, R2;
};

__device__ __forceinline__ uint64_t dmul(uint64_t a, uint64_t b, ModArgs m) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    uint64_t mm = lo * m.np;
    uint64_t mp_lo = mm * m.p;
    uint64_t mp_hi = __umul64hi(mm, m.p);
    uint64_t s = lo + mp_lo;
    uint64_t carry = (s < lo) ? 1ull : 0ull;
    uint64_t t = hi + mp_hi + carry;
    if (t >= m.p) t -= m.p;
    return t;
}

__device__ __forceinline__ uint64_t dadd(uint64_t a, uint64_t b, ModArgs m) {
    uint64_t s = a + b;
    return s >= m.p ? s - m.p : s;
}

__device__ __forceinline__ uint64_t dsub(uint64_t a, uint64_t b, ModArgs m) {
    return a >= b ? a - b : a + (m.p - b);
}

/// base is Montgomery form; result is `base^e * R mod p`.
__device__ __forceinline__ uint64_t dpow_mont(uint64_t base, uint64_t e,
                                             ModArgs m) {
    uint64_t r = m.R;
    while (e) {
        if (e & 1ull) r = dmul(r, base, m);
        base = dmul(base, base, m);
        e >>= 1;
    }
    return r;
}

/// `a*b mod p` without needing a device `__int128`.
__device__ __forceinline__ uint64_t dmulmod(uint64_t a, uint64_t b, ModArgs m) {
    return dmul(dmul(a, b, m), m.R2, m);
}
