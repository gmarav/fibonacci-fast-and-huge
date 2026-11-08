// Pure host-side planner and memory model for the 2^28 block engine.
//
// No CUDA and no compiler intrinsics, so this header is unit-testable with any
// C++17 compiler.  It is used by the GPU driver (`fib.cpp`) before every
// allocation and by `plan_test.cpp` for the readiness/memory table.

#ifndef BLOCK28_PLAN_MODEL_H
#define BLOCK28_PLAN_MODEL_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "block28_constants.h"

namespace blk28plan {

// Deliberately conservative: log2(phi) < 7/10.  The true value is
// 0.6942419136..., so this overestimates by ~0.8% and never under-sizes an
// operand.  No floating point is used.
static const size_t kLog2PhiNum = 7;
static const size_t kLog2PhiDen = 10;

static const int kMaxLog2 = BLK28_MAX_LOG2;   // 28
static const int kBlockLog2 = BLK28_BLOCK_LOG2;  // 27

struct Plan {
    int lg;          // transform log2
    size_t B;        // block length in u32 limbs
    size_t N;        // transform length = 2*B
    int na, nb;      // operand block counts
    int K;           // number of block sums = na+nb-1 (0 if empty)
    size_t need;     // la+lb
    bool block_mode; // true when B == 2^27
};

static inline int ceillog2(size_t x) {
    int lg = 0;
    size_t v = 1;
    while (v < x) {
        v <<= 1;
        ++lg;
    }
    return lg;
}

// Upper bound on the limb count of the largest fast-doubling operand
// (F(floor(n/2)) and F(floor(n/2)+1)).  Never an underestimate.
static inline size_t operand_limb_bound(uint64_t n) {
    if (n == 0) return 0;
    uint64_t half = n / 2;
    uint64_t bits = (half / kLog2PhiDen) * kLog2PhiNum +
                    ((half % kLog2PhiDen) * kLog2PhiNum) / kLog2PhiDen + 2;
    return (size_t)(bits / 32) + 2;
}

// Smallest transform that holds la+lb, capped at 2^28 with B = 2^27.
static inline Plan plan_for(size_t la, size_t lb) {
    Plan p{};
    p.need = la + lb;
    p.na = p.nb = 0;
    p.K = 0;
    if (p.need == 0) {
        p.lg = 1;
        p.B = 1;
        p.N = 2;
        p.block_mode = false;
        return p;
    }
    if (p.need <= ((size_t)1 << kMaxLog2)) {
        p.lg = ceillog2(p.need);
        if (p.lg < 1) p.lg = 1;
        p.B = (size_t)1 << (p.lg - 1);
        p.N = (size_t)1 << p.lg;
        p.block_mode = (p.lg == kMaxLog2);
    } else {
        p.lg = kMaxLog2;
        p.B = (size_t)1 << kBlockLog2;
        p.N = (size_t)1 << kMaxLog2;
        p.block_mode = true;
    }
    p.na = (int)((la + p.B - 1) / p.B);
    p.nb = (int)((lb + p.B - 1) / p.B);
    p.K = p.na + p.nb - 1;
    if (p.K < 0) p.K = 0;
    return p;
}

// Hard structural limit: the engine has three transform arrays and two
// staging buffers; each operand may span at most `max_blocks` blocks.
static inline bool plan_supported(const Plan& p, int max_blocks) {
    return p.na <= max_blocks && p.nb <= max_blocks;
}

struct MemEstimate {
    size_t device_arrays;    // 3 u64 transform arrays
    size_t device_staging;   // a, b staging buffers, max_blocks each
    size_t device_twiddles;
    size_t device_context;   // measured ~1.31 GiB CUDA context
    size_t host_operands;    // a, b, t=2b-a
    size_t host_download;    // low/high per prime (4 x B u64)
    size_t host_segwork;     // d64 carry scratch
    size_t host_output;      // one exact product
};

static inline MemEstimate estimate(const Plan& p, size_t la, size_t lb) {
    MemEstimate e{};
    e.device_arrays = (size_t)3 * p.N * 8;
    // The driver allocates exactly the block count of the largest step.
    size_t blocks = (size_t)(p.na > p.nb ? p.na : p.nb);
    if (blocks < 1) blocks = 1;
    const size_t cap = blocks * p.B;
    e.device_staging = 2 * cap * 4;
    // Both primes, forward+inverse: rootR + 256-entry tiles + tail tables.
    e.device_twiddles = (size_t)2 * (2 * (kMaxLog2 + 1) * 8 +
                                     2 * (kMaxLog2 + 1) * 256 * 8 +
                                     2 * 4096 * 8);
    e.device_context = (size_t)(1.31 * 1024 * 1024 * 1024);
    const size_t lt = la + 1;
    e.host_operands = (la + lb + lt) * 4;
    e.host_download = 4 * p.B * 8;
    e.host_segwork = p.B * 8;
    e.host_output = p.need * 4;
    return e;
}

static inline size_t device_total(const MemEstimate& e) {
    return e.device_arrays + e.device_staging + e.device_twiddles +
           e.device_context;
}
static inline size_t host_total(const MemEstimate& e) {
    return e.host_operands + e.host_download + e.host_segwork + e.host_output;
}

static inline double gib(size_t b) {
    return (double)b / (1024.0 * 1024.0 * 1024.0);
}

// Conservative index ceiling for a device budget, using the planner's own
// limb bound and max_blocks.  Returns 0 if even n=1 does not fit.
static inline uint64_t max_index_for_device(size_t budget_bytes, int max_blocks) {
    uint64_t best = 0;
    // 64-bit n; the operand bound is monotone, so a coarse scan then refine.
    for (uint64_t n = 1000000ull; n < 1000000000000ull; n = n * 101 / 100) {
        size_t la = operand_limb_bound(n);
        Plan p = plan_for(la, la + 1);
        if (!plan_supported(p, max_blocks)) break;
        MemEstimate e = estimate(p, la, la + 1);
        if (device_total(e) > budget_bytes) break;
        best = n;
    }
    return best;
}

// Same for a host-RAM budget.
static inline uint64_t max_index_for_host(size_t budget_bytes, int max_blocks) {
    uint64_t best = 0;
    for (uint64_t n = 1000000ull; n < 1000000000000ull; n = n * 101 / 100) {
        size_t la = operand_limb_bound(n);
        Plan p = plan_for(la, la + 1);
        if (!plan_supported(p, max_blocks)) break;
        MemEstimate e = estimate(p, la, la + 1);
        if (host_total(e) > budget_bytes) break;
        best = n;
    }
    return best;
}

// Coefficient-wise exactness ceiling: the full product coefficient is at most
// (la+lb)*(2^32-1)^2 and must stay below P = P0*P1.  Returns the largest n
// under the planner's conservative limb bound.
static inline uint64_t max_index_for_coefficient_bound() {
    uint64_t best = 0;
    for (uint64_t n = 1000000ull; n < 1000000000000ull; n = n * 1001 / 1000) {
        size_t la = operand_limb_bound(n);
        if (2 * la > BLK28_MAX_LIMBS) break;
        best = n;
    }
    return best;
}

}  // namespace blk28plan

#endif  // BLOCK28_PLAN_MODEL_H
