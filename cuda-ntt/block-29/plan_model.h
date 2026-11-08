// Pure host-side planner and memory model for the 2^29 block engine.
// No CUDA, no compiler intrinsics: this header is unit-testable with any C++17
// compiler and is used by both the driver and the readiness report.

#ifndef BLOCK_PLAN_MODEL_H
#define BLOCK_PLAN_MODEL_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace blkplan {

// A deliberately conservative bound: log2(phi) < 7/10.  The true value is
// 0.6942419136..., so this overestimates by 0.8% and can never under-size a
// transform.  No floating point is used.
static const size_t kLog2PhiNum = 7;
static const size_t kLog2PhiDen = 10;

struct Plan {
    int lg;             // transform log2
    size_t B;           // block length in limbs
    size_t N;           // transform length
    bool block_mode;    // true if BL=2^28 and lg=29
    int na, nb;         // operand blocks
    int K;              // block sums = na+nb-1
    size_t need;        // la+lb
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

// Upper bound on the limb count of the largest doubling operand, F(n/2) and
// F(n/2+1).  Uses 0.6942419136 > log2(phi) plus guard limbs, so it is never
// an underestimate.  Returns 0 for n == 0.
static inline size_t operand_limb_bound(uint64_t n) {
    if (n == 0) return 0;
    // bits(F(n/2)) < (n/2) * 7/10 + 1; half < 2^36 for the target range, so
    // the multiplication stays inside uint64_t.
    uint64_t half = n / 2;
    uint64_t bits = (half / kLog2PhiDen) * kLog2PhiNum +
                    ((half % kLog2PhiDen) * kLog2PhiNum) / kLog2PhiDen + 2;
    return (size_t)(bits / 32) + 2;
}

static inline Plan plan_for(size_t la, size_t lb) {
    Plan p{};
    p.need = la + lb;
    if (p.need == 0) {
        p.lg = 1;
        p.B = 1;
        p.N = 2;
        p.block_mode = false;
        p.na = p.nb = 0;
        p.K = 0;
        return p;
    }
    size_t product = p.need;
    if (product <= ((size_t)1 << 29)) {
        p.lg = ceillog2(product);
        if (p.lg < 1) p.lg = 1;
        p.B = (size_t)1 << (p.lg - 1);
        p.N = (size_t)1 << p.lg;
        p.block_mode = false;
    } else {
        p.lg = 29;
        p.B = (size_t)1 << 28;
        p.N = (size_t)1 << 29;
        p.block_mode = true;
    }
    p.na = (int)((la + p.B - 1) / p.B);
    p.nb = (int)((lb + p.B - 1) / p.B);
    if (p.na < 1) p.na = 0;
    if (p.nb < 1) p.nb = 0;
    p.K = p.na + p.nb - 1;
    if (p.K < 0) p.K = 0;
    return p;
}

static inline bool plan_supported(const Plan& p) {
    // The device engine has three resident left-operand buffers (0..2), one
    // right-operand workspace (3) and one accumulator (4), so each operand
    // may use at most three 2^28 blocks.  A 6-block (3+3) product is the
    // largest supported plan.
    if (p.na > 3 || p.nb > 3) return false;
    return (size_t)p.na + (size_t)p.nb <= 6;
}

struct MemEstimate {
    size_t device_arrays;
    size_t device_stage;
    size_t device_twiddles;
    size_t device_context;   // measured 1.30 GiB overhead
    size_t host_operands;
    size_t host_seg0;        // packed 6-byte prime-0 segments
    size_t host_working;     // seg1 + download + d
    size_t host_output;      // one exact product
};

static inline MemEstimate estimate(const Plan& p, size_t la, size_t lb) {
    MemEstimate e{};
    e.device_arrays = (size_t)5 * p.N * 8;
    size_t stage_cap = p.N / 2;
    if (stage_cap > ((size_t)1 << 28)) stage_cap = (size_t)1 << 28;
    e.device_stage = stage_cap * 4;
    e.device_twiddles = 2 * 30 * 8 + 2 * 30 * 256 * 8 + 4 * 4096 * 8;
    e.device_context = (size_t)(1.30 * 1024 * 1024 * 1024);
    e.host_operands = (la + lb) * 4;
    e.host_seg0 = (size_t)(p.K + 1) * p.B * 6;
    e.host_working = p.B * 8 + p.N * 8 + p.B * 8;
    e.host_output = p.need * 4;
    return e;
}

static inline double gib(size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); }

}  // namespace blkplan

#endif  // BLOCK_PLAN_MODEL_H
