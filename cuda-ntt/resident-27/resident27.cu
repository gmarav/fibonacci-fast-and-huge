// resident-27: GPU-resident exact Fibonacci fast doubling on the ntt-alt
// transform core.
//
// This is a merge of two working implementations:
//
//   * cuda-ntt/ntt-alt  -- the faster exact 3-prime NTT multiplication core:
//     Montgomery 32-bit arithmetic, DIF/DIT with no bit-reversal pass,
//     per-stage contiguous twiddle tables, a shared-memory tail kernel for the
//     last 12 stages of every transform, Shoup/Garner CRT, two-level carry.
//   * cuda-ntt/resident -- the fused doubling loop that keeps the whole
//     (a,b) state and all workspaces on the device for the entire bit loop,
//     and uses 5 NTT transforms per prime per step instead of 9.
//
// The doubling algebra (unchanged, exact):
//
//   t  = 2b - a                     (F(k+1) + F(k-1), nonnegative)
//   c  = a * t                      (F(2k))
//   d  = a*a + b*b                  (F(2k+1))
//   bit == 0: (a,b) = (c,d)
//   bit == 1: (a,b) = (d, c+d)
//
//   NTT(a*a + b*b) = NTT(a)^2 + NTT(b)^2  is valid pointwise because every
//   coefficient is nonnegative, so d needs one inverse transform, not two.
//   t is materialised with an exact borrow chain and transformed (the limb
//   vector of t is not 2*b_vec - a_vec), so a step is
//
//       3 forward + 2 inverse = 5 transforms per prime
//
//   All 3 scans of one step are fused into one launch (blockIdx.y), and the
//   transform length is re-planned per bit, so the cost follows the
//   O(N log N) fast-doubling schedule rather than paying for one fixed
//   maximum-size transform at every bit.
//
// What changed relative to cuda-ntt/resident:
//
//   * primes: 2281701377, 3489660929, 3892314113 (ascending, all > 2^31).
//     Their 2-adicities are 27, 28, 27, so all three support a 2^27
//     transform.  P = p0*p1*p2 ~= 2^94.6459 and exceeds the 2^27-limb
//     coefficient bound by 12.52x.  Every modulus is above 2^31, which is why
//     the Montgomery product, the Shoup product and the butterfly sum use the
//     widened 32-bit forms (see below).
//   * forward/inverse: DIF/DIT radix-2 with no bit-reversal pass.  A transform
//     is (lg - 12) global stage launches plus one shared-memory tail launch
//     that runs the last 12 stages of each 4096-element tile in place,
//     instead of lg global passes.  At n = 2^20 that is 9 passes instead of
//     20 per transform.
//   * twiddles: per-stage contiguous tables (stage `half` owns
//     [cap-2*half, cap-half)), so every butterfly reads stride-1 twiddles.
//     The table for a stage depends only on `half`, not on the transform
//     length, so one capacity-sized table serves every n <= cap.
//   * CRT: alt's Shoup/Garner reconstruction, writing the redundant digit
//     array T[i] = w0[i] + w1[i-1] + w2[i-2] directly (no lo/mid/hi
//     scratch).  c+d is formed as T_c + T_d.
//   * carry: alt's exact two-level block-transition scan.  A chunk of 2048
//     u64 digits is summarised by its exact carry-out function for carry-in
//     0..5 (c+d digits are < 6*2^32), the block summaries are composed by a
//     single serial scan, and a replay pass writes the limbs.  This is exact
//     for any data; it does not assume carries are rare.
//   * memory: the six u32 lo/mid/hi buffers and three u64 acc buffers of the
//     old core are replaced by three u64 T buffers, so the post-transform
//     working set is smaller.
//   * fusion (adopted from cuda-ntt/radix4, which measured the same ideas):
//     the residue reduction is folded into the first forward pass (the
//     borrow chain no longer writes nine residue buffers), and the pointwise
//     product is folded into the inverse tail's tile load (c overwrites the
//     dead T-hat buffers, d the dead B-hat buffers; A-hat is read-only).
//     The radix-16 register/tile transform of radix4 was measured and not
//     adopted: its ~270 GB/s tile pass is slower than the ntt-alt-style
//     global stages plus 4096-element tail used here (~1.6 TB/s effective).
//
// No floating point is used anywhere.

#include "resident27.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;

#define MAX_NTT_LOG2 27
#define MAX_STEPS 80

// ---------------------------------------------------------------------------
// Error plumbing
// ---------------------------------------------------------------------------

static char g_last_error[512];

static bool set_error_text(const char* what, cudaError_t err) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", what,
                  cudaGetErrorString(err));
    std::fprintf(stderr, "resident27: %s\n", g_last_error);
    return false;
}

static bool set_error_msg(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
    std::fprintf(stderr, "resident27: %s\n", g_last_error);
    return false;
}

#define CUDA_TRY(call)                                                         \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) return set_error_text(#call, err_);           \
    } while (0)

#define CUDA_TRY_I(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            set_error_text(#call, err_);                                       \
            return 3;                                                          \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device arithmetic (copied from cuda-ntt/ntt-alt, widened for p > 2^31)
//
//   p0 = 2281701377 = 17 * 2^27 + 1, root 3
//   p1 = 3489660929 = 13 * 2^28 + 1, root 3
//   p2 = 3892314113 = 29 * 2^27 + 1, root 3
//   P  = 30992022342066952614946275329 ~= 2^94.6459
//
// Below 2^31 there is exactly one prime with 2-adicity >= 27 (2013265921),
// so a 2^27 transform is impossible with all moduli below 2^31.  These three
// all lie above 2^31, and every routine below is written for the full
// 32-bit-wide range:
//
//   * dmul:    t + m*p can exceed 2^64 and (t + m*p)>>32 can exceed 2^32, so
//              the 33rd bit is kept explicitly and the result is reduced in
//              u64 before truncation.  A compile-time assertion rejects any
//              modulus >= 2^32.
//   * dshoup:  q*p must be subtracted in u64; the remainder is < 2p < 2^33.
//   * dadd:    u + v can exceed 2^32, so the sum is formed in u64 and reduced
//              once (u + v < 2p < 2^33).  Subtraction never overflows a u32
//              for p < 2^32 and is left as is.
//
// The primes are stored in ASCENDING order.  The Garner reconstruction below
// subtracts a0 (a residue modulo p0) from a1 and a2 inside a single
// conditional correction; that is only valid when p0 is the smallest of the
// three primes.  Ordering the table differently silently breaks the CRT.
//
// All three primes are above 2^31, so a raw u32 limb is already < 2^32 < 2p:
// the raw-limb reduction needs exactly one conditional subtraction.
// ---------------------------------------------------------------------------

constexpr int kNumPrimes = 3;
constexpr u32 kP[kNumPrimes] = {2281701377u, 3489660929u, 3892314113u};
constexpr u32 kRoot[kNumPrimes] = {3u, 3u, 3u};

static_assert((u64)kP[0] < (u64)kP[1] && (u64)kP[0] < (u64)kP[2],
              "Garner requires p0 to be the smallest prime");
static_assert((u64)kP[1] < (u64)kP[2], "prime table must be ascending");
static_assert((u64)kP[2] < (1ull << 32),
              "the widened 32-bit Montgomery path requires p < 2^32");
static_assert((u64)kP[0] > (1ull << 31) && (u64)kP[1] > (1ull << 31) &&
                  (u64)kP[2] > (1ull << 31),
              "this set exercises the all-wide >2^31 path");
static_assert(0xffffffffull / (u64)kP[0] <= 1ull,
              "dred must provide enough conditional subtractions for p0");
static_assert((u64)kP[0] - 1 == (17ull << 27), "p0 must be 17*2^27+1");
static_assert((u64)kP[1] - 1 == (13ull << 28), "p1 must be 13*2^28+1");
static_assert((u64)kP[2] - 1 == (29ull << 27), "p2 must be 29*2^27+1");

// Shared-memory tile used by the transform tail kernel.
constexpr int kTile = 4096;
constexpr int kStageThreads = 256;

// CRT block size (one thread per coefficient).
constexpr int kCrtBlock = 512;

// Carry propagation: 2048 digits per block, 8 per thread, 256 threads.
constexpr int kCarryBlock = 2048;
constexpr int kCarryThreads = 256;
constexpr int kCarryPer = kCarryBlock / kCarryThreads;
// Maximum digit value is 6*(2^32-1), so a carry is in 0..5 (6 states).

// 32x32 -> 64 Montgomery product for any p < 2^32.
//
// t + m*p is divisible by 2^32 but can be as large as 2^64 + 2^32*p, so the
// high half is 33 bits wide.  The carry out of the u64 sum is materialised as
// bit 32 of `u`; the true quotient is < 2p, so exactly one conditional
// subtraction normalises it.  For p < 2^31 the carry is always 0 and the
// compiler folds this back into the narrow form.
static_assert((u64)kP[2] < (1ull << 32), "dmul requires p < 2^32");

__device__ __forceinline__ u32 dmul(u32 a, u32 b, u32 p, u32 np) {
    u64 t = (u64)a * (u64)b;
    u32 m = (u32)t * np;
    u64 s = t + (u64)m * (u64)p;
    u64 carry = (s < t) ? 1ull : 0ull;
    u64 u = (s >> 32) | (carry << 32);
    if (u >= p) u -= p;
    return (u32)u;
}

// Modular addition for p < 2^32: u + v can exceed 2^32, so the sum is formed
// in u64 and reduced once (u + v < 2p < 2^33).
__device__ __forceinline__ u32 dadd(u32 a, u32 b, u32 p) {
    u64 s = (u64)a + (u64)b;
    if (s >= p) s -= p;
    return (u32)s;
}

// v < 2^32.  Every prime in this set is above 2^31, so a raw limb is already
// below 2p: one conditional subtraction is the exact reduction.  (The 2^26
// experiment needed nine because its p0 was 469762049 < 2^30; that failure
// mode cannot occur here and the legacy diagnostic hook is not carried over.)
__device__ __forceinline__ u32 dred(u32 v, u32 p) {
    v -= (v >= p) ? p : 0u;
    return v;
}

// Shoup multiplication by a fixed constant w (ws = floor(w * 2^32 / p)).
// For p > 2^31 the remainder w*x - q*p is below 2p < 2^33, so the subtraction
// must be done in u64; the 32-bit form silently drops the 33rd bit.
__device__ __forceinline__ u32 dshoup(u32 x, u32 w, u32 ws, u32 p) {
    u32 q = (u32)(((u64)ws * x) >> 32);
    u64 r = (u64)w * (u64)x - (u64)q * (u64)p;
    if (r >= p) r -= p;
    return (u32)r;
}

__device__ __forceinline__ u32 dlog2(u32 v) {
    return 31u - (u32)__clz(v);
}

// ---------------------------------------------------------------------------
// Prime metadata and host-side integer helpers (no 128-bit type is needed)
// ---------------------------------------------------------------------------

struct PrimeInfo {
    u32 p;
    u32 np; // -p^-1 mod 2^32
    u32 r;  // 2^32 mod p
    u32 r2; // 2^64 mod p
};

static PrimeInfo g_prime[kNumPrimes];
static bool g_prime_ready = false;

static u32 hpowmod(u32 b, u64 e, u32 p) {
    u64 r = 1, bb = (u64)b % p;
    while (e) {
        if (e & 1ull) r = (r * bb) % p;
        bb = (bb * bb) % p;
        e >>= 1;
    }
    return (u32)r;
}

static u32 host_mont_np(u32 p) {
    u32 inv = 1u;
    for (int i = 0; i < 5; ++i) inv *= 2u - p * inv;
    return (u32)(0u - inv);
}

static u32 host_mont_mul(u32 a, u32 b, u32 p, u32 np) {
    u64 t = (u64)a * b;
    u32 m = (u32)t * np;
    u64 s = t + (u64)m * p;
    u64 carry = (s < t) ? 1ull : 0ull;
    u64 u = (s >> 32) | (carry << 32);
    if (u >= p) u -= p;
    return (u32)u;
}

struct CrtArgs {
    u32 P0, P1, P2;
    u32 np0, np1, np2;
    u32 inv01;    // p0^-1 mod p1
    u32 inv01_sh; // floor(inv01 * 2^32 / p1)
    u32 p0m2;     // p0 mod p2
    u32 p0m2_sh;  // floor(p0m2 * 2^32 / p2)
    u32 invp01;   // (p0*p1)^-1 mod p2
    u32 invp01_sh;
};

static CrtArgs g_crt;

// Host-side guard diagnostics: 2-adicity, primitive roots, the ascending CRT
// ordering, and the exact 3-prime product against the 2^MAX_NTT_LOG2
// coefficient bound.
// This never runs on the arithmetic path and never uses floating point.
static int ctz_u64(u64 v) {
    int n = 0;
    while (v && (v & 1ull) == 0) {
        v >>= 1;
        ++n;
    }
    return n;
}

static bool verify_prime_set() {
    bool ok = true;
    for (int i = 0; i < kNumPrimes; ++i) {
        const u64 p = kP[i];
        const int v2 = ctz_u64(p - 1);
        if (v2 < MAX_NTT_LOG2) {
            std::fprintf(stderr,
                         "resident27: prime %u supports only 2^%d (need 2^%d)\n",
                         kP[i], v2, MAX_NTT_LOG2);
            ok = false;
        }
        // Trial-divide p-1; sqrt(2^32) < 65536 iterations, host-only.
        u64 m = p - 1;
        u64 factors[16];
        int nf = 0;
        for (u64 d = 2; d * d <= m; ++d) {
            if (m % d == 0) {
                if (nf < 16) factors[nf++] = d;
                while (m % d == 0) m /= d;
            }
        }
        if (m > 1 && nf < 16) factors[nf++] = m;
        for (int j = 0; j < nf; ++j) {
            if (hpowmod(kRoot[i], (p - 1) / factors[j], (u32)p) == 1u) {
                std::fprintf(stderr,
                             "resident27: %u is not a primitive root of %u "
                             "(fails at factor %llu)\n",
                             kRoot[i], kP[i],
                             (unsigned long long)factors[j]);
                ok = false;
            }
        }
    }

    // P = p0*p1*p2 as three base-2^32 words.
    const u64 lo = (u64)kP[0] * kP[1]; // p0,p1 < 2^32 -> < 2^64
    const u64 m0 = (lo & 0xffffffffull) * kP[2];
    const u64 m1 = (lo >> 32) * kP[2];
    const u64 mid = (m0 >> 32) + (m1 & 0xffffffffull);
    const u64 w0 = m0 & 0xffffffffull;
    const u64 w1 = mid & 0xffffffffull;
    const u64 w2 = (m1 >> 32) + (mid >> 32);

    // Exact coefficient bound for the tightest 2^MAX_NTT_LOG2 step.  The
    // scheduler guarantees operands of at most 2^(MAX_NTT_LOG2-1) - 1 limbs,
    // so every convolution coefficient of d = a*a + b*b is below
    //   bound = 2 * (2^(MAX_NTT_LOG2-1) - 1) * (2^32 - 1)^2.
    // Compare P and bound as three base-2^32 words: no floating point and no
    // truncation, and no fourth prime is silently assumed.
    const u64 mlimbs = ((u64)1 << (MAX_NTT_LOG2 - 1)) - 1ull;
    const u64 k2 = 2ull * mlimbs;
    const u64 bsq = 0xfffffffe00000001ull; // (2^32 - 1)^2 = 2^64 - 2^33 + 1
    const u64 q0 = (bsq & 0xffffffffull) * k2;
    const u64 q1 = (bsq >> 32) * k2;
    const u64 qmid = (q0 >> 32) + (q1 & 0xffffffffull);
    const u64 c0 = q0 & 0xffffffffull;
    const u64 c1 = qmid & 0xffffffffull;
    const u64 c2 = (q1 >> 32) + (qmid >> 32);
    // Two independent tests: the power-of-two bound required by the spec
    // (P > 2^(MAX_NTT_LOG2+64), i.e. word 2 above 2^MAX_NTT_LOG2) and the
    // exact coefficient bound above.
#if defined(_MSC_VER)
    // kP is a compile-time constant, so /analyze folds these bounds and
    // reports C6285/C6326.  The checks are intentional and must stay.
#pragma warning(suppress : 6285 6326)
#endif
    const bool product_pow2_ok =
        (w2 > (1ull << MAX_NTT_LOG2)) ||
        (w2 == (1ull << MAX_NTT_LOG2) && (w1 != 0 || w0 != 0));
#if defined(_MSC_VER)
#pragma warning(suppress : 6285 6326)
#endif
    const bool product_ok =
        (w2 > c2) || (w2 == c2 && (w1 > c1 || (w1 == c1 && w0 > c0)));
#if defined(_MSC_VER)
#pragma warning(suppress : 6285 6326)
#endif
    if (!product_pow2_ok || !product_ok) {
        std::fprintf(stderr,
                     "resident27: 3-prime product is too small for a 2^%d "
                     "transform (P > 2^%d: %s, P > bound: %s)\n",
                     MAX_NTT_LOG2, MAX_NTT_LOG2 + 64,
                     product_pow2_ok ? "yes" : "no",
                     product_ok ? "yes" : "no");
        ok = false;
    }
    const char* diag = std::getenv("RESIDENT27_DIAG");
    if (diag && *diag && *diag != '0') {
        std::fprintf(stderr,
                     "resident27: primes %u, %u, %u; roots %u, %u, %u; "
                     "P words [%llu, %llu, %llu]; bound words "
                     "[%llu, %llu, %llu]; P > 2^%d: %s; P > bound: %s\n",
                     kP[0], kP[1], kP[2], kRoot[0], kRoot[1], kRoot[2],
                     (unsigned long long)w0, (unsigned long long)w1,
                     (unsigned long long)w2, (unsigned long long)c0,
                     (unsigned long long)c1, (unsigned long long)c2,
                     MAX_NTT_LOG2 + 64, product_pow2_ok ? "yes" : "no",
                     product_ok ? "yes" : "no");
    }
    return ok;
}

static bool init_primes() {
    if (g_prime_ready) return true;
    if (!verify_prime_set()) return false;
    for (int i = 0; i < kNumPrimes; ++i) {
        u32 p = kP[i];
        g_prime[i].p = p;
        g_prime[i].np = host_mont_np(p);
        g_prime[i].r = (u32)((1ull << 32) % p);
        g_prime[i].r2 = (u32)(((u64)g_prime[i].r * g_prime[i].r) % p);
    }

    const u32 p0 = kP[0], p1 = kP[1], p2 = kP[2];
    g_crt.P0 = p0;
    g_crt.P1 = p1;
    g_crt.P2 = p2;
    g_crt.np0 = g_prime[0].np;
    g_crt.np1 = g_prime[1].np;
    g_crt.np2 = g_prime[2].np;

    u32 inv01 = hpowmod(p0 % p1, (u64)p1 - 2, p1);
    g_crt.inv01 = inv01;
    g_crt.inv01_sh = (u32)(((u64)inv01 << 32) / p1);

    u32 p0m2 = p0 % p2;
    g_crt.p0m2 = p0m2;
    g_crt.p0m2_sh = (u32)(((u64)p0m2 << 32) / p2);

    u64 p01 = (u64)p0 * p1; // < 2^62
    u32 p01m2 = (u32)(p01 % p2);
    u32 invp01 = hpowmod(p01m2, (u64)p2 - 2, p2);
    g_crt.invp01 = invp01;
    g_crt.invp01_sh = (u32)(((u64)invp01 << 32) / p2);

    g_prime_ready = true;
    return true;
}

// ---------------------------------------------------------------------------
// Per-step descriptor (device memory)
//
// Only the buffers that change identity between steps live here.  Every other
// kernel argument is constant for a given transform length, which lets one
// CUDA graph per length be reused for every step of that length.
// ---------------------------------------------------------------------------

struct StepDesc {
    const u32* a;
    const u32* b;
    u32 len_a;
    u32 len_b;
    u32 len_t;
    u32 L;        // number of T entries (limbs) to produce
    u32 bound[3]; // Binet limb upper bounds for c, d, c+d
};

// ---------------------------------------------------------------------------
// Transforms
//
// Twiddle table layout (same as ntt-alt): stage `half` owns
// [cap - 2*half, cap - half).  Because the twiddle values of a DIF/DIT stage
// depend only on `half` (W = g^((p-1)/(2*half))) and not on the transform
// length, the full-capacity table is a valid suffix-addressed table for any
// n <= cap.  This is what makes one graph per lg enough.
// ---------------------------------------------------------------------------

struct NttArgs {
    u32* data[9];     // forward: A0,A1,A2,B0,B1,B2,T0,T1,T2
                      // inverse: C0,C1,C2,D0,D1,D2
    const u32* tw[3];
    u32 p[3];
    u32 np[3];
    u32 n;
    u32 cap;
    u32 half;
    u32 off;
    u32 lh;
    u32 tile;
    u32 start_half;
    // Residue reduction folded into the first forward pass: when is_first is
    // set, the stage reads the raw limbs named by the per-step descriptor and
    // reduces them mod p instead of reading a pre-filled residue buffer.
    // Reading the descriptor on the device keeps a per-lg graph reusable for
    // calls with different operand lengths (the step ABI).
    const StepDesc* d;
    const u32* raw_t;
    int is_first;
    // Fused pointwise inverse tail only.
    const u32* pw_a[3]; // A-hat (never overwritten)
    const u32* pw_t[3]; // T-hat, read by the c product
    const u32* pw_b[3]; // B-hat, read by the d product
    u32 scale[3];       // 2^64 * n^-1 mod p
};

// Exact reduction of one raw limb for the first forward pass.
__device__ __forceinline__ u32 load_raw(const NttArgs& arg, int b, u32 pos,
                                        u32 p) {
    const int which = b / 3;
    const u32* src = which == 0 ? arg.d->a
                                : (which == 1 ? arg.d->b : arg.raw_t);
    const u32 len = which == 0 ? arg.d->len_a
                               : (which == 1 ? arg.d->len_b : arg.d->len_t);
    const u32 x = (pos < len) ? __ldg(src + pos) : 0u;
    return dred(x, p);
}

// One global radix-2 stage over all operands/primes in blockIdx.y.
// INV=0: DIF butterfly (natural -> bit-reversed).  INV=1: DIT butterfly.
template <int INV>
__global__ void k_stage_multi(NttArgs arg) {
    const int b = blockIdx.y;
    const int pr = b % 3;
    u32* a = arg.data[b];
    const u32* w = arg.tw[pr];
    const u32 p = arg.p[pr];
    const u32 np = arg.np[pr];

    u32 k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (arg.n >> 1)) return;
    const u32 half = arg.half;
    const u32 j = k & (half - 1u);
    const u32 grp = k >> arg.lh;
    const u32 i0 = (grp << (arg.lh + 1u)) | j;
    const u32 i1 = i0 + half;
    const u32 u = arg.is_first ? load_raw(arg, b, i0, p) : a[i0];
    const u32 v = arg.is_first ? load_raw(arg, b, i1, p) : a[i1];
    const u32 wt = w[arg.off + j];
    if (INV) {
        u32 vw = dmul(v, wt, p, np);
        u32 s = dadd(u, vw, p);
        u32 d = u - vw;
        if (u < vw) d += p;
        a[i0] = s;
        a[i1] = d;
    } else {
        u32 s = dadd(u, v, p);
        u32 d = u - v;
        if (u < v) d += p;
        a[i0] = s;
        a[i1] = dmul(d, wt, p, np);
    }
}

// Last stages of every transform, executed inside one 4096-element shared
// tile per block ("one block per tile").
// INV=0 walks half = start_half .. 1 (DIF), INV=1 walks 1 .. start_half (DIT).
template <int INV>
__global__ void k_tail_multi(NttArgs arg) {
    const int b = blockIdx.y;
    const int pr = b % 3;
    u32* a = arg.data[b];
    const u32* w = arg.tw[pr];
    const u32 p = arg.p[pr];
    const u32 np = arg.np[pr];
    const u32 tile = arg.tile;

    extern __shared__ u32 sh[];
    u32* s = sh;
    u32* sw = sh + tile;
    const u32 base = blockIdx.x * tile;
    if (arg.is_first) {
        for (u32 i = threadIdx.x; i < tile; i += blockDim.x)
            s[i] = load_raw(arg, b, base + i, p);
    } else {
        for (u32 i = threadIdx.x; i < tile; i += blockDim.x) s[i] = a[base + i];
    }
    __syncthreads();

    if (!INV) {
        for (u32 half = arg.start_half; half >= 1u; half >>= 1) {
            const u32 lh = dlog2(half);
            const u32 off = arg.cap - (half << 1);
            for (u32 j = threadIdx.x; j < half; j += blockDim.x)
                sw[j] = w[off + j];
            __syncthreads();
            for (u32 k = threadIdx.x; k < (tile >> 1); k += blockDim.x) {
                const u32 j = k & (half - 1u);
                const u32 grp = k >> lh;
                const u32 i0 = (grp << (lh + 1u)) | j;
                const u32 i1 = i0 + half;
                const u32 u = s[i0];
                const u32 v = s[i1];
                u32 su = dadd(u, v, p);
                u32 d = u - v;
                if (u < v) d += p;
                s[i0] = su;
                s[i1] = dmul(d, sw[j], p, np);
            }
            __syncthreads();
        }
    } else {
        for (u32 half = 1u; half <= arg.start_half; half <<= 1) {
            const u32 lh = dlog2(half);
            const u32 off = arg.cap - (half << 1);
            for (u32 j = threadIdx.x; j < half; j += blockDim.x)
                sw[j] = w[off + j];
            __syncthreads();
            for (u32 k = threadIdx.x; k < (tile >> 1); k += blockDim.x) {
                const u32 j = k & (half - 1u);
                const u32 grp = k >> lh;
                const u32 i0 = (grp << (lh + 1u)) | j;
                const u32 i1 = i0 + half;
                const u32 u = s[i0];
                const u32 v = dmul(s[i1], sw[j], p, np);
                u32 su = dadd(u, v, p);
                u32 d = u - v;
                if (u < v) d += p;
                s[i0] = su;
                s[i1] = d;
            }
            __syncthreads();
        }
    }

    for (u32 i = threadIdx.x; i < tile; i += blockDim.x) a[base + i] = s[i];
}

// Inverse DIT tail with the pointwise product fused into the tile load.
//   blockIdx.y < 3:  s = A * T * n^-1        (c), written over the T-hat buffers
//   blockIdx.y >= 3: s = (A^2 + B^2) * n^-1  (d), written over the B-hat buffers
//
// A-hat is read by both products and is never written.  Each block loads its
// whole tile before writing it back, and tiles are disjoint, so overwriting
// T-hat (c) and B-hat (d) in place is race-free; T-hat and B-hat are dead
// after the pointwise product.
__global__ void k_tail_pointwise(NttArgs arg) {
    const int b = blockIdx.y;
    const int pr = b % 3;
    u32* a = arg.data[b];
    const u32* A = arg.pw_a[pr];
    const u32* O = (b < 3) ? arg.pw_t[pr] : arg.pw_b[pr];
    const u32* w = arg.tw[pr];
    const u32 p = arg.p[pr];
    const u32 np = arg.np[pr];
    const u32 scale = arg.scale[pr];
    const u32 tile = arg.tile;

    extern __shared__ u32 sh[];
    u32* s = sh;
    u32* sw = sh + tile;
    const u32 base = blockIdx.x * tile;

    if (b < 3) {
        for (u32 i = threadIdx.x; i < tile; i += blockDim.x) {
            const u32 av = A[base + i];
            const u32 ov = O[base + i];
            s[i] = dmul(dmul(av, ov, p, np), scale, p, np);
        }
    } else {
        for (u32 i = threadIdx.x; i < tile; i += blockDim.x) {
            const u32 av = A[base + i];
            const u32 bv = O[base + i];
            const u32 aa = dmul(av, av, p, np);
            const u32 bb = dmul(bv, bv, p, np);
            u32 sum = dadd(aa, bb, p);
            s[i] = dmul(sum, scale, p, np);
        }
    }
    __syncthreads();

    for (u32 half = 1u; half <= arg.start_half; half <<= 1) {
        const u32 lh = dlog2(half);
        const u32 off = arg.cap - (half << 1);
        for (u32 j = threadIdx.x; j < half; j += blockDim.x)
            sw[j] = w[off + j];
        __syncthreads();
        for (u32 k = threadIdx.x; k < (tile >> 1); k += blockDim.x) {
            const u32 j = k & (half - 1u);
            const u32 grp = k >> lh;
            const u32 i0 = (grp << (lh + 1u)) | j;
            const u32 i1 = i0 + half;
            const u32 u = s[i0];
            const u32 v = dmul(s[i1], sw[j], p, np);
            u32 su = dadd(u, v, p);
            u32 d = u - v;
            if (u < v) d += p;
            s[i0] = su;
            s[i1] = d;
        }
        __syncthreads();
    }

    for (u32 i = threadIdx.x; i < tile; i += blockDim.x) a[base + i] = s[i];
}


// Garner reconstruction of one coefficient into three 32-bit words
// (copied from ntt-alt; plain residues, Shoup multiplications).
__device__ __forceinline__ void crt_words(u32 a0, u32 a1, u32 a2,
                                          const CrtArgs& c, u32& w0, u32& w1,
                                          u32& w2) {
    u32 d1 = a1 - a0;
    if (a1 < a0) d1 += c.P1;
    u32 t1 = dshoup(d1, c.inv01, c.inv01_sh, c.P1);

    u32 u = dshoup(t1, c.p0m2, c.p0m2_sh, c.P2);
    u32 e = dadd(a0, u, c.P2);

    u32 d2 = a2 - e;
    if (a2 < e) d2 += c.P2;
    u32 t2 = dshoup(d2, c.invp01, c.invp01_sh, c.P2);

    // x = a0 + p0*t1 + (p0*p1)*t2, all without a 128-bit type.
    u64 lo = (u64)c.P0 * t1 + a0; // < 2^62
    u64 A = (u64)c.P0 * c.P1;     // < 2^62
    u64 alo = A & 0xffffffffull;
    u64 ahi = A >> 32;
    u64 plo = alo * t2; // < 2^63
    u64 phi = ahi * t2; // < 2^61
    u64 b0 = plo & 0xffffffffull;
    u64 mid = (plo >> 32) + (phi & 0xffffffffull);
    u64 b1 = mid & 0xffffffffull;
    u64 b2 = (phi >> 32) + (mid >> 32);

    u64 s = b0 + lo;
    w0 = (u32)s;
    u64 carry = s >> 32;
    s = b1 + carry;
    w1 = (u32)s;
    carry = s >> 32;
    w2 = (u32)(b2 + carry);
}

// CRT + digit combination for c (blockIdx.y = 0) and d (blockIdx.y = 1):
//   T[i] = w0[i] + w1[i-1] + w2[i-2]   (< 3*2^32)
// The two coefficients before each block are recomputed in the shared-memory
// halo, exactly as ntt-alt does.
__global__ void k_crt(u64* __restrict__ Tc, u64* __restrict__ Td,
                      const u32* __restrict__ rc0,
                      const u32* __restrict__ rc1,
                      const u32* __restrict__ rc2,
                      const u32* __restrict__ rd0,
                      const u32* __restrict__ rd1,
                      const u32* __restrict__ rd2, u32 nout, u32 L,
                      CrtArgs c) {
    const int prod = blockIdx.y;
    const u32* r0 = prod ? rd0 : rc0;
    const u32* r1 = prod ? rd1 : rc1;
    const u32* r2 = prod ? rd2 : rc2;
    u64* T = prod ? Td : Tc;

    __shared__ u32 s0[kCrtBlock + 2];
    __shared__ u32 s1[kCrtBlock + 2];
    __shared__ u32 s2[kCrtBlock + 2];

    const u32 t = threadIdx.x;
    const u32 base = blockIdx.x * kCrtBlock;
    const u32 idx = base + t;

    if (idx < nout) {
        u32 w0, w1, w2;
        crt_words(r0[idx], r1[idx], r2[idx], c, w0, w1, w2);
        s0[t + 2] = w0;
        s1[t + 2] = w1;
        s2[t + 2] = w2;
    } else {
        s0[t + 2] = 0u;
        s1[t + 2] = 0u;
        s2[t + 2] = 0u;
    }

    if (t < 2u) {
        i32 jj = (i32)base - 2 + (i32)t;
        if (jj >= 0 && (u32)jj < nout) {
            u32 w0, w1, w2;
            crt_words(r0[jj], r1[jj], r2[jj], c, w0, w1, w2);
            s0[t] = w0;
            s1[t] = w1;
            s2[t] = w2;
        } else {
            s0[t] = 0u;
            s1[t] = 0u;
            s2[t] = 0u;
        }
    }
    __syncthreads();

    if (idx < L) T[idx] = (u64)s0[t + 2] + (u64)s1[t + 1] + (u64)s2[t];
}

// T_cd[i] = T_c[i] + T_d[i]  (< 6*2^32).
__global__ void k_combine_cd(u64* __restrict__ Tcd, const u64* __restrict__ Tc,
                             const u64* __restrict__ Td, u32 L) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    Tcd[i] = Tc[i] + Td[i];
}

// ---------------------------------------------------------------------------
// Carry propagation (ntt-alt's exact two-level block-transition scan)
//
// A digit T[i] < 6*2^32 and the carry into any position is in 0..5, so a
// chunk's exact behaviour is the function f(x) = carry out for carry in x,
// x = 0..5.  Each 2048-digit block composes its 256 thread functions, a
// single serial scan composes the block functions, and a replay writes the
// limbs with the block carry-in.  This is exact for any data.
// ---------------------------------------------------------------------------

__device__ __forceinline__ u32 pack_states(u32 f0, u32 f1, u32 f2, u32 f3,
                                           u32 f4, u32 f5) {
    return f0 | (f1 << 3) | (f2 << 6) | (f3 << 9) | (f4 << 12) | (f5 << 15);
}

__device__ __forceinline__ u32 state_at(u32 s, u32 x) {
    return (s >> (3u * x)) & 7u;
}

__global__ void k_carry_sum(const u64* __restrict__ T0,
                            const u64* __restrict__ T1,
                            const u64* __restrict__ T2,
                            u32* __restrict__ sums, u32 L, u32 nb) {
    __shared__ u64 sT[kCarryBlock];
    __shared__ u32 sF[kCarryThreads];
    const int prod = blockIdx.y;
    const u64* T = prod == 0 ? T0 : (prod == 1 ? T1 : T2);
    const u32 t = threadIdx.x;
    const u32 base = blockIdx.x * kCarryBlock;
    for (u32 u = t; u < kCarryBlock; u += blockDim.x) {
        u32 idx = base + u;
        sT[u] = (idx < L) ? T[idx] : 0ull;
    }
    __syncthreads();

    const u32 s = t * kCarryPer;
    u32 c0 = 0u, c1 = 1u, c2 = 2u, c3 = 3u, c4 = 4u, c5 = 5u;
    for (u32 u = s; u < s + kCarryPer; ++u) {
        u64 v = sT[u];
        c0 = (u32)((v + c0) >> 32);
        c1 = (u32)((v + c1) >> 32);
        c2 = (u32)((v + c2) >> 32);
        c3 = (u32)((v + c3) >> 32);
        c4 = (u32)((v + c4) >> 32);
        c5 = (u32)((v + c5) >> 32);
    }
    sF[t] = pack_states(c0, c1, c2, c3, c4, c5);
    __syncthreads();

    if (t == 0u) {
        u32 a0 = 0u, a1 = 1u, a2 = 2u, a3 = 3u, a4 = 4u, a5 = 5u;
        for (u32 u = 0; u < kCarryThreads; ++u) {
            u32 sm = sF[u];
            u32 n0 = state_at(sm, a0);
            u32 n1 = state_at(sm, a1);
            u32 n2 = state_at(sm, a2);
            u32 n3 = state_at(sm, a3);
            u32 n4 = state_at(sm, a4);
            u32 n5 = state_at(sm, a5);
            a0 = n0;
            a1 = n1;
            a2 = n2;
            a3 = n3;
            a4 = n4;
            a5 = n5;
        }
        sums[prod * nb + blockIdx.x] =
            pack_states(a0, a1, a2, a3, a4, a5);
    }
}

// Compose the block functions, one independent serial chain per product.
__global__ void k_carry_scan(const u32* __restrict__ sums, u32* __restrict__ cin,
                             u32* __restrict__ err, u32 nb) {
    const int prod = threadIdx.x;
    if (prod >= 3) return;
    u32 c = 0u;
    for (u32 b = 0; b < nb; ++b) {
        cin[prod * (nb + 1) + b] = c;
        u32 s = sums[prod * nb + b];
        c = state_at(s, c);
    }
    cin[prod * (nb + 1) + nb] = c;
    // Every product fits in `L` limbs, so the final carry must vanish.
    if (c != 0u) atomicOr(err, 1u << (4 + prod));
}

__global__ void k_carry_apply(u32* __restrict__ out0, u32* __restrict__ out1,
                              u32* __restrict__ out2,
                              const u64* __restrict__ T0,
                              const u64* __restrict__ T1,
                              const u64* __restrict__ T2,
                              const u32* __restrict__ bcin,
                              const StepDesc* __restrict__ d,
                              u32* __restrict__ err, u32 L, u32 nb) {
    __shared__ u64 sT[kCarryBlock];
    __shared__ u32 sL[kCarryBlock];
    __shared__ u32 sF[kCarryThreads];
    __shared__ u32 sCin[kCarryThreads];
    __shared__ u32 sFinal;

    const int prod = blockIdx.y;
    const u64* T = prod == 0 ? T0 : (prod == 1 ? T1 : T2);
    u32* out = prod == 0 ? out0 : (prod == 1 ? out1 : out2);
    const u32 t = threadIdx.x;
    const u32 base = blockIdx.x * kCarryBlock;
    for (u32 u = t; u < kCarryBlock; u += blockDim.x) {
        u32 idx = base + u;
        sT[u] = (idx < L) ? T[idx] : 0ull;
    }
    __syncthreads();

    const u32 s = t * kCarryPer;
    u32 c0 = 0u, c1 = 1u, c2 = 2u, c3 = 3u, c4 = 4u, c5 = 5u;
    for (u32 u = s; u < s + kCarryPer; ++u) {
        u64 v = sT[u];
        c0 = (u32)((v + c0) >> 32);
        c1 = (u32)((v + c1) >> 32);
        c2 = (u32)((v + c2) >> 32);
        c3 = (u32)((v + c3) >> 32);
        c4 = (u32)((v + c4) >> 32);
        c5 = (u32)((v + c5) >> 32);
    }
    sF[t] = pack_states(c0, c1, c2, c3, c4, c5);
    __syncthreads();

    if (t == 0u) {
        u32 c = bcin[prod * (nb + 1) + blockIdx.x];
        for (u32 u = 0; u < kCarryThreads; ++u) {
            sCin[u] = c;
            u32 sm = sF[u];
            c = state_at(sm, c);
        }
        if (blockIdx.x == gridDim.x - 1u) sFinal = c;
    }
    __syncthreads();

    u32 c = sCin[t];
    for (u32 u = s; u < s + kCarryPer; ++u) {
        u64 x = sT[u] + c;
        sL[u] = (u32)x;
        c = (u32)(x >> 32);
    }
    __syncthreads();

    const u32 bound = d->bound[prod];
    for (u32 u = t; u < kCarryBlock; u += blockDim.x) {
        u32 idx = base + u;
        if (idx < L) {
            u32 limb = sL[u];
            out[idx] = limb;
            if (idx >= bound && limb != 0u)
                atomicOr(err, 1u << (8 + prod));
        }
    }
    if (t == 0u && blockIdx.x == gridDim.x - 1u) out[L] = sFinal;
}

// ---------------------------------------------------------------------------
// t = 2b - a with an exact block-scan borrow chain, fused with the residue
// conversion.
//
// Per limb, with borrow st in {-1,0,1},
//   s = 2*B[i] + st - A[i];  limb = s mod 2^32;  st' = floor(s / 2^32),
// and st' stays in {-1,0,1}.  This is the same exact block-transition idea as
// the alt carry: a block of 2048 limbs summarises its carry-out function for
// carry-in -1,0,1 (2 bits per state), one serial scan composes the blocks,
// and a replay pass applies the block carry-in.
//
// The same passes also produce the nine plain-residue transform inputs:
// pass 1 writes a mod p and b mod p, pass 3 writes t mod p.  That removes the
// separate reduction kernel and its second read of all three operands.
// ---------------------------------------------------------------------------

constexpr int kSubBlock = 2048;
constexpr int kSubThreads = 256;
constexpr int kSubPer = kSubBlock / kSubThreads;

struct SubArgs {
    const StepDesc* d;
    u32* t;
    u32* sums; // [nsub]
    u32* cin;  // [nsub+1]
    u32* err;
    u32 n;
};

__device__ __forceinline__ i32 sub_limb(u32 av, u32 bv, i32 st, u32& limb) {
    long long s = 2ll * (long long)bv + (long long)st - (long long)av;
    if (s < 0) {
        limb = (u32)(s + (1ll << 32));
        return -1;
    }
    limb = (u32)s;
    return (i32)(s >> 32);
}

// Pack three 2-bit state transitions (state 0 = borrow -1, 1 = 0, 2 = +1).
__device__ __forceinline__ u32 pack_sub3(u32 f0, u32 f1, u32 f2) {
    return f0 | (f1 << 2) | (f2 << 4);
}

__device__ __forceinline__ u32 sub_state_at(u32 s, u32 x) {
    return (s >> (2u * x)) & 3u;
}

__global__ void k_sub_sum(SubArgs args) {
    __shared__ u32 sF[kSubThreads];
    const StepDesc* d = args.d;
    const u32 la = d->len_a;
    const u32 lb = d->len_b;
    const u32 n = args.n;
    const u32 t = threadIdx.x;
    const u32 start = blockIdx.x * kSubBlock + t * kSubPer;

    i32 c0 = -1, c1 = 0, c2 = 1;
    for (u32 j = 0; j < kSubPer; ++j) {
        const u32 i = start + j;
        u32 av = 0u, bv = 0u;
        if (i < n) {
            av = (i < la) ? d->a[i] : 0u;
            bv = (i < lb) ? d->b[i] : 0u;
        }
        u32 limb;
        c0 = sub_limb(av, bv, c0, limb);
        c1 = sub_limb(av, bv, c1, limb);
        c2 = sub_limb(av, bv, c2, limb);
    }
    sF[t] = pack_sub3((u32)(c0 + 1), (u32)(c1 + 1), (u32)(c2 + 1));
    __syncthreads();

    if (t == 0u) {
        u32 a0 = 0u, a1 = 1u, a2 = 2u;
        for (u32 u = 0; u < kSubThreads; ++u) {
            const u32 sm = sF[u];
            const u32 n0 = sub_state_at(sm, a0);
            const u32 n1 = sub_state_at(sm, a1);
            const u32 n2 = sub_state_at(sm, a2);
            a0 = n0;
            a1 = n1;
            a2 = n2;
        }
        args.sums[blockIdx.x] = pack_sub3(a0, a1, a2);
    }
}

__global__ void k_sub_scan(const u32* __restrict__ sums,
                           u32* __restrict__ cin, u32* __restrict__ err,
                           u32 nb) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    u32 c = 1u; // state index of borrow 0
    for (u32 b = 0; b < nb; ++b) {
        cin[b] = c;
        c = sub_state_at(sums[b], c);
    }
    cin[nb] = c;
    // t = 2b - a >= 0, so the final borrow must be zero (state index 1).
    if (c != 1u) atomicOr(err, 1u);
}

__global__ void k_sub_apply(SubArgs args) {
    __shared__ u32 sF[kSubThreads];
    __shared__ u32 sCin[kSubThreads];
    const StepDesc* d = args.d;
    const u32 la = d->len_a;
    const u32 lb = d->len_b;
    const u32 n = args.n;
    const u32 t = threadIdx.x;
    const u32 start = blockIdx.x * kSubBlock + t * kSubPer;

    i32 c0 = -1, c1 = 0, c2 = 1;
    for (u32 j = 0; j < kSubPer; ++j) {
        const u32 i = start + j;
        u32 av = 0u, bv = 0u;
        if (i < n) {
            av = (i < la) ? d->a[i] : 0u;
            bv = (i < lb) ? d->b[i] : 0u;
        }
        u32 limb;
        c0 = sub_limb(av, bv, c0, limb);
        c1 = sub_limb(av, bv, c1, limb);
        c2 = sub_limb(av, bv, c2, limb);
    }
    sF[t] = pack_sub3((u32)(c0 + 1), (u32)(c1 + 1), (u32)(c2 + 1));
    __syncthreads();

    if (t == 0u) {
        u32 c = args.cin[blockIdx.x];
        for (u32 u = 0; u < kSubThreads; ++u) {
            sCin[u] = c;
            c = sub_state_at(sF[u], c);
        }
    }
    __syncthreads();

    i32 st = (i32)sCin[t] - 1;
    for (u32 j = 0; j < kSubPer; ++j) {
        const u32 i = start + j;
        u32 av = 0u, bv = 0u;
        if (i < n) {
            av = (i < la) ? d->a[i] : 0u;
            bv = (i < lb) ? d->b[i] : 0u;
        }
        u32 limb;
        st = sub_limb(av, bv, st, limb);
        if (i < n) args.t[i] = limb;
    }
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

struct Engine {
    int cap_log2 = 0;
    int cap = 0;
    u32* ntt[9] = {};
    u32* tw[3] = {};
    u32* twi[3] = {};
    u32* raw[5] = {}; // 0=a, 1=b, 2=c, 3=d, 4=c+d
    u32* raw_t = nullptr;
    u64* T[3] = {}; // Tc, Td, Tcd
    u32* b_sums = nullptr; // borrow block summaries
    u32* b_cin = nullptr;  // borrow block carry-in
    u32* c_sums = nullptr;
    u32* c_cin = nullptr;
    u32* err = nullptr;
    StepDesc* desc = nullptr;   // single-step descriptor (step ABI / per-lg graph)
    StepDesc* d_desc = nullptr; // whole-schedule descriptor array
    std::map<uint64_t, cudaGraphExec_t> loop_graph; // keyed by Fibonacci index
    StepDesc* pin_desc = nullptr;
    u32* pin_in[2] = {};
    u32* pin_out = nullptr;
    u32* pin_one = nullptr;
    u32 scales[MAX_NTT_LOG2 + 1][3] = {};
    cudaGraphExec_t graph[MAX_NTT_LOG2 + 1] = {};
    bool graph_valid[MAX_NTT_LOG2 + 1] = {};
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[2] = {};
    // Optional phase breakdown (RESIDENT27_PROFILE=1, no-graph path only).
    cudaEvent_t pev[12] = {};
    cudaEvent_t sev[MAX_STEPS + 1] = {};
    double prof[7] = {};
    int nbmax = 0;
    gpu_fib_stats stats = {};
};

static Engine g_engine;
static bool g_ready = false;
static bool g_use_graph = true;
static bool g_profile = false;

static int ilog2_u32(u32 v) {
    int l = 0;
    while ((1u << l) < v) ++l;
    return l;
}

static void free_engine(Engine* e) {
    for (auto& kv : e->loop_graph) cudaGraphExecDestroy(kv.second);
    e->loop_graph.clear();
    for (int k = 0; k <= MAX_NTT_LOG2; ++k) {
        if (e->graph_valid[k]) {
            cudaGraphExecDestroy(e->graph[k]);
            e->graph[k] = nullptr;
            e->graph_valid[k] = false;
        }
    }
    for (int k = 0; k < 9; ++k) {
        if (e->ntt[k]) cudaFree(e->ntt[k]);
        e->ntt[k] = nullptr;
    }
    for (int k = 0; k < 3; ++k) {
        if (e->tw[k]) cudaFree(e->tw[k]);
        if (e->twi[k]) cudaFree(e->twi[k]);
        e->tw[k] = e->twi[k] = nullptr;
        if (e->T[k]) cudaFree(e->T[k]);
        e->T[k] = nullptr;
    }
    for (int k = 0; k < 5; ++k) {
        if (e->raw[k]) cudaFree(e->raw[k]);
        e->raw[k] = nullptr;
    }
    if (e->raw_t) cudaFree(e->raw_t);
    e->raw_t = nullptr;
    if (e->b_sums) cudaFree(e->b_sums);
    if (e->b_cin) cudaFree(e->b_cin);
    if (e->c_sums) cudaFree(e->c_sums);
    if (e->c_cin) cudaFree(e->c_cin);
    if (e->err) cudaFree(e->err);
    if (e->desc) cudaFree(e->desc);
    if (e->d_desc) cudaFree(e->d_desc);
    e->b_sums = nullptr;
    e->b_cin = nullptr;
    e->c_sums = nullptr;
    e->c_cin = nullptr;
    e->err = nullptr;
    e->desc = nullptr;
    e->d_desc = nullptr;
    if (e->pin_desc) cudaFreeHost(e->pin_desc);
    for (int k = 0; k < 2; ++k) {
        if (e->pin_in[k]) cudaFreeHost(e->pin_in[k]);
        e->pin_in[k] = nullptr;
    }
    if (e->pin_out) cudaFreeHost(e->pin_out);
    if (e->pin_one) cudaFreeHost(e->pin_one);
    e->pin_desc = nullptr;
    e->pin_out = nullptr;
    e->pin_one = nullptr;
    e->cap = 0;
    e->cap_log2 = 0;
    e->nbmax = 0;
}

static bool alloc_engine(Engine* e, int lg) {
    const int cap = 1 << lg;
    if (cap > (1 << MAX_NTT_LOG2)) return set_error_msg("NTT length too large");
    e->cap = cap;
    e->cap_log2 = lg;

    const int total = cap + 4;
    for (int k = 0; k < 9; ++k) CUDA_TRY(cudaMalloc(&e->ntt[k], (size_t)cap * 4));
    for (int k = 0; k < 3; ++k) {
        CUDA_TRY(cudaMalloc(&e->tw[k], (size_t)(cap - 1) * 4));
        CUDA_TRY(cudaMalloc(&e->twi[k], (size_t)(cap - 1) * 4));
    }
    for (int k = 0; k < 5; ++k)
        CUDA_TRY(cudaMalloc(&e->raw[k], (size_t)total * 4));
    CUDA_TRY(cudaMalloc(&e->raw_t, (size_t)total * 4));
    for (int k = 0; k < 3; ++k)
        CUDA_TRY(cudaMalloc(&e->T[k], (size_t)total * 8));

    e->nbmax = (cap + 1 + kCarryBlock - 1) / kCarryBlock + 1;
    CUDA_TRY(cudaMalloc(&e->b_sums, (size_t)e->nbmax * 4));
    CUDA_TRY(cudaMalloc(&e->b_cin, (size_t)(e->nbmax + 1) * 4));
    CUDA_TRY(cudaMalloc(&e->c_sums, (size_t)(3 * e->nbmax) * 4));
    CUDA_TRY(cudaMalloc(&e->c_cin, (size_t)(3 * (e->nbmax + 1)) * 4));
    CUDA_TRY(cudaMalloc(&e->err, 4));
    CUDA_TRY(cudaMalloc(&e->desc, sizeof(StepDesc)));
    CUDA_TRY(cudaMalloc(&e->d_desc, sizeof(StepDesc) * MAX_STEPS));

    CUDA_TRY(cudaMallocHost(&e->pin_desc, sizeof(StepDesc) * MAX_STEPS));
    for (int k = 0; k < 2; ++k)
        CUDA_TRY(cudaMallocHost(&e->pin_in[k], (size_t)total * 4));
    CUDA_TRY(cudaMallocHost(&e->pin_out, (size_t)total * 4));
    CUDA_TRY(cudaMallocHost(&e->pin_one, 4));
    return true;
}

// Per-stage contiguous twiddle tables (copied from ntt-alt), built for the
// whole engine capacity.  Stage half owns [cap-2*half, cap-half).
static void make_twiddle_tables(int cap, int pi, std::vector<u32>& fwd,
                                std::vector<u32>& inv) {
    const PrimeInfo& q = g_prime[pi];
    const u32 p = q.p;
    const u32 rmod = q.r;
    const size_t sz = (size_t)(cap - 1);
    fwd.assign(sz, 0u);
    inv.assign(sz, 0u);
    for (u32 h = (u32)cap >> 1; h >= 1u; h >>= 1) {
        const u32 off = (u32)cap - (h << 1);
        const u32 W = hpowmod(kRoot[pi], (u64)(p - 1) / (2u * h), p);
        const u32 Wm = (u32)(((u64)W * rmod) % p);
        fwd[off] = rmod;
        for (u32 j = 1; j < h; ++j)
            fwd[off + j] = host_mont_mul(fwd[off + j - 1], Wm, p, q.np);
        inv[off] = rmod;
        for (u32 j = 1; j < h; ++j) inv[off + j] = p - fwd[off + (h - j)];
    }
}

static bool build_twiddles(Engine* e) {
    const int cap = e->cap;
    const size_t bytes = (size_t)(cap - 1) * 4;
    for (int k = 0; k < 3; ++k) {
        std::vector<u32> tf, ti;
        make_twiddle_tables(cap, k, tf, ti);
        CUDA_TRY(cudaMemcpy(e->tw[k], tf.data(), bytes,
                            cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemcpy(e->twi[k], ti.data(), bytes,
                            cudaMemcpyHostToDevice));
    }
    return true;
}

static bool ensure_capacity(Engine* e, int lg) {
    if (e->cap_log2 >= lg) return true;
    free_engine(e);
    if (!alloc_engine(e, lg)) return false;
    if (!build_twiddles(e)) return false;
    const char* mi = std::getenv("RESIDENT27_MEMINFO");
    if (mi && *mi && *mi != '0') {
        size_t freeb = 0, totalb = 0;
        cudaMemGetInfo(&freeb, &totalb);
        std::fprintf(stderr,
                     "resident27: cap=2^%d, process device memory in use "
                     "%.1f MiB (GPU total %.1f MiB)\n",
                     lg, (double)(totalb - freeb) / 1048576.0,
                     (double)totalb / 1048576.0);
    }
    return true;
}

static bool engine_init() {
    if (g_ready) return true;
    if (!init_primes()) return false;

    for (int lg = 1; lg <= MAX_NTT_LOG2; ++lg) {
        const u32 n = 1u << lg;
        for (int k = 0; k < 3; ++k) {
            const PrimeInfo& pi = g_prime[k];
            const u32 ninv = hpowmod(n, (u64)pi.p - 2, pi.p);
            g_engine.scales[lg][k] = (u32)(((u64)pi.r2 * ninv) % pi.p);
        }
    }

    if (!g_engine.stream) {
        CUDA_TRY(cudaStreamCreateWithFlags(&g_engine.stream,
                                           cudaStreamNonBlocking));
        CUDA_TRY(cudaEventCreate(&g_engine.ev[0]));
        CUDA_TRY(cudaEventCreate(&g_engine.ev[1]));
        for (int i = 0; i < 12; ++i) CUDA_TRY(cudaEventCreate(&g_engine.pev[i]));
        for (int i = 0; i <= MAX_STEPS; ++i)
            CUDA_TRY(cudaEventCreate(&g_engine.sev[i]));
    }
    const char* ng = std::getenv("RESIDENT27_NO_GRAPH");
    g_use_graph = !(ng && *ng && *ng != '0');
    const char* pf = std::getenv("RESIDENT27_PROFILE");
    g_profile = (pf && *pf && *pf != '0');
    g_ready = true;
    return true;
}

// ---------------------------------------------------------------------------
// One fused doubling step: enqueue every kernel for transform length 2^lg.
// All arguments are constant for a given lg, so this function is also the
// body of the per-length CUDA graph.  The per-step values that do change
// (input pointers, lengths, Binet bounds, digit extent) live in StepDesc.
// ---------------------------------------------------------------------------

static bool enqueue_step(Engine* e, int lg, const StepDesc* desc, u32 L) {
    const u32 n = 1u << lg;
    const u32 tile = (n < (u32)kTile) ? n : (u32)kTile;
    const int tl = ilog2_u32(tile);
    const int g = (int)lg - tl; // number of global stages per transform
    if (L > n + 1) L = n + 1;   // T entries (plus the final carry)
    const u32 nb = (L + kCarryBlock - 1) / kCarryBlock;
    const u32 nsub = (n + kSubBlock - 1) / kSubBlock;

    const int grid_b = (int)((n / 2 + kStageThreads - 1) / kStageThreads);
    const int grid_crt = (int)((L + kCrtBlock - 1) / kCrtBlock);

    if (nb > (u32)e->nbmax) return set_error_msg("chunk overflow");

    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[0], e->stream);

    // 1. t = 2b - a (exact block-scan borrow chain), fused with the
    //    plain-residue conversion of a, b and t.
    SubArgs sa{};
    sa.d = desc;
    sa.t = e->raw_t;
    sa.sums = e->b_sums;
    sa.cin = e->b_cin;
    sa.err = e->err;
    sa.n = n;
    k_sub_sum<<<nsub, kSubThreads, 0, e->stream>>>(sa);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[1], e->stream);
    k_sub_scan<<<1, 1, 0, e->stream>>>(e->b_sums, e->b_cin, e->err, nsub);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[2], e->stream);
    k_sub_apply<<<nsub, kSubThreads, 0, e->stream>>>(sa);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[3], e->stream);

    // 2. forward NTT of a, b, t (DIF, natural -> bit-reversed)
    NttArgs fwd{};
    for (int k = 0; k < 9; ++k) fwd.data[k] = e->ntt[k];
    for (int k = 0; k < 3; ++k) {
        fwd.tw[k] = e->tw[k];
        fwd.p[k] = g_prime[k].p;
        fwd.np[k] = g_prime[k].np;
    }
    fwd.n = n;
    fwd.cap = (u32)e->cap;
    fwd.tile = tile;
    fwd.start_half = tile >> 1;
    // The first forward pass reads the raw limb sources named by the step
    // descriptor and reduces them; the borrow passes therefore no longer
    // write the nine residue buffers.
    fwd.d = desc;
    fwd.raw_t = e->raw_t;
    for (int i = 0; i < g; ++i) {
        const u32 half = (n >> 1) >> i;
        fwd.half = half;
        fwd.off = (u32)e->cap - (half << 1);
        fwd.lh = (u32)(lg - 1 - i);
        fwd.is_first = (i == 0) ? 1 : 0;
        k_stage_multi<0><<<dim3(grid_b, 9), kStageThreads, 0, e->stream>>>(fwd);
    }
    {
        const size_t shmem = (size_t)(tile + (tile >> 1)) * 4;
        // With no global stages the tail is the first (and only) pass.
        fwd.is_first = (g == 0) ? 1 : 0;
        k_tail_multi<0><<<dim3(n / tile, 9), kStageThreads, shmem, e->stream>>>(
            fwd);
    }
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[4], e->stream);

    // 4. inverse NTT of c and d (DIT, bit-reversed -> natural) with the
    //    pointwise product fused into the tail tile load.
    //    c is written over the T-hat buffers (ntt[6..8]) and d over the B-hat
    //    buffers (ntt[3..5]); A-hat (ntt[0..2]) is read by both and never
    //    written, so the in-place overwrite is race-free.
    NttArgs inv{};
    inv.data[0] = e->ntt[6];
    inv.data[1] = e->ntt[7];
    inv.data[2] = e->ntt[8];
    inv.data[3] = e->ntt[3];
    inv.data[4] = e->ntt[4];
    inv.data[5] = e->ntt[5];
    for (int k = 0; k < 3; ++k) {
        inv.tw[k] = e->twi[k];
        inv.p[k] = g_prime[k].p;
        inv.np[k] = g_prime[k].np;
        inv.pw_a[k] = e->ntt[k];
        inv.pw_t[k] = e->ntt[k + 6];
        inv.pw_b[k] = e->ntt[k + 3];
        inv.scale[k] = e->scales[lg][k];
    }
    inv.n = n;
    inv.cap = (u32)e->cap;
    inv.tile = tile;
    inv.start_half = tile >> 1;
    {
        const size_t shmem = (size_t)(tile + (tile >> 1)) * 4;
        k_tail_pointwise<<<dim3(n / tile, 6), kStageThreads, shmem,
                           e->stream>>>(inv);
    }
    for (int i = 0; i < g; ++i) {
        const u32 half = tile << i;
        inv.half = half;
        inv.off = (u32)e->cap - (half << 1);
        inv.lh = (u32)(tl + i);
        k_stage_multi<1><<<dim3(grid_b, 6), kStageThreads, 0, e->stream>>>(inv);
    }
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[5], e->stream);

    // 5. CRT digit arrays for c and d, then T_cd = T_c + T_d
    k_crt<<<dim3(grid_crt, 2), kCrtBlock, 0, e->stream>>>(
        e->T[0], e->T[1], e->ntt[6], e->ntt[7], e->ntt[8], e->ntt[3],
        e->ntt[4], e->ntt[5], n, L, g_crt);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[6], e->stream);

    // 6. T_cd = T_c + T_d
    {
        const int gcb = (int)((L + 255) / 256);
        k_combine_cd<<<gcb, 256, 0, e->stream>>>(e->T[2], e->T[0], e->T[1],
                                                 L);
    }
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[7], e->stream);

    // 7. exact two-level carry into raw_c, raw_d, raw_cd
    k_carry_sum<<<dim3(nb, 3), kCarryThreads, 0, e->stream>>>(
        e->T[0], e->T[1], e->T[2], e->c_sums, L, nb);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[8], e->stream);
    k_carry_scan<<<1, 32, 0, e->stream>>>(e->c_sums, e->c_cin, e->err, nb);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[9], e->stream);
    k_carry_apply<<<dim3(nb, 3), kCarryThreads, 0, e->stream>>>(
        e->raw[2], e->raw[3], e->raw[4], e->T[0], e->T[1], e->T[2], e->c_cin,
        desc, e->err, L, nb);
    if (g_profile && !g_use_graph) cudaEventRecord(e->pev[10], e->stream);

    CUDA_TRY(cudaGetLastError());
    return true;
}

struct StepPlan {
    int bit;
    int lg;
    int src_a;
    int src_b;
};

static bool launch_step(Engine* e, int lg) {
    if (!g_use_graph) return enqueue_step(e, lg, e->desc, (1u << lg) + 1);
    if (!e->graph_valid[lg]) {
        CUDA_TRY(cudaStreamSynchronize(e->stream));
        cudaGraph_t graph = nullptr;
        CUDA_TRY(cudaStreamBeginCapture(e->stream,
                                        cudaStreamCaptureModeThreadLocal));
        if (!enqueue_step(e, lg, e->desc, (1u << lg) + 1)) {
            cudaStreamEndCapture(e->stream, &graph);
            if (graph) cudaGraphDestroy(graph);
            return false;
        }
        CUDA_TRY(cudaStreamEndCapture(e->stream, &graph));
        CUDA_TRY(cudaGraphInstantiate(&e->graph[lg], graph, 0ull));
        cudaGraphDestroy(graph);
        e->graph_valid[lg] = true;
    }
    CUDA_TRY(cudaGraphLaunch(e->graph[lg], e->stream));
    return true;
}

// Whole-schedule graph: the bit loop, the per-step descriptors and every
// buffer identity are deterministic for a given index, so the entire resident
// loop can be captured once and replayed with a single launch.  This removes
// the per-step cudaGraphLaunch and StepDesc upload from the timed path.
static bool launch_loop(Engine* e, uint64_t n,
                        const std::vector<StepPlan>& plans) {
    auto it = e->loop_graph.find(n);
    if (it != e->loop_graph.end()) {
        CUDA_TRY(cudaGraphLaunch(it->second, e->stream));
        return true;
    }
    CUDA_TRY(cudaStreamSynchronize(e->stream));
    cudaGraph_t graph = nullptr;
    CUDA_TRY(cudaStreamBeginCapture(e->stream,
                                    cudaStreamCaptureModeThreadLocal));
    for (size_t i = 0; i < plans.size(); ++i) {
        if (!enqueue_step(e, plans[i].lg, e->d_desc + i,
                          e->pin_desc[i].L)) {
            cudaStreamEndCapture(e->stream, &graph);
            if (graph) cudaGraphDestroy(graph);
            return false;
        }
    }
    CUDA_TRY(cudaStreamEndCapture(e->stream, &graph));
    cudaGraphExec_t ex = nullptr;
    CUDA_TRY(cudaGraphInstantiate(&ex, graph, 0ull));
    cudaGraphDestroy(graph);
    e->loop_graph[n] = ex;
    CUDA_TRY(cudaGraphLaunch(ex, e->stream));
    return true;
}

// ---------------------------------------------------------------------------
// Schedule: the whole bit loop is deterministic, so lengths, transform sizes
// and buffer identities are planned on the host before any kernel runs.
//
// F(j) < phi^j < 2^(0.694242*j), hence bits(F(j)) <= floor(0.694242*j)+1.
// ---------------------------------------------------------------------------

static u32 limb_bound(u64 index) {
    if (index == 0) return 0;
    u64 bits = (index * 694242ull) / 1000000ull + 1;
    return (u32)((bits + 31) / 32);
}

static bool build_plans(u64 n, std::vector<StepPlan>& plans, int& max_lg,
                        u32& result_limbs) {
    plans.clear();
    max_lg = 0;
    result_limbs = limb_bound(n);
    if (n == 0) return true;
    int bits = 0;
    for (u64 v = n; v; v >>= 1) ++bits;
    u64 k = 0;
    int src_a = 0; // raw_a: F(0) = 0
    int src_b = 1; // raw_b: F(1) = 1
    for (int i = bits - 1; i >= 0; --i) {
        int bit = (int)((n >> i) & 1ull);
        u32 la = limb_bound(k);
        u32 lb = limb_bound(k + 1);
        u32 m = std::max(la, lb);
        u32 req = 2 * m + 2;
        if (req < 4) req = 4;
        int lg = ilog2_u32(req);
        if (lg > MAX_NTT_LOG2) {
            char msg[96]; std::snprintf(msg, sizeof(msg), "index needs an NTT longer than 2^%d", MAX_NTT_LOG2); return set_error_msg(msg);
        }
        StepPlan p;
        p.bit = bit;
        p.lg = lg;
        p.src_a = src_a;
        p.src_b = src_b;
        plans.push_back(p);
        if (lg > max_lg) max_lg = lg;
        if (bit == 0) {
            src_a = 2; // raw_c
            src_b = 3; // raw_d
        } else {
            src_a = 3; // raw_d
            src_b = 4; // raw_cd
        }
        k = 2 * k + (u64)bit;
    }
    return true;
}

static bool sync_and_check(Engine* e) {
    CUDA_TRY(cudaStreamSynchronize(e->stream));
    u32 flags = 0;
    CUDA_TRY(cudaMemcpy(&flags, e->err, 4, cudaMemcpyDeviceToHost));
    if (flags != 0) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "device self-check failed (error flags 0x%x)", flags);
        return set_error_msg(msg);
    }
    return true;
}

static u32* dup_pinned(const u32* src, size_t count, size_t* out_count) {
    size_t n = count;
    while (n > 0 && src[n - 1] == 0) --n;
    if (n == 0) {
        *out_count = 0;
        return nullptr;
    }
    u32* buf = (u32*)std::malloc(n * 4);
    if (!buf) {
        *out_count = 0;
        return nullptr;
    }
    std::memcpy(buf, src, n * 4);
    *out_count = n;
    return buf;
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" RESIDENT_API int gpu_fib_doubling(uint64_t n, uint32_t** out_limbs,
                                             size_t* out_count) {
    if (!out_limbs || !out_count) return 1;
    *out_limbs = nullptr;
    *out_count = 0;

    Engine* e = &g_engine;
    e->stats = gpu_fib_stats();
    if (!engine_init()) return 3;
    if (n == 0) return 0; // F(0) = 0, canonical empty vector

    std::vector<StepPlan> plans;
    int max_lg = 0;
    u32 result_limbs = 0;
    if (!build_plans(n, plans, max_lg, result_limbs)) return 2;
    if (!ensure_capacity(e, max_lg)) return 3;

    // Fill the pinned descriptor schedule (the loop is deterministic).
    u64 k = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        StepPlan& p = plans[i];
        StepDesc d{};
        const u32 nstep = 1u << p.lg;
        d.a = e->raw[p.src_a];
        d.b = e->raw[p.src_b];
        d.len_a = limb_bound(k);
        d.len_b = limb_bound(k + 1);
        d.len_t = std::max(d.len_a, d.len_b) + 1;
        d.bound[0] = limb_bound(2 * k);
        d.bound[1] = limb_bound(2 * k + 1);
        d.bound[2] = limb_bound(2 * k + 2);
        // c+d is the longest product; +1 guard limb, clamped to the
        // transform output extent.
        u32 L = d.bound[2] + 2;
        if (L > nstep + 1) L = nstep + 1;
        d.L = L;
        e->pin_desc[i] = d;
        k = 2 * k + (u64)p.bit;
    }

    CUDA_TRY_I(cudaMemsetAsync(e->err, 0, 4, e->stream));
    *e->pin_one = 1u;
    // Seed: b = 1 in raw_b; a = F(0) is never read (len_a = 0).
    CUDA_TRY_I(cudaMemcpyAsync(e->raw[1], e->pin_one, 4,
                               cudaMemcpyHostToDevice, e->stream));
    // One upload for the whole deterministic descriptor schedule.
    CUDA_TRY_I(cudaMemcpyAsync(e->d_desc, e->pin_desc,
                               plans.size() * sizeof(StepDesc),
                               cudaMemcpyHostToDevice, e->stream));

    cudaEventRecord(e->ev[0], e->stream);
    const auto host_start = std::chrono::steady_clock::now();
    if (g_use_graph) {
        if (!launch_loop(e, n, plans)) return 3;
    } else {
        for (size_t i = 0; i < plans.size(); ++i) {
            if (g_profile) cudaEventRecord(e->sev[i], e->stream);
            if (!enqueue_step(e, plans[i].lg, e->d_desc + i, e->pin_desc[i].L)) return 3;
        }
        if (g_profile) cudaEventRecord(e->sev[plans.size()], e->stream);
    }
    for (size_t i = 0; i < plans.size(); ++i) {
        const int tl = ilog2_u32((u32)std::min(
            (u32)kTile, (u32)(1u << plans[i].lg)));
        const uint64_t passes = (uint64_t)(2 * (plans[i].lg - tl + 1));
        e->stats.steps += 1;
        e->stats.transforms += 15; // 3 forward + 2 inverse, times 3 primes
        e->stats.stage_kernels += passes;
        e->stats.other_kernels += 8;
        e->stats.launches += passes + 8;
        if ((uint64_t)1u << plans[i].lg > e->stats.max_ntt_length) {
            e->stats.max_ntt_length = (uint64_t)1u << plans[i].lg;
        }
    }

    // F(n) is the `a` state after the last step: raw_c when the last bit is
    // zero, raw_d when it is one.
    const StepPlan& last = plans.back();
    const int final_src = last.bit ? 3 : 2;
    CUDA_TRY_I(cudaMemcpyAsync(e->pin_out, e->raw[final_src],
                               (size_t)result_limbs * 4,
                               cudaMemcpyDeviceToHost, e->stream));
    cudaEventRecord(e->ev[1], e->stream);
    if (!sync_and_check(e)) return 4;

    if (g_profile) {
        double tot = 0.0;
        std::fprintf(stderr, "resident27 step ms (last %zu):",
                     plans.size() < (size_t)8 ? plans.size() : (size_t)8);
        for (size_t i = 0; i < plans.size(); ++i) {
            float t = 0.f;
            cudaEventElapsedTime(&t, e->sev[i], e->sev[i + 1]);
            tot += t;
            if (i + 8 >= plans.size()) std::fprintf(stderr, " %.3f", t);
        }
        std::fprintf(stderr, "  sum=%.3f\n", tot);
    }

    if (g_profile) {
        static const char* names[10] = {
            "sub_sum", "sub_scan", "sub_apply", "fwd",    "inv+pw",
            "crt",     "combine",  "carry_sum", "carry_scan", "carry_apply"};
        float cur = 0.f;
        std::fprintf(stderr, "resident27 phase breakdown (last step, ms):");
        for (int i = 0; i < 10; ++i) {
            cudaEventElapsedTime(&cur, e->pev[i], e->pev[i + 1]);
            std::fprintf(stderr, " %s=%.3f", names[i], cur);
        }
        std::fprintf(stderr, "\n");
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, e->ev[0], e->ev[1]);
    const auto host_end = std::chrono::steady_clock::now();
    e->stats.gpu_ms = ms;
    e->stats.host_ms =
        std::chrono::duration<double, std::milli>(host_end - host_start)
            .count();
    e->stats.peak_limbs = result_limbs;

    size_t count = 0;
    u32* buf = dup_pinned(e->pin_out, result_limbs, &count);
    if (count != 0 && !buf) return 4;
    *out_limbs = buf;
    *out_count = count;
    return 0;
}

extern "C" RESIDENT_API int gpu_ntt_doubling_step(
    const uint32_t* a, size_t a_limbs, const uint32_t* b, size_t b_limbs,
    uint32_t bit, uint32_t** out_a, size_t* out_a_limbs, uint32_t** out_b,
    size_t* out_b_limbs) {
    if (!out_a || !out_a_limbs || !out_b || !out_b_limbs) return 1;
    *out_a = nullptr;
    *out_b = nullptr;
    *out_a_limbs = 0;
    *out_b_limbs = 0;

    Engine* e = &g_engine;
    e->stats = gpu_fib_stats();
    if (!engine_init()) return 3;

    size_t la = a_limbs;
    while (la > 0 && a[la - 1] == 0) --la;
    size_t lb = b_limbs;
    while (lb > 0 && b[lb - 1] == 0) --lb;
    if (la > 0xffffffffull || lb > 0xffffffffull) return 1;

    u32 m = (u32)std::max(la, lb);
    u32 req = 2 * m + 2;
    if (req < 4) req = 4;
    int lg = ilog2_u32(req);
    if (lg > MAX_NTT_LOG2) return 2;
    if (!ensure_capacity(e, lg)) return 3;

    const int total = (1 << lg) + 4;
    std::memset(e->pin_in[0], 0, (size_t)total * 4);
    std::memset(e->pin_in[1], 0, (size_t)total * 4);
    if (la) std::memcpy(e->pin_in[0], a, la * 4);
    if (lb) std::memcpy(e->pin_in[1], b, lb * 4);
    CUDA_TRY_I(cudaMemcpyAsync(e->raw[0], e->pin_in[0], (size_t)total * 4,
                               cudaMemcpyHostToDevice, e->stream));
    CUDA_TRY_I(cudaMemcpyAsync(e->raw[1], e->pin_in[1], (size_t)total * 4,
                               cudaMemcpyHostToDevice, e->stream));

    StepDesc d{};
    const u32 n = 1u << lg;
    d.a = e->raw[0];
    d.b = e->raw[1];
    d.len_a = (u32)la;
    d.len_b = (u32)lb;
    d.len_t = m + 1;
    // The single-step ABI does not know the index, so the Binet bounds are
    // unavailable; the buffers hold n+2 limbs by construction.
    d.bound[0] = d.bound[1] = d.bound[2] = n + 2;
    d.L = n + 1;
    e->pin_desc[0] = d;

    CUDA_TRY_I(cudaMemsetAsync(e->err, 0, 4, e->stream));
    CUDA_TRY_I(cudaMemcpyAsync(e->desc, &e->pin_desc[0], sizeof(StepDesc),
                               cudaMemcpyHostToDevice, e->stream));
    cudaEventRecord(e->ev[0], e->stream);
    if (!launch_step(e, lg)) return 3;

    const int nres = (int)n + 2;
    const int src_a = bit ? 3 : 2;
    const int src_b = bit ? 4 : 3;
    CUDA_TRY_I(cudaMemcpyAsync(e->pin_out, e->raw[src_a], (size_t)nres * 4,
                               cudaMemcpyDeviceToHost, e->stream));
    CUDA_TRY_I(cudaMemcpyAsync(e->pin_in[0], e->raw[src_b], (size_t)nres * 4,
                               cudaMemcpyDeviceToHost, e->stream));
    cudaEventRecord(e->ev[1], e->stream);
    if (!sync_and_check(e)) return 4;

    float ms = 0;
    cudaEventElapsedTime(&ms, e->ev[0], e->ev[1]);
    const int tl = ilog2_u32((u32)std::min((u32)kTile, n));
    const uint64_t passes = (uint64_t)(2 * (lg - tl + 1));
    e->stats.gpu_ms = ms;
    e->stats.steps = 1;
    e->stats.transforms = 15;
    e->stats.stage_kernels = passes;
    e->stats.other_kernels = 8;
    e->stats.launches = passes + 8;
    e->stats.max_ntt_length = n;

    size_t ca = 0, cb = 0;
    u32* ba = dup_pinned(e->pin_out, (size_t)nres, &ca);
    u32* bb = dup_pinned(e->pin_in[0], (size_t)nres, &cb);
    if ((ca != 0 && !ba) || (cb != 0 && !bb)) {
        if (ba) std::free(ba);
        if (bb) std::free(bb);
        return 4;
    }
    *out_a = ba;
    *out_a_limbs = ca;
    *out_b = bb;
    *out_b_limbs = cb;
    return 0;
}

extern "C" RESIDENT_API void gpu_fib_free_limbs(uint32_t* limbs) {
    std::free(limbs);
}

extern "C" RESIDENT_API int gpu_fib_get_stats(gpu_fib_stats* out) {
    if (!out) return 1;
    *out = g_engine.stats;
    return 0;
}

extern "C" RESIDENT_API void gpu_fib_shutdown(void) {
    if (g_engine.stream) {
        cudaStreamSynchronize(g_engine.stream);
    }
    free_engine(&g_engine);
}

extern "C" RESIDENT_API const char* gpu_fib_backend_name(void) {
    return "resident-27-27 3-prime NTT (2281701377, 3489660929, 3892314113), "
           "sm_120, DIF/DIT + 12-stage shared tail, 5 transforms per prime "
           "per doubling step, widened 32-bit Montgomery (p < 2^32), "
           "MAX_NTT_LOG2=27";
}
