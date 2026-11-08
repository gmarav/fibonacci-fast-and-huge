// block-27: exact blockwise multiplication / fast doubling on top of the
// validated cuda-ntt/resident-27 2^27 3x32-bit NTT core.
//
// The resident engine multiplies operands of at most 2^26 - 1 limbs with one
// 2^27 transform.  This file keeps that core unchanged (it is included
// verbatim) and adds a block-convolution driver for much larger operands:
//
//   * a block is B = 2^26 32-bit limbs (256 MiB); two blocks convolve into
//     exactly 2^27 coefficients, so one 3-prime 2^27 NTT pair is exact with
//     the same 3-prime product P ~= 2^94.65 and a 25x coefficient margin;
//   * an operand of up to k blocks is multiplied by schoolbook block
//     convolution; the accumulator is indexed by the diagonal s = i + j, so
//     only a 2-block window (2B coefficients) is live at a time;
//   * each block pair is forward-transformed on the device straight from the
//     operand limbs (the first forward pass already folds the mod-p
//     reduction), pointwise-multiplied and inverse-transformed on the device;
//   * the CRT result is accumulated as 64-bit redundant digits directly into
//     the diagonal window, so no per-pair carry chain is needed;
//   * one exact two-level carry pass per diagonal normalises the window, the
//     finalised low block is copied to the host, and the high block becomes
//     the carry-in of the next diagonal.
//
// All arithmetic is the validated 32-bit Montgomery / Shoup / Garner path
// from resident-27.  No floating point is used anywhere.

#include "../resident-27/resident27.cu"

// ---------------------------------------------------------------------------
// Extra kernels
// ---------------------------------------------------------------------------

// Fused inverse tail for the square-sum product:
//   out = (A_i * A_j + B_i * B_j) * n^-1
// The pointwise product is folded into the tile load exactly like
// k_tail_pointwise, and the DIT tail then runs in shared memory.  The output
// overwrites the B_j set; that set is dead after this pair, and tiles are
// disjoint, so the in-place overwrite is race-free.
struct PairArgs {
    u32* out[3];
    const u32* a0[3];
    const u32* a1[3];
    const u32* b0[3];
    const u32* b1[3];
    const u32* tw[3];
    u32 p[3];
    u32 np[3];
    u32 scale[3];
    u32 n;
    u32 cap;
    u32 tile;
    u32 start_half;
    u32 have_a;
    u32 have_b;
};

__global__ void k_tail_pw_pair(PairArgs arg) {
    const int pr = blockIdx.y % 3;
    u32* a = arg.out[pr];
    const u32* tw = arg.tw[pr];
    const u32 p = arg.p[pr];
    const u32 np = arg.np[pr];
    const u32 scale = arg.scale[pr];
    const u32 tile = arg.tile;
    const u32 have_a = arg.have_a;
    const u32 have_b = arg.have_b;

    extern __shared__ u32 sh[];
    u32* s = sh;
    u32* sw = sh + tile;
    const u32 base = blockIdx.x * tile;

    for (u32 i = threadIdx.x; i < tile; i += blockDim.x) {
        u32 acc = 0u;
        if (have_a)
            acc = dmul(arg.a0[pr][base + i], arg.a1[pr][base + i], p, np);
        if (have_b) {
            const u32 bb = dmul(arg.b0[pr][base + i], arg.b1[pr][base + i], p, np);
            acc = dadd(acc, bb, p);
        }
        s[i] = dmul(acc, scale, p, np);
    }
    __syncthreads();

    for (u32 half = 1u; half <= arg.start_half; half <<= 1) {
        const u32 lh = dlog2(half);
        const u32 off = arg.cap - (half << 1);
        for (u32 j = threadIdx.x; j < half; j += blockDim.x) sw[j] = tw[off + j];
        __syncthreads();
        for (u32 k = threadIdx.x; k < (tile >> 1); k += blockDim.x) {
            const u32 j = k & (half - 1u);
            const u32 grp = k >> lh;
            const u32 i0 = (grp << (lh + 1u)) | j;
            const u32 i1 = i0 + half;
            const u32 u = s[i0];
            const u32 v = dmul(s[i1], sw[j], p, np);
            const u32 su = dadd(u, v, p);
            u32 d = u - v;
            if (u < v) d += p;
            s[i0] = su;
            s[i1] = d;
        }
        __syncthreads();
    }

    for (u32 i = threadIdx.x; i < tile; i += blockDim.x) a[base + i] = s[i];
}

// CRT of one pair's three residue arrays and accumulation into the diagonal
// window as 64-bit redundant digits:
//   acc[i] += weight * (w0[i] + w1[i-1] + w2[i-2])
// The two halo coefficients before each block are recomputed exactly as in
// k_crt.  After a full diagonal of at most 10 pairs with weight <= 2 every
// digit is below 60*2^32, which the fold + two-level carry normalises exactly.
__global__ void k_crt_acc(u64* __restrict__ acc, const u32* __restrict__ r0,
                          const u32* __restrict__ r1,
                          const u32* __restrict__ r2, u32 nout, u32 L,
                          u32 weight, CrtArgs c) {
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

    if (idx < L) {
        const u64 v = (u64)s0[t + 2] + (u64)s1[t + 1] + (u64)s2[t];
        acc[idx] += (u64)weight * v;
    }
}

// Split the 64-bit redundant digit of a diagonal window: hi[i] = acc[i] >> 32.
__global__ void k_fold_hi(const u64* __restrict__ acc, u32* __restrict__ hi,
                          u32 L) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    hi[i] = (u32)(acc[i] >> 32);
}

// acc[i] = (acc[i] mod 2^32) + hi[i-1].  After this every digit is below
// 2^32 + 59, i.e. below 6*2^32, so the validated carry chain (digits below
// 6*2^32, carries 0..5) applies unchanged.
__global__ void k_fold_lo(u64* __restrict__ acc, const u32* __restrict__ hi,
                          u32 L) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    acc[i] = (acc[i] & 0xffffffffull) + (i ? (u64)hi[i - 1] : 0ull);
}

// Carry the still-live high block of a finished diagonal into the next
// diagonal's window:  acc[i] = limb[B + i] for i <= B, zero above.
__global__ void k_acc_reinit(u64* __restrict__ acc,
                             const u32* __restrict__ limb, u32 B, u32 L) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    acc[i] = (i <= B) ? (u64)limb[B + i] : 0ull;
}

// ---------------------------------------------------------------------------
// Block engine state
// ---------------------------------------------------------------------------

struct BlockCtx {
    int B_log2 = 0;
    u32 B = 0;
    u32 n = 0;   // transform length = 2*B
    u32 cap = 0; // twiddle table capacity = n
    u32* ntt[12] = {};
    u32* tw[3] = {};
    u32* twi[3] = {};
    u64* acc = nullptr;
    u32* limb = nullptr; // n + 8
    u32* hi = nullptr;   // n + 8 fold scratch
    u32* err = nullptr;
    u32* err_carry = nullptr;
    u32* b_sums = nullptr;
    u32* b_cin = nullptr;
    size_t nsub_cap = 0;
    u32* op[3] = {}; // a, b, t (device, block padded)
    size_t op_cap[3] = {};
    StepDesc* desc_dev = nullptr;
    StepDesc* desc_host = nullptr;
    size_t desc_cap = 0;
    u32* pin_stage = nullptr;
    size_t pin_stage_limbs = 0;
    const StepDesc* bound_desc = nullptr; // dummy descriptor (carry bounds)
    cudaStream_t stream = nullptr;
    cudaEvent_t dev_ev[2] = {};
    double device_ms = 0.0;
    uint64_t pcie_h2d = 0, pcie_d2h = 0;
    u32 scales[MAX_NTT_LOG2 + 1][3] = {};
    bool ready = false;
    uint64_t pairs = 0;
    uint64_t pcie_bytes = 0;
    double ms_fwd = 0.0;
    double ms_inv = 0.0;
    double ms_crt = 0.0;
    double ms_carry = 0.0;
    double ms_xfer = 0.0;
    uint64_t kernels = 0;
};

static BlockCtx g_blk;

static bool blk_diag_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* s = std::getenv("BLOCK27_DIAG");
        v = (s && *s && *s != '0') ? 1 : 0;
    }
    return v == 1;
}

static bool blk_error(const char* msg) {
    std::fprintf(stderr, "block27: %s\n", msg);
    return false;
}

static bool blk_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "block27: %s: %s\n", what, cudaGetErrorString(err));
    return false;
}

static void blk_free() {
    BlockCtx& c = g_blk;
    if (c.stream) cudaStreamSynchronize(c.stream);
    for (int i = 0; i < 2; ++i) {
        if (c.dev_ev[i]) cudaEventDestroy(c.dev_ev[i]);
        c.dev_ev[i] = nullptr;
    }
    for (int i = 0; i < 12; ++i) {
        if (c.ntt[i]) cudaFree(c.ntt[i]);
        c.ntt[i] = nullptr;
    }
    for (int i = 0; i < 3; ++i) {
        if (c.tw[i]) cudaFree(c.tw[i]);
        if (c.twi[i]) cudaFree(c.twi[i]);
        c.tw[i] = c.twi[i] = nullptr;
        if (c.op[i]) cudaFree(c.op[i]);
        c.op[i] = nullptr;
        c.op_cap[i] = 0;
    }
    if (c.acc) cudaFree(c.acc);
    if (c.limb) cudaFree(c.limb);
    if (c.hi) cudaFree(c.hi);
    if (c.err) cudaFree(c.err);
    if (c.err_carry) cudaFree(c.err_carry);
    if (c.b_sums) cudaFree(c.b_sums);
    if (c.b_cin) cudaFree(c.b_cin);
    if (c.desc_dev) cudaFree(c.desc_dev);
    if (c.desc_host) cudaFreeHost(c.desc_host);
    if (c.pin_stage) cudaFreeHost(c.pin_stage);
    c.acc = nullptr;
    c.limb = nullptr;
    c.hi = nullptr;
    c.err = nullptr;
    c.err_carry = nullptr;
    c.b_sums = nullptr;
    c.b_cin = nullptr;
    c.nsub_cap = 0;
    c.desc_dev = nullptr;
    c.desc_host = nullptr;
    c.pin_stage = nullptr;
    c.desc_cap = 0;
    c.pin_stage_limbs = 0;
    c.ready = false;
    c.B_log2 = 0;
}

// Twiddles and scales -------------------------------------------------------

static bool blk_build_tables(BlockCtx& c) {
    const int cap = (int)c.cap;
    const size_t bytes = (size_t)(cap - 1) * 4;
    for (int k = 0; k < 3; ++k) {
        std::vector<u32> tf, ti;
        make_twiddle_tables(cap, k, tf, ti);
        if (!blk_cuda(cudaMemcpy(c.tw[k], tf.data(), bytes, cudaMemcpyHostToDevice),
                      "tw upload"))
            return false;
        if (!blk_cuda(cudaMemcpy(c.twi[k], ti.data(), bytes, cudaMemcpyHostToDevice),
                      "twi upload"))
            return false;
    }
    for (int lg = 1; lg <= MAX_NTT_LOG2; ++lg) {
        const u32 nn = 1u << lg;
        for (int k = 0; k < 3; ++k) {
            const PrimeInfo& pi = g_prime[k];
            const u32 ninv = hpowmod(nn, (u64)pi.p - 2, pi.p);
            c.scales[lg][k] = (u32)(((u64)pi.r2 * ninv) % pi.p);
        }
    }
    return true;
}

// Configure the engine for block size 2^B_log2.  Transform length is 2*B.
static bool blk_configure(int B_log2) {
    BlockCtx& c = g_blk;
    if (B_log2 < 8 || B_log2 + 1 > MAX_NTT_LOG2)
        return blk_error("bad block size");
    if (c.ready && c.B_log2 == B_log2) return true;
    blk_free();
    if (!init_primes()) return false;
    if (!c.stream) {
        if (!blk_cuda(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking),
                      "stream"))
            return false;
    }
    if (!c.dev_ev[0]) {
        if (!blk_cuda(cudaEventCreate(&c.dev_ev[0]), "ev0")) return false;
        if (!blk_cuda(cudaEventCreate(&c.dev_ev[1]), "ev1")) return false;
    }
    c.B_log2 = B_log2;
    c.B = 1u << B_log2;
    c.n = c.B << 1;
    c.cap = c.n;

    for (int i = 0; i < 12; ++i) {
        if (!blk_cuda(cudaMalloc(&c.ntt[i], ((size_t)c.n + 8) * 4), "ntt malloc"))
            return false;
        if (!blk_cuda(cudaMemset(c.ntt[i] + c.n, 0, 8 * 4), "ntt guard")) return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!blk_cuda(cudaMalloc(&c.tw[i], (size_t)(c.cap - 1) * 4), "tw malloc"))
            return false;
        if (!blk_cuda(cudaMalloc(&c.twi[i], (size_t)(c.cap - 1) * 4), "twi malloc"))
            return false;
    }
    if (!blk_cuda(cudaMalloc(&c.acc, (size_t)(c.n + 8) * 8), "acc")) return false;
    if (!blk_cuda(cudaMalloc(&c.limb, (size_t)(c.n + 8) * 4), "limb")) return false;
    if (!blk_cuda(cudaMalloc(&c.hi, (size_t)(c.n + 8) * 4), "hi")) return false;
    if (!blk_cuda(cudaMalloc(&c.err, 4), "err")) return false;
    if (!blk_cuda(cudaMalloc(&c.err_carry, 4), "err_carry")) return false;
    if (!blk_build_tables(c)) return false;
    c.ready = true;
    return true;
}

// (Re)allocate the operand buffers for a step with la / lb / lt limbs.
static bool blk_alloc_ops(BlockCtx& c, size_t la, size_t lb, size_t lt,
                          bool want_t) {
    const size_t need[3] = {la, lb, want_t ? lt : 0};
    for (int i = 0; i < 3; ++i) {
        size_t want = ((need[i] + c.B - 1) / c.B) * (size_t)c.B;
        if (want == 0) want = (size_t)c.B;
        if (c.op_cap[i] >= want && c.op[i]) continue;
        if (c.op[i]) cudaFree(c.op[i]);
        c.op[i] = nullptr;
        if (!blk_cuda(cudaMalloc(&c.op[i], want * 4), "op malloc")) return false;
        c.op_cap[i] = want;
    }
    const size_t maxlen = want_t ? (lt + 1) : (std::max(la, lb) + 1);
    const size_t nsub = (maxlen + kSubBlock - 1) / kSubBlock + 1;
    if (c.nsub_cap < nsub) {
        if (c.b_sums) cudaFree(c.b_sums);
        if (c.b_cin) cudaFree(c.b_cin);
        c.b_sums = nullptr;
        c.b_cin = nullptr;
        if (!blk_cuda(cudaMalloc(&c.b_sums, nsub * 4), "bsums")) return false;
        if (!blk_cuda(cudaMalloc(&c.b_cin, (nsub + 1) * 4), "bcin")) return false;
        c.nsub_cap = nsub;
    }
    if (c.pin_stage_limbs < (size_t)2 * c.B + 8) {
        if (c.pin_stage) cudaFreeHost(c.pin_stage);
        c.pin_stage = nullptr;
        if (!blk_cuda(cudaMallocHost(&c.pin_stage, ((size_t)2 * c.B + 8) * 4),
                      "pin"))
            return false;
        c.pin_stage_limbs = (size_t)2 * c.B + 8;
    }
    return true;
}

// Descriptor scratch: one entry per block of each operand plus two whole-value
// entries (borrow chain, carry bound).
static bool blk_alloc_desc(BlockCtx& c, size_t entries) {
    if (c.desc_cap >= entries) return true;
    if (c.desc_dev) cudaFree(c.desc_dev);
    if (c.desc_host) cudaFreeHost(c.desc_host);
    c.desc_dev = nullptr;
    c.desc_host = nullptr;
    if (!blk_cuda(cudaMalloc(&c.desc_dev, entries * sizeof(StepDesc)), "desc"))
        return false;
    if (!blk_cuda(cudaMallocHost(&c.desc_host, entries * sizeof(StepDesc)),
                  "desc pin"))
        return false;
    c.desc_cap = entries;
    return true;
}

// ---------------------------------------------------------------------------
// Transform launches (one operand / one output at a time, 3 primes in y)
// ---------------------------------------------------------------------------

static void blk_launch_fwd(BlockCtx& c, int set, const StepDesc* dptr, int lg) {
    const u32 n = 1u << lg;
    const u32 tile = (n < (u32)kTile) ? n : (u32)kTile;
    const int tl = ilog2_u32(tile);
    const int g = lg - tl;
    const int grid_b = (int)((n / 2 + kStageThreads - 1) / kStageThreads);

    NttArgs a{};
    for (int k = 0; k < 3; ++k) {
        a.data[k] = c.ntt[3 * set + k];
        a.tw[k] = c.tw[k];
        a.p[k] = g_prime[k].p;
        a.np[k] = g_prime[k].np;
    }
    a.n = n;
    a.cap = (u32)c.cap;
    a.tile = tile;
    a.start_half = tile >> 1;
    a.d = dptr;
    a.raw_t = c.op[2];

    for (int i = 0; i < g; ++i) {
        const u32 half = (n >> 1) >> i;
        a.half = half;
        a.off = (u32)c.cap - (half << 1);
        a.lh = (u32)(lg - 1 - i);
        a.is_first = (i == 0) ? 1 : 0;
        k_stage_multi<0><<<dim3(grid_b, 3), kStageThreads, 0, c.stream>>>(a);
        ++c.kernels;
    }
    const size_t shmem = (size_t)(tile + (tile >> 1)) * 4;
    a.is_first = (g == 0) ? 1 : 0;
    k_tail_multi<0><<<dim3(n / tile, 3), kStageThreads, shmem, c.stream>>>(a);
    ++c.kernels;
}

// Inverse c = A * T (pointwise fused into the DIT tail, T set overwritten).
static void blk_launch_inv_c(BlockCtx& c, int set_a, int set_t, int lg) {
    const u32 n = 1u << lg;
    const u32 tile = (n < (u32)kTile) ? n : (u32)kTile;
    const int tl = ilog2_u32(tile);
    const int g = lg - tl;
    const int grid_b = (int)((n / 2 + kStageThreads - 1) / kStageThreads);

    NttArgs a{};
    for (int k = 0; k < 3; ++k) {
        a.data[k] = c.ntt[3 * set_t + k];
        a.tw[k] = c.twi[k];
        a.p[k] = g_prime[k].p;
        a.np[k] = g_prime[k].np;
        a.pw_a[k] = c.ntt[3 * set_a + k];
        a.pw_t[k] = c.ntt[3 * set_t + k];
        a.pw_b[k] = c.ntt[3 * set_t + k];
        a.scale[k] = c.scales[lg][k];
    }
    a.n = n;
    a.cap = (u32)c.cap;
    a.tile = tile;
    a.start_half = tile >> 1;

    const size_t shmem = (size_t)(tile + (tile >> 1)) * 4;
    k_tail_pointwise<<<dim3(n / tile, 3), kStageThreads, shmem, c.stream>>>(a);
    ++c.kernels;
    for (int i = 0; i < g; ++i) {
        const u32 half = tile << i;
        a.half = half;
        a.off = (u32)c.cap - (half << 1);
        a.lh = (u32)(tl + i);
        k_stage_multi<1><<<dim3(grid_b, 3), kStageThreads, 0, c.stream>>>(a);
        ++c.kernels;
    }
}

// Inverse d = A_i*A_j + B_i*B_j, written over the B_j (set_out) arrays.
static void blk_launch_inv_pair(BlockCtx& c, int set_ai, int set_aj,
                                int set_bi, int set_bj, int set_out,
                                u32 have_a, u32 have_b, int lg) {
    const u32 n = 1u << lg;
    const u32 tile = (n < (u32)kTile) ? n : (u32)kTile;
    const int tl = ilog2_u32(tile);
    const int g = lg - tl;
    const int grid_b = (int)((n / 2 + kStageThreads - 1) / kStageThreads);

    PairArgs p{};
    for (int k = 0; k < 3; ++k) {
        p.out[k] = c.ntt[3 * set_out + k];
        p.a0[k] = c.ntt[3 * set_ai + k];
        p.a1[k] = c.ntt[3 * set_aj + k];
        p.b0[k] = c.ntt[3 * set_bi + k];
        p.b1[k] = c.ntt[3 * set_bj + k];
        p.tw[k] = c.twi[k];
        p.p[k] = g_prime[k].p;
        p.np[k] = g_prime[k].np;
        p.scale[k] = c.scales[lg][k];
    }
    p.n = n;
    p.cap = (u32)c.cap;
    p.tile = tile;
    p.start_half = tile >> 1;
    p.have_a = have_a;
    p.have_b = have_b;

    const size_t shmem = (size_t)(tile + (tile >> 1)) * 4;
    k_tail_pw_pair<<<dim3(n / tile, 3), kStageThreads, shmem, c.stream>>>(p);
    ++c.kernels;

    NttArgs a{};
    for (int k = 0; k < 3; ++k) {
        a.data[k] = c.ntt[3 * set_out + k];
        a.tw[k] = c.twi[k];
        a.p[k] = g_prime[k].p;
        a.np[k] = g_prime[k].np;
    }
    a.n = n;
    a.cap = (u32)c.cap;
    a.tile = tile;
    a.start_half = tile >> 1;
    for (int i = 0; i < g; ++i) {
        const u32 half = tile << i;
        a.half = half;
        a.off = (u32)c.cap - (half << 1);
        a.lh = (u32)(tl + i);
        k_stage_multi<1><<<dim3(grid_b, 3), kStageThreads, 0, c.stream>>>(a);
        ++c.kernels;
    }
}

// CRT + accumulate for one pair: set_out holds the three residue arrays.
// nout = n+1 and L = n+1 so the w2 word of the top coefficient (shifted by
// two digit positions) is captured, which is exactly what makes a^2+b^2 one
// digit wider than a single product.  The transform arrays are allocated with
// 8 guard elements that stay zero.
static void blk_launch_crt_acc(BlockCtx& c, int set_out, u32 L, u32 weight) {
    const u32 nout = c.n + 1;
    const int grid = (int)((nout + kCrtBlock - 1) / kCrtBlock);
    k_crt_acc<<<grid, kCrtBlock, 0, c.stream>>>(
        c.acc, c.ntt[3 * set_out + 0], c.ntt[3 * set_out + 1],
        c.ntt[3 * set_out + 2], nout, L, weight, g_crt);
    ++c.kernels;
}

// ---------------------------------------------------------------------------
// Diagonal finalisation: fold, exact carry, reinit, download
// ---------------------------------------------------------------------------

// Normalise the diagonal window in acc, leave the limbs in c.limb, and return
// the carry out of the window (limb[n]).
static bool blk_finalize_diag(BlockCtx& c, u32* carry_out) {
    const u32 n = c.n;
    // A diagonal can be the sum of up to 10 pair products, so its value can
    // exceed 2^(32*2B); process one extra digit to capture that overflow.
    const u32 L = n + 1;
    const u32 nb = (L + kCarryBlock - 1) / kCarryBlock;
    const int threads = 256;
    const int grid = (int)((L + threads - 1) / threads);

    u32 flags = 0;
    if (!blk_cuda(cudaMemsetAsync(c.err, 0, 4, c.stream), "err clear"))
        return false;

    // The u32 fold scratch is a dedicated buffer (L = n+1 digits).
    k_fold_hi<<<grid, threads, 0, c.stream>>>(c.acc, c.hi, L);
    k_fold_lo<<<grid, threads, 0, c.stream>>>(c.acc, c.hi, L);
    c.kernels += 2;
    if (blk_diag_enabled()) {
        cudaStreamSynchronize(c.stream);
        u64 a23 = 0, a24 = 0;
        cudaMemcpy(&a23, c.acc + n - 1, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&a24, c.acc + n, 8, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "    fold: acc[n-1]=%llx acc[n]=%llx\n",
                     (unsigned long long)a23, (unsigned long long)a24);
    }

    k_carry_sum<<<dim3(nb, 1), kCarryThreads, 0, c.stream>>>(
        c.acc, c.acc, c.acc, c.ntt[7], L, nb);
    k_carry_scan<<<1, 1, 0, c.stream>>>(c.ntt[7], c.ntt[8], c.err_carry, nb);
    k_carry_apply<<<dim3(nb, 1), kCarryThreads, 0, c.stream>>>(
        c.limb, c.limb, c.limb, c.acc, c.acc, c.acc, c.ntt[8], c.bound_desc,
        c.err, L, nb);
    c.kernels += 3;

    if (!blk_cuda(cudaStreamSynchronize(c.stream), "finalize sync")) return false;
    if (blk_diag_enabled()) {
        u32 l23 = 0, l24 = 0, l25 = 0;
        cudaMemcpy(&l23, c.limb + n - 1, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&l24, c.limb + n, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&l25, c.limb + n + 1, 4, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "    carry: limb[n-1]=%08x limb[n]=%08x limb[n+1]=%08x\n",
                     l23, l24, l25);
    }
    if (!blk_cuda(cudaMemcpy(&flags, c.err, 4, cudaMemcpyDeviceToHost), "flags"))
        return false;
    if (flags != 0) {
        std::fprintf(stderr, "block27: diagonal self-check flags 0x%x\n", flags);
        return false;
    }
    u32 carry = 0;
    if (!blk_cuda(cudaMemcpy(&carry, c.limb + L, 4, cudaMemcpyDeviceToHost),
                  "carry"))
        return false;
    *carry_out = carry;
    return true;
}

static bool blk_download(BlockCtx& c, u32* out, u32 offset_limbs, u32 count) {
    if (!blk_cuda(cudaMemcpyAsync(c.pin_stage, c.limb, (size_t)count * 4,
                                  cudaMemcpyDeviceToHost, c.stream),
                  "d2h"))
        return false;
    if (!blk_cuda(cudaStreamSynchronize(c.stream), "d2h sync")) return false;
    std::memcpy(out + offset_limbs, c.pin_stage, (size_t)count * 4);
    c.pcie_bytes += (uint64_t)count * 4;
    c.pcie_d2h += (uint64_t)count * 4;
    return true;
}

static void blk_reinit_acc(BlockCtx& c) {
    const u32 L = c.n + 1;
    const int threads = 256;
    const int grid = (int)((L + threads - 1) / threads);
    k_acc_reinit<<<grid, threads, 0, c.stream>>>(c.acc, c.limb, c.B, L);
    ++c.kernels;
    if (blk_diag_enabled()) {
        cudaStreamSynchronize(c.stream);
        u64 v = 0, lim = 0;
        cudaMemcpy(&v, c.acc + c.B, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&lim, c.limb + 2 * c.B, 4, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "    reinit: acc[B]=%llu limb[2B]=%llu\n",
                     (unsigned long long)v, (unsigned long long)lim);
    }
}

// ---------------------------------------------------------------------------
// Descriptors, upload and t = 2b - a
// ---------------------------------------------------------------------------

// Descriptor layout: [0,ka) a blocks, [ka,ka+kb) b blocks,
// [ka+kb,ka+kb+kt) t blocks, then whole (borrow chain) and dummy (carry).
struct DescPlan {
    size_t ka, kb, kt, whole_idx, dummy_idx, entries;
};

static DescPlan blk_plan(size_t ka, size_t kb, size_t kt) {
    DescPlan p{};
    p.ka = ka;
    p.kb = kb;
    p.kt = kt;
    const size_t kmax = std::max(ka, std::max(kb, kt));
    p.whole_idx = 3 * kmax;
    p.dummy_idx = p.whole_idx + 1;
    p.entries = p.dummy_idx + 1;
    return p;
}

static void blk_fill_desc(BlockCtx& c, const DescPlan& p, size_t la, size_t lb,
                          size_t lt) {
    for (size_t i = 0; i < p.ka; ++i) {
        StepDesc d{};
        d.a = c.op[0] + i * c.B;
        d.len_a = (u32)std::min<size_t>(c.B, la - i * c.B);
        c.desc_host[i] = d;
    }
    for (size_t i = 0; i < p.kb; ++i) {
        StepDesc d{};
        d.a = c.op[1] + i * c.B;
        d.len_a = (u32)std::min<size_t>(c.B, lb - i * c.B);
        c.desc_host[p.ka + i] = d;
    }
    for (size_t i = 0; i < p.kt; ++i) {
        StepDesc d{};
        d.a = c.op[2] + i * c.B;
        d.len_a = (u32)std::min<size_t>(c.B, lt - i * c.B);
        c.desc_host[p.ka + p.kb + i] = d;
    }
    StepDesc whole{};
    whole.a = c.op[0];
    whole.b = c.op[1];
    whole.len_a = (u32)la;
    whole.len_b = (u32)lb;
    whole.len_t = (u32)lt;
    c.desc_host[p.whole_idx] = whole;
    StepDesc dummy{};
    // The carry pass writes L = n + 1 digits (the extra digit holds the
    // diagonal overflow), so the bound check must allow index n.
    dummy.bound[0] = c.n + 8;
    dummy.bound[1] = c.n + 8;
    dummy.bound[2] = c.n + 8;
    c.desc_host[p.dummy_idx] = dummy;
}

static bool blk_upload(const u32* a, size_t la, const u32* b, size_t lb,
                       size_t lt, const DescPlan& p) {
    BlockCtx& c = g_blk;
    const auto t0 = std::chrono::steady_clock::now();
    if (la) {
        if (!blk_cuda(cudaMemcpyAsync(c.op[0], a, la * 4,
                                      cudaMemcpyHostToDevice, c.stream),
                      "upload a"))
            return false;
    }
    if (lb) {
        if (!blk_cuda(cudaMemcpyAsync(c.op[1], b, lb * 4,
                                      cudaMemcpyHostToDevice, c.stream),
                      "upload b"))
            return false;
    }
    c.pcie_bytes += (uint64_t)(la + lb) * 4;
    c.pcie_h2d += (uint64_t)(la + lb) * 4;

    // t = 2b - a, exact block-scan borrow chain (same kernels as the resident
    // engine's fused step).
    const StepDesc whole = c.desc_host[p.whole_idx];
    if (!blk_cuda(cudaMemcpyAsync(c.desc_dev + p.whole_idx, &whole,
                                  sizeof(whole), cudaMemcpyHostToDevice,
                                  c.stream),
                  "whole desc upload"))
        return false;

    const u32 nsub = (u32)((lt + kSubBlock - 1) / kSubBlock);
    SubArgs sa{};
    sa.d = c.desc_dev + p.whole_idx;
    sa.t = c.op[2];
    sa.sums = c.b_sums;
    sa.cin = c.b_cin;
    sa.err = c.err;
    sa.n = (u32)lt;
    k_sub_sum<<<nsub, kSubThreads, 0, c.stream>>>(sa);
    k_sub_scan<<<1, 1, 0, c.stream>>>(c.b_sums, c.b_cin, c.err, nsub);
    k_sub_apply<<<nsub, kSubThreads, 0, c.stream>>>(sa);
    c.kernels += 3;
    if (!blk_cuda(cudaStreamSynchronize(c.stream), "sub sync")) return false;

    u32 flags = 0;
    if (!blk_cuda(cudaMemcpy(&flags, c.err, 4, cudaMemcpyDeviceToHost), "flags"))
        return false;
    if (flags != 0) {
        std::fprintf(stderr, "block27: borrow self-check flags 0x%x\n", flags);
        return false;
    }
    c.ms_xfer += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return true;
}

static bool blk_upload_desc(BlockCtx& c, const DescPlan& p) {
    return blk_cuda(cudaMemcpyAsync(c.desc_dev, c.desc_host,
                                    p.entries * sizeof(StepDesc),
                                    cudaMemcpyHostToDevice, c.stream),
                    "desc upload") &&
           blk_cuda(cudaStreamSynchronize(c.stream), "desc sync");
}

// ---------------------------------------------------------------------------
// Schoolbook block convolution: product = A * B
// ---------------------------------------------------------------------------

static bool blk_conv(BlockCtx& c, int baseA, int nA, int baseB, int nB,
                     u32* out, int lg, uint32_t* out_limbs) {
    const int n_out = nA + nB;
    const u32 B = c.B;

    if (!blk_cuda(cudaMemsetAsync(c.acc, 0, (size_t)(c.n + 8) * 8, c.stream),
                  "acc clear"))
        return false;

    for (int s = 0; s <= nA + nB - 2; ++s) {
        const int ilo = std::max(0, s - (nB - 1));
        const int ihi = std::min(nA - 1, s);
        for (int i = ilo; i <= ihi; ++i) {
            const int j = s - i;
            auto t0 = std::chrono::steady_clock::now();
            blk_launch_fwd(c, 0, c.desc_dev + (baseA + i), lg);
            blk_launch_fwd(c, 1, c.desc_dev + (baseB + j), lg);
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "fwd sync"))
                return false;
            c.ms_fwd += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

            t0 = std::chrono::steady_clock::now();
            blk_launch_inv_c(c, 0, 1, lg);
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "inv sync"))
                return false;
            c.ms_inv += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

            t0 = std::chrono::steady_clock::now();
            blk_launch_crt_acc(c, 1, c.n + 1, 1u);
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "crt sync"))
                return false;
            c.ms_crt += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            ++c.pairs;
        }

        auto tc = std::chrono::steady_clock::now();
        u32 carry = 0;
        if (!blk_finalize_diag(c, &carry)) return false;
        const bool last = (s == nA + nB - 2);
        if (blk_diag_enabled()) {
            u32 top = 0, lo = 0, mid = 0;
            cudaMemcpy(&top, c.limb + c.n, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&lo, c.limb, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&mid, c.limb + c.B, 4, cudaMemcpyDeviceToHost);
            std::fprintf(stderr,
                         "  conv diag s=%d carry=%u top=%08x lo=%08x limbB=%08x\n",
                         s, carry, top, lo, mid);
        }
        if (carry >= 64u) {
            std::fprintf(stderr,
                         "block27: diagonal %d carry %u out of range\n", s, carry);
            return false;
        }
        if (last && carry != 0u) {
            std::fprintf(stderr,
                         "block27: final diagonal carry %u is not zero\n", carry);
            return false;
        }
        u32 count = last ? 2u * B : B;
        if (last) {
            // A single product fits 2B limbs; the top digit must be zero.
            u32 top = 0;
            cudaMemcpy(&top, c.limb + c.n, 4, cudaMemcpyDeviceToHost);
            if (top != 0u) {
                std::fprintf(stderr,
                             "block27: conv final digit %08x is not zero\n", top);
                return false;
            }
        }
        if (!blk_download(c, out, (u32)s * B, count)) return false;
        if (!last) blk_reinit_acc(c);
        if (!blk_cuda(cudaStreamSynchronize(c.stream), "diag sync")) return false;
        c.ms_carry += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tc)
                          .count();
    }
    if (out_limbs) *out_limbs = (uint32_t)n_out * B;
    return true;
}

// ---------------------------------------------------------------------------
// d = a^2 + b^2, symmetric block pairs (i <= j, weight 2 for i < j)
// ---------------------------------------------------------------------------

static bool blk_sum_sq(BlockCtx& c, int ka, int kb, u32* out, int lg,
                       uint32_t* out_limbs) {
    const int K = std::max(ka, kb);
    const u32 B = c.B;

    if (!blk_cuda(cudaMemsetAsync(c.acc, 0, (size_t)(c.n + 8) * 8, c.stream),
                  "acc clear"))
        return false;

    for (int s = 0; s <= 2 * K - 2; ++s) {
        const int ilo = std::max(0, s - (K - 1));
        const int ihi = s / 2;
        for (int i = ilo; i <= ihi; ++i) {
            const int j = s - i;
            const bool have_a = (i < ka && j < ka);
            const bool have_b = (i < kb && j < kb);
            if (!have_a && !have_b) continue;
            const u32 weight = (i == j) ? 1u : 2u;

            auto t0 = std::chrono::steady_clock::now();
            if (have_a) {
                blk_launch_fwd(c, 0, c.desc_dev + i, lg);
                if (i != j) blk_launch_fwd(c, 1, c.desc_dev + j, lg);
            }
            if (have_b) {
                blk_launch_fwd(c, 2, c.desc_dev + (ka + i), lg);
                if (i != j) blk_launch_fwd(c, 3, c.desc_dev + (ka + j), lg);
            }
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "fwd sync"))
                return false;
            c.ms_fwd += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

            t0 = std::chrono::steady_clock::now();
            const int set_ai = 0;
            const int set_aj = (i != j) ? 1 : 0;
            const int set_bi = 2;
            const int set_bj = (i != j) ? 3 : 2;
            blk_launch_inv_pair(c, set_ai, set_aj, set_bi, set_bj, 3,
                                have_a ? 1u : 0u, have_b ? 1u : 0u, lg);
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "inv sync"))
                return false;
            c.ms_inv += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();

            t0 = std::chrono::steady_clock::now();
            blk_launch_crt_acc(c, 3, c.n + 1, weight);
            if (!blk_cuda(cudaStreamSynchronize(c.stream), "crt sync"))
                return false;
            c.ms_crt += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            ++c.pairs;
        }

        auto tc = std::chrono::steady_clock::now();
        u32 carry = 0;
        if (!blk_finalize_diag(c, &carry)) return false;
        const bool last = (s == 2 * K - 2);
        if (blk_diag_enabled()) {
            u32 top = 0, lo = 0, mid = 0;
            cudaMemcpy(&top, c.limb + c.n, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&lo, c.limb, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&mid, c.limb + c.B, 4, cudaMemcpyDeviceToHost);
            std::fprintf(stderr,
                         "  sumsq diag s=%d carry=%u top=%08x lo=%08x limbB=%08x\n",
                         s, carry, top, lo, mid);
            std::vector<u32> dump(c.n + 1);
            cudaMemcpy(dump.data(), c.limb, (size_t)(c.n + 1) * 4,
                       cudaMemcpyDeviceToHost);
            char name[64];
            std::snprintf(name, sizeof(name), "sumsq_diag_%d.bin", s);
            FILE* f = std::fopen(name, "wb");
            if (f) {
                std::fwrite(dump.data(), 4, dump.size(), f);
                std::fclose(f);
            }
        }
        if (carry >= 64u) {
            std::fprintf(stderr,
                         "block27: sum-sq diagonal %d carry %u out of range\n", s,
                         carry);
            return false;
        }
        if (last && carry != 0u) {
            std::fprintf(stderr,
                         "block27: sum-sq final carry %u is not zero\n", carry);
            return false;
        }
        const u32 count = last ? 2u * B + 1u : B;
        if (!blk_download(c, out, (u32)s * B, count)) return false;
        if (!last) blk_reinit_acc(c);
        if (!blk_cuda(cudaStreamSynchronize(c.stream), "diag sync")) return false;
        c.ms_carry += std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tc)
                          .count();
    }
    // a^2 + b^2 can be one digit wider than 2*K*B limbs.
    if (out_limbs) *out_limbs = (uint32_t)(2 * K) * B + 1u;
    return true;
}

// ---------------------------------------------------------------------------
// Internal driver API
// ---------------------------------------------------------------------------

static bool block_product(const u32* a, size_t la, const u32* b, size_t lb,
                          int B_log2, std::vector<u32>* out) {
    BlockCtx& c = g_blk;
    if (!blk_configure(B_log2)) return false;
    if (la == 0 || lb == 0) {
        out->clear();
        return true;
    }
    const size_t B = c.B;
    const size_t ka = (la + B - 1) / B;
    const size_t kb = (lb + B - 1) / B;
    const DescPlan p = blk_plan(ka, kb, 0);
    if (!blk_alloc_desc(c, p.entries)) return false;
    if (!blk_alloc_ops(c, la, lb, 1, false)) return false;
    blk_fill_desc(c, p, la, lb, 0);

    // Upload just the operands; the generic product does not need t.
    const auto t0 = std::chrono::steady_clock::now();
    if (la && !blk_cuda(cudaMemcpyAsync(c.op[0], a, la * 4,
                                        cudaMemcpyHostToDevice, c.stream),
                        "upload a"))
        return false;
    if (lb && !blk_cuda(cudaMemcpyAsync(c.op[1], b, lb * 4,
                                        cudaMemcpyHostToDevice, c.stream),
                        "upload b"))
        return false;
    c.pcie_bytes += (uint64_t)(la + lb) * 4;
    c.pcie_h2d += (uint64_t)(la + lb) * 4;
    if (!blk_upload_desc(c, p)) return false;
    c.bound_desc = c.desc_dev + p.dummy_idx;

    out->assign((ka + kb) * B, 0u);
    uint32_t olen = 0;
    if (!blk_conv(c, 0, (int)ka, (int)ka, (int)kb, out->data(), c.B_log2 + 1,
                  &olen))
        return false;
    out->resize(olen);
    while (!out->empty() && out->back() == 0u) out->pop_back();
    c.ms_xfer += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return true;
}

static bool block_step(const u32* a, size_t la, const u32* b, size_t lb,
                       int B_log2, bool want_d, std::vector<u32>* outc,
                       std::vector<u32>* outd) {
    BlockCtx& c = g_blk;
    if (!blk_configure(B_log2)) return false;
    if (la == 0 || lb == 0) return blk_error("empty block-step operand");
    const size_t B = c.B;
    const size_t ka = (la + B - 1) / B;
    const size_t kb = (lb + B - 1) / B;
    const size_t lt = std::max(la, lb) + 1;
    const size_t kt = (lt + B - 1) / B;
    const DescPlan p = blk_plan(ka, kb, kt);
    if (!blk_alloc_desc(c, p.entries)) return false;
    if (!blk_alloc_ops(c, la, lb, lt, true)) return false;
    blk_fill_desc(c, p, la, lb, lt);
    if (c.dev_ev[0]) cudaEventRecord(c.dev_ev[0], c.stream);
    if (!blk_upload(a, la, b, lb, lt, p)) return false;
    if (!blk_upload_desc(c, p)) return false;
    c.bound_desc = c.desc_dev + p.dummy_idx;

    // c = a * t:  a blocks start at 0, t blocks start at ka + kb.
    outc->assign((ka + kt) * B, 0u);
    uint32_t lc = 0;
    if (!blk_conv(c, 0, (int)ka, (int)(ka + kb), (int)kt, outc->data(),
                  c.B_log2 + 1, &lc))
        return false;
    outc->resize(lc);
    while (!outc->empty() && outc->back() == 0u) outc->pop_back();

    if (want_d) {
        outd->assign(2 * std::max(ka, kb) * B + 1, 0u);
        uint32_t ld = 0;
        if (!blk_sum_sq(c, (int)ka, (int)kb, outd->data(), c.B_log2 + 1, &ld))
            return false;
        outd->resize(ld);
        while (!outd->empty() && outd->back() == 0u) outd->pop_back();
    } else {
        outd->clear();
    }
    if (c.dev_ev[0]) {
        cudaEventRecord(c.dev_ev[1], c.stream);
        cudaStreamSynchronize(c.stream);
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, c.dev_ev[0], c.dev_ev[1]) == cudaSuccess)
            c.device_ms += ms;
    }
    return true;
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" void block27_shutdown(void) { blk_free(); }

extern "C" int block27_is_ready(void) { return g_blk.ready ? 1 : 0; }

extern "C" int block27_configure(int B_log2) {
    return blk_configure(B_log2) ? 0 : 1;
}

extern "C" int block27_mul(const u32* a, size_t la, const u32* b, size_t lb,
                           int B_log2, u32** out, size_t* out_count) {
    if (!out || !out_count) return 1;
    *out = nullptr;
    *out_count = 0;
    std::vector<u32> v;
    if (!block_product(a, la, b, lb, B_log2, &v)) return 3;
    if (!v.empty()) {
        u32* p = (u32*)std::malloc(v.size() * 4);
        if (!p) return 4;
        std::memcpy(p, v.data(), v.size() * 4);
        *out = p;
        *out_count = v.size();
    }
    return 0;
}

extern "C" int block27_step(const u32* a, size_t la, const u32* b, size_t lb,
                            int B_log2, int want_d, u32** out_c,
                            size_t* out_c_count, u32** out_d,
                            size_t* out_d_count) {
    if (!out_c || !out_c_count || !out_d || !out_d_count) return 1;
    *out_c = nullptr;
    *out_d = nullptr;
    *out_c_count = 0;
    *out_d_count = 0;
    std::vector<u32> vc, vd;
    if (!block_step(a, la, b, lb, B_log2, want_d != 0, &vc, &vd)) return 3;
    if (!vc.empty()) {
        u32* p = (u32*)std::malloc(vc.size() * 4);
        if (!p) return 4;
        std::memcpy(p, vc.data(), vc.size() * 4);
        *out_c = p;
        *out_c_count = vc.size();
    }
    if (want_d && !vd.empty()) {
        u32* p = (u32*)std::malloc(vd.size() * 4);
        if (!p) return 4;
        std::memcpy(p, vd.data(), vd.size() * 4);
        *out_d = p;
        *out_d_count = vd.size();
    }
    return 0;
}

extern "C" void block27_get_stats(uint64_t* pairs, uint64_t* pcie_bytes,
                                  uint64_t* kernels, double* ms_fwd,
                                  double* ms_inv, double* ms_crt,
                                  double* ms_carry, double* ms_xfer) {
    if (pairs) *pairs = g_blk.pairs;
    if (pcie_bytes) *pcie_bytes = g_blk.pcie_bytes;
    if (kernels) *kernels = g_blk.kernels;
    if (ms_fwd) *ms_fwd = g_blk.ms_fwd;
    if (ms_inv) *ms_inv = g_blk.ms_inv;
    if (ms_crt) *ms_crt = g_blk.ms_crt;
    if (ms_carry) *ms_carry = g_blk.ms_carry;
    if (ms_xfer) *ms_xfer = g_blk.ms_xfer;
}

extern "C" void block27_get_stats2(uint64_t* h2d, uint64_t* d2h,
                                   double* device_ms) {
    if (h2d) *h2d = g_blk.pcie_h2d;
    if (d2h) *d2h = g_blk.pcie_d2h;
    if (device_ms) *device_ms = g_blk.device_ms;
}

extern "C" void block27_reset_stats(void) {
    g_blk.pairs = 0;
    g_blk.pcie_bytes = 0;
    g_blk.pcie_h2d = 0;
    g_blk.pcie_d2h = 0;
    g_blk.kernels = 0;
    g_blk.device_ms = 0.0;
    g_blk.ms_fwd = g_blk.ms_inv = g_blk.ms_crt = g_blk.ms_carry = g_blk.ms_xfer =
        0.0;
}

extern "C" int block27_vram_info(size_t* used_bytes, size_t* total_bytes) {
    size_t freeb = 0, totalb = 0;
    cudaError_t e = cudaMemGetInfo(&freeb, &totalb);
    if (e != cudaSuccess) return 1;
    if (used_bytes) *used_bytes = totalb - freeb;
    if (total_bytes) *total_bytes = totalb;
    return 0;
}
