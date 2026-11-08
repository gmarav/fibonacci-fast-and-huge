// 2^29, two-prime 48-bit block-convolution NTT engine (CUDA).
//
// Arithmetic: Montgomery multiplication with R = 2^64 and manual two-limb
// reduction (__umul64hi); CUDA 13 / sm_120 has no usable device __int128.
//
// NTT schedule:
//   forward: DIF global stages len = 2^lg .. 2^(T+1), then one shared-memory
//            tail per 2^T contiguous chunk for stages 2^T .. 2;
//   inverse: reverse order, DIT stages 2 .. 2^T in shared memory, then
//            global stages 2^(T+1) .. 2^lg.  No final scale: the pointwise
//            kernel multiplies by n^-1 (verified against the host reference).
//
// Twiddles are derived on the fly from the primitive 2^29-th root: per stage
// s only a 256-entry small table and one per-block exponentiation are used.
// No 2^29-element twiddle table exists.

#include "block29_engine.h"
#include "block_constants.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

#define BLK_TAIL_LOG2 12
#define BLK_TW_TILE 256
#define BLK_MAX_LOG2_LOCAL 29
#define BLK_EV_POOL 32

#include "../common/mont48_kernels.cuh"

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Host helpers
// ---------------------------------------------------------------------------

static u64 host_mulmod(u64 a, u64 b, u64 p) {
#if defined(_MSC_VER) && defined(_M_X64)
    u64 hi;
    u64 lo = _umul128(a, b, &hi);
    u64 rem;
    _udiv128(hi, lo, p, &rem);
    return rem;
#else
    return (u64)(((unsigned __int128)a * b) % p);
#endif
}

static u64 host_powmod(u64 b, u64 e, u64 p) {
    u64 r = 1;
    b %= p;
    while (e) {
        if (e & 1) r = host_mulmod(r, b, p);
        b = host_mulmod(b, b, p);
        e >>= 1;
    }
    return r;
}

static char g_error[512];
static bool set_err(const char* what, cudaError_t err) {
    std::snprintf(g_error, sizeof g_error, "%s: %s", what,
                  cudaGetErrorString(err));
    return false;
}
static bool set_msg(const char* msg) {
    std::snprintf(g_error, sizeof g_error, "%s", msg);
    return false;
}

struct Engine {
    int lg = 0;
    size_t n = 0;
    u64* buf[BLK_BUFFERS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    u32* stage = nullptr;
    size_t stage_cap = 0;
    u64* d_rootR = nullptr;     // [2][30]
    u64* d_small = nullptr;     // [2][30*BLK_TW_TILE]
    u64* d_rootR_inv = nullptr; // [2][30]
    u64* d_small_inv = nullptr; // [2][30*BLK_TW_TILE]
    u64* d_tail_fwd = nullptr;  // [2][1<<BLK_TAIL_LOG2]
    u64* d_tail_inv = nullptr;  // [2][1<<BLK_TAIL_LOG2]
    cudaStream_t stream = nullptr;
    uint64_t launches = 0;
    double gpu_ms = 0.0;
    // Event pool for per-call GPU time accounting.
    cudaEvent_t ev_start[BLK_EV_POOL];
    cudaEvent_t ev_stop[BLK_EV_POOL];
    bool ev_ready = false;
};

struct EvPending {
    int slot;
    bool copy;
};

static EvPending g_pending[BLK_EV_POOL];
static int g_npending = 0;
static double g_kernel_ms = 0.0;
static double g_copy_ms = 0.0;

static Engine g_e;
static bool g_ready = false;

// ---------------------------------------------------------------------------
// GPU-time accounting: one event pair per stream operation group.  Pending
// pairs are consumed at be_sync() (which the driver calls after every block
// sum), so the elapsed time excludes host CRT/carry work.
// ---------------------------------------------------------------------------

static void ev_flush(bool sync_stream) {
    if (sync_stream && g_ready && g_e.stream) cudaStreamSynchronize(g_e.stream);
    if (g_e.ev_ready) {
        for (int i = 0; i < g_npending; ++i) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, g_e.ev_start[g_pending[i].slot],
                                 g_e.ev_stop[g_pending[i].slot]);
            if (g_pending[i].copy)
                g_copy_ms += ms;
            else
                g_kernel_ms += ms;
        }
    }
    g_npending = 0;
}

static int ev_begin(void) {
    if (!g_e.ev_ready) return -1;
    if (g_npending >= BLK_EV_POOL) ev_flush(true);
    // The mask keeps the index provably inside the pool for static analysis;
    // ev_flush above guarantees g_npending < BLK_EV_POOL.
    int slot = g_npending & (BLK_EV_POOL - 1);
    cudaEventRecord(g_e.ev_start[slot], g_e.stream);
    return slot;
}

static void ev_end(int slot, bool copy) {
    if (slot < 0) return;
    cudaEventRecord(g_e.ev_stop[slot], g_e.stream);
    g_pending[g_npending].slot = slot;
    g_pending[g_npending].copy = copy;
    ++g_npending;
}

static const u64 kPrime[2] = {P0, P1};
static const u64 kNP[2] = {P0_NP, P1_NP};
static const u64 kR[2] = {P0_R, P1_R};
static const u64 kR2[2] = {P0_R2, P1_R2};
static const u64 kW[2] = {P0_W, P1_W};

static ModArgs mod_args(int prime) {
    ModArgs m;
    m.p = kPrime[prime];
    m.np = kNP[prime];
    m.R = kR[prime];
    m.R2 = kR2[prime];
    return m;
}

static u64 ninv_cache[2][BLK_MAX_LOG2_LOCAL + 1];
static bool ninv_ready[2][BLK_MAX_LOG2_LOCAL + 1];

static u64 get_ninv(int prime, int lg) {
    if (!ninv_ready[prime][lg]) {
        u64 n = (u64)1 << lg;
        ninv_cache[prime][lg] = host_powmod(n % kPrime[prime],
                                            kPrime[prime] - 2, kPrime[prime]);
        ninv_ready[prime][lg] = true;
    }
    return ninv_cache[prime][lg];
}

static void free_engine() {
    for (int i = 0; i < BLK_BUFFERS; ++i) {
        if (g_e.buf[i]) cudaFree(g_e.buf[i]);
        g_e.buf[i] = nullptr;
    }
    if (g_e.stage) cudaFree(g_e.stage);
    if (g_e.d_rootR) cudaFree(g_e.d_rootR);
    if (g_e.d_small) cudaFree(g_e.d_small);
    if (g_e.d_rootR_inv) cudaFree(g_e.d_rootR_inv);
    if (g_e.d_small_inv) cudaFree(g_e.d_small_inv);
    if (g_e.d_tail_fwd) cudaFree(g_e.d_tail_fwd);
    if (g_e.d_tail_inv) cudaFree(g_e.d_tail_inv);
    g_e.stage = nullptr;
    g_e.d_rootR = nullptr;
    g_e.d_small = nullptr;
    g_e.d_rootR_inv = nullptr;
    g_e.d_small_inv = nullptr;
    g_e.d_tail_fwd = nullptr;
    g_e.d_tail_inv = nullptr;
    g_e.lg = 0;
    g_e.n = 0;
    g_e.stage_cap = 0;
    if (g_e.ev_ready) {
        for (int i = 0; i < BLK_EV_POOL; ++i) {
            cudaEventDestroy(g_e.ev_start[i]);
            cudaEventDestroy(g_e.ev_stop[i]);
        }
        g_e.ev_ready = false;
    }
    g_npending = 0;
    if (g_e.stream) cudaStreamDestroy(g_e.stream);
    g_e.stream = nullptr;
    g_ready = false;
}

static bool build_twiddles() {
    const int L = BLK_MAX_LOG2_LOCAL;
    std::vector<u64> rootR(2 * (L + 1), 0);
    std::vector<u64> small((size_t)2 * (L + 1) * BLK_TW_TILE, 0);
    std::vector<u64> rootRi(2 * (L + 1), 0);
    std::vector<u64> smalli((size_t)2 * (L + 1) * BLK_TW_TILE, 0);
    std::vector<u64> tailF(2 * (1 << BLK_TAIL_LOG2), 0);
    std::vector<u64> tailI(2 * (1 << BLK_TAIL_LOG2), 0);

    for (int pi = 0; pi < 2; ++pi) {
        const u64 p = kPrime[pi];
        const u64 R = kR[pi];
        for (int s = 1; s <= L; ++s) {
            u64 root = host_powmod(kW[pi], 1ull << (L - s), p);
            u64 invroot = host_powmod(root, p - 2, p);
            rootR[(size_t)pi * (L + 1) + s] = host_mulmod(root, R, p);
            rootRi[(size_t)pi * (L + 1) + s] = host_mulmod(invroot, R, p);

            u64 acc = R;   // root^0 * R
            for (int t = 0; t < BLK_TW_TILE; ++t) {
                small[((size_t)pi * (L + 1) + s) * BLK_TW_TILE + t] = acc;
                acc = host_mulmod(acc, root, p);
            }
            acc = R;
            for (int t = 0; t < BLK_TW_TILE; ++t) {
                smalli[((size_t)pi * (L + 1) + s) * BLK_TW_TILE + t] = acc;
                acc = host_mulmod(acc, invroot, p);
            }
            if (s <= BLK_TAIL_LOG2) {
                size_t off = ((size_t)1 << (s - 1)) - 1;
                acc = R;
                for (size_t j = 0; j < ((size_t)1 << (s - 1)); ++j) {
                    tailF[(size_t)pi * (1 << BLK_TAIL_LOG2) + off + j] = acc;
                    acc = host_mulmod(acc, root, p);
                }
                acc = R;
                for (size_t j = 0; j < ((size_t)1 << (s - 1)); ++j) {
                    tailI[(size_t)pi * (1 << BLK_TAIL_LOG2) + off + j] = acc;
                    acc = host_mulmod(acc, invroot, p);
                }
            }
        }
    }

    if (cudaMalloc((void**)&g_e.d_rootR, rootR.size() * 8) != cudaSuccess) return false;
    if (cudaMalloc((void**)&g_e.d_small, small.size() * 8) != cudaSuccess) return false;
    if (cudaMalloc((void**)&g_e.d_rootR_inv, rootRi.size() * 8) != cudaSuccess)
        return false;
    if (cudaMalloc((void**)&g_e.d_small_inv, smalli.size() * 8) != cudaSuccess)
        return false;
    if (cudaMalloc((void**)&g_e.d_tail_fwd, tailF.size() * 8) != cudaSuccess)
        return false;
    if (cudaMalloc((void**)&g_e.d_tail_inv, tailI.size() * 8) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_rootR, rootR.data(), rootR.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_small, small.data(), small.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_rootR_inv, rootRi.data(), rootRi.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_small_inv, smalli.data(), smalli.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_tail_fwd, tailF.data(), tailF.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    if (cudaMemcpy(g_e.d_tail_inv, tailI.data(), tailI.size() * 8,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}

extern "C" BLK_API int be_init(int max_lg) {
    if (max_lg < 1 || max_lg > BLK_MAX_LOG2_LOCAL) {
        set_msg("be_init: lg out of range");
        return 1;
    }
    if (g_ready && g_e.lg >= max_lg) return 0;
    free_engine();
    if (cudaStreamCreate(&g_e.stream) != cudaSuccess) {
        set_err("cudaStreamCreate", cudaGetLastError());
        return 1;
    }
    for (int i = 0; i < BLK_EV_POOL; ++i) {
        if (cudaEventCreateWithFlags(&g_e.ev_start[i], cudaEventDefault) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&g_e.ev_stop[i], cudaEventDefault) !=
                cudaSuccess) {
            set_err("cudaEventCreate", cudaGetLastError());
            free_engine();
            return 1;
        }
    }
    g_e.ev_ready = true;
    g_npending = 0;
    g_kernel_ms = 0.0;
    g_copy_ms = 0.0;
    g_e.lg = max_lg;
    g_e.n = (size_t)1 << max_lg;
    for (int i = 0; i < BLK_BUFFERS; ++i) {
        if (cudaMalloc((void**)&g_e.buf[i], g_e.n * 8) != cudaSuccess) {
            set_err("cudaMalloc transform buffer", cudaGetLastError());
            free_engine();
            return 1;
        }
    }
    g_e.stage_cap = g_e.n / 2;
    if (g_e.stage_cap > BLK_BLOCK_LIMBS) g_e.stage_cap = BLK_BLOCK_LIMBS;
    if (g_e.stage_cap < 1) g_e.stage_cap = 1;
    if (cudaMalloc((void**)&g_e.stage, g_e.stage_cap * 4) != cudaSuccess) {
        set_err("cudaMalloc staging", cudaGetLastError());
        free_engine();
        return 1;
    }
    if (!build_twiddles()) {
        set_err("twiddle upload", cudaGetLastError());
        free_engine();
        return 1;
    }
    g_ready = true;
    return 0;
}

extern "C" BLK_API int be_grow(int lg) {
    if (!g_ready || g_e.lg < lg) return be_init(lg);
    return 0;
}

extern "C" BLK_API void be_shutdown(void) { free_engine(); }

static bool check_buf(int b) {
    if (!g_ready) return set_msg("engine not initialised");
    if (b < 0 || b >= BLK_BUFFERS) return set_msg("bad buffer index");
    return true;
}

extern "C" BLK_API int be_load(int dst, int prime, const uint32_t* host,
                               size_t limbs, int lg) {
    if (!check_buf(dst) || prime < 0 || prime > 1) return 1;
    if (lg < 1 || lg > g_e.lg) return set_msg("be_load: bad lg");
    if (limbs > g_e.stage_cap) return set_msg("be_load: block too long");
    ModArgs m = mod_args(prime);
    size_t n = (size_t)1 << lg;

    if (limbs) {
        cudaError_t e = cudaMemcpyAsync(g_e.stage, host, limbs * 4,
                                        cudaMemcpyHostToDevice, g_e.stream);
        if (e != cudaSuccess) return set_err("be_load: memcpy", e), 1;
    }
    int threads = 256;
    size_t blocks64 = (n + threads - 1) / threads;
    int blocks = (int)(blocks64 > 65535 ? 65535 : blocks64);
    k_load<<<blocks, threads, 0, g_e.stream>>>(g_e.buf[dst], g_e.stage,
                                               (unsigned)limbs, lg, m);
    g_e.launches++;
    return 0;
}

extern "C" BLK_API int be_zero(int buf, int lg) {
    if (!check_buf(buf) || lg < 1 || lg > g_e.lg) return 1;
    size_t n = (size_t)1 << lg;
    cudaError_t e =
        cudaMemsetAsync(g_e.buf[buf], 0, n * 8, g_e.stream);
    return e == cudaSuccess ? 0 : (set_err("be_zero", e), 1);
}

static void launch_global(int buf, int prime, int lg, bool inv) {
    ModArgs m = mod_args(prime);
    const u64* rootR =
        (inv ? g_e.d_rootR_inv : g_e.d_rootR) +
        (size_t)prime * (BLK_MAX_LOG2_LOCAL + 1);
    const u64* small =
        (inv ? g_e.d_small_inv : g_e.d_small) +
        (size_t)prime * (BLK_MAX_LOG2_LOCAL + 1) * BLK_TW_TILE;
    int T = lg < BLK_TAIL_LOG2 ? lg : BLK_TAIL_LOG2;
    if (inv) {
        for (int s = T + 1; s <= lg; ++s) {
            int half = 1 << (s - 1);
            int groups = 1 << (lg - s);
            int G = (groups >= 4) ? 4 : 1;
            dim3 block(BLK_TW_TILE, G);
            dim3 grid((half + BLK_TW_TILE - 1) / BLK_TW_TILE,
                      (groups + G - 1) / G);
            k_stage<1><<<grid, block, 0, g_e.stream>>>(
                g_e.buf[buf], lg, s, m, rootR, small);
            g_e.launches++;
        }
    } else {
        for (int s = lg; s >= T + 1; --s) {
            int half = 1 << (s - 1);
            int groups = 1 << (lg - s);
            int G = (groups >= 4) ? 4 : 1;
            dim3 block(BLK_TW_TILE, G);
            dim3 grid((half + BLK_TW_TILE - 1) / BLK_TW_TILE,
                      (groups + G - 1) / G);
            k_stage<0><<<grid, block, 0, g_e.stream>>>(
                g_e.buf[buf], lg, s, m, rootR, small);
            g_e.launches++;
        }
    }
}

static void launch_tail(int buf, int prime, int lg, bool inv) {
    ModArgs m = mod_args(prime);
    int t = lg < BLK_TAIL_LOG2 ? lg : BLK_TAIL_LOG2;
    size_t nt = (size_t)1 << t;
    size_t n = (size_t)1 << lg;
    const u64* tw = inv
                        ? g_e.d_tail_inv + (size_t)prime * (1 << BLK_TAIL_LOG2)
                        : g_e.d_tail_fwd + (size_t)prime * (1 << BLK_TAIL_LOG2);
    dim3 block(256);
    dim3 grid((unsigned)(n / nt));
    if (inv)
        k_tail<1><<<grid, block, 0, g_e.stream>>>(g_e.buf[buf], lg, m, tw);
    else
        k_tail<0><<<grid, block, 0, g_e.stream>>>(g_e.buf[buf], lg, m, tw);
    g_e.launches++;
}

extern "C" BLK_API int be_fwd(int buf, int prime, int lg) {
    if (!check_buf(buf) || prime < 0 || prime > 1) return 1;
    if (lg < 1 || lg > g_e.lg) return set_msg("be_fwd: bad lg");
    launch_global(buf, prime, lg, false);
    launch_tail(buf, prime, lg, false);
    return 0;
}

extern "C" BLK_API int be_inv(int buf, int prime, int lg) {
    if (!check_buf(buf) || prime < 0 || prime > 1) return 1;
    if (lg < 1 || lg > g_e.lg) return set_msg("be_inv: bad lg");
    launch_tail(buf, prime, lg, true);
    launch_global(buf, prime, lg, true);
    return 0;
}

extern "C" BLK_API int be_pointwise(int dst, int a, int b, int prime, int lg,
                                    int mode) {
    if (!check_buf(dst) || !check_buf(a) || !check_buf(b)) return 1;
    if (prime < 0 || prime > 1) return set_msg("be_pointwise: bad prime");
    if (lg < 1 || lg > g_e.lg) return set_msg("be_pointwise: bad lg");
    ModArgs m = mod_args(prime);
    u64 ninv = get_ninv(prime, lg);
    size_t n = (size_t)1 << lg;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    k_pointwise<<<blocks, threads, 0, g_e.stream>>>(g_e.buf[dst], g_e.buf[a],
                                                    g_e.buf[b], n, m, ninv,
                                                    mode);
    g_e.launches++;
    return 0;
}

extern "C" BLK_API int be_download(int src, uint64_t* host, int lg) {
    if (!check_buf(src) || lg < 1 || lg > g_e.lg) return 1;
    size_t n = (size_t)1 << lg;
    cudaError_t e = cudaMemcpyAsync(host, g_e.buf[src], n * 8,
                                    cudaMemcpyDeviceToHost, g_e.stream);
    return e == cudaSuccess ? 0 : (set_err("be_download", e), 1);
}

extern "C" BLK_API int be_download_range(int src, uint64_t* host,
                                         size_t start, size_t count, int lg) {
    if (!check_buf(src) || lg < 1 || lg > g_e.lg) return 1;
    size_t n = (size_t)1 << lg;
    if (start + count > n) return set_msg("be_download_range: out of range");
    cudaError_t e = cudaMemcpyAsync(host, g_e.buf[src] + start, count * 8,
                                    cudaMemcpyDeviceToHost, g_e.stream);
    return e == cudaSuccess ? 0 : (set_err("be_download_range", e), 1);
}

extern "C" BLK_API int be_sync(void) {
    if (!g_ready) return 1;
    cudaError_t e = cudaStreamSynchronize(g_e.stream);
    ev_flush(false);
    return e == cudaSuccess ? 0 : (set_err("be_sync", e), 1);
}

extern "C" BLK_API void be_gpu_timer_reset(void) {
    ev_flush(true);
    g_kernel_ms = 0.0;
    g_copy_ms = 0.0;
}

// Coarse GPU-phase marks used by the driver.  A mark covers all stream work
// (kernels and H2D/D2H copies) enqueued between begin and end; the pending
// pair is consumed at the next be_sync().  Do not nest marks.
extern "C" BLK_API int be_gpu_mark_begin(void) { return ev_begin(); }

extern "C" BLK_API void be_gpu_mark_end(int token) { ev_end(token, false); }

extern "C" BLK_API const char* be_error(void) { return g_error; }

extern "C" BLK_API const char* be_backend_name(void) {
    return "block-29 (two 48-bit primes, R=2^64, 2^29 NTT)";
}

extern "C" BLK_API int be_get_stats(be_stats* out) {
    if (!out) return 1;
    out->max_lg = g_e.lg;
    out->bytes_arrays = (size_t)BLK_BUFFERS * g_e.n * 8;
    out->bytes_staging = g_e.stage_cap * 4;
    out->bytes_twiddles = (size_t)4 * (BLK_MAX_LOG2_LOCAL + 1) * 8 +
                          (size_t)4 * (BLK_MAX_LOG2_LOCAL + 1) * BLK_TW_TILE * 8 +
                          (size_t)4 * (1 << BLK_TAIL_LOG2) * 8;
    out->launches = g_e.launches;
    out->gpu_ms = g_kernel_ms;
    out->gpu_copy_ms = g_copy_ms;
    return 0;
}

extern "C" BLK_API int be_meminfo(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes) return 1;
    cudaError_t e = cudaMemGetInfo(free_bytes, total_bytes);
    return e == cudaSuccess ? 0 : (set_err("cudaMemGetInfo", e), 1);
}

// ---------------------------------------------------------------------------
// Device self-test (explicit only)
// ---------------------------------------------------------------------------

static bool read_vec_u64(const std::string& path, std::vector<u64>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)sz / 8);
    bool ok = std::fread(v.data(), 8, v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}

static bool read_vec_u32(const std::string& path, std::vector<u32>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)sz / 4);
    bool ok = std::fread(v.data(), 4, v.size(), f) == v.size();
    std::fclose(f);
    return ok;
}

extern "C" BLK_API int be_selftest(const char* outdir, int mode, int lg) {
    std::string dir = outdir ? outdir : "out";
    if (be_init(lg > 0 ? lg : 12) != 0) return 1;

    if (mode == 0) {
        for (int pi = 0; pi < 2; ++pi) {
            std::string path = dir + (pi ? "/mul_p1.bin" : "/mul_p0.bin");
            std::vector<u64> rec;
            if (!read_vec_u64(path, rec) || rec.size() % 3) {
                std::fprintf(stderr, "selftest: bad %s\n", path.c_str());
                return 1;
            }
            size_t n = rec.size() / 3;
            std::vector<u64> a(n), b(n), e(n);
            for (size_t i = 0; i < n; ++i) {
                a[i] = rec[3 * i];
                b[i] = rec[3 * i + 1];
                e[i] = rec[3 * i + 2];
            }
            u64 *da, *db, *de;
            unsigned long long* dbad;
            cudaMalloc((void**)&da, n * 8);
            cudaMalloc((void**)&db, n * 8);
            cudaMalloc((void**)&de, n * 8);
            cudaMalloc((void**)&dbad, 8);
            cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice);
            cudaMemcpy(db, b.data(), n * 8, cudaMemcpyHostToDevice);
            cudaMemcpy(de, e.data(), n * 8, cudaMemcpyHostToDevice);
            cudaMemset(dbad, 0, 8);
            k_mulcheck<<<64, 256>>>(da, db, de, n, mod_args(pi), dbad);
            unsigned long long bad = 0;
            cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost);
            cudaFree(da);
            cudaFree(db);
            cudaFree(de);
            cudaFree(dbad);
            std::printf("device mont %s: %zu vectors, %llu mismatches\n",
                        pi ? "p1" : "p0", n, bad);
            if (bad) return 1;
        }
        std::printf("DEVICE MONT SELFTEST PASS\n");
        return 0;
    }

    if (mode == 1) {
        int L = lg > 0 ? lg : 12;
        size_t n = (size_t)1 << L;
        std::vector<u32> x, y;
        std::string base = dir + "/ntt_" + std::to_string(L);
        if (!read_vec_u32(base + "_x.bin", x) ||
            !read_vec_u32(base + "_y.bin", y)) {
            std::fprintf(stderr, "selftest: missing %s_*.bin\n", base.c_str());
            return 1;
        }
        if (x.size() != n / 2 || y.size() != n / 2) {
            std::fprintf(stderr, "selftest: bad input sizes\n");
            return 1;
        }
        if (be_init(L) != 0) return 1;
        for (int pi = 0; pi < 2; ++pi) {
            if (be_load(0, pi, x.data(), x.size(), L) != 0) return 1;
            if (be_load(1, pi, y.data(), y.size(), L) != 0) return 1;
            if (be_fwd(0, pi, L) != 0 || be_fwd(1, pi, L) != 0) return 1;
            if (be_pointwise(2, 0, 1, pi, L, 0) != 0) return 1;
            if (be_inv(2, pi, L) != 0) return 1;
            std::vector<u64> got(n);
            if (be_download(2, got.data(), L) != 0 || be_sync() != 0) return 1;
            std::vector<u64> exp;
            std::string ep = base + (pi ? "_p1.bin" : "_p0.bin");
            if (!read_vec_u64(ep, exp) || exp.size() != n) {
                std::fprintf(stderr, "selftest: missing %s\n", ep.c_str());
                return 1;
            }
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i) bad += (got[i] != exp[i]);
            std::printf("device ntt n=%zu prime %d: %zu mismatches\n", n, pi,
                        bad);
            if (bad) return 1;
        }
        std::printf("DEVICE NTT SELFTEST PASS\n");
        return 0;
    }

    std::printf("selftest mode %d not implemented\n", mode);
    return 1;
}




