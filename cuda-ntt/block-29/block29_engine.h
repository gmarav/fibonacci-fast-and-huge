// C ABI for the 2^29, two-prime 48-bit block-convolution engine.
//
// The engine owns five transform buffers of 2^lg u64 residues plus a u32
// staging buffer for limb blocks.  All calls are synchronous with respect to
// the host call that launches them (they enqueue on one stream); call
// be_sync() before reading results.
//
// Conventions (validated by the host reference and verify_python.py):
//   * residues are Montgomery form (x*R, R = 2^64) in the transform buffers;
//   * the forward NTT is DIF, the inverse is DIT with no final scale;
//   * the pointwise kernel applies n^-1, so the inverse output is the plain
//     convolution residue mod p;
//   * CRT and base-2^32 carry normalisation are performed on the host in the
//     prototype driver (a device CRT kernel is provided for validation).

#ifndef BLOCK_ENGINE_H
#define BLOCK_ENGINE_H

#include <stddef.h>
#include <stdint.h>

// The engine is a process-backed executable, so no DLL export decoration is
// needed.  The macro is kept to document the C ABI entry points.
#define BLK_API

#ifdef __cplusplus
extern "C" {
#endif

#define BLK_BUFFERS 5

typedef struct be_stats {
    int max_lg;             /* capacity log2 currently allocated */
    size_t bytes_arrays;    /* five transform buffers */
    size_t bytes_staging;   /* u32 limb staging buffer */
    size_t bytes_twiddles;  /* small/tail tables */
    uint64_t launches;      /* kernels launched since init */
    double gpu_ms;          /* measured kernel time of the last call */
    double gpu_copy_ms;     /* measured H2D/D2H stream copy time */
} be_stats;

// Allocate five buffers of 2^max_lg u64, the staging buffer and twiddles.
// Returns 0 on success.  Calling again with a larger lg grows the buffers.
BLK_API int be_init(int max_lg);
BLK_API int be_grow(int lg);
BLK_API void be_shutdown(void);

// Load `limbs` base-2^32 limbs from host memory into transform buffer `dst`
// for prime `prime` (0 or 1), converting to Montgomery form and zero padding
// to 2^lg.  `limbs` must be <= min(2^(lg-1), 2^28).
BLK_API int be_load(int dst, int prime, const uint32_t* host, size_t limbs,
                    int lg);

// In-place forward (DIF) / inverse (DIT, no final scale) NTT of length 2^lg.
BLK_API int be_fwd(int buf, int prime, int lg);
BLK_API int be_inv(int buf, int prime, int lg);

// dst = mode&1 ? dst + s*a*b : s*a*b, where s = 2^-lg mod p and, when
// mode&2 is set, the product is doubled before scaling (symmetric pairs).
BLK_API int be_pointwise(int dst, int a, int b, int prime, int lg, int mode);

BLK_API int be_zero(int buf, int lg);
BLK_API int be_download(int src, uint64_t* host, int lg);
// Copy `count` u64 residues starting at `start` from transform buffer `src`.
BLK_API int be_download_range(int src, uint64_t* host, size_t start,
                              size_t count, int lg);
BLK_API int be_sync(void);
BLK_API void be_gpu_timer_reset(void);
BLK_API int be_gpu_mark_begin(void);
BLK_API void be_gpu_mark_end(int token);

BLK_API const char* be_error(void);
BLK_API const char* be_backend_name(void);
BLK_API int be_get_stats(be_stats* out);
// Current CUDA free/total device bytes (creates a context if none exists).
BLK_API int be_meminfo(size_t* free_bytes, size_t* total_bytes);

// Device arithmetic / NTT self-test (never run automatically; the driver's
// `selftest` command calls it explicitly and only after a GPU go-ahead).
//   mode 0: Montgomery vectors from out/mul_{p0,p1}.bin
//   mode 1: convolution at length 2^lg against out/ntt_<lg>_{x,y,p0,p1}.bin
//   mode 2: block-convolution vectors from out/block_*.bin
BLK_API int be_selftest(const char* outdir, int mode, int lg);

#ifdef __cplusplus
}
#endif

#endif  // BLOCK_ENGINE_H
