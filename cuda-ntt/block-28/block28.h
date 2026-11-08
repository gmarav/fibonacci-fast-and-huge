// C ABI for the block28 exact blockwise multiply / fast-doubling engine.
//
// Design (option A):
//   * NTT length 2^28, block size B = 2^27 32-bit limbs;
//   * two 48-bit primes (P0, P1) with R = 2^64 Montgomery arithmetic;
//   * three in-place u64 transform buffers A, B, C and two u32 operand
//     staging buffers (a, b) -- the block-27 `t` buffer is not present;
//   * on-the-fly twiddles, no 2^28-entry tables;
//   * results are staged to the host, never kept whole on the device.
//
// Limb format everywhere: little-endian base-2^32 limbs, canonical (no
// trailing zero limbs).  No floating point is used on the arithmetic path.

#ifndef BLOCK28_H
#define BLOCK28_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLK28_BUFFERS 3

typedef struct block28_stats {
    int max_lg;
    size_t bytes_arrays;
    size_t bytes_staging;
    size_t bytes_twiddles;
    uint64_t launches;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t pairs;        // block-pair pointwise accumulations
    uint64_t block_sums;   // inverse transforms of accumulated sums
    double gpu_ms;         // CUDA-event kernel time
    double gpu_copy_ms;    // CUDA-event H2D/D2H time
    double ms_fwd;
    double ms_inv;
    double ms_pointwise;
} block28_stats;

// Allocate the three u64 transform arrays for transforms up to 2^max_lg.
int be_init(int max_lg);
// Ensure the two u32 operand staging buffers hold at least `limbs` limbs.
int be_reserve(size_t limbs);
// Upload an operand (0 = a, 1 = b) into its staging buffer.
int be_upload(int which, const uint32_t* host, size_t limbs);
// Load block `off_limbs` of staging operand `which` into transform buffer
// `dst`, reduce to Montgomery form, and run the forward NTT.
int be_fwd_block(int dst, int prime, int which, size_t off_limbs,
                 size_t limbs, int lg);
int be_zero(int buf, int lg);
// C = (mode&1 ? C : 0) + (mode&2 ? 2 : 1) * n^-1 * A * B.
int be_pointwise_acc(int dst, int a, int b, int prime, int lg, uint64_t ninv,
                     int mode);
int be_inv(int buf, int prime, int lg);
int be_fwd(int buf, int prime, int lg);
uint64_t be_ninv(int prime, int lg);
int be_download_range(int src, uint64_t* host, size_t start, size_t count,
                      int lg);
int be_sync(void);

void be_gpu_timer_reset(void);
int be_gpu_mark_begin(void);
void be_gpu_mark_end(int token);

const char* be_error(void);
const char* be_backend_name(void);
int be_get_stats(block28_stats* out);
int be_meminfo(size_t* free_bytes, size_t* total_bytes);

// mode 0: device Montgomery vectors from <outdir>/mul_p{0,1}.bin
// mode 1: device NTT replay from <outdir>/ntt_<lg>_{x,y,p0,p1}.bin
int be_selftest(const char* outdir, int mode, int lg);

void be_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif  // BLOCK28_H
