// C ABI for the block-27 exact blockwise multiply / fast-doubling engine.
//
// Limb format: little-endian base-2^32 limbs, canonical (no trailing zero
// limbs).  No floating point is used anywhere on the arithmetic path.

#ifndef BLOCK27_H
#define BLOCK27_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct block27_stats {
    uint64_t block_pairs;   // block-pair convolutions executed
    uint64_t kernels;       // kernel launches
    uint64_t pcie_bytes;    // host<->device bytes moved (operands + results)
    double ms_fwd;          // forward transform wall time (synced)
    double ms_inv;          // inverse transform wall time (synced)
    double ms_crt;          // CRT + accumulate wall time (synced)
    double ms_carry;        // diagonal finalisation wall time
    double ms_xfer;         // upload / borrow-chain wall time
} block27_stats;

// Block size is 2^B_log2 limbs, transform length 2*B.  The production value is
// 26 (256 MiB blocks, 2^27 transforms).  Returns 0 on success.
int block27_configure(int B_log2);

// Exact product c = a * b.  On success *out is malloc'd (free with free()).
int block27_mul(const uint32_t* a, size_t a_limbs, const uint32_t* b,
                size_t b_limbs, int B_log2, uint32_t** out,
                size_t* out_limbs);

// One fast-doubling step: c = a*(2b-a) = F(2k), d = a^2+b^2 = F(2k+1).
// want_d = 0 skips d (only valid when the caller does not need F(2k+1)).
int block27_step(const uint32_t* a, size_t a_limbs, const uint32_t* b,
                 size_t b_limbs, int B_log2, int want_d, uint32_t** out_c,
                 size_t* out_c_limbs, uint32_t** out_d, size_t* out_d_limbs);

int block27_vram_info(size_t* used_bytes, size_t* total_bytes);

void block27_get_stats(uint64_t* pairs, uint64_t* pcie_bytes,
                       uint64_t* kernels, double* ms_fwd, double* ms_inv,
                       double* ms_crt, double* ms_carry, double* ms_xfer);

typedef struct block27_stats2 {
    uint64_t pcie_h2d;
    uint64_t pcie_d2h;
    double device_ms; // CUDA-event time of all block-step stream work
} block27_stats2;

void block27_get_stats2(uint64_t* h2d, uint64_t* d2h, double* device_ms);

void block27_reset_stats(void);

void block27_shutdown(void);

int block27_is_ready(void);

#ifdef __cplusplus
}
#endif

#endif // BLOCK27_H
