// C ABI for the GPU-resident exact fast-doubling Fibonacci engine built on the
// ntt-alt transform core.
//
// Little-endian base-2^32 limbs, canonical form (no trailing zero limbs,
// zero is a zero-length value).  No floating point is used anywhere.
//
// Two entry points are provided:
//
//   gpu_fib_doubling(n, ...)
//       One call computes F(n) with the whole fast-doubling bit loop resident
//       on the GPU.  Only the seed (0,1) is uploaded and only F(n) is
//       downloaded.
//
//   gpu_ntt_doubling_step(a, b, bit, ...)
//       One fused doubling step: (a,b) -> (a',b') with the given bit, using
//       five NTT transforms per prime instead of the nine that three
//       independent multiplications would need.  The caller supplies the
//       operands and receives the two results; the device workspaces stay
//       allocated inside the engine between calls.
//
// Both entry points share one lazily created engine (device buffers, twiddle
// tables and cached CUDA graphs).  The engine grows its workspace on demand
// and is released by gpu_fib_shutdown().

#ifndef RESIDENT27_H
#define RESIDENT27_H

#include <stddef.h>
#include <stdint.h>

// The engine is a process-backed executable, so no DLL export decoration is
// needed.  The macro is kept to document the C ABI entry points.
#define RESIDENT_API

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_fib_stats {
    uint64_t steps;          /* doubling steps executed (last call) */
    uint64_t transforms;     /* forward+inverse NTT instances (prime copies) */
    uint64_t stage_kernels;  /* butterfly stage launches */
    uint64_t other_kernels;  /* reduction, pointwise, CRT, carry launches */
    uint64_t launches;       /* total kernel launches == stage + other */
    uint64_t max_ntt_length; /* largest transform length used */
    uint64_t peak_limbs;     /* limbs in the largest downloaded value */
    double gpu_ms;           /* device busy time of the last call */
    double host_ms;          /* host time of the last call, enqueue to sync */
} gpu_fib_stats;

// Full resident fast-doubling loop: F(n).
// Returns 0 on success, nonzero on failure.  On success *out_limbs points to a
// buffer allocated with malloc by this module (free it with
// gpu_fib_free_limbs) and *out_count is the canonical limb count (0 for zero).
RESIDENT_API int gpu_fib_doubling(uint64_t n, uint32_t** out_limbs,
                                  size_t* out_count);

// One fused doubling step.  a and b are canonical limb vectors; bit is the
// bit of the index that selects (c,d) or (d,c+d):
//   bit == 0:  (a',b') = (c,d),   c = a*(2b-a), d = a^2+b^2
//   bit == 1:  (a',b') = (d,c+d)
// Both outputs are freshly allocated (free with gpu_fib_free_limbs).
RESIDENT_API int gpu_ntt_doubling_step(const uint32_t* a, size_t a_limbs,
                                       const uint32_t* b, size_t b_limbs,
                                       uint32_t bit, uint32_t** out_a,
                                       size_t* out_a_limbs, uint32_t** out_b,
                                       size_t* out_b_limbs);

RESIDENT_API void gpu_fib_free_limbs(uint32_t* limbs);

// Statistics for the most recent call (either entry point).
RESIDENT_API int gpu_fib_get_stats(gpu_fib_stats* out);

// Free all device memory, cached graphs and pinned buffers.  Safe to call at
// any time; the next call recreates the engine.
RESIDENT_API void gpu_fib_shutdown(void);

// Human readable description of the backend (static storage).
RESIDENT_API const char* gpu_fib_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif // RESIDENT27_H
