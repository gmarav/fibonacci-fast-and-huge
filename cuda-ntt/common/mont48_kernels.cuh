// Shared 48-bit NTT kernels for block-28 and block-29.
//
// The five live kernels are identical in both engines.  Engine-specific plan
// logic, buffer counts, and host schedules stay in the individual .cu files.
// BLK_TAIL_LOG2 and BLK_TW_TILE must be defined before including this header.
#pragma once

#include "mont48_device.cuh"

#ifndef BLK_TAIL_LOG2
#error "define BLK_TAIL_LOG2 before including mont48_kernels.cuh"
#endif
#ifndef BLK_TW_TILE
#error "define BLK_TW_TILE before including mont48_kernels.cuh"
#endif

__global__ void k_load(uint64_t* __restrict__ dst, const uint32_t* __restrict__ src,
                       unsigned limbs, int lg, ModArgs m) {
    size_t n = (size_t)1 << lg;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        dst[i] = (i < (size_t)limbs) ? dmul((uint64_t)src[i], m.R2, m) : 0ull;
    }
}

template <int INV>
__global__ void k_stage(uint64_t* __restrict__ a, int lg_n, int s, ModArgs m,
                        const uint64_t* __restrict__ rootR,
                        const uint64_t* __restrict__ small) {
    const int TILE = BLK_TW_TILE;
    int half = 1 << (s - 1);
    int len = 1 << s;
    int groups = 1 << (lg_n - s);
    int j = blockIdx.x * TILE + threadIdx.x;

    __shared__ uint64_t sh_start;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sh_start = dpow_mont(rootR[s], (uint64_t)blockIdx.x * TILE, m);
    }
    __syncthreads();
    if (j >= half) return;

    int g = blockIdx.y * blockDim.y + threadIdx.y;
    if (g >= groups) return;

    uint64_t tw = dmul(sh_start, small[s * TILE + threadIdx.x], m);
    {
        size_t i = (size_t)g * len + j;
        uint64_t u = a[i], v = a[i + half];
        if (INV == 0) {
            a[i] = dadd(u, v, m);
            a[i + half] = dmul(dsub(u, v, m), tw, m);
        } else {
            uint64_t t = dmul(v, tw, m);
            a[i] = dadd(u, t, m);
            a[i + half] = dsub(u, t, m);
        }
    }
}

template <int INV>
__global__ void k_tail(uint64_t* __restrict__ a, int lg_n, ModArgs m,
                       const uint64_t* __restrict__ tw) {
    __shared__ uint64_t sm[1 << BLK_TAIL_LOG2];
    int t = lg_n < BLK_TAIL_LOG2 ? lg_n : BLK_TAIL_LOG2;
    size_t nt = (size_t)1 << t;
    size_t base = (size_t)blockIdx.x * nt;
    for (size_t i = threadIdx.x; i < nt; i += blockDim.x) sm[i] = a[base + i];
    __syncthreads();

    if (INV == 0) {
        for (int s = t; s >= 1; --s) {
            int half = 1 << (s - 1);
            size_t len = (size_t)1 << s;
            const uint64_t* tws = tw + (((size_t)1 << (s - 1)) - 1);
            for (size_t idx = threadIdx.x; idx < (nt >> 1);
                 idx += blockDim.x) {
                size_t blk = idx >> (s - 1);
                size_t j = idx - (blk << (s - 1));
                size_t ii = blk * len + j;
                uint64_t u = sm[ii], v = sm[ii + half];
                sm[ii] = dadd(u, v, m);
                sm[ii + half] = dmul(dsub(u, v, m), tws[j], m);
            }
            __syncthreads();
        }
    } else {
        for (int s = 1; s <= t; ++s) {
            int half = 1 << (s - 1);
            size_t len = (size_t)1 << s;
            const uint64_t* tws = tw + (((size_t)1 << (s - 1)) - 1);
            for (size_t idx = threadIdx.x; idx < (nt >> 1);
                 idx += blockDim.x) {
                size_t blk = idx >> (s - 1);
                size_t j = idx - (blk << (s - 1));
                size_t ii = blk * len + j;
                uint64_t u = sm[ii], v = dmul(sm[ii + half], tws[j], m);
                sm[ii] = dadd(u, v, m);
                sm[ii + half] = dsub(u, v, m);
            }
            __syncthreads();
        }
    }

    for (size_t i = threadIdx.x; i < nt; i += blockDim.x) a[base + i] = sm[i];
}

/// c = (mode&1 ? c : 0) + (mode&2 ? 2 : 1) * n^-1 * a * b  (mod p)
__global__ void k_pointwise(uint64_t* __restrict__ c,
                            const uint64_t* __restrict__ a,
                            const uint64_t* __restrict__ b, size_t n,
                            ModArgs m, uint64_t ninv, int mode) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        uint64_t v = dmul(dmul(a[i], b[i], m), ninv, m);
        if (mode & 2) v = dadd(v, v, m);
        if (mode & 1) v = dadd(c[i], v, m);
        c[i] = v;
    }
}

__global__ void k_mulcheck(const uint64_t* __restrict__ a,
                           const uint64_t* __restrict__ b,
                           const uint64_t* __restrict__ exp, size_t n,
                           ModArgs m, unsigned long long* bad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        if (dmul(a[i], b[i], m) != exp[i]) atomicAdd(bad, 1ull);
    }
}
