// CPU-only exactness test for segment_norm.h (the driver's CRT/carry path).
//
//   g++ -O2 -std=c++17 segment_test.cpp -o build\segment_test.exe
//   .\build\segment_test.exe
//
// It generates random 96-bit coefficient sequences x_j with the same trailing
// word constraints as a real product (x_{M-1} < 2^32, x_{M-2} < 2^64), feeds
// their residues (mod P0, mod P1) through the exact code the GPU driver uses,
// and compares the resulting base-2^32 limbs against a direct __int128 carry
// reference.  No GPU and no CUDA runtime are involved.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "block_constants.h"
#include "segment_norm.h"

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint64_t u64;
typedef unsigned __int128 u128;

static u64 g_state = 0x243F6A8885A308D3ull;

static u64 rnd() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

static void trim(std::vector<u32>& v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}

static std::vector<u32> reference(const std::vector<u128>& x) {
    std::vector<u32> out;
    out.reserve(x.size() + 4);
    u128 carry = 0;
    for (size_t j = 0; j < x.size(); ++j) {
        u128 total = x[j] + carry;
        out.push_back((u32)(total & 0xFFFFFFFFull));
        carry = total >> 32;
    }
    while (carry) {
        out.push_back((u32)(carry & 0xFFFFFFFFull));
        carry >>= 32;
    }
    trim(out);
    return out;
}

static bool run_case(const char* name, std::vector<u128> x, size_t B,
                     int threads) {
    const size_t M = x.size();
    const size_t nseg = (M + B - 1) / B;
    // Enforce the trailing-word constraints of a real product.
    if (M >= 1) x[M - 1] &= 0xFFFFFFFFull;
    if (M >= 2) x[M - 2] &= 0xFFFFFFFFFFFFFFFFull;

    std::vector<u32> lo(M), out(M, 0);
    std::vector<u16> hi(M);
    std::vector<u64> seg1(M);
    for (size_t j = 0; j < M; ++j) {
        u64 a0 = (u64)(x[j] % P0);
        u64 a1 = (u64)(x[j] % P1);
        lo[j] = (u32)a0;
        hi[j] = (u16)(a0 >> 32);
        seg1[j] = a1;
    }
    std::vector<u64> d(B, 0);
    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;
    for (size_t k = 0; k < nseg; ++k) {
        size_t off = k * B;
        size_t lim = (M - off < B) ? (M - off) : B;
        bool full = (lim == B);
        segnorm::process_segment(lo.data() + off, hi.data() + off,
                                 seg1.data() + off, lim, d.data(),
                                 out.data() + off, carry, sb1, sc1, sc2, full,
                                 P0, P1, BLK_INV_P0_MOD_P1, threads);
    }
    if (carry != 0) {
        out.push_back(carry);
    }
    std::vector<u32> want = reference(x);
    trim(out);
    if (out != want) {
        std::fprintf(stderr,
                     "FAIL %s: limbs differ (got %zu, want %zu)\n", name,
                     out.size(), want.size());
        size_t n = std::min(out.size(), want.size());
        for (size_t i = 0; i < n; ++i) {
            if (out[i] != want[i]) {
                std::fprintf(stderr, "  first mismatch at %zu: got %08x want %08x\n",
                             i, out[i], want[i]);
                break;
            }
        }
        return false;
    }
    std::printf("segment %-26s M=%-7zu B=%-6zu chunks=%zu PASS\n", name, M, B,
                nseg);
    return true;
}

int main() {
    const int threads = 16;

    // CRT round trip.
    for (int it = 0; it < 20000; ++it) {
        u128 xv;
        if (it == 0) xv = 0;
        else if (it == 1) xv = 1;
        else if (it == 2) xv = (u128)P0 * P1 - 1;
        else xv = ((u128)rnd() << 64) ^ (u128)rnd();
        xv %= (u128)P0 * P1;
        u64 a0 = (u64)(xv % P0);
        u64 a1 = (u64)(xv % P1);
        segnorm::U96 r = segnorm::crt(a0, a1, P0, P1, BLK_INV_P0_MOD_P1);
        u128 got = (u128)r.lo | ((u128)r.hi << 64);
        if (got != xv) {
            std::fprintf(stderr, "CRT FAIL at %d\n", it);
            return 1;
        }
    }
    std::printf("CRT 20000/20000 exact PASS\n");

    bool ok = true;
    // Random sequences, several segment sizes and partial tails.
    for (int mode = 0; mode < 4 && ok; ++mode) {
        for (size_t B : {(size_t)1024, (size_t)4096, (size_t)1 << 16}) {
            for (int seg = 1; seg <= 6 && ok; ++seg) {
                size_t M = (size_t)seg * B;
                if (mode == 0) M -= B / 3;      // partial last segment
                if (mode == 1) M -= 1;
                if (mode == 2) M = (size_t)seg * B;  // exact
                if (M == 0) continue;
                std::vector<u128> x(M);
                for (size_t j = 0; j < M; ++j) {
                    if (mode == 3) {
                        x[j] = (u128)P0 * P1 - 1;   // all maximal, long carries
                    } else {
                        x[j] = ((u128)rnd() << 64) | rnd();
                        x[j] %= (u128)P0 * P1;
                    }
                }
                char name[64];
                std::snprintf(name, sizeof name, "mode%d seg%d", mode, seg);
                ok = run_case(name, std::move(x), B, threads);
            }
        }
    }

    // All-zero and all-ones edge cases.
    for (size_t B : {(size_t)1024, (size_t)4096}) {
        std::vector<u128> z((size_t)3 * B, 0);
        ok = ok && run_case("zero", std::move(z), B, threads);
        std::vector<u128> o((size_t)3 * B, (u128)1);
        ok = ok && run_case("ones", std::move(o), B, threads);
    }

    if (ok) std::printf("SEGMENT NORMALISATION TESTS PASSED\n");
    return ok ? 0 : 1;
}
