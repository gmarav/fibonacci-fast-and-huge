// Exact test of segment CRT + base-2^32 carry normalisation.
//
// Generates coefficient sequences (including the worst case P-1 everywhere),
// splits them into segments, runs the same process_segment path the GPU driver
// uses, and compares the resulting limbs and final carry with a direct
// unsigned __int128 normalisation of the whole sequence.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "block28_constants.h"
#include "segment_norm.h"

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

static u64 g_rng = 0x5EE28ull;
static u64 rnd64() {
    u64 z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Direct exact reference: spread each 96-bit coefficient over three base-2^32
// words (d_i = low(x_i) + high(x_{i-1}) + top(x_{i-2})) and propagate the
// limb carry.  This is the identity process_segment implements.
static std::vector<u32> direct_norm(const std::vector<u128>& x, u32& carry) {
    std::vector<u32> out(x.size());
    u64 c = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        u64 lo = (u64)(x[i] & 0xFFFFFFFFu);
        u64 mid = (i >= 1) ? (u64)((x[i - 1] >> 32) & 0xFFFFFFFFu) : 0ull;
        u64 top = (i >= 2) ? (u64)(x[i - 2] >> 64) : 0ull;
        u64 d = lo + mid + top;
        u64 t = d + c;
        out[i] = (u32)t;
        c = t >> 32;
    }
    carry = (u32)c;
    return out;
}

static int run_case(size_t B, size_t segs, size_t last_lim, int mode) {
    const size_t L = (segs - 1) * B + last_lim;
    std::vector<u128> x(L);
    for (size_t i = 0; i < L; ++i) {
        if (mode == 0) {
            x[i] = (u128)(rnd64() % P0) + (u128)P0 * (rnd64() % P1);
            x[i] %= (u128)P0 * P1;
        } else if (mode == 1) {
            x[i] = (u128)P0 * P1 - 1;   // every coefficient maximal
        } else if (mode == 2) {
            x[i] = 0xFFFFFFFFull;       // long carry chains
        } else {
            x[i] = (rnd64() & 1) ? (u128)P0 * P1 - 1 : (u128)0;
        }
    }
    std::vector<u64> res0(L), res1(L);
    for (size_t i = 0; i < L; ++i) {
        res0[i] = (u64)(x[i] % P0);
        res1[i] = (u64)(x[i] % P1);
    }
    u32 want_carry = 0;
    std::vector<u32> want = direct_norm(x, want_carry);

    std::vector<u32> got(L);
    std::vector<u64> d(B);
    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;
    size_t off = 0;
    for (size_t s = 0; s < segs; ++s) {
        size_t lim = (s + 1 == segs) ? last_lim : B;
        bool full = (lim == B);
        segnorm::process_segment(res0.data() + off, res1.data() + off, lim,
                                 d.data(), got.data() + off, carry, sb1, sc1,
                                 sc2, full, P0, P1, BLK28_INV_P0_MOD_P1, 1);
        off += lim;
    }
    int bad = 0;
    for (size_t i = 0; i < L; ++i)
        if (got[i] != want[i]) ++bad;
    if (carry != want_carry) ++bad;
    if (bad) {
        std::printf("segment B=%zu segs=%zu last=%zu mode=%d FAIL (%d)\n", B,
                    segs, last_lim, mode, bad);
        return 1;
    }
    return 0;
}

int main() {
    int bad = 0, cases = 0;
    const size_t Bs[] = {2, 3, 16, 63, 64, 255, 1024, 4096};
    for (size_t B : Bs) {
        for (size_t segs = 1; segs <= 6; ++segs) {
            for (size_t last : {B, B - 1 > 1 ? B - 1 : (size_t)1,
                                (size_t)1, (size_t)2}) {
                if (last < 1 || last > B) continue;
                for (int mode = 0; mode < 4; ++mode) {
                    ++cases;
                    bad += run_case(B, segs, last, mode);
                }
            }
        }
    }
    std::printf("segment CRT/carry: %d cases, %s\n", cases,
                bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
