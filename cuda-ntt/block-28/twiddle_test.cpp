// Exact test of the on-the-fly twiddle derivation used by the CUDA stage
// kernels, at every stage of a 2^28 transform.
//
// The device derives each thread's twiddle as
//     start = root^j0 * R            (Montgomery exponentiation)
//     tw    = start * small[t]       (Montgomery product)
// with small[t] = root^t * R, which must equal root^(j0+t) * R.  The inverse
// direction uses the same construction with root^-1.

#include <cstdio>
#include <vector>

#include "arith_ref.h"

using namespace blk28;

// Mirror of the device dpow_mont: base is in Montgomery form, result is
// base^e * R mod p.
static u64 dpow_mont(u64 base, u64 e, const Mod& m) {
    u64 r = m.R;
    while (e) {
        if (e & 1) r = mont48(r, base, m.p, m.np);
        base = mont48(base, base, m.p, m.np);
        e >>= 1;
    }
    return r;
}

static int check_prime(const Mod& m, const char* name) {
    size_t checked = 0, bad = 0;
    for (int s = 1; s <= BLK28_MAX_LOG2; ++s) {
        const u64 root = stage_root(m, s);
        const u64 rootR = mulmod(root, m.R, m.p);
        // order check
        if (mulmod(root, powmod(root, ((u64)1 << (s - 1)) - 1, m.p), m.p) !=
                m.p - 1) {
            std::printf("  %s s=%d root order FAIL\n", name, s);
            ++bad;
        }
        // 256-entry tile
        std::vector<u64> small(256);
        u64 acc = m.R;
        for (int t = 0; t < 256; ++t) {
            small[t] = acc;
            acc = mulmod(acc, root, m.p);
        }
        const u64 half = (u64)1 << (s - 1);
        u64 starts[6] = {0, 256, 4096, half > 2048 ? half - 2048 : 0,
                         half > 256 ? half - 256 : 0, half > 0 ? half - 1 : 0};
        for (u64 j0 : starts) {
            u64 start = dpow_mont(rootR, j0, m);
            for (int t = 0; t < 256; ++t) {
                if (j0 + (u64)t >= half) break;
                u64 got = mont48(start, small[t], m.p, m.np);
                u64 want = twiddle(m, root, j0 + (u64)t);
                ++checked;
                if (got != want) ++bad;
            }
        }
        // inverse root
        const u64 iroot = powmod(m.winv, 1ull << (BLK28_ROOT_LOG2 - s), m.p);
        const u64 irootR = mulmod(iroot, m.R, m.p);
        u64 istart = dpow_mont(irootR, 4096, m);
        u64 ismall = m.R;
        for (int t = 0; t < 256; ++t) {
            u64 got = mont48(istart, ismall, m.p, m.np);
            u64 want = twiddle(m, iroot, 4096 + (u64)t);
            ++checked;
            if (got != want) ++bad;
            ismall = mulmod(ismall, iroot, m.p);
        }
        // shared-memory tail tables: stages 1..12
        if (s <= 12) {
            acc = m.R;
            for (u64 j = 0; j < ((u64)1 << (s - 1)); ++j) {
                u64 want = twiddle(m, root, j);
                ++checked;
                if (acc != want) ++bad;
                acc = mulmod(acc, root, m.p);
            }
        }
    }
    std::printf("twiddles %s: %zu checks, %zu mismatches %s\n", name, checked,
                bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main() {
    int bad = 0;
    bad += check_prime(mod0(), "p0");
    bad += check_prime(mod1(), "p1");
    if (bad) {
        std::printf("TWIDDLE TEST FAIL\n");
        return 1;
    }
    std::printf("TWIDDLE TEST PASS\n");
    return 0;
}
