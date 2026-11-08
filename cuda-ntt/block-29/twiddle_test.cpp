// CPU check of the on-the-fly twiddle derivation used by block29_engine.cu.
//
// The device does not store a 2^29 twiddle table.  For a global stage s it
// keeps a 256-entry table small[s][t] = root_s^t * R and computes the tile
// start by Montgomery exponentiation:
//
//     start = dpow_mont(root_s * R, j0)                 // root_s^{j0} * R
//     tw    = dmul(start, small[s][t])                  // root_s^{j0+t} * R
//
// This test reproduces that arithmetic on the host and compares it with the
// direct value mulmod(root_s^{j0+t}, R, p).  It also checks the tail tables
// and the inverse roots.
//
//   g++ -O2 -std=c++17 twiddle_test.cpp -o build\twiddle_test.exe
//   .\build\twiddle_test.exe

#include <cstdio>
#include <cstdint>

#include "arith_ref.h"

using namespace blk;

typedef uint64_t u64;

static u64 dpow_mont(u64 baseR, u64 e, const Mod& m) {
    u64 r = m.R;
    while (e) {
        if (e & 1) r = mont48(r, baseR, m.p, m.np);
        baseR = mont48(baseR, baseR, m.p, m.np);
        e >>= 1;
    }
    return r;
}

int main() {
    const Mod mods[2] = {mod0(), mod1()};
    const int L = BLK_MAX_LOG2;
    const int TILE = 256;
    int bad = 0;
    long checks = 0;

    for (int pi = 0; pi < 2; ++pi) {
        const Mod& m = mods[pi];
        for (int s = 1; s <= L; ++s) {
            u64 root = powmod(m.w, 1ull << (L - s), m.p);
            u64 invroot = powmod(root, m.p - 2, m.p);
            u64 rootR = mulmod(root, m.R, m.p);
            u64 rootRi = mulmod(invroot, m.R, m.p);

            // 256-entry small tables.
            u64 accF = m.R, accI = m.R;
            for (int t = 0; t < TILE; ++t) {
                u64 wantF = mulmod(powmod(root, (u64)t, m.p), m.R, m.p);
                u64 wantI = mulmod(powmod(invroot, (u64)t, m.p), m.R, m.p);
                if (accF != wantF || accI != wantI) ++bad;
                checks += 2;
                accF = mulmod(accF, root, m.p);
                accI = mulmod(accI, invroot, m.p);
            }

            // Tile starts at a few j0 values, then combine with small[t].
            const u64 j0s[] = {0, 1, 255, 256, 257, 1024, 12345, 1ull << 20};
            const int ts[] = {0, 1, 7, 128, 255};
            for (u64 j0 : j0s) {
                u64 startF = dpow_mont(rootR, j0, m);
                u64 startI = dpow_mont(rootRi, j0, m);
                u64 wantStartF = mulmod(powmod(root, j0 % m.p, m.p), m.R, m.p);
                if (startF != wantStartF) ++bad;
                ++checks;
                for (int t : ts) {
                    u64 smallF = mulmod(powmod(root, (u64)t, m.p), m.R, m.p);
                    u64 smallI = mulmod(powmod(invroot, (u64)t, m.p), m.R, m.p);
                    u64 twF = mont48(startF, smallF, m.p, m.np);
                    u64 twI = mont48(startI, smallI, m.p, m.np);
                    u64 wantF = mulmod(powmod(root, (j0 + (u64)t) % m.p, m.p),
                                       m.R, m.p);
                    u64 wantI = mulmod(powmod(invroot, (j0 + (u64)t) % m.p, m.p),
                                       m.R, m.p);
                    if (twF != wantF || twI != wantI) ++bad;
                    checks += 2;
                }
            }

            // Tail tables for s <= 12: fwd[j] = root^j R, inv[j] = invroot^j R.
            if (s <= 12) {
                u64 off = (1ull << (s - 1)) - 1;
                accF = m.R;
                accI = m.R;
                for (u64 j = 0; j < (1ull << (s - 1)); ++j) {
                    u64 wantF = mulmod(powmod(root, j % m.p, m.p), m.R, m.p);
                    u64 wantI = mulmod(powmod(invroot, j % m.p, m.p), m.R, m.p);
                    if (accF != wantF || accI != wantI) ++bad;
                    checks += 2;
                    accF = mulmod(accF, root, m.p);
                    accI = mulmod(accI, invroot, m.p);
                }
                (void)off;
            }
        }
    }

    std::printf("twiddle derivations: %ld checks, %d mismatches\n", checks, bad);
    if (!bad) std::printf("TWIDDLE DERIVATION TESTS PASSED\n");
    return bad ? 1 : 0;
}
