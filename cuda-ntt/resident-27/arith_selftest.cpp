// Host-side exactness self-test for the widened 32-bit arithmetic used by
// resident-27-27.  It mirrors dmul / dadd / dshoup / dred / crt_words from
// resident27.cu and checks them against __int128 reference arithmetic.
//
// Build (MinGW g++) and run:
//   g++ -O2 -std=c++17 arith_selftest.cpp -o build\arith_selftest.exe
//   .\build\arith_selftest.exe
//
// This never touches the GPU and uses no floating point.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

constexpr int kNumPrimes = 3;
constexpr u32 kP[kNumPrimes] = {2281701377u, 3489660929u, 3892314113u};

static u128 P3() {
    u128 p = 1;
    for (int i = 0; i < kNumPrimes; ++i) p *= kP[i];
    return p;
}

static u32 host_mont_np(u32 p) {
    u32 inv = 1u;
    for (int i = 0; i < 5; ++i) inv *= 2u - p * inv;
    return (u32)(0u - inv);
}

static u32 dmul(u32 a, u32 b, u32 p, u32 np) {
    u64 t = (u64)a * (u64)b;
    u32 m = (u32)t * np;
    u64 s = t + (u64)m * (u64)p;
    u64 carry = (s < t) ? 1ull : 0ull;
    u64 u = (s >> 32) | (carry << 32);
    if (u >= p) u -= p;
    return (u32)u;
}

static u32 dadd(u32 a, u32 b, u32 p) {
    u64 s = (u64)a + (u64)b;
    if (s >= p) s -= p;
    return (u32)s;
}

static u32 dsub(u32 a, u32 b, u32 p) {
    u32 d = a - b;
    if (a < b) d += p;
    return d;
}

static u32 dred(u32 v, u32 p) {
    v -= (v >= p) ? p : 0u;
    return v;
}

static u32 dshoup(u32 x, u32 w, u32 ws, u32 p) {
    u32 q = (u32)(((u64)ws * x) >> 32);
    u64 r = (u64)w * (u64)x - (u64)q * (u64)p;
    if (r >= p) r -= p;
    return (u32)r;
}

struct CrtArgs {
    u32 P0, P1, P2;
    u32 np0, np1, np2;
    u32 inv01, inv01_sh;
    u32 p0m2, p0m2_sh;
    u32 invp01, invp01_sh;
};

static u32 hpowmod(u32 b, u64 e, u32 p) {
    u64 r = 1, bb = (u64)b % p;
    while (e) {
        if (e & 1ull) r = (r * bb) % p;
        bb = (bb * bb) % p;
        e >>= 1;
    }
    return (u32)r;
}

static CrtArgs make_crt() {
    CrtArgs c{};
    const u32 p0 = kP[0], p1 = kP[1], p2 = kP[2];
    c.P0 = p0; c.P1 = p1; c.P2 = p2;
    c.np0 = host_mont_np(p0);
    c.np1 = host_mont_np(p1);
    c.np2 = host_mont_np(p2);
    c.inv01 = hpowmod(p0 % p1, (u64)p1 - 2, p1);
    c.inv01_sh = (u32)(((u64)c.inv01 << 32) / p1);
    c.p0m2 = p0 % p2;
    c.p0m2_sh = (u32)(((u64)c.p0m2 << 32) / p2);
    u64 p01 = (u64)p0 * p1;
    c.invp01 = hpowmod((u32)(p01 % p2), (u64)p2 - 2, p2);
    c.invp01_sh = (u32)(((u64)c.invp01 << 32) / p2);
    return c;
}

static void crt_words(u32 a0, u32 a1, u32 a2, const CrtArgs& c, u32& w0,
                      u32& w1, u32& w2) {
    u32 d1 = a1 - a0;
    if (a1 < a0) d1 += c.P1;
    u32 t1 = dshoup(d1, c.inv01, c.inv01_sh, c.P1);

    u32 u = dshoup(t1, c.p0m2, c.p0m2_sh, c.P2);
    u32 e = dadd(a0, u, c.P2);

    u32 d2 = a2 - e;
    if (a2 < e) d2 += c.P2;
    u32 t2 = dshoup(d2, c.invp01, c.invp01_sh, c.P2);

    u64 lo = (u64)c.P0 * t1 + a0;
    u64 A = (u64)c.P0 * c.P1;
    u64 alo = A & 0xffffffffull;
    u64 ahi = A >> 32;
    u64 plo = alo * t2;
    u64 phi = ahi * t2;
    u64 b0 = plo & 0xffffffffull;
    u64 mid = (plo >> 32) + (phi & 0xffffffffull);
    u64 b1 = mid & 0xffffffffull;
    u64 b2 = (phi >> 32) + (mid >> 32);

    u64 s = b0 + lo;
    w0 = (u32)s;
    u64 carry = s >> 32;
    s = b1 + carry;
    w1 = (u32)s;
    carry = s >> 32;
    w2 = (u32)(b2 + carry);
}

static int bitlen(u128 v) {
    int n = 0;
    while (v) {
        v >>= 1;
        ++n;
    }
    return n;
}

static int g_fail = 0;

static void expect(bool ok, const char* what, u64 a, u64 b, u64 got, u64 want) {
    if (!ok) {
        std::fprintf(stderr,
                     "FAIL %s a=%llu b=%llu got=%llu want=%llu\n", what,
                     (unsigned long long)a, (unsigned long long)b,
                     (unsigned long long)got, (unsigned long long)want);
        if (++g_fail > 20) std::exit(1);
    }
}

int main() {
    const u128 P = P3();
    const u32 rmod = (u32)(((u128)1 << 32) % kP[0]);
    (void)rmod;
    CrtArgs c = make_crt();

    std::printf("primes %u %u %u\n", kP[0], kP[1], kP[2]);
    std::printf("np     %u %u %u\n", c.np0, c.np1, c.np2);

    // Constants from the bigbits survey / task brief.
    const u32 exp_np[3] = {2281701375u, 3489660927u, 3892314111u};
    const u32 exp_R[3] = {2013265919u, 805306367u, 402653183u};
    const u32 exp_R2[3] = {773725730u, 1961643719u, 1703176688u};
    for (int i = 0; i < kNumPrimes; ++i) {
        const u32 p = kP[i];
        const u32 r = (u32)(((u128)1 << 32) % p);
        const u32 r2 = (u32)(((u128)r * r) % p);
        expect(host_mont_np(p) == exp_np[i], "np const", p, 0,
               host_mont_np(p), exp_np[i]);
        expect(r == exp_R[i], "R const", p, 0, r, exp_R[i]);
        expect(r2 == exp_R2[i], "R2 const", p, 0, r2, exp_R2[i]);
    }
    std::printf("P bits %d\n", bitlen(P));

    // Exhaustive-ish boundaries plus random sampling for every prime.
    std::mt19937_64 rng(0x5EED1234ull);
    for (int pi = 0; pi < kNumPrimes; ++pi) {
        const u32 p = kP[pi];
        const u32 r = (u32)(((u128)1 << 32) % p);
        const u32 r2 = (u32)(((u128)r * r) % p);
        const u32 np = host_mont_np(p);
        // Montgomery round trip: mont(a, r2) == a*R mod p.
        for (int it = 0; it < 200000; ++it) {
            u32 a = (u32)rng();
            u32 got = dmul(a, r2, p, np);
            u32 want = (u32)(((u128)a << 32) % p);
            expect(got == want, "mont-to-plain", a, r2, got, want);
        }
        // dmul vs the u128 reference for a*b*R^-1 mod p.
        const u32 Rinv = hpowmod(r, (u64)p - 2, p);
        for (int it = 0; it < 200000; ++it) {
            u32 a = it < 8 ? (u32)(p - 1 - it) : (u32)rng();
            u32 b = it < 8 ? (u32)(p - 1 - it) : (u32)rng();
            a %= p;
            b %= p;
            u32 got = dmul(a, b, p, np);
            u32 want = (u32)(((u128)a * b) % p * Rinv % p);
            expect(got == want, "dmul", a, b, got, want);
        }
        // dadd / dsub vs u128.
        const u32 edges[] = {0u, 1u, 2u, p - 1, p - 2, p / 2, p / 2 + 1,
                             0x80000000u % p, 0xffffffffu % p};
        for (u32 a0 : edges)
            for (u32 b0 : edges) {
                u32 a = a0 % p, b = b0 % p;
                u32 want = (u32)(((u128)a + b) % p);
                expect(dadd(a, b, p) == want, "dadd", a, b, dadd(a, b, p), want);
                want = (u32)(((u128)a + p - b) % p);
                expect(dsub(a, b, p) == want, "dsub", a, b, dsub(a, b, p), want);
            }
        for (int it = 0; it < 200000; ++it) {
            u32 a = (u32)(rng() % p), b = (u32)(rng() % p);
            u32 want = (u32)(((u128)a + b) % p);
            expect(dadd(a, b, p) == want, "dadd", a, b, dadd(a, b, p), want);
            want = (u32)(((u128)a + p - b) % p);
            expect(dsub(a, b, p) == want, "dsub", a, b, dsub(a, b, p), want);
        }
        // dred must be the canonical reduction of a full 32-bit value.
        for (int it = 0; it < 200000; ++it) {
            u32 v = (u32)rng();
            u32 want = v % p;
            expect(dred(v, p) == want, "dred", v, 0, dred(v, p), want);
        }
        // Shoup: w fixed, x < p, ws = floor(w*2^32/p).
        for (int it = 0; it < 100000; ++it) {
            u32 w = (u32)(rng() % p);
            u32 ws = (u32)(((u128)w << 32) / p);
            u32 x = (u32)(rng() % p);
            u32 got = dshoup(x, w, ws, p);
            u32 want = (u32)(((u128)w * x) % p);
            expect(got == want, "dshoup", x, w, got, want);
        }
    }

    // CRT round trip over the full modulus: random x, residues, reconstruct.
    const u64 Pw0 = (u64)(P & 0xffffffffull);
    const u64 Pw1 = (u64)((P >> 32) & 0xffffffffull);
    const u64 Pw2 = (u64)((P >> 64) & 0xffffffffull);
    std::printf("P words %llu %llu %llu\n", (unsigned long long)Pw0,
                (unsigned long long)Pw1, (unsigned long long)Pw2);
    for (int it = 0; it < 400000; ++it) {
        u128 x;
        if (it < 6) {
            // Boundary cases: 0, 1, P-1, and just below each prime.
            const u128 cand[] = {(u128)0, (u128)1, P - 1, P - 2,
                                 (u128)kP[0] - 1, (u128)kP[1] - 1};
            x = cand[it];
        } else if (it < 12) {
            x = ((u128)kP[0] * kP[1]) * (u128)(rng() % kP[2]) + (rng() % kP[0]);
        } else {
            x = ((u128)rng() << 64) ^ (u128)rng();
            x %= P;
        }
        u32 a0 = (u32)(x % kP[0]);
        u32 a1 = (u32)(x % kP[1]);
        u32 a2 = (u32)(x % kP[2]);
        u32 w0, w1, w2;
        crt_words(a0, a1, a2, c, w0, w1, w2);
        u128 got = (u128)w0 | ((u128)w1 << 32) | ((u128)w2 << 64);
        expect(got == x, "crt", (u64)x, (u64)(x >> 64), (u64)(got == x),
               (u64)1);
    }

    // Exact coefficient-bound arithmetic used by verify_prime_set().
    const u64 mlimbs = ((u64)1 << 26) - 1ull;
    const u64 k2 = 2ull * mlimbs;
    const u64 bsq = 0xfffffffe00000001ull;
    const u64 q0 = (bsq & 0xffffffffull) * k2;
    const u64 q1 = (bsq >> 32) * k2;
    const u64 qmid = (q0 >> 32) + (q1 & 0xffffffffull);
    const u64 c0 = q0 & 0xffffffffull;
    const u64 c1 = qmid & 0xffffffffull;
    const u64 c2 = (q1 >> 32) + (qmid >> 32);
    u128 bound = ((u128)c2 << 64) | ((u128)c1 << 32) | (u128)c0;
    u128 exact = (u128)2 * mlimbs * (u128)0xffffffffull * 0xffffffffull;
    std::printf("bound 3-word %llu %llu %llu\n", (unsigned long long)c0,
                (unsigned long long)c1, (unsigned long long)c2);
    std::printf("bound bits %d\n", bitlen(exact));
    expect(bound == exact, "bound-words", (u64)exact, 0,
           (u64)(bound == exact), 1);
    const bool product_ok =
        (Pw2 > c2) || (Pw2 == c2 && (Pw1 > c1 || (Pw1 == c1 && Pw0 > c0)));
    expect(product_ok, "P>bound", 0, 0, product_ok, 1);
    std::printf("P > bound: %s (P/bound = %.4f)\n", product_ok ? "yes" : "no",
                (double)P / (double)exact);

    if (g_fail == 0) std::printf("ALL ARITHMETIC TESTS PASSED\n");
    return g_fail == 0 ? 0 : 1;
}
