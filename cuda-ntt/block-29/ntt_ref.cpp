// Host-only exactness harness for the 2^29 two-prime 48-bit engine.
//
// Modes:
//   mont           100k Montgomery products per prime, binary vectors + check
//   ntt   <lg>     length-2^lg convolution for both primes + exact check
//   block          small block-convolution cases + exact check
//   fib   <n>      fast doubling on the CPU with the same arithmetic
//
// Build:
//   g++ -O2 -std=c++17 ntt_ref.cpp -o build\ntt_ref.exe
//
// The `mont` and `ntt` modes write `out/*.bin` for the independent Python
// checker (verify_python.py).  Nothing here touches a GPU.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "arith_ref.h"

using namespace blk;

static u64 g_rng = 0x9E3779B97F4A7C15ull;

static u64 next_rand() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

static bool write_file(const std::string& path, const void* data, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    size_t n = std::fwrite(data, 1, bytes, f);
    std::fclose(f);
    return n == bytes;
}

static int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// ---------------------------------------------------------------------------
// Montgomery multiplication vectors
// ---------------------------------------------------------------------------

static int run_mont(const std::string& dir) {
    const Mod mods[2] = {mod0(), mod1()};
    const char* tag[2] = {"p0", "p1"};
    for (int pi = 0; pi < 2; ++pi) {
        const Mod& m = mods[pi];
        const u64 Rinv = powmod(m.R, m.p - 2, m.p);
        std::vector<u64> rec;
        rec.reserve(3 * 100000);
        for (int it = 0; it < 100000; ++it) {
            u64 a = (it < 8) ? (u64)(m.p - 1 - it) : (next_rand() % m.p);
            u64 b = (it < 8) ? (u64)(m.p - 1 - it) : (next_rand() % m.p);
            u64 got = mont48(a, b, m.p, m.np);
            u64 want = mulmod(mulmod(a, b, m.p), Rinv, m.p);
            if (got != want) {
                std::fprintf(stderr,
                             "%s mont mismatch a=%llu b=%llu got=%llu want=%llu\n",
                             tag[pi], (unsigned long long)a,
                             (unsigned long long)b, (unsigned long long)got,
                             (unsigned long long)want);
                return 1;
            }
            // Conversion round trip: mont(a, R2) == a*R, from_mont undoes it.
            u64 ar = mont48(a, m.R2, m.p, m.np);
            if (ar != mulmod(a, m.R, m.p) || from_mont(ar, m) != a) {
                return fail("mont conversion round trip");
            }
            rec.push_back(a);
            rec.push_back(b);
            rec.push_back(got);
        }
        std::string path = dir + "/mul_" + tag[pi] + ".bin";
        if (!write_file(path, rec.data(), rec.size() * 8)) return 1;
    }
    std::printf("mont: 100000/100000 exact PASS per prime\n");
    return 0;
}

// ---------------------------------------------------------------------------
// NTT convolution vectors
// ---------------------------------------------------------------------------

static int run_ntt(const std::string& dir, int lg) {
    const size_t n = (size_t)1 << lg;
    const size_t half = n / 2;
    std::vector<u32> x(half), y(half);
    // Values < 512 keep the exact convolution below 2^62 for the Python
    // int64 cross-check and below the CRT product by an enormous margin.
    for (size_t i = 0; i < half; ++i) {
        x[i] = (u32)(next_rand() % 512);
        y[i] = (u32)(next_rand() % 512);
    }
    // The O(n^2) schoolbook product is only used for the internal exact check
    // at n <= 2^16.  Larger lengths are verified independently by
    // verify_python.py, which multiplies the full integers with Python big
    // integers (Kronecker substitution) instead of O(n^2) convolution.
    std::vector<u32> exact;
    if (lg <= 16) exact = mul_schoolbook(x, y);

    const Mod mods[2] = {mod0(), mod1()};
    const char* tag[2] = {"p0", "p1"};
    std::vector<std::vector<u64>> res(2);
    for (int pi = 0; pi < 2; ++pi) {
        conv_single(x, y, lg, mods[pi], res[pi]);
        std::vector<u64> scaled;
        conv_single_scaled(x, y, lg, mods[pi], scaled);
        if (scaled != res[pi]) return fail("scaled pointwise convention");
        if (lg <= 16) {
            std::vector<u32> exact_pad(n, 0);
            std::copy(exact.begin(), exact.end(), exact_pad.begin());
            for (size_t t = 0; t < n; ++t) {
                if (res[pi][t] != (u64)(exact_pad[t] % mods[pi].p)) {
                    std::fprintf(stderr, "ntt %s n=%zu mismatch at %zu\n",
                                 tag[pi], n, t);
                    return 1;
                }
            }
        }
    }
    if (lg <= 16) {
        // CRT + carry must reproduce the exact limbs.
        std::vector<u32> got = conv_single_crt(x, y, lg, mods[0], mods[1]);
        if (cmp_limbs(got, exact) != 0) return fail("ntt CRT exact limbs");
    }

    char buf[64];
    std::snprintf(buf, sizeof buf, "/ntt_%d_x.bin", lg);
    if (!write_file(dir + buf, x.data(), x.size() * 4)) return 1;
    std::snprintf(buf, sizeof buf, "/ntt_%d_y.bin", lg);
    if (!write_file(dir + buf, y.data(), y.size() * 4)) return 1;
    for (int pi = 0; pi < 2; ++pi) {
        std::snprintf(buf, sizeof buf, "/ntt_%d_%s.bin", lg, tag[pi]);
        if (!write_file(dir + buf, res[pi].data(), n * 8)) return 1;
    }
    if (lg <= 16) {
        std::snprintf(buf, sizeof buf, "/ntt_%d_conv.bin", lg);
        if (!write_file(dir + buf, exact.data(), exact.size() * 4)) return 1;
    }

    std::printf(
        "ntt n=%zu: %zu residues written for both primes%s\n", n, n,
        lg <= 16 ? " (exact schoolbook PASS)" : " (Python verification)");
    return 0;
}

// ---------------------------------------------------------------------------
// Block-convolution cases
// ---------------------------------------------------------------------------

static int run_block(const std::string& dir) {
    (void)dir;
    const Mod m0 = mod0(), m1 = mod1();
    const int cases[][4] = {
        // bl, la, lb, seed
        {4, 23, 17, 1},   {5, 64, 40, 2},   {6, 100, 77, 3},
        {7, 250, 181, 4}, {8, 600, 333, 5}, {5, 96, 96, 6},
    };
    for (const auto& c : cases) {
        const int bl = c[0];
        const size_t la = (size_t)c[1], lb = (size_t)c[2];
        g_rng = 0x1234567890ABCDEFull + (u64)c[3] * 0x9E3779B97F4A7C15ull;
        std::vector<u32> a(la), b(lb);
        for (auto& v : a) v = (u32)(next_rand() % 512);
        for (auto& v : b) v = (u32)(next_rand() % 512);
        // Make the top limb nonzero so the length is exercised.
        a.back() |= 1;
        b.back() |= 1;
        BlockPlan bp = make_block_plan(bl);
        std::vector<u32> got = conv_blocks(a, b, bp, m0, m1);
        std::vector<u32> want = mul_schoolbook(a, b);
        if (cmp_limbs(got, want) != 0) {
            std::fprintf(stderr,
                         "block mismatch bl=%d la=%zu lb=%zu got=%zu want=%zu\n",
                         bl, la, lb, got.size(), want.size());
            return 1;
        }
        std::printf("block bl=%2d la=%3zu lb=%3zu: %zu limbs exact PASS\n", bl,
                    la, lb, got.size());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// CPU fast doubling using the same arithmetic
// ---------------------------------------------------------------------------

static std::vector<u32> mul_exact(const std::vector<u32>& a,
                                  const std::vector<u32>& b) {
    const Mod m0 = mod0(), m1 = mod1();
    if (a.empty() || b.empty()) return {};
    size_t need = a.size() + b.size();
    int lg = 1;
    while (((size_t)1 << lg) < need) ++lg;
    if (lg > BLK_MAX_LOG2) {
        std::fprintf(stderr, "mul_exact: length %zu exceeds 2^29\n", need);
        std::exit(2);
    }
    return conv_single_crt(a, b, lg, m0, m1);
}

static int run_fib(const std::string& dir, uint64_t n) {
    // Invariant: (a,b) = (F(k), F(k+1)).  After processing all bits of n,
    // a = F(n) and b = F(n+1).
    std::vector<u32> a;      // F(0) = 0
    std::vector<u32> b = {1};  // F(1) = 1
    if (n != 0) {
        int top = 63;
        while (!((n >> top) & 1)) --top;
        for (int bit = top; bit >= 0; --bit) {
            // c = a*(2b-a), d = a*a + b*b
            std::vector<u32> t = shl1_limbs(b);
            t = sub_limbs(t, a);
            std::vector<u32> c = mul_exact(a, t);
            std::vector<u32> aa = mul_exact(a, a);
            std::vector<u32> bb = mul_exact(b, b);
            std::vector<u32> d = add_limbs(aa, bb);
            if (((n >> bit) & 1) == 0) {
                a = std::move(c);
                b = std::move(d);
            } else {
                std::vector<u32> cd = add_limbs(c, d);
                a = std::move(d);
                b = std::move(cd);
            }
        }
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "/fib_%llu.bin", (unsigned long long)n);
    FILE* f = std::fopen((dir + buf).c_str(), "wb");
    if (!f) return fail("open fib output");
    u64 cnt = a.size();
    std::fwrite(&cnt, 8, 1, f);
    if (cnt) std::fwrite(a.data(), 4, (size_t)cnt, f);
    std::fclose(f);
    std::printf("fib n=%llu: %zu limbs (%.1f bits) -> fib_%llu.bin\n",
                (unsigned long long)n, a.size(), a.size() * 32.0,
                (unsigned long long)n);
    return 0;
}

int main(int argc, char** argv) {
    std::string dir = "out";
    if (argc < 2) {
        std::printf("usage: %s mont|ntt <lg>|block|fib <n> [outdir]\n", argv[0]);
        return 2;
    }
    std::string mode = argv[1];
    if (argc >= 3 && std::string(argv[argc - 1]).rfind("-", 0) != 0) {
        // optional trailing outdir, detected by a non-numeric last argument
        std::string last = argv[argc - 1];
        bool numeric = !last.empty() &&
                       last.find_first_not_of("0123456789") == std::string::npos;
        if (!numeric) dir = last;
    }
    if (mode == "mont") return run_mont(dir);
    if (mode == "ntt") {
        int lg = (argc >= 3) ? std::atoi(argv[2]) : 12;
        return run_ntt(dir, lg);
    }
    if (mode == "block") return run_block(dir);
    if (mode == "fib") {
        uint64_t n = (argc >= 3) ? std::strtoull(argv[2], nullptr, 10) : 10000;
        return run_fib(dir, n);
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
