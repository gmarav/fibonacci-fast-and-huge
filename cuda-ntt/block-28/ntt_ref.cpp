// Host-side exactness harness for the 2^28 block engine (CPU only).
//
//   ntt_ref mont  [outdir]     100,000 Montgomery products per prime
//   ntt_ref ntt <lg> [outdir]  one-block convolution vectors + residues
//   ntt_ref block [outdir]     block convolution vs schoolbook
//   ntt_ref fib <n> [outdir]   exact fast doubling through the block engine
//
// Nothing here uses CUDA; `verify_python.py` re-derives every written vector
// from Python integers.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "arith_ref.h"
#include "plan_model.h"

using namespace blk28;

static u64 g_rng = 0x28B10C28B10C28B1ull;
static u64 rnd64() {
    u64 z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static u32 rnd32() { return (u32)rnd64(); }

static bool write_file(const std::string& path, const void* p, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    if (bytes && std::fwrite(p, 1, bytes, f) != bytes) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

static bool write_u64(const std::string& path, const std::vector<u64>& v) {
    return write_file(path, v.data(), v.size() * 8);
}
static bool write_u32(const std::string& path, const std::vector<u32>& v) {
    return write_file(path, v.data(), v.size() * 4);
}

static bool write_limbs(const std::string& path, const std::vector<u32>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    u64 n = v.size();
    std::fwrite(&n, 8, 1, f);
    if (n) std::fwrite(v.data(), 4, (size_t)n, f);
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// mode mont
// ---------------------------------------------------------------------------
static int mode_mont(const std::string& dir) {
    const size_t N = 100000;
    for (int pi = 0; pi < 2; ++pi) {
        const Mod m = pi ? mod1() : mod0();
        std::vector<u64> rec(3 * N);
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i) {
            u64 a = rnd64() % m.p;
            u64 b = rnd64() % m.p;
            u64 got = mont48(a, b, m.p, m.np);
            // Direct oracle: a*b*R^-1 mod p with R^-1 computed by Fermat.
            u64 rinv = powmod(m.R % m.p, m.p - 2, m.p);
            u64 want = (u64)(((u128)a * b % m.p) * (u128)rinv % m.p);
            if (got != want) ++bad;
            rec[3 * i] = a;
            rec[3 * i + 1] = b;
            rec[3 * i + 2] = got;
        }
        // Conversion round trips.
        for (size_t i = 0; i < 10000; ++i) {
            u64 x = rnd64() % m.p;
            if (from_mont(to_mont(x, m), m) != x) ++bad;
        }
        std::string path = dir + (pi ? "/mul_p1.bin" : "/mul_p0.bin");
        if (!write_u64(path, rec)) return 1;
        std::printf("mont %s: %zu vectors, %zu mismatches %s\n", pi ? "p1" : "p0",
                    N, bad, bad ? "FAIL" : "PASS");
        if (bad) return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// mode ntt
// ---------------------------------------------------------------------------
static int mode_ntt(int lg, const std::string& dir) {
    const size_t n = (size_t)1 << lg;
    std::vector<u32> x(n / 2), y(n / 2);
    for (auto& v : x) v = (u32)(rnd64() & 0xFFFF);
    for (auto& v : y) v = (u32)(rnd64() & 0xFFFF);
    std::string base = dir + "/ntt_" + std::to_string(lg);
    if (!write_u32(base + "_x.bin", x) || !write_u32(base + "_y.bin", y))
        return 1;
    for (int pi = 0; pi < 2; ++pi) {
        const Mod m = pi ? mod1() : mod0();
        std::vector<u64> res;
        conv_single_scaled(x, y, lg, m, res);
        if (!write_u64(base + (pi ? "_p1.bin" : "_p0.bin"), res)) return 1;
        std::printf("ntt n=%zu prime %d written\n", n, pi);
    }
    if (lg <= 16) {
        std::vector<u32> prod =
            conv_single_crt(x, y, lg, mod0(), mod1());
        if (!write_u32(base + "_conv.bin", prod)) return 1;
        std::printf("ntt n=%zu CRT product written (%zu limbs)\n", n, prod.size());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// mode block
// ---------------------------------------------------------------------------
static int mode_block(const std::string& dir) {
    (void)dir;
    const Mod m0 = mod0(), m1 = mod1();
    size_t cases = 0, bad = 0;
    for (int bl : {5, 6, 7, 8, 9, 10}) {
        for (int ka : {1, 2, 3, 4}) {
            for (int kb : {1, 2, 3}) {
                const size_t B = (size_t)1 << bl;
                const size_t la = (size_t)ka * B - (rnd32() % (B / 4 + 1));
                const size_t lb = (size_t)kb * B - (rnd32() % (B / 4 + 1));
                std::vector<u32> a(la), b(lb);
                for (auto& v : a) v = rnd32();
                for (auto& v : b) v = rnd32();
                std::vector<u32> want = mul_schoolbook(a, b);
                std::vector<u32> got = conv_blocks(a, b, make_block_plan(bl), m0, m1);
                ++cases;
                if (got != want) {
                    ++bad;
                    std::printf("block bl=%d ka=%d kb=%d la=%zu lb=%zu FAIL "
                                "(got %zu want %zu)\n", bl, ka, kb, la, lb,
                                got.size(), want.size());
                }
            }
        }
    }
    std::printf("block convolution: %zu/%zu cases exact %s\n", cases - bad,
                cases, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// mode fib
// ---------------------------------------------------------------------------
static std::vector<u32> mul_ref(const std::vector<u32>& a,
                                const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    if (a.size() + b.size() <= 2048) return mul_schoolbook(a, b);
    BlockPlan bp;
    size_t need = a.size() + b.size();
    if (need <= ((size_t)1 << BLK28_MAX_LOG2)) {
        int lg = blk28plan::ceillog2(need);
        bp = make_block_plan(lg - 1);
    } else {
        bp = make_block_plan(BLK28_BLOCK_LOG2);
    }
    return conv_blocks(a, b, bp, mod0(), mod1());
}

static int mode_fib(uint64_t n, const std::string& dir) {
    std::vector<u32> a, b{1};
    if (n == 0) {
        if (!write_limbs(dir + "/fib_0.bin", a)) return 1;
        std::printf("fib n=0 limbs=0\n");
        return 0;
    }
    int top = 63;
    while (!((n >> top) & 1)) --top;
    for (int bit = top; bit >= 0; --bit) {
        std::vector<u32> t2 = shl1_limbs(b);
        t2 = sub_limbs(t2, a);
        std::vector<u32> c = mul_ref(a, t2);
        std::vector<u32> d;
        bool want_d = !(bit == 0 && ((n >> bit) & 1) == 0);
        if (want_d) {
            std::vector<u32> aa = mul_ref(a, a);
            std::vector<u32> bb = mul_ref(b, b);
            d = add_limbs(aa, bb);
        }
        if (((n >> bit) & 1) == 0) {
            a = std::move(c);
            b = std::move(d);
        } else {
            std::vector<u32> s = add_limbs(c, d);
            a = std::move(d);
            b = std::move(s);
        }
    }
    while (!a.empty() && a.back() == 0) a.pop_back();
    if (!write_limbs(dir + "/fib_" + std::to_string(n) + ".bin", a)) return 1;
    std::printf("fib n=%" PRIu64 " limbs=%zu written\n", n, a.size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: ntt_ref mont|block [outdir] | ntt <lg> [outdir] | "
                     "fib <n> [outdir]\n");
        return 2;
    }
    std::string mode = argv[1];
    if (mode == "mont") return mode_mont(argc > 2 ? argv[2] : "out");
    if (mode == "block") return mode_block(argc > 2 ? argv[2] : "out");
    if (mode == "ntt") {
        if (argc < 3) return 2;
        return mode_ntt(std::atoi(argv[2]), argc > 3 ? argv[3] : "out");
    }
    if (mode == "fib") {
        if (argc < 3) return 2;
        return mode_fib(std::strtoull(argv[2], nullptr, 10),
                        argc > 3 ? argv[3] : "out");
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
