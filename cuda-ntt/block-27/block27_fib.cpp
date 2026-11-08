// Host driver for the block-27 exact blockwise Fibonacci engine.
//
//   block27_fib fib --n N [--out FILE] [--B LOG2] [--force-block] [--oracle PATH]
//   block27_fib mul --a A.bin --b B.bin --out C.bin [--B LOG2]
//   block27_fib selftest [--iters N] [--B LOG2]
//   block27_fib bench --sizes LIST [--runs R]
//   block27_fib info
//
// Small steps (operands that fit one 2^27 transform) use the validated
// resident-27 fast-doubling step; larger steps use the blockwise driver.
// Limb format: u64 count followed by little-endian u32 limbs.

#include "block27.h"
#include "../resident-27/resident27.h"
#include "../common/limb_io.h"
#include "../common/limb_math.h"
#include "../common/report.h"
#include "../common/run_options.h"
#include "../common/stats_json.h"
#include "../common/time_util.h"
#include "../common/usage.h"

using namespace fibcommon;

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;

namespace {

constexpr u64 kResidentMaxLimbs = ((u64)1 << 26) - 1; // 2*m+2 <= 2^27

// ---------------------------------------------------------------------------
// SHA-256 (for reporting file hashes; not used by the arithmetic path)
// ---------------------------------------------------------------------------

struct Sha256 {
    u32 h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    u64 total = 0;
    unsigned char buf[64];
    size_t used = 0;

    static u32 ror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const unsigned char* p) {
        static const u32 K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        u32 w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((u32)p[4 * i] << 24) | ((u32)p[4 * i + 1] << 16) |
                   ((u32)p[4 * i + 2] << 8) | (u32)p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            u32 s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            u32 s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
            g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = hh + S1 + ch + K[i] + w[i];
            u32 S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            u32 mj = (a & b) ^ (a & c) ^ (b & c);
            u32 t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(const void* data, size_t len) {
        const unsigned char* p = (const unsigned char*)data;
        total += len;
        while (len) {
            size_t take = std::min(len, sizeof(buf) - used);
            std::memcpy(buf + used, p, take);
            used += take;
            p += take;
            len -= take;
            if (used == sizeof(buf)) {
                block(buf);
                used = 0;
            }
        }
    }

    std::string hex() {
        unsigned char pad[72];
        size_t padlen = 0;
        u64 bits = total * 8;
        pad[padlen++] = 0x80;
        while ((total + padlen) % 64 != 56) pad[padlen++] = 0;
        for (int i = 7; i >= 0; --i) pad[padlen++] = (unsigned char)(bits >> (8 * i));
        update(pad, padlen);
        char out[65];
        for (int i = 0; i < 8; ++i)
            std::snprintf(out + 8 * i, 9, "%08X", h[i]);
        out[64] = 0;
        return std::string(out);
    }
};

std::string hash_limbs(const std::vector<u32>& v) {
    Sha256 s;
    u64 n = v.size();
    s.update(&n, 8);
    s.update(v.data(), v.size() * 4);
    return s.hex();
}

// ---------------------------------------------------------------------------
// Limb files and host arithmetic
// ---------------------------------------------------------------------------

// t = 2b - a (the caller guarantees 2b >= a).
std::vector<u32> twice_b_minus_a(const std::vector<u32>& a,
                                 const std::vector<u32>& b) {
    const size_t n = std::max(a.size(), b.size()) + 1;
    std::vector<u32> r(n, 0u);
    long long st = 0;
    for (size_t i = 0; i < n; ++i) {
        long long bv = (i < b.size()) ? (long long)b[i] : 0ll;
        long long av = (i < a.size()) ? (long long)a[i] : 0ll;
        long long s = 2 * bv + st - av;
        if (s < 0) {
            r[i] = (u32)(s + (1ll << 32));
            st = -1;
        } else {
            r[i] = (u32)s;
            st = s >> 32;
        }
    }
    if (st != 0) {
        std::fprintf(stderr, "twice_b_minus_a: negative result\n");
        std::exit(1);
    }
    trim(r);
    return r;
}

// Exact reference multiply for the self-test (base 2^32 schoolbook).
std::vector<u32> ref_mul(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<u32> c(a.size() + b.size() + 1, 0u);
    for (size_t i = 0; i < a.size(); ++i) {
        u64 carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            u64 cur = (u64)a[i] * b[j] + c[i + j] + carry;
            c[i + j] = (u32)cur;
            carry = cur >> 32;
        }
        size_t k = i + b.size();
        while (carry) {
            u64 cur = (u64)c[k] + carry;
            c[k] = (u32)cur;
            carry = cur >> 32;
            ++k;
        }
    }
    trim(c);
    return c;
}

// ---------------------------------------------------------------------------
// Random generator
// ---------------------------------------------------------------------------

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    u64 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    u32 limb() { return (u32)(next() >> 32); }
    size_t range(size_t lo, size_t hi) { // inclusive
        return lo + (size_t)(next() % (hi - lo + 1));
    }
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct BlockStatsSnap {
    u64 pairs = 0, pcie = 0, h2d = 0, d2h = 0, kernels = 0;
    double fwd = 0, inv = 0, crt = 0, carry = 0, xfer = 0, device = 0;
};

BlockStatsSnap snap_stats() {
    BlockStatsSnap s;
    block27_get_stats(&s.pairs, &s.pcie, &s.kernels, &s.fwd, &s.inv, &s.crt,
                      &s.carry, &s.xfer);
    block27_get_stats2(&s.h2d, &s.d2h, &s.device);
    return s;
}

BlockStatsSnap diff_stats(const BlockStatsSnap& a, const BlockStatsSnap& b) {
    BlockStatsSnap d;
    d.pairs = b.pairs - a.pairs;
    d.pcie = b.pcie - a.pcie;
    d.h2d = b.h2d - a.h2d;
    d.d2h = b.d2h - a.d2h;
    d.kernels = b.kernels - a.kernels;
    d.fwd = b.fwd - a.fwd;
    d.inv = b.inv - a.inv;
    d.crt = b.crt - a.crt;
    d.carry = b.carry - a.carry;
    d.xfer = b.xfer - a.xfer;
    d.device = b.device - a.device;
    return d;
}

struct VramSampler {
    std::atomic<bool> done{false};
    std::thread th;
    size_t peak = 0, total = 0;
    void start() {
        th = std::thread([this]() {
            for (;;) {
                size_t u = 0, t = 0;
                if (block27_vram_info(&u, &t) == 0) {
                    if (u > peak) peak = u;
                    total = t;
                }
                if (done.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
    }
    void stop() {
        done.store(true);
        if (th.joinable()) th.join();
    }
};

// ---------------------------------------------------------------------------
// Fast doubling loop
// ---------------------------------------------------------------------------

struct FibRun {
    std::vector<u32> value;
    double wall_ms = 0.0;
    double native_device_ms = 0.0;
    int native_steps = 0;
    int block_steps = 0;
    int max_block_count = 0;
    BlockStatsSnap bs;
    size_t vram_used = 0, vram_total = 0;
};

FibRun run_fib(u64 n, int B_log2, bool force_block, bool verbose) {
    FibRun r;
    const double t0 = now_ms();
    if (n == 0) {
        r.wall_ms = now_ms() - t0;
        return r;
    }

    std::vector<u32> a; // F(0)
    std::vector<u32> b = {1u};
    bool block_ready = false;
    bool native_freed = false;

    const BlockStatsSnap s0 = snap_stats();
    int bits = 0;
    for (u64 v = n; v; v >>= 1) ++bits;

    for (int i = bits - 1; i >= 0; --i) {
        const int bit = (int)((n >> i) & 1ull);
        const size_t m = std::max(a.size(), b.size());
        const bool want_d = !(i == 0 && bit == 0);
        const bool use_native = !force_block && m <= kResidentMaxLimbs;

        // k = 0 seed: (c, d) = (F(0), F(1)) = (0, 1).  No multiplication is
        // needed (and the block driver rejects empty operands).
        if (a.empty() && b.size() == 1 && b[0] == 1u) {
            std::vector<u32> c;
            std::vector<u32> d = {1u};
            if (i == 0) {
                r.value = (bit == 0) ? std::move(c) : std::move(d);
            } else if (bit == 0) {
                a = std::move(c);
                b = std::move(d);
            } else {
                a = std::move(d);
                b = {1u}; // F(1), F(2) = (1, 1)
            }
            continue;
        }

        const double ts = now_ms();
        if (use_native) {
            if (block_ready || block27_is_ready()) {
                block27_shutdown();
                block_ready = false;
            }
            uint32_t* oa = nullptr;
            uint32_t* ob = nullptr;
            size_t la = 0, lb = 0;
            int st = gpu_ntt_doubling_step(
                a.empty() ? nullptr : a.data(), a.size(),
                b.empty() ? nullptr : b.data(), b.size(), (uint32_t)bit, &oa,
                &la, &ob, &lb);
            if (st != 0) {
                std::fprintf(stderr, "resident step failed: %d at k-bit %d\n", st,
                             i);
                std::exit(4);
            }
            std::vector<u32> na(oa, oa + la), nb(ob, ob + lb);
            gpu_fib_free_limbs(oa);
            gpu_fib_free_limbs(ob);
            trim(na);
            trim(nb);
            a.swap(na);
            b.swap(nb);
            ++r.native_steps;
            {
                gpu_fib_stats gs{};
                if (gpu_fib_get_stats(&gs) == 0) r.native_device_ms += gs.gpu_ms;
            }
            if (verbose)
                std::fprintf(stderr,
                             "  step bit=%d native  m=%zu limbs  %.1f ms\n", bit,
                             m, now_ms() - ts);
        } else {
            if (!native_freed) {
                gpu_fib_shutdown();
                native_freed = true;
            }
            if (!block_ready) {
                if (block27_configure(B_log2) != 0) {
                    std::fprintf(stderr, "block configure failed\n");
                    std::exit(4);
                }
                block_ready = true;
            }
            uint32_t* pc = nullptr;
            uint32_t* pd = nullptr;
            size_t lc = 0, ld = 0;
            int st = block27_step(a.data(), a.size(), b.data(), b.size(), B_log2,
                                  want_d ? 1 : 0, &pc, &lc, &pd, &ld);
            if (st != 0) {
                std::fprintf(stderr, "block step failed: %d at k-bit %d\n", st, i);
                std::exit(4);
            }
            std::vector<u32> c(pc, pc + lc);
            std::vector<u32> d;
            if (pd) d.assign(pd, pd + ld);
            std::free(pc);
            std::free(pd);
            trim(c);
            trim(d);
            if (i == 0) {
                r.value = (bit == 0) ? std::move(c) : std::move(d);
            } else if (bit == 0) {
                a = std::move(c);
                b = std::move(d);
            } else {
                std::vector<u32> cd = add_limbs(c, d);
                a = std::move(d);
                b = std::move(cd);
            }
            ++r.block_steps;
            const int kc =
                (int)((m + (size_t{1} << B_log2) - 1) >> B_log2);
            r.max_block_count = std::max(r.max_block_count, kc);
            if (verbose)
                std::fprintf(stderr,
                             "  step bit=%d block   m=%zu limbs (%d blocks)  "
                             "%.1f ms\n",
                             bit, m, kc, now_ms() - ts);
        }
    }

    if (block_ready) {
        // Leave the engine allocated for stats/VRAM reporting; the caller or
        // process exit releases it.
    }
    if (r.value.empty() && n != 0) r.value = std::move(a);
    r.bs = diff_stats(s0, snap_stats());
    block27_vram_info(&r.vram_used, &r.vram_total);
    r.wall_ms = now_ms() - t0;
    return r;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

int cmd_fib(int argc, char** argv) {
    fibcommon::RunOptions run;
    int B = 26;
    bool force_block = false;
    bool verbose = false;
    std::string oracle;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (fibcommon::parse_run_option(a, i, argc, argv, run)) {
            continue;
        } else if (a == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (a == "--force-block") force_block = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--oracle" && i + 1 < argc) oracle = argv[++i];
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (!run.have_n) {
        std::fprintf(stderr, "fib needs --n\n");
        return 2;
    }
    const u64 n = run.n;
    std::string out_path = run.out_path;
    std::string stats_path = run.stats_path;
    std::string progress_path = run.progress_path;
    int repeat = run.iterations;
    if (repeat < 1) repeat = 1;
    write_progress(progress_path, 0, repeat, "starting");

    VramSampler vram;
    vram.start();
    FibRun r;
    std::vector<double> samples_ms;
    std::vector<double> stream_ms;
    std::vector<double> call_ms;
    std::vector<double> total_ms;
    for (int rep = 0; rep < repeat; ++rep) {
        r = run_fib(n, B, force_block, verbose);
        samples_ms.push_back(r.wall_ms);
        stream_ms.push_back(r.native_device_ms + r.bs.device);
        call_ms.push_back(r.wall_ms);
        total_ms.push_back(r.wall_ms);
        fibcommon::print_fib_header(
            "block-27", n, rep + 1, repeat, r.value.size(),
            limb_bits(r.value));
        fibcommon::print_timing(r.wall_ms, r.wall_ms,
                                r.native_device_ms + r.bs.device);
        fibcommon::print_memory((double)r.vram_used / 1048576.0, 0.0);
        char extras[512];
        std::snprintf(
            extras, sizeof(extras),
            "native_steps=%d native_device_ms=%.1f block_steps=%d "
            "max_blocks=%d pairs=%llu kernels=%llu h2d_mib=%.1f d2h_mib=%.1f "
            "fwd_ms=%.1f inv_ms=%.1f crt_ms=%.1f carry_ms=%.1f "
            "xfer_ms=%.1f vram_total_mib=%.1f",
            r.native_steps, r.native_device_ms, r.block_steps,
            r.max_block_count, (unsigned long long)r.bs.pairs,
            (unsigned long long)r.bs.kernels,
            (double)r.bs.h2d / 1048576.0, (double)r.bs.d2h / 1048576.0,
            r.bs.fwd, r.bs.inv, r.bs.crt, r.bs.carry, r.bs.xfer,
            (double)r.vram_total / 1048576.0);
        fibcommon::print_engine(extras);
        std::fflush(stdout);
        char progress_detail[64];
        std::snprintf(progress_detail, sizeof(progress_detail), "%.1f ms",
                      r.wall_ms);
        write_progress(progress_path, rep + 1, repeat, progress_detail);
        block27_reset_stats();
    }
    vram.stop();
    fibcommon::print_memory((double)vram.peak / 1048576.0, 0.0);

    if (!out_path.empty()) {
        if (!write_limbs(out_path, r.value)) {
            std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
            return 1;
        }
        long long fsz = -1;
        if (FILE* f = std::fopen(out_path.c_str(), "rb")) {
            if (_fseeki64(f, 0, SEEK_END) == 0) fsz = _ftelli64(f);
            std::fclose(f);
        }
        fibcommon::print_written(
            out_path.c_str(), fsz,
            (long long)(8 + 4 * (long long)r.value.size()));
        fibcommon::print_sha256(hash_limbs(r.value).c_str());
    }
    if (!write_stats_json(stats_path, "block-27", n, samples_ms, stream_ms,
                          call_ms, total_ms, (double)vram.peak / 1048576.0,
                          0.0)) {
        return 1;
    }

    if (!oracle.empty()) {
        std::string tmp = out_path.empty() ? std::string("oracle_tmp.bin")
                                           : out_path + ".oracle";
        char cmd[1024];
        std::snprintf(cmd, sizeof(cmd), "\"%s\" fib %llu \"%s\"", oracle.c_str(),
                      (unsigned long long)n, tmp.c_str());
        int rc = std::system(cmd);
        if (rc != 0) {
            std::fprintf(stderr, "oracle failed (rc=%d)\n", rc);
            return 1;
        }
        std::vector<u32> ref;
        if (!read_limbs(tmp, ref)) return 1;
        trim(ref);
        const size_t d = first_diff(r.value, ref);
        if (d == (size_t)-1) {
            std::printf("oracle: MATCH (%zu limbs)\n", ref.size());
        } else {
            std::printf("oracle: MISMATCH at limb %zu (%zu vs %zu limbs)\n", d,
                        r.value.size(), ref.size());
            return 1;
        }
        if (out_path.empty()) std::remove(tmp.c_str());
    }
    return 0;
}

int cmd_mul(int argc, char** argv) {
    std::string pa, pb, po;
    int B = 26;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--a" && i + 1 < argc) pa = argv[++i];
        else if (a == "--b" && i + 1 < argc) pb = argv[++i];
        else if (a == "--out" && i + 1 < argc) po = argv[++i];
        else if (a == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (pa.empty() || pb.empty() || po.empty()) return 2;
    std::vector<u32> a, b;
    if (!read_limbs(pa, a) || !read_limbs(pb, b)) return 1;
    const BlockStatsSnap s0 = snap_stats();
    const double t0 = now_ms();
    uint32_t* pc = nullptr;
    size_t lc = 0;
    int st = block27_mul(a.data(), a.size(), b.data(), b.size(), B, &pc, &lc);
    if (st != 0) return 3;
    std::vector<u32> c(pc, pc + lc);
    std::free(pc);
    trim(c);
    const double ms = now_ms() - t0;
    const BlockStatsSnap d = diff_stats(s0, snap_stats());
    if (!write_limbs(po, c)) return 1;
    std::printf("product: %zu limbs, %.3f ms, pairs %llu, pcie %.1f MiB, "
                "sha256 %s\n",
                c.size(), ms, (unsigned long long)d.pairs,
                (double)d.pcie / 1048576.0, hash_limbs(c).c_str());
    std::printf("phase: fwd %.1f inv %.1f crt %.1f carry %.1f xfer %.1f ms\n",
                d.fwd, d.inv, d.crt, d.carry, d.xfer);
    return 0;
}

int cmd_selftest(int argc, char** argv) {
    int iters = 100000;
    int B = 9;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    Rng rng(0x5eed1234u);
    int failures = 0;

    // Phase 1: many single-block products (the atomic block convolution).
    {
        const size_t block = (size_t)1 << B;
        const double t0 = now_ms();
        for (int it = 0; it < iters; ++it) {
            const size_t la = rng.range(1, block);
            const size_t lb = rng.range(1, block);
            std::vector<u32> a(la), b(lb);
            for (auto& v : a) v = rng.limb();
            for (auto& v : b) v = rng.limb();
            if (it % 3 == 0) a[la - 1] = 0xffffffffu;
            if (it % 5 == 0) b[lb - 1] = 0xffffffffu;
            trim(a);
            trim(b);
            uint32_t* pc = nullptr;
            size_t lc = 0;
            int st = block27_mul(a.data(), a.size(), b.data(), b.size(), B, &pc,
                                 &lc);
            if (st != 0) {
                ++failures;
                std::fprintf(stderr, "mul status %d it=%d\n", st, it);
                break;
            }
            std::vector<u32> got(pc, pc + lc), want = ref_mul(a, b);
            std::free(pc);
            trim(got);
            if (got != want) {
                ++failures;
                std::fprintf(stderr, "single-block mismatch it=%d (%zu vs %zu)\n",
                             it, got.size(), want.size());
                if (failures > 5) break;
            }
        }
        std::printf("phase1: %d single-block products in %.1f ms, failures %d\n",
                    iters, now_ms() - t0, failures);
    }

    // Phase 2: multi-block products (diagonal accumulation, overlap-add).
    {
        const int per = std::max(1, iters / 100);
        const double t0 = now_ms();
        int done = 0;
        for (int it = 0; it < per && failures == 0; ++it) {
            const int ka = 2 + (int)(it % 4);
            const int kb = 2 + (int)((it / 4) % 4);
            const size_t block = (size_t)1 << B;
            const size_t la = (size_t)(ka - 1) * block + rng.range(1, block);
            const size_t lb = (size_t)(kb - 1) * block + rng.range(1, block);
            std::vector<u32> a(la), b(lb);
            for (auto& v : a) v = rng.limb();
            for (auto& v : b) v = rng.limb();
            trim(a);
            trim(b);
            uint32_t* pc = nullptr;
            size_t lc = 0;
            if (block27_mul(a.data(), a.size(), b.data(), b.size(), B, &pc, &lc) != 0) {
                ++failures;
                break;
            }
            std::vector<u32> got(pc, pc + lc), want = ref_mul(a, b);
            std::free(pc);
            trim(got);
            if (got != want) {
                ++failures;
                const size_t d = first_diff(got, want);
                std::fprintf(stderr,
                             "multi-block mismatch it=%d ka=%d kb=%d la=%zu lb=%zu "
                             "diff=%zu got=%zu want=%zu (got %08x want %08x)\n",
                             it, ka, kb, la, lb, d, got.size(), want.size(),
                             d < got.size() ? got[d] : 0u,
                             d < want.size() ? want[d] : 0u);
            }
            ++done;
        }
        std::printf("phase2: %d multi-block products in %.1f ms, failures %d\n",
                    done, now_ms() - t0, failures);
    }

    // Phase 3: full step (c = a*t, d = a^2+b^2) against the host reference.
    {
        const int per = std::max(1, iters / 100);
        const double t0 = now_ms();
        int done = 0;
        for (int it = 0; it < per && failures == 0; ++it) {
            const int ka = 2 + (int)(it % 3);
            const size_t block = (size_t)1 << B;
            const size_t la = (size_t)(ka - 1) * block + rng.range(1, block);
            const size_t lb = la;
            std::vector<u32> b(lb);
            for (auto& v : b) v = rng.limb();
            b[lb - 1] &= 0x7fffffffu;
            // a <= b guarantees 2b - a >= 0.
            std::vector<u32> a = b;
            for (size_t i = 0; i < a.size(); ++i)
                if ((rng.next() & 3) == 0) a[i] = rng.limb() >> 1;
            a.back() = std::min(a.back(), b.back());
            trim(a);
            trim(b);
            // Require 2b >= a for a nonnegative t.
            std::vector<u32> t = twice_b_minus_a(a, b);
            uint32_t* pc = nullptr;
            uint32_t* pd = nullptr;
            size_t lc = 0, ld = 0;
            if (block27_step(a.data(), a.size(), b.data(), b.size(), B, 1, &pc,
                             &lc, &pd, &ld) != 0) {
                ++failures;
                break;
            }
            std::vector<u32> c(pc, pc + lc), d(pd, pd + ld);
            std::free(pc);
            std::free(pd);
            trim(c);
            trim(d);
            std::vector<u32> wc = ref_mul(a, t);
            std::vector<u32> aa = ref_mul(a, a);
            std::vector<u32> bb = ref_mul(b, b);
            std::vector<u32> wd = add_limbs(aa, bb);
            const bool okc = (c == wc);
            const bool okd = (d == wd);
            if (!okc || !okd) {
                ++failures;
                const size_t di = first_diff(d, wd);
                std::fprintf(stderr,
                             "step mismatch it=%d ka=%d la=%zu c=%d d=%d "
                             "(c %zu/%zu diff %zu, d %zu/%zu diff %zu)\n",
                             it, ka, la, (int)okc, (int)okd, c.size(), wc.size(),
                             first_diff(c, wc), d.size(), wd.size(), di);
                for (size_t q = di; q < di + 4 && q < d.size(); ++q)
                    std::fprintf(stderr, "   limb %zu got %08x want %08x\n", q,
                                 d[q], q < wd.size() ? wd[q] : 0u);
                // Isolate: square-sum via the generic product path.
                uint32_t* p1 = nullptr;
                uint32_t* p2 = nullptr;
                size_t l1 = 0, l2 = 0;
                block27_mul(a.data(), a.size(), a.data(), a.size(), B, &p1, &l1);
                block27_mul(b.data(), b.size(), b.data(), b.size(), B, &p2, &l2);
                std::vector<u32> ga(p1, p1 + l1), gb(p2, p2 + l2);
                std::free(p1);
                std::free(p2);
                trim(ga);
                trim(gb);
                std::fprintf(stderr, "   generic a*a %s, b*b %s, a*a+b*b %s\n",
                             ga == aa ? "ok" : "BAD", gb == bb ? "ok" : "BAD",
                             add_limbs(ga, gb) == wd ? "ok" : "BAD");
            }
            ++done;
        }
        std::printf("phase3: %d full steps in %.1f ms, failures %d\n", done,
                    now_ms() - t0, failures);
    }

    if (failures == 0) std::printf("BLOCK27 SELFTEST PASSED\n");
    return failures == 0 ? 0 : 1;
}

int cmd_bench(int argc, char** argv) {
    std::string sizes = "1048576,33554432,1024000000,2048000000,4096000000,6.186B";
    int runs = 2;
    int B = 26;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--sizes" && i + 1 < argc) sizes = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    std::vector<u64> list;
    size_t pos = 0;
    while (pos < sizes.size()) {
        size_t comma = sizes.find(',', pos);
        std::string tok = sizes.substr(pos, comma == std::string::npos
                                                ? std::string::npos
                                                : comma - pos);
        u64 mult = 1;
        if (!tok.empty()) {
            const char c = tok.back();
            if (c == 'K' || c == 'k') { mult = 1024ull; tok.pop_back(); }
            else if (c == 'M' || c == 'm') { mult = 1024ull * 1024; tok.pop_back(); }
            else if (c == 'G' || c == 'g') { mult = 1024ull * 1024 * 1024; tok.pop_back(); }
        }
        list.push_back(std::strtoull(tok.c_str(), nullptr, 0) * mult);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    std::printf("| n | F limbs | wall ms (median) | wall ms (min) | "
                "native/block steps | pairs | vram MiB |\n");
    std::printf("|---:|---:|---:|---:|---|---:|---:|\n");
    for (u64 n : list) {
        std::vector<double> times;
        FibRun last;
        for (int r = 0; r < runs + 1; ++r) {
            FibRun rr = run_fib(n, B, false, false);
            if (r == 0) continue; // warm-up
            times.push_back(rr.wall_ms);
            last = std::move(rr);
        }
        std::sort(times.begin(), times.end());
        const double med = times[times.size() / 2];
        std::printf("| %llu | %zu | %.3f | %.3f | %d/%d | %llu | %.0f |\n",
                    (unsigned long long)n, last.value.size(), med, times.front(),
                    last.native_steps, last.block_steps,
                    (unsigned long long)last.bs.pairs,
                    (double)last.vram_used / 1048576.0);
        std::printf("  phases: fwd %.1f inv %.1f crt %.1f carry %.1f xfer %.1f "
                    "ms, pcie %.1f MiB\n",
                    last.bs.fwd, last.bs.inv, last.bs.crt, last.bs.carry,
                    last.bs.xfer, (double)last.bs.pcie / 1048576.0);
        std::fflush(stdout);
    }
    return 0;
}

int cmd_sqtest(int argc, char** argv) {
    int B = 9;
    int blocks = 4;
    bool random = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--B" && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (a == "--blocks" && i + 1 < argc) blocks = std::atoi(argv[++i]);
        else if (a == "--random") random = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const size_t block = (size_t)1 << B;
    const size_t la = (size_t)blocks * block;
    std::vector<u32> a(la, 0xffffffffu), b(la, 0xffffffffu);
    if (random) {
        Rng rng(0x12345678u);
        for (auto& v : a) v = rng.limb();
        for (auto& v : b) v = rng.limb();
        b = a;
    }
    uint32_t* pc = nullptr;
    uint32_t* pd = nullptr;
    size_t lc = 0, ld = 0;
    if (block27_step(a.data(), a.size(), b.data(), b.size(), B, 1, &pc, &lc,
                     &pd, &ld) != 0)
        return 3;
    std::vector<u32> d(pd, pd + ld);
    std::free(pc);
    std::free(pd);
    trim(d);
    std::vector<u32> aa = ref_mul(a, a);
    std::vector<u32> want = add_limbs(aa, aa);
    const size_t diff = first_diff(d, want);
    std::printf("sqtest B=%d blocks=%d len=%zu: d %zu limbs, want %zu, diff %zd\n",
                B, blocks, la, d.size(), want.size(), (ptrdiff_t)diff);
    if (diff != (size_t)-1) {
        int shown = 0;
        for (size_t i = diff; i < d.size() && shown < 8; ++i) {
            if (i >= want.size() || d[i] != want[i]) {
                std::printf("  limb %zu (block %zu off %zu) got %08x want %08x\n",
                            i, i / block, i % block, d[i],
                            i < want.size() ? want[i] : 0u);
                ++shown;
            }
        }
        return 1;
    }
    std::printf("SQT EST OK\n");
    return 0;
}

void usage() {
    static const fibcommon::UsageCommand commands[] = {
        {"fib --n N [--out FILE] [--B LOG2] [--iterations R] "
         "[--stats-json PATH] [--force-block] [--oracle PATH]",
         "exact F(N) on the GPU"},
        {"mul --a A.bin --b B.bin --out C.bin [--B LOG2]",
         "exact product of two limb files"},
        {"selftest [--iters N] [--B LOG2]", "CPU/GPU selftest"},
        {"sqtest [--B LOG2]", "square-path selftest"},
        {"bench --sizes LIST [--runs R] [--B LOG2]", "benchmark"},
        {"info", "engine information"},
    };
    fibcommon::print_usage(
        "block27_fib", "2^27 three-prime 32-bit block engine", commands,
        sizeof(commands) / sizeof(commands[0]),
        "  --verbose  --oracle PATH\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "fib") return cmd_fib(argc, argv);
    if (mode == "mul") return cmd_mul(argc, argv);
    if (mode == "selftest") return cmd_selftest(argc, argv);
    if (mode == "sqtest") return cmd_sqtest(argc, argv);
    if (mode == "bench") return cmd_bench(argc, argv);
    if (mode == "info") {
        size_t used = 0, total = 0;
        block27_vram_info(&used, &total);
        std::printf("vram used %.1f MiB of %.1f MiB\n", (double)used / 1048576.0,
                    (double)total / 1048576.0);
        return 0;
    }
    usage();
    return 2;
}
