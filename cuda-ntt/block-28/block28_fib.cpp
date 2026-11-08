// Driver for the 2^28, two-prime 48-bit option-A block engine.
//
// Schedule (host side):
//   * exact fast doubling (a,b) -> (c,d), c = a*(2b-a), d = a^2+b^2;
//   * each product is a schoolbook block convolution with B = 2^27 limbs and
//     transform length 2^28 (or the smallest power of two that fits);
//   * the two 48-bit primes are processed per output segment: for block sum k,
//     prime 0 computes C_k, then prime 1 computes C_k, and the segment is
//     immediately CRT'd and carried into base-2^32 limbs.  This keeps the host
//     residue storage at two segments instead of the whole product;
//   * the device keeps only three transform buffers (A, B, C) plus the two
//     operand staging buffers (a, b).  Each block pair re-transforms both
//     blocks, which is the price of option A's memory bound;
//   * results are downloaded block-by-block; the full result is never resident
//     on the device.
//
// No floating point is used on the arithmetic path.

#include "block28.h"
#include "block28_constants.h"
#include "plan_model.h"
#include "segment_norm.h"
#include "../common/limb_io.h"
#include "../common/limb_math.h"
#include "../common/memory.h"
#include "../common/report.h"
#include "../common/run_options.h"
#include "../common/stats_json.h"
#include "../common/usage.h"

using namespace fibcommon;

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint64_t u64;

static int g_threads = 16;
static bool g_verbose = false;
static int g_force_bl = 0;

// ---------------------------------------------------------------------------
// Limb helpers
// ---------------------------------------------------------------------------

// GPU block multiplication (device schedule)
// ---------------------------------------------------------------------------

static bool load_ok(const char* what) {
    std::fprintf(stderr, "%s: %s\n", what, be_error());
    return false;
}

// Exact product of x and y.  `self` means x == y and uses symmetric block
// pairs (weight 2 for i != j).
static bool mul_gpu(const std::vector<u32>& x, const std::vector<u32>& y,
                    bool self, std::vector<u32>& out, const char* tag) {
    out.clear();
    if (x.empty() || y.empty()) return true;
    if (x.size() + y.size() <= 4096) {
        out = mul_small(x, y);
        return true;
    }

    blk28plan::Plan p = blk28plan::plan_for(x.size(), y.size());
    if (g_force_bl > 0) {
        p.B = (size_t)1 << g_force_bl;
        p.lg = g_force_bl + 1;
        p.N = (size_t)1 << p.lg;
        p.block_mode = (p.lg == BLK28_MAX_LOG2);
        p.need = x.size() + y.size();
        p.na = (int)((x.size() + p.B - 1) / p.B);
        p.nb = (int)((y.size() + p.B - 1) / p.B);
        p.K = p.na + p.nb - 1;
    }
    if (!blk28plan::plan_supported(p, 16)) {
        std::fprintf(stderr, "%s: product needs %d+%d blocks (max 16)\n", tag,
                     p.na, p.nb);
        return false;
    }
    if (be_init(p.lg) != 0) return load_ok("be_init");

    const size_t B = p.B;
    const int lg = p.lg;
    const int na = p.na;
    const int nb = self ? na : p.nb;
    const int K = na + nb - 1;
    const size_t need = p.need;
    const int Kneed = (int)((need + B - 1) / B);

    const size_t blocks = (size_t)std::max(na, nb);
    if (be_reserve(blocks * B) != 0) return load_ok("be_reserve");
    int mk = be_gpu_mark_begin();
    if (be_upload(0, x.data(), x.size()) != 0) return load_ok("be_upload a");
    if (!self && be_upload(1, y.data(), y.size()) != 0)
        return load_ok("be_upload b");
    be_gpu_mark_end(mk);

    std::vector<u64> low[2] = {std::vector<u64>(B, 0), std::vector<u64>(B, 0)};
    std::vector<u64> prev[2] = {std::vector<u64>(B, 0), std::vector<u64>(B, 0)};
    std::vector<u64> d64(B, 0);
    out.assign(need, 0);

    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;
    const u64 ninv[2] = {be_ninv(0, lg), be_ninv(1, lg)};

    if (g_verbose) {
        std::fprintf(stderr,
                     "%s: lg=%d B=2^%d na=%d nb=%d K=%d Kneed=%d need=%zu "
                     "self=%d\n",
                     tag, lg, p.lg - 1, na, nb, K, Kneed, need, (int)self);
    }

    for (int k = 0; k < Kneed; ++k) {
        const bool has_sum = (k < K);
        if (has_sum) {
            for (int pi = 0; pi < 2; ++pi) {
                int mk_pair = be_gpu_mark_begin();
                bool first = true;
                if (!self) {
                    const int ilo = std::max(0, k - (nb - 1));
                    const int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        const int j = k - i;
                        if (j < 0 || j >= nb) continue;
                        const size_t len_i =
                            std::min(B, x.size() - (size_t)i * B);
                        const size_t len_j =
                            std::min(B, y.size() - (size_t)j * B);
                        if (be_fwd_block(0, pi, 0, (size_t)i * B, len_i, lg) ||
                            be_fwd_block(1, pi, 1, (size_t)j * B, len_j, lg) ||
                            be_pointwise_acc(2, 0, 1, pi, lg, ninv[pi],
                                             first ? 0 : 1))
                            return load_ok("block pair");
                        first = false;
                    }
                } else {
                    const int ilo = std::max(0, k - (na - 1));
                    const int ihi = k / 2;
                    for (int i = ilo; i <= ihi; ++i) {
                        const int j = k - i;
                        if (j >= na) continue;
                        const size_t len_i =
                            std::min(B, x.size() - (size_t)i * B);
                        const size_t len_j =
                            std::min(B, x.size() - (size_t)j * B);
                        if (be_fwd_block(0, pi, 0, (size_t)i * B, len_i, lg))
                            return load_ok("self pair i");
                        if (i == j) {
                            if (be_pointwise_acc(2, 0, 0, pi, lg, ninv[pi],
                                                 first ? 0 : 1))
                                return load_ok("self pair diag");
                        } else {
                            if (be_fwd_block(1, pi, 0, (size_t)j * B, len_j,
                                             lg) ||
                                be_pointwise_acc(2, 0, 1, pi, lg, ninv[pi],
                                                 (first ? 0 : 1) | 2))
                                return load_ok("self pair offdiag");
                        }
                        first = false;
                    }
                }
                if (be_inv(2, pi, lg) != 0) return load_ok("be_inv");
                if (be_download_range(2, low[pi].data(), 0, B, lg) != 0)
                    return load_ok("download low");
                be_gpu_mark_end(mk_pair);
                if (be_sync() != 0) return load_ok("be_sync");
                // Fold C_k[0..B) + C_{k-1}[B..2B) in the residue domain.  The
                // fold must happen before the high download because the buffer
                // still holds this prime's result.
                segnorm::fold_add(low[pi].data(), prev[pi].data(), B,
                                  pi ? P1 : P0, g_threads);
                if (const char* dd = std::getenv("BLOCK28_DUMP")) {
                    char nm[512];
                    std::snprintf(nm, sizeof nm, "%s/low_k%d_p%d.bin", dd, k, pi);
                    FILE* f = std::fopen(nm, "wb");
                    if (f) {
                        std::fwrite(low[pi].data(), 8, B, f);
                        std::fclose(f);
                    }
                }
                int mk2 = be_gpu_mark_begin();
                if (be_download_range(2, prev[pi].data(), B, B, lg) != 0)
                    return load_ok("download high");
                be_gpu_mark_end(mk2);
            }
        } else {
            if (be_sync() != 0) return load_ok("be_sync high");
            std::swap(low[0], prev[0]);
            std::swap(low[1], prev[1]);
        }

        const size_t off = (size_t)k * B;
        if (off >= need) break;
        const size_t lim = std::min(B, need - off);
        segnorm::process_segment(low[0].data(), low[1].data(), lim, d64.data(),
                                 out.data() + off, carry, sb1, sc1, sc2,
                                 lim == B, P0, P1, BLK28_INV_P0_MOD_P1,
                                 g_threads);
        if (!has_sum) break;
    }

    if (carry != 0) {
        std::fprintf(stderr, "%s: non-zero final carry %u\n", tag, carry);
        return false;
    }
    trim(out);
    return true;
}

// ---------------------------------------------------------------------------
// Fast doubling
// ---------------------------------------------------------------------------

static bool fib_gpu(uint64_t n, std::vector<u32>& result, double* wall_ms) {
    auto t0 = std::chrono::steady_clock::now();
    result.clear();
    if (n == 0) {
        if (wall_ms) *wall_ms = 0.0;
        return true;
    }

    size_t L = blk28plan::operand_limb_bound(n);
    blk28plan::Plan p = blk28plan::plan_for(L, L + 1);
    if (!blk28plan::plan_supported(p, 16)) {
        std::fprintf(stderr, "index needs more than 16 blocks of 2^%d limbs\n",
                     BLK28_BLOCK_LOG2);
        return false;
    }
    if (be_init(p.lg) != 0) return load_ok("be_init");
    const size_t blocks = (size_t)std::max(p.na, p.nb);
    if (be_reserve(blocks * p.B) != 0) return load_ok("be_reserve");
    be_gpu_timer_reset();

    std::vector<u32> a;      // F(0)
    std::vector<u32> b{1};   // F(1)
    int top = 63;
    while (!((n >> top) & 1)) --top;

    for (int bit = top; bit >= 0; --bit) {
        std::vector<u32> t2 = shl1_limbs(b);
        t2 = sub_limbs(t2, a);              // 2b - a
        std::vector<u32> c;
        if (!mul_gpu(a, t2, false, c, "c=a*(2b-a)")) return false;
        t2.clear();
        t2.shrink_to_fit();

        std::vector<u32> d;
        if (!(bit == 0 && ((n >> bit) & 1) == 0)) {
            std::vector<u32> aa, bb;
            if (!mul_gpu(a, a, true, aa, "aa=a*a")) return false;
            if (!mul_gpu(b, b, true, bb, "bb=b*b")) return false;
            d = add_limbs(aa, bb);          // d = a^2 + b^2
            bb.clear();
            bb.shrink_to_fit();
        }

        if (((n >> bit) & 1) == 0) {
            a = std::move(c);
            b = std::move(d);
        } else {
            add_inplace(c, d);              // c + d
            a = std::move(d);
            b = std::move(c);
        }
    }

    result = std::move(a);
    trim(result);
    auto t1 = std::chrono::steady_clock::now();
    if (wall_ms)
        *wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage() {
    static const fibcommon::UsageCommand commands[] = {
        {"plan --n N [--n N ...]", "print schedule and memory model"},
        {"fib --n N [--out FILE] [--iterations R] [--stats-json PATH]",
         "exact F(N) on the GPU"},
        {"mul --a A.bin --b B.bin [--out FILE] [--self] [--force-bl L]",
         "exact product of two limb files"},
        {"bench --sizes a,b,c --runs R [--warmup W]", "benchmark"},
        {"selftest --mode 0|1 --lg L [--outdir DIR]", "engine selftest"},
        {"info", "free VRAM / host RAM"},
    };
    fibcommon::print_usage(
        "block28_fib", "2^28 two-prime 48-bit block engine", commands,
        sizeof(commands) / sizeof(commands[0]),
        "  --threads T  --verbose\n");
}

static bool parse_u64_list(const char* s, std::vector<uint64_t>& out) {
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        std::string part = str.substr(pos, comma == std::string::npos
                                               ? std::string::npos
                                               : comma - pos);
        if (!part.empty())
            out.push_back(std::strtoull(part.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return !out.empty();
}

static void print_plan(uint64_t n) {
    size_t L = blk28plan::operand_limb_bound(n);
    blk28plan::Plan p = blk28plan::plan_for(L, L + 1);
    blk28plan::MemEstimate e = blk28plan::estimate(p, L, L + 1);
    std::printf("n=%llu operand<=%zu limbs product<=%zu lg=%d B=2^%d na=%d "
                "nb=%d K=%d block_mode=%d\n",
                (unsigned long long)n, L, 2 * L, p.lg, p.lg - 1, p.na, p.nb,
                p.K, (int)p.block_mode);
    std::printf("  device arrays %.2f + staging %.2f + tw %.4f + ctx %.2f = "
                "%.2f GiB\n",
                blk28plan::gib(e.device_arrays),
                blk28plan::gib(e.device_staging),
                blk28plan::gib(e.device_twiddles),
                blk28plan::gib(e.device_context),
                blk28plan::gib(blk28plan::device_total(e)));
    std::printf("  host operands %.2f + download %.2f + work %.2f + one "
                "product %.2f = %.2f GiB\n",
                blk28plan::gib(e.host_operands),
                blk28plan::gib(e.host_download),
                blk28plan::gib(e.host_segwork),
                blk28plan::gib(e.host_output),
                blk28plan::gib(blk28plan::host_total(e)));
}

// Refuse to start if the device or host model does not fit with reserve.
static bool preflight(uint64_t n) {
    size_t L = blk28plan::operand_limb_bound(n);
    blk28plan::Plan p = blk28plan::plan_for(L, L + 1);
    blk28plan::MemEstimate e = blk28plan::estimate(p, L, L + 1);
    // cudaMemGetInfo's "free" already excludes the CUDA context, so the
    // preflight compares it against the allocations we are about to make.
    const size_t dev_alloc = e.device_arrays + e.device_staging +
                             e.device_twiddles;
    const size_t dev_model = blk28plan::device_total(e);
    const size_t reserve = (size_t)1 << 30;
    size_t dfree = 0, dtotal = 0;
    if (be_meminfo(&dfree, &dtotal) != 0) return load_ok("meminfo");
    const size_t used_now = dtotal - dfree;
    std::printf("preflight: alloc=%.2f GiB model_total=%.2f GiB "
                "device_now_used=%.2f GiB free=%.2f GiB reserve=1.00 GiB\n",
                blk28plan::gib(dev_alloc), blk28plan::gib(dev_model),
                blk28plan::gib(used_now), blk28plan::gib(dfree));
    if (dfree < dev_alloc + reserve) {
        std::fprintf(stderr,
                     "preflight: device budget exceeded (need %.2f + 1 GiB "
                     "reserve, free %.2f GiB)\n",
                     blk28plan::gib(dev_alloc), blk28plan::gib(dfree));
        return false;
    }
    const size_t host_need = blk28plan::host_total(e);
    const u64 avail = host_avail_bytes();
    std::printf("preflight: host_need=%.2f GiB host_avail=%.2f GiB\n",
                blk28plan::gib(host_need), blk28plan::gib((size_t)avail));
    if (avail && avail < host_need + ((size_t)2 << 30)) {
        std::fprintf(stderr, "preflight: host RAM budget exceeded\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    std::string cmd = argv[1];
    std::vector<uint64_t> ns;
    fibcommon::RunOptions run;
    std::string outdir = "out";
    std::string a_path, b_path;
    bool self_flag = false;
    int runs = 1, warmup = 1, mode = 0, lg_cli = 0;
    int threads = (int)std::thread::hardware_concurrency();
    if (threads <= 0) threads = 8;
    if (threads > 64) threads = 64;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (fibcommon::parse_run_option(a, i, argc, argv, run)) {
            if (run.have_n) {
                ns.push_back(run.n);
                run.have_n = false;
            }
            continue;
        } else if (a == "--sizes") parse_u64_list(next("--sizes"), ns);
        else if (a == "--outdir") outdir = next("--outdir");
        else if (a == "--runs") runs = std::atoi(next("--runs"));
        else if (a == "--warmup") warmup = std::atoi(next("--warmup"));
        else if (a == "--mode") mode = std::atoi(next("--mode"));
        else if (a == "--lg") lg_cli = std::atoi(next("--lg"));
        else if (a == "--threads") threads = std::atoi(next("--threads"));
        else if (a == "--a") a_path = next("--a");
        else if (a == "--b") b_path = next("--b");
        else if (a == "--self") self_flag = true;
        else if (a == "--force-bl") g_force_bl = std::atoi(next("--force-bl"));
        else if (a == "--verbose") g_verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    g_threads = threads < 1 ? 1 : threads;
    const std::string out_path = run.out_path;
    int iterations = run.iterations;
    const std::string stats_path = run.stats_path;
    const std::string progress_path = run.progress_path;

    if (cmd == "plan") {
        if (ns.empty()) ns = {100000000000ull};
        for (uint64_t n : ns) print_plan(n);
        return 0;
    }

    if (cmd == "info") {
        size_t f = 0, t = 0;
        if (be_meminfo(&f, &t) != 0) return load_ok("meminfo");
        std::printf("VRAM total=%.3f MiB used=%.3f MiB free=%.3f MiB\n",
                    t / 1048576.0, (t - f) / 1048576.0, f / 1048576.0);
        std::printf("host available=%.3f MiB\n", host_avail_bytes() / 1048576.0);
        return 0;
    }

    if (cmd == "selftest") {
        size_t f0 = 0, t0 = 0;
        be_meminfo(&f0, &t0);
        auto ts0 = std::chrono::steady_clock::now();
        int rc = be_selftest(outdir.c_str(), mode, lg_cli);
        auto ts1 = std::chrono::steady_clock::now();
        size_t f1 = 0, t1 = 0;
        if (be_meminfo(&f1, &t1) != 0) {
            f1 = f0;
            t1 = t0;
        }
        block28_stats st{};
        be_get_stats(&st);
        double ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        std::printf("selftest: rc=%d wall=%.3f ms cudaMemGetInfo total=%.3f "
                    "MiB used=%.3f MiB engine_arrays=%.3f MiB staging=%.3f "
                    "MiB twiddles=%.3f KiB\n",
                    rc, ms, t1 / 1048576.0, (t1 - f1) / 1048576.0,
                    st.bytes_arrays / 1048576.0, st.bytes_staging / 1048576.0,
                    st.bytes_twiddles / 1024.0);
        if (rc) std::fprintf(stderr, "selftest failed: %s\n", be_error());
        be_shutdown();
        return rc;
    }

    if (cmd == "dumpblock") {
        // Debug: one block pair (i = j = 0) of A and B, full N residues for
        // both primes, written as raw u64 arrays.
        if (a_path.empty() || b_path.empty()) return 2;
        std::vector<u32> a, b;
        if (!read_limbs(a_path, a) || !read_limbs(b_path, b)) return 1;
        int bl = g_force_bl > 0 ? g_force_bl : 10;
        int lg = bl + 1;
        size_t B = (size_t)1 << bl;
        if (be_init(lg) != 0) return load_ok("be_init");
        if (be_reserve(B) != 0) return load_ok("be_reserve");
        if (be_upload(0, a.data(), std::min(B, a.size())) != 0 ||
            be_upload(1, b.data(), std::min(B, b.size())) != 0)
            return load_ok("upload");
        for (int pi = 0; pi < 2; ++pi) {
            if (be_fwd_block(0, pi, 0, 0, std::min(B, a.size()), lg) ||
                be_fwd_block(1, pi, 1, 0, std::min(B, b.size()), lg) ||
                be_pointwise_acc(2, 0, 1, pi, lg, be_ninv(pi, lg), 0) ||
                be_inv(2, pi, lg))
                return load_ok("block");
            std::vector<u64> got((size_t)1 << lg);
            if (be_download_range(2, got.data(), 0, got.size(), lg) ||
                be_sync())
                return load_ok("download");
            std::string path = out_path.empty()
                                   ? (std::string("dump_p") + char('0' + pi) +
                                      ".bin")
                                   : out_path + "_p" + char('0' + pi) + ".bin";
            FILE* f = std::fopen(path.c_str(), "wb");
            if (!f) return 1;
            std::fwrite(got.data(), 8, got.size(), f);
            std::fclose(f);
            std::printf("dumpblock: wrote %s (%zu u64)\n", path.c_str(),
                        got.size());
        }
        be_shutdown();
        return 0;
    }

    if (cmd == "fib") {
        if (ns.empty()) {
            std::fprintf(stderr, "fib needs --n\n");
            return 2;
        }
        uint64_t n = ns[0];
        if (!preflight(n)) return 1;
        if (iterations < 1) iterations = 1;
        write_progress(progress_path, 0, iterations, "starting");
        std::vector<u32> r;
        std::vector<double> samples_ms, stream_ms, call_ms, total_ms;
        double peak_vram_mib = 0.0;
        double peak_host_mib = 0.0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            double ms = 0;
            auto total0 = std::chrono::steady_clock::now();
            if (!fib_gpu(n, r, &ms)) {
                be_shutdown();
                return 1;
            }
            block28_stats st{};
            be_get_stats(&st);
            size_t f = 0, t = 0;
            be_meminfo(&f, &t);
            bool wrote = true;
            if (!out_path.empty()) wrote = write_limbs(out_path, r);
            auto total1 = std::chrono::steady_clock::now();
            double total_run_ms =
                std::chrono::duration<double, std::milli>(total1 - total0).count();
            const double used = (t - f) / 1048576.0;
            const double host_peak = (double)host_peak_mib();
            peak_vram_mib = std::max(peak_vram_mib, used);
            peak_host_mib = std::max(peak_host_mib, host_peak);
            samples_ms.push_back(ms);
            stream_ms.push_back(st.gpu_ms + st.gpu_copy_ms);
            call_ms.push_back(ms);
            total_ms.push_back(total_run_ms);
            fibcommon::print_fib_header(
                "block-28", n, iteration + 1, iterations, r.size(),
                (uint64_t)r.size() * 32);
            fibcommon::print_timing(ms, total_run_ms,
                                    st.gpu_ms + st.gpu_copy_ms);
            fibcommon::print_memory(used, host_peak);
            char extras[256];
            std::snprintf(
                extras, sizeof(extras),
                "h2d_mib=%.1f d2h_mib=%.1f pairs=%llu block_sums=%llu "
                "launches=%llu lg=%d",
                st.h2d_bytes / 1048576.0, st.d2h_bytes / 1048576.0,
                (unsigned long long)st.pairs,
                (unsigned long long)st.block_sums,
                (unsigned long long)st.launches, st.max_lg);
            fibcommon::print_engine(extras);
            char progress_detail[96];
            std::snprintf(progress_detail, sizeof(progress_detail),
                          "%.1f ms · %.1f GiB", ms, used / 1024.0);
            write_progress(progress_path, iteration + 1, iterations,
                           progress_detail);
            if (!wrote) {
                be_shutdown();
                return 1;
            }
        }
        if (!write_stats_json(stats_path, "block-28", n, samples_ms, stream_ms,
                              call_ms, total_ms, peak_vram_mib, peak_host_mib)) {
            be_shutdown();
            return 1;
        }
        be_shutdown();
        return 0;
    }

    if (cmd == "mul") {
        if (a_path.empty() || (!self_flag && b_path.empty())) {
            std::fprintf(stderr, "mul needs --a and --b (or --a --self)\n");
            return 2;
        }
        std::vector<u32> a, b;
        if (!read_limbs(a_path, a)) {
            std::fprintf(stderr, "mul: cannot read %s\n", a_path.c_str());
            return 1;
        }
        if (self_flag) {
            b = a;
        } else if (!read_limbs(b_path, b)) {
            std::fprintf(stderr, "mul: cannot read %s\n", b_path.c_str());
            return 1;
        }
        blk28plan::Plan p = blk28plan::plan_for(a.size(), b.size());
        blk28plan::MemEstimate e = blk28plan::estimate(p, a.size(), b.size());
        size_t dfree = 0, dtotal = 0;
        be_meminfo(&dfree, &dtotal);
        std::printf("mul: la=%zu lb=%zu plan lg=%d na=%d nb=%d device=%.2f "
                    "GiB free=%.2f GiB\n",
                    a.size(), b.size(), p.lg, p.na, p.nb,
                    blk28plan::gib(blk28plan::device_total(e)),
                    blk28plan::gib(dfree));
        be_gpu_timer_reset();
        auto ts0 = std::chrono::steady_clock::now();
        std::vector<u32> prod;
        bool ok = mul_gpu(a, b, self_flag, prod, "mul");
        auto ts1 = std::chrono::steady_clock::now();
        block28_stats st{};
        be_get_stats(&st);
        double ms =
            std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        std::printf(
            "mul la=%zu lb=%zu -> %zu limbs wall=%.3f ms gpu_stream_ms=%.3f "
            "h2d_mib=%.1f d2h_mib=%.1f pairs=%llu block_sums=%llu "
            "launches=%llu host_peak_mib=%zu\n",
            a.size(), b.size(), prod.size(), ms, st.gpu_ms + st.gpu_copy_ms,
            st.h2d_bytes / 1048576.0, st.d2h_bytes / 1048576.0,
            (unsigned long long)st.pairs, (unsigned long long)st.block_sums,
            (unsigned long long)st.launches, host_peak_mib());
        be_shutdown();
        if (!ok) {
            std::fprintf(stderr, "mul failed: %s\n", be_error());
            return 1;
        }
        if (!out_path.empty() && !write_limbs(out_path, prod)) return 1;
        return 0;
    }

    if (cmd == "bench") {
        if (ns.empty())
            ns = {1048576ull, 33554432ull, 1024000000ull, 2048000000ull,
                  4096000000ull, 6186556317ull, 61865563170ull};
        for (uint64_t n : ns) {
            std::vector<u32> r;
            double ms = 0;
            for (int w = 0; w < warmup; ++w) {
                if (!fib_gpu(n, r, &ms)) {
                    be_shutdown();
                    return 1;
                }
            }
            std::vector<double> times, gpu;
            size_t limbs = r.size();
            for (int it = 0; it < runs; ++it) {
                if (!fib_gpu(n, r, &ms)) {
                    be_shutdown();
                    return 1;
                }
                block28_stats s{};
                be_get_stats(&s);
                times.push_back(ms);
                gpu.push_back(s.gpu_ms + s.gpu_copy_ms);
            }
            std::sort(times.begin(), times.end());
            std::sort(gpu.begin(), gpu.end());
            block28_stats st{};
            be_get_stats(&st);
            std::printf("n=%llu limbs=%zu median=%.3f ms min=%.3f ms "
                        "gpu_stream_median=%.3f ms lg=%d arrays=%.2f GiB "
                        "launches=%llu host_peak_mib=%zu\n",
                        (unsigned long long)n, limbs, times[times.size() / 2],
                        times.front(), gpu[gpu.size() / 2], st.max_lg,
                        blk28plan::gib(st.bytes_arrays),
                        (unsigned long long)st.launches, host_peak_mib());
        }
        be_shutdown();
        return 0;
    }

    usage();
    return 2;
}