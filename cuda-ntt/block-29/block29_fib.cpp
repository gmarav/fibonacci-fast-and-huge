// Driver for the 2^29, two-prime 48-bit block-convolution engine.
//
// This is the host-side implementation of the block schedule described in
// README.md:
//
//   * exact fast doubling (a,b) -> (c,d) with c = a*(2b-a), d = a^2+b^2;
//   * each product is evaluated by block convolution: B = 2^28 limbs,
//     transform length 2^29 (or the smallest power of two that fits when the
//     operands are smaller);
//   * the two 48-bit primes are processed sequentially.  Prime 0's folded
//     segment residues are stored packed in 6 bytes per coefficient while
//     prime 1 is computed; the CRT and base-2^32 carry normalisation happen
//     per segment, keeping host memory bounded;
//   * the overlap-add fold is done on the host, so the device only ever
//     holds the five transform buffers.
//
// No GPU kernel is launched by the `plan` command; `fib`/`bench`/`selftest`
// require a GPU and must only be run after the explicit GPU go-ahead.

#include "block29_engine.h"
#include "block_constants.h"
#include "plan_model.h"
#include "segment_norm.h"
#include "../common/limb_io.h"
#include "../common/limb_math.h"
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

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint64_t u64;

static int g_threads = 16;
static bool g_verbose = false;
static int g_force_bl = 0;   // test override: block log2 (0 = natural plan)

// ---------------------------------------------------------------------------
// Host integer helpers (no __int128: the driver is compiled by nvcc/MSVC)
// ---------------------------------------------------------------------------

static inline int ctz64(u64 v) {
    int n = 0;
    while ((v & 1ull) == 0) {
        v >>= 1;
        ++n;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Limb helpers
// ---------------------------------------------------------------------------

// GPU block multiplication
// ---------------------------------------------------------------------------

static bool load_block(int dst, int prime, const std::vector<u32>& v,
                       size_t block_index, size_t B, int lg) {
    size_t off = block_index * B;
    size_t len = std::min(B, v.size() - off);
    if (be_load(dst, prime, v.data() + off, len, lg) != 0) {
        std::fprintf(stderr, "be_load: %s\n", be_error());
        return false;
    }
    if (be_fwd(dst, prime, lg) != 0) {
        std::fprintf(stderr, "be_fwd: %s\n", be_error());
        return false;
    }
    return true;
}

// Exact product of x and y.  `self` means the two operands are identical and
// the resident transformed blocks can be reused for every block sum.
static bool mul_gpu(const std::vector<u32>& x, const std::vector<u32>& y,
                    bool self, std::vector<u32>& out, const char* tag) {
    out.clear();
    if (x.empty() || y.empty()) return true;
    if (x.size() + y.size() <= 4096) {
        out = mul_small(x, y);
        return true;
    }

    blkplan::Plan p = blkplan::plan_for(x.size(), y.size());
    if (g_force_bl > 0) {
        p.B = (size_t)1 << g_force_bl;
        p.lg = g_force_bl + 1;
        p.N = (size_t)1 << p.lg;
        p.block_mode = (p.lg == 29);
        p.need = x.size() + y.size();
        p.na = (int)((x.size() + p.B - 1) / p.B);
        p.nb = (int)((y.size() + p.B - 1) / p.B);
        p.K = p.na + p.nb - 1;
    }
    if (!blkplan::plan_supported(p)) {
        std::fprintf(stderr, "%s: product needs %d blocks (max 6)\n", tag,
                     p.na + p.nb);
        return false;
    }
    if (be_init(p.lg) != 0) {
        std::fprintf(stderr, "be_init(%d): %s\n", p.lg, be_error());
        return false;
    }
    const size_t B = p.B;
    const size_t need = p.need;
    const int lg = p.lg;
    const int na = p.na;
    const int nb = p.nb;
    const int K = p.K;
    const int Kneed = (int)((need + B - 1) / B);

    std::vector<u32> seg0_lo((size_t)Kneed * B, 0);
    std::vector<u16> seg0_hi((size_t)Kneed * B, 0);
    std::vector<u64> dl(B, 0);
    std::vector<u64> fold_a(B, 0), fold_b(B, 0);
    std::vector<u64> d64(B, 0);
    out.assign(need, 0);

    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;

    if (g_verbose) {
        std::fprintf(stderr,
                     "%s: lg=%d B=2^%d na=%d nb=%d K=%d Kneed=%d need=%zu\n",
                     tag, lg, ctz64((u64)p.B), na, nb, K, Kneed, need);
    }

    for (int prime = 0; prime < 2; ++prime) {
        const u64 pmod = prime ? P1 : P0;
        std::fill(fold_a.begin(), fold_a.end(), 0);

        // Transform the X blocks once per prime; they stay resident.
        int mx = be_gpu_mark_begin();
        for (int i = 0; i < na; ++i) {
            if (!load_block(i, prime, x, (size_t)i, B, lg)) return false;
        }
        be_gpu_mark_end(mx);

        for (int k = 0; k < Kneed; ++k) {
            bool has_sum = (k < K);
            if (has_sum) {
                int mk = be_gpu_mark_begin();
                if (be_zero(4, lg) != 0) return false;
                if (self) {
                    int ilo = std::max(0, k - (na - 1));
                    int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        // Every ordered pair (i, k-i) contributes once.
                        if (be_pointwise(4, i, k - i, prime, lg, 1) != 0) {
                            std::fprintf(stderr, "pointwise: %s\n",
                                         be_error());
                            return false;
                        }
                    }
                } else {
                    int ilo = std::max(0, k - (nb - 1));
                    int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        int j = k - i;
                        if (j < 0 || j >= nb) continue;
                        if (!load_block(3, prime, y, (size_t)j, B, lg))
                            return false;
                        if (be_pointwise(4, i, 3, prime, lg, 1) != 0) {
                            std::fprintf(stderr, "pointwise: %s\n",
                                         be_error());
                            return false;
                        }
                    }
                }
                if (be_inv(4, prime, lg) != 0) {
                    std::fprintf(stderr, "be_inv: %s\n", be_error());
                    return false;
                }
                if (be_download_range(4, dl.data(), 0, B, lg) != 0 ||
                    be_download_range(4, fold_b.data(), B, B, lg) != 0) {
                    std::fprintf(stderr, "download: %s\n", be_error());
                    return false;
                }
                be_gpu_mark_end(mk);
                if (be_sync() != 0) {
                    std::fprintf(stderr, "be_sync: %s\n", be_error());
                    return false;
                }
                segnorm::fold_add(dl.data(), fold_a.data(), dl.size(), pmod, g_threads);
            }

            const std::vector<u64>& seg = has_sum ? dl : fold_a;
            size_t off = (size_t)k * B;
            size_t lim = (off >= need) ? 0 : std::min(B, need - off);
            if (lim == 0) break;

            if (prime == 0) {
                segnorm::pack(seg0_lo.data() + off, seg0_hi.data() + off, seg.data(), lim, pmod, g_threads);
            } else {
                const u32* lo = seg0_lo.data() + off;
                const u16* hi = seg0_hi.data() + off;
                bool full = (lim == B);
                segnorm::process_segment(lo, hi, seg.data(), lim, d64.data(),
                                        out.data() + off, carry, sb1, sc1, sc2,
                                        full, P0, P1, BLK_INV_P0_MOD_P1,
                                        g_threads);
            }
            if (has_sum) std::swap(fold_a, fold_b);
        }
    }

    if (carry != 0) {
        std::fprintf(stderr,
                     "%s: non-zero final carry %u (product longer than %zu)\n",
                     tag, carry, need);
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

    size_t L = blkplan::operand_limb_bound(n);
    blkplan::Plan p = blkplan::plan_for(L, L);
    if (!blkplan::plan_supported(p)) {
        std::fprintf(stderr, "index needs more than 6 blocks of 2^28 limbs\n");
        return false;
    }
    if (be_init(p.lg) != 0) {
        std::fprintf(stderr, "be_init(%d): %s\n", p.lg, be_error());
        return false;
    }
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

        std::vector<u32> aa;
        if (!mul_gpu(a, a, true, aa, "aa=a*a")) return false;
        std::vector<u32> bb;
        if (!mul_gpu(b, b, true, bb, "bb=b*b")) return false;
        add_inplace(aa, bb);                 // d = a^2 + b^2
        bb.clear();
        bb.shrink_to_fit();
        std::vector<u32> d = std::move(aa);

        if (((n >> bit) & 1) == 0) {
            a = std::move(c);
            b = std::move(d);
        } else {
            add_inplace(c, d);               // c + d
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
// File I/O (canonical limb format: u64 count + little-endian u32 limbs)
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
    };
    fibcommon::print_usage(
        "block29_fib", "2^29 two-prime 48-bit block engine", commands,
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
        if (!part.empty()) out.push_back(std::strtoull(part.c_str(), nullptr, 10));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return !out.empty();
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
        if (ns.empty()) ns = {61865563170ull};
        for (uint64_t n : ns) {
            size_t L = blkplan::operand_limb_bound(n);
            blkplan::Plan p = blkplan::plan_for(L, L);
            blkplan::MemEstimate e = blkplan::estimate(p, L, L);
            std::printf(
                "n=%llu operand<=%zu limbs product<=%zu lg=%d block=%d "
                "na=%d nb=%d K=%d\n",
                (unsigned long long)n, L, 2 * L, p.lg, (int)p.block_mode, p.na,
                p.nb, p.K);
            std::printf(
                "  device arrays %.2f + stage %.2f + tw %.3f + ctx %.2f = "
                "%.2f GiB (23.89 GiB total)\n",
                blkplan::gib(e.device_arrays), blkplan::gib(e.device_stage),
                blkplan::gib(e.device_twiddles), blkplan::gib(e.device_context),
                blkplan::gib(e.device_arrays + e.device_stage +
                             e.device_twiddles + e.device_context));
            std::printf(
                "  host operands %.2f + seg0 %.2f + working %.2f + one product "
                "%.2f = %.2f GiB\n",
                blkplan::gib(e.host_operands), blkplan::gib(e.host_seg0),
                blkplan::gib(e.host_working), blkplan::gib(e.host_output),
                blkplan::gib(e.host_operands + e.host_seg0 + e.host_working +
                             e.host_output));
        }
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
        be_stats st{};
        be_get_stats(&st);
        double ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        std::printf(
            "selftest: rc=%d wall=%.3f ms  cudaMemGetInfo: total=%.3f MiB "
            "used_before_context=%.3f MiB used_after=%.3f MiB "
            "engine_delta=%.3f MiB (arrays %.3f + staging %.3f + twiddles "
            "%.3f)\n",
            rc, ms, t1 / 1048576.0, (t0 - f0) / 1048576.0,
            (t1 - f1) / 1048576.0,
            ((t1 - f1) - (t0 - f0)) / 1048576.0,
            st.bytes_arrays / 1048576.0, st.bytes_staging / 1048576.0,
            st.bytes_twiddles / 1048576.0);
        if (rc) std::fprintf(stderr, "selftest failed: %s\n", be_error());
        be_shutdown();
        return rc;
    }

    if (cmd == "fib") {
        if (ns.empty()) {
            std::fprintf(stderr, "fib needs --n\n");
            return 2;
        }
        uint64_t n = ns[0];
        if (iterations < 1) iterations = 1;
        write_progress(progress_path, 0, iterations, "starting");
        std::vector<u32> r;
        std::vector<double> samples_ms, stream_ms, call_ms, total_ms;
        double peak_vram_mib = 0.0;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            double ms = 0;
            auto total0 = std::chrono::steady_clock::now();
            if (!fib_gpu(n, r, &ms)) {
                be_shutdown();
                return 1;
            }
            be_stats st{};
            be_get_stats(&st);
            size_t f = 0, t = 0;
            be_meminfo(&f, &t);
            bool wrote = true;
            if (!out_path.empty()) wrote = write_limbs(out_path, r);
            auto total1 = std::chrono::steady_clock::now();
            double total_run_ms =
                std::chrono::duration<double, std::milli>(total1 - total0).count();
            const double used = (t - f) / 1048576.0;
            peak_vram_mib = std::max(peak_vram_mib, used);
            samples_ms.push_back(ms);
            stream_ms.push_back(st.gpu_ms + st.gpu_copy_ms);
            call_ms.push_back(ms);
            total_ms.push_back(total_run_ms);
            fibcommon::print_fib_header(
                "block-29", n, iteration + 1, iterations, r.size(),
                (uint64_t)r.size() * 32);
            fibcommon::print_timing(ms, total_run_ms,
                                    st.gpu_ms + st.gpu_copy_ms);
            fibcommon::print_memory(used, 0.0);
            char extras[128];
            std::snprintf(extras, sizeof(extras), "launches=%llu lg=%d",
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
        if (!write_stats_json(stats_path, "block-29", n, samples_ms, stream_ms,
                              call_ms, total_ms, peak_vram_mib, 0.0)) {
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
        size_t f0 = 0, t0 = 0;
        be_meminfo(&f0, &t0);
        be_gpu_timer_reset();
        auto ts0 = std::chrono::steady_clock::now();
        std::vector<u32> prod;
        bool ok = mul_gpu(a, b, self_flag, prod, "mul");
        auto ts1 = std::chrono::steady_clock::now();
        size_t f1 = 0, t1 = 0;
        be_meminfo(&f1, &t1);
        be_stats st{};
        be_get_stats(&st);
        double ms =
            std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        std::printf(
            "mul la=%zu lb=%zu -> %zu limbs  wall=%.3f ms  gpu_stream_ms=%.3f  "
            "plan_lg=%d arrays=%.3f MiB  cuda used: context=%.3f MiB "
            "peak=%.3f MiB\n",
            a.size(), b.size(), prod.size(), ms, st.gpu_ms + st.gpu_copy_ms,
            st.max_lg, st.bytes_arrays / 1048576.0, (t0 - f0) / 1048576.0,
            (t1 - f1) / 1048576.0);
        be_shutdown();
        if (!ok) {
            std::fprintf(stderr, "mul failed: %s\n", be_error());
            return 1;
        }
        if (!out_path.empty() && !write_limbs(out_path, prod)) return 1;
        return 0;
    }

    if (cmd == "bench") {
        if (ns.empty()) ns = {1048576ull, 33554432ull, 1024000000ull,
                              2048000000ull, 4096000000ull, 6186556317ull,
                              12373112634ull, 61865563170ull};
        for (uint64_t n : ns) {
            // Warm-up (also builds the engine and all buffers).
            std::vector<u32> r;
            double ms = 0;
            for (int w = 0; w < warmup; ++w) {
                if (!fib_gpu(n, r, &ms)) {
                    be_shutdown();
                    return 1;
                }
            }
            std::vector<double> times, gk, gc;
            size_t limbs = r.size();
            for (int it = 0; it < runs; ++it) {
                if (!fib_gpu(n, r, &ms)) {
                    be_shutdown();
                    return 1;
                }
                be_stats s{};
                be_get_stats(&s);
                times.push_back(ms);
                gk.push_back(s.gpu_ms);
                gc.push_back(s.gpu_copy_ms);
            }
            std::sort(times.begin(), times.end());
            std::sort(gk.begin(), gk.end());
            std::sort(gc.begin(), gc.end());
            be_stats st{};
            be_get_stats(&st);
            std::printf(
                "n=%llu limbs=%zu median=%.3f ms min=%.3f ms "
                "gpu_stream_median=%.3f ms "
                "lg=%d arrays=%.2f GiB launches=%llu\n",
                (unsigned long long)n, limbs, times[times.size() / 2],
                times.front(), gk[gk.size() / 2] + gc[gc.size() / 2], st.max_lg,
                blkplan::gib(st.bytes_arrays),
                (unsigned long long)st.launches);
        }
        be_shutdown();
        return 0;
    }

    usage();
    return 2;
}

