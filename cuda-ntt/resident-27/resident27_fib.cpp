// Host driver for the GPU-resident exact Fibonacci doubling engine built on
// the ntt-alt transform core (cuda-ntt/resident-27).
//
// This program owns no big-integer arithmetic: every product, every limb
// addition and the whole bit loop live inside resident27.cu / .obj.  The
// driver only
//   * parses the CLI and the index list,
//   * calls the C ABI (full resident loop or one fused step at a time),
//   * times the calls and prints transform/launch counts,
//   * compares results limb-for-limb against the GMP oracle executable.
//
// Limb format (identical to cuda-ntt/mul_ntt.cu and cuda-ntt/fib): a u64 limb
// count followed by that many little-endian u32 limbs, least significant
// first.  Zero is a zero-length value.

#include "resident27.h"
#include "../common/limb_io.h"
#include "../common/report.h"
#include "../common/run_options.h"
#include "../common/stats_json.h"
#include "../common/time_util.h"
#include "../common/usage.h"

using namespace fibcommon;

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

typedef uint32_t u32;
typedef uint64_t u64;

namespace {

constexpr u64 kKib = 1024;
constexpr u64 kMib = 1024 * 1024;
constexpr u64 kGib = 1024 * 1024 * 1024;

const u64 kValidationIndices[] = {0, 1, 2, 3, 10, 100, 1000, 10000, 100000};

std::vector<u32> take(uint32_t* limbs, size_t count) {
    std::vector<u32> out;
    if (limbs && count) out.assign(limbs, limbs + count);
    if (limbs) gpu_fib_free_limbs(limbs);
    trim(out);
    return out;
}

// ---------------------------------------------------------------------------
// GMP oracle (external executable, built with MinGW + static libgmp)
// ---------------------------------------------------------------------------

std::string g_oracle;

std::string temp_path(const char* suffix) {
    const char* tmp = std::getenv("TEMP");
    std::string dir = tmp ? tmp : ".";
    char name[64];
    std::snprintf(name, sizeof(name), "\\resident27_%lu_%s",
                  (unsigned long)std::rand(), suffix);
    return dir + name;
}

bool oracle_fib(u64 n, std::vector<u32>& out, double* ms) {
    if (g_oracle.empty()) return false;
    std::string file = temp_path("oracle.bin");
    // cmd /c needs one extra pair of quotes around the whole command when the
    // program path is quoted.
    std::string cmd = std::string("\"\"") + g_oracle + "\" fib " +
                      std::to_string(n) + " \"" + file + "\"\"";
    FILE* p = POPEN(cmd.c_str(), "r");
    if (!p) return false;
    char line[512];
    double t = -1;
    while (std::fgets(line, sizeof(line), p)) {
        double v;
        if (std::sscanf(line, "FIB_MS %lf", &v) == 1) t = v;
    }
    int rc = PCLOSE(p);
    bool ok = (rc == 0) && read_limbs(file, out);
    std::remove(file.c_str());
    if (ok && ms) *ms = t;
    return ok;
}

// ---------------------------------------------------------------------------
// Size parsing
// ---------------------------------------------------------------------------

bool parse_size_token(const std::string& token, u64& value) {
    if (token.empty()) return false;
    u64 mult = 1;
    size_t digits = token.size();
    char suffix = token.back();
    if (suffix == 'k' || suffix == 'K') {
        mult = kKib;
        --digits;
    } else if (suffix == 'm' || suffix == 'M') {
        mult = kMib;
        --digits;
    } else if (suffix == 'g' || suffix == 'G') {
        mult = kGib;
        --digits;
    }
    if (digits == 0) return false;
    u64 parsed = 0;
    for (size_t i = 0; i < digits; ++i) {
        char c = token[i];
        if (c < '0' || c > '9') return false;
        parsed = parsed * 10 + (u64)(c - '0');
    }
    value = parsed * mult;
    return true;
}

std::vector<u64> parse_sizes(const std::string& text) {
    std::vector<u64> out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t comma = text.find(',', start);
        size_t end = comma == std::string::npos ? text.size() : comma;
        std::string token = text.substr(start, end - start);
        u64 v = 0;
        if (!parse_size_token(token, v)) {
            std::fprintf(stderr, "invalid index '%s'\n", token.c_str());
            std::exit(2);
        }
        if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (out.empty()) {
        std::fprintf(stderr, "--sizes needs at least one index\n");
        std::exit(2);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

struct RunResult {
    std::vector<u32> value;
    double wall_ms = 0;
    gpu_fib_stats stats{};
    int status = 0;
};

RunResult run_resident(u64 n) {
    RunResult r;
    uint32_t* limbs = nullptr;
    size_t count = 0;
    const double t0 = now_ms();
    r.status = gpu_fib_doubling(n, &limbs, &count);
    r.wall_ms = now_ms() - t0;
    r.value = take(limbs, count);
    gpu_fib_get_stats(&r.stats);
    return r;
}

// Same math, but the host drives the loop with the single-step ABI: every bit
// uploads (a,b) and downloads (a',b').  This is the "fused but not resident"
// fallback, measured for comparison.
RunResult run_hostloop(u64 n) {
    RunResult r;
    if (n == 0) return r;
    std::vector<u32> a;    // F(0) = 0
    std::vector<u32> b{1}; // F(1) = 1
    gpu_fib_stats acc{};
    const double t0 = now_ms();
    int bits = 0;
    for (u64 v = n; v; v >>= 1) ++bits;
    for (int i = bits - 1; i >= 0; --i) {
        const uint32_t bit = (uint32_t)((n >> i) & 1ull);
        uint32_t* oa = nullptr;
        uint32_t* ob = nullptr;
        size_t la = 0, lb = 0;
        int st = gpu_ntt_doubling_step(a.empty() ? nullptr : a.data(),
                                       a.size(),
                                       b.empty() ? nullptr : b.data(),
                                       b.size(), bit, &oa, &la, &ob, &lb);
        if (st != 0) {
            r.status = st;
            return r;
        }
        a = take(oa, la);
        b = take(ob, lb);
        gpu_fib_stats one{};
        gpu_fib_get_stats(&one);
        acc.steps += one.steps;
        acc.transforms += one.transforms;
        acc.stage_kernels += one.stage_kernels;
        acc.other_kernels += one.other_kernels;
        acc.launches += one.launches;
        acc.gpu_ms += one.gpu_ms;
        if (one.max_ntt_length > acc.max_ntt_length) {
            acc.max_ntt_length = one.max_ntt_length;
        }
        if ((uint64_t)a.size() > acc.peak_limbs) acc.peak_limbs = a.size();
    }
    r.wall_ms = now_ms() - t0;
    r.value = a;
    r.stats = acc;
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() & 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// The RTX 5090 Laptop idles at a low SM clock and boosts under load, so a
// short warm-up leaves the first measured index at low clocks.  Burn roughly
// half a second of the exact workload first; it is excluded from every
// reported time.
void burn_gpu() {
    const double start = now_ms();
    std::printf("warming up (about 0.5 s of F(1M))...\n");
    std::fflush(stdout);
    int calls = 0;
    do {
        RunResult r = run_resident(1ull << 20);
        if (r.status != 0) break;
        ++calls;
    } while (calls < 40 || now_ms() - start < 500.0);
    std::printf("warm-up done: %d calls in %.0f ms\n", calls, now_ms() - start);
    std::fflush(stdout);
}

void print_usage() {
    static const fibcommon::UsageCommand commands[] = {
        {"step --a A.bin --b B.bin --bit 0|1 [--out-a OA.bin] [--out-b OB.bin]",
         "one fused doubling step"},
        {"fib --n N [--out F.bin] [--oracle PATH] [--iterations R] "
         "[--stats-json PATH] [--hostloop]",
         "compute F(N); indices accept K/M/G suffixes"},
        {"validate [--oracle PATH] [--sizes LIST] [--full]",
         "validate against the GMP oracle"},
        {"bench [--sizes LIST] [--runs R] [--oracle PATH] [--csv] [--hostloop]",
         "benchmark"},
    };
    fibcommon::print_usage(
        "resident27_fib", "2^27 GPU-resident fast-doubling engine", commands,
        sizeof(commands) / sizeof(commands[0]),
        "  --oracle PATH  --hostloop  --csv  --full\n");
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

int mode_step(int argc, char** argv) {
    std::string fa, fb, foa, fob;
    int bit = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--a" && i + 1 < argc) fa = argv[++i];
        else if (a == "--b" && i + 1 < argc) fb = argv[++i];
        else if (a == "--bit" && i + 1 < argc) bit = std::atoi(argv[++i]);
        else if (a == "--out-a" && i + 1 < argc) foa = argv[++i];
        else if (a == "--out-b" && i + 1 < argc) fob = argv[++i];
    }
    if (fa.empty() || fb.empty()) {
        print_usage();
        return 2;
    }
    std::vector<u32> A, B;
    if (!read_limbs(fa, A) || !read_limbs(fb, B)) {
        std::fprintf(stderr, "cannot read input limb files\n");
        return 1;
    }
    uint32_t* oa = nullptr;
    uint32_t* ob = nullptr;
    size_t la = 0, lb = 0;
    const double t0 = now_ms();
    int st = gpu_ntt_doubling_step(A.empty() ? nullptr : A.data(), A.size(),
                                   B.empty() ? nullptr : B.data(), B.size(),
                                   (uint32_t)(bit ? 1 : 0), &oa, &la, &ob, &lb);
    const double ms = now_ms() - t0;
    if (st != 0) {
        std::fprintf(stderr, "gpu_ntt_doubling_step failed: %d\n", st);
        return 1;
    }
    std::vector<u32> OA = take(oa, la);
    std::vector<u32> OB = take(ob, lb);
    gpu_fib_stats stats{};
    gpu_fib_get_stats(&stats);
    std::printf("step bit=%d: a' %zu limbs, b' %zu limbs\n", bit, OA.size(),
                OB.size());
    std::printf("wall %.3f ms, device %.3f ms, ntt length %llu, "
                "transforms %llu (stage kernels %llu)\n",
                ms, stats.gpu_ms, (unsigned long long)stats.max_ntt_length,
                (unsigned long long)stats.transforms,
                (unsigned long long)stats.stage_kernels);
    if (!foa.empty() && !write_limbs(foa, OA)) return 1;
    if (!fob.empty() && !write_limbs(fob, OB)) return 1;
    if (foa.empty() && fob.empty()) {
        std::printf("a' limbs (least significant first, hex):\n");
        for (size_t i = 0; i < OA.size(); ++i) {
            std::printf("%08x%c", OA[i], ((i + 1) % 8 == 0) ? '\n' : ' ');
        }
        std::printf("\nb' limbs:\n");
        for (size_t i = 0; i < OB.size(); ++i) {
            std::printf("%08x%c", OB[i], ((i + 1) % 8 == 0) ? '\n' : ' ');
        }
        std::printf("\n");
    }
    return 0;
}

int mode_fib(int argc, char** argv) {
    fibcommon::RunOptions run;
    bool hostloop = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (fibcommon::parse_run_option(a, i, argc, argv, run)) {
            continue;
        } else if (a == "--oracle" && i + 1 < argc) {
            g_oracle = argv[++i];
        } else if (a == "--hostloop") {
            hostloop = true;
        }
    }
    if (!run.have_n) {
        print_usage();
        return 2;
    }
    const u64 n = run.n;
    std::string out_file = run.out_path;
    std::string stats_path = run.stats_path;
    std::string progress_path = run.progress_path;
    int repeat = run.iterations;
    if (repeat < 1) repeat = 1;
    write_progress(progress_path, 0, repeat, "starting");

    // Warm up the CUDA context, cached graphs, and the GPU clock once, outside
    // the samples. This matches the old in-process burn-in.
    if (!hostloop) {
        burn_gpu();
    }

    double best = 0;
    RunResult r;
    std::vector<double> samples_ms;
    std::vector<double> stream_ms;
    std::vector<double> call_ms;
    std::vector<double> total_ms;
    samples_ms.reserve(repeat);
    stream_ms.reserve(repeat);
    call_ms.reserve(repeat);
    total_ms.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        r = hostloop ? run_hostloop(n) : run_resident(n);
        if (r.status != 0) {
            std::fprintf(stderr, "n=%llu failed with status %d\n",
                         (unsigned long long)n, r.status);
            return 1;
        }
        samples_ms.push_back(r.wall_ms);
        stream_ms.push_back(r.stats.gpu_ms);
        call_ms.push_back(r.wall_ms);
        total_ms.push_back(r.wall_ms);
        char detail[64];
        std::snprintf(detail, sizeof(detail), "%.1f ms", r.wall_ms);
        write_progress(progress_path, i + 1, repeat, detail);
        if (i == 0 || r.wall_ms < best) best = r.wall_ms;
    }
    fibcommon::print_fib_header("resident", n, 1, 1, r.value.size(),
                                limb_bits(r.value));
    fibcommon::print_timing(best, best, r.stats.gpu_ms);
    char extras[320];
    std::snprintf(
        extras, sizeof(extras),
        "path=%s best_call_ms=%.3f transforms=%llu stage_kernels=%llu "
        "launches=%llu max_ntt_length=%llu",
        hostloop ? "hostloop" : "resident",
        best, (unsigned long long)r.stats.transforms,
        (unsigned long long)r.stats.stage_kernels,
        (unsigned long long)r.stats.launches,
        (unsigned long long)r.stats.max_ntt_length);
    fibcommon::print_engine(extras);
    if (!out_file.empty() && !write_limbs(out_file, r.value)) return 1;
    if (!write_stats_json(stats_path, "resident", n, samples_ms, stream_ms,
                          call_ms, total_ms, 0.0, 0.0)) {
        return 1;
    }
    if (!g_oracle.empty()) {
        std::vector<u32> expected;
        double oms = 0;
        if (!oracle_fib(n, expected, &oms)) {
            std::fprintf(stderr, "oracle failed\n");
            return 1;
        }
        bool ok = expected == r.value;
        std::printf("GMP oracle: %.3f ms, %s\n", oms,
                    ok ? "MATCH" : "MISMATCH");
        if (!ok) {
            std::printf("first difference at limb %zu\n",
                        first_difference(r.value, expected));
            return 1;
        }
    }
    return 0;
}

int mode_validate(int argc, char** argv) {
    std::vector<u64> sizes;
    bool full = false;
    bool csv = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--oracle" && i + 1 < argc) g_oracle = argv[++i];
        else if (a == "--sizes" && i + 1 < argc) sizes = parse_sizes(argv[++i]);
        else if (a == "--full") full = true;
        else if (a == "--csv") csv = true;
    }
    std::printf("resident27_fib validate\n");
    std::printf("backend: %s\n", gpu_fib_backend_name());
    if (g_oracle.empty()) {
        std::printf("WARNING: no --oracle given, exactness cannot be checked\n");
    }

    std::vector<u64> indices(std::begin(kValidationIndices),
                             std::end(kValidationIndices));
    if (full) {
        for (u64 v : sizes) indices.push_back(v);
    }
    if (indices.empty()) {
        std::printf("nothing to do\n");
        return 0;
    }

    burn_gpu();

    bool all_ok = true;
    if (!csv) {
        std::printf("\n| n | F bits | limbs | resident ms | step-loop ms | "
                    "exact resident | exact step |\n");
        std::printf("| ---: | ---: | ---: | ---: | ---: | :--- | :--- |\n");
    }
    for (u64 n : indices) {
        RunResult r = run_resident(n);
        if (r.status != 0) {
            std::printf("| %llu | FAILED status %d | | | | | |\n",
                        (unsigned long long)n, r.status);
            all_ok = false;
            continue;
        }
        // The same step ABI driven from the host, for an independent check of
        // the fused single-step entry point.
        RunResult s = run_hostloop(n);
        bool ok_res = true;
        bool ok_step = true;
        if (!g_oracle.empty()) {
            std::vector<u32> expected;
            double oms = 0;
            if (!oracle_fib(n, expected, &oms)) {
                std::printf("oracle failed for n=%llu\n",
                            (unsigned long long)n);
                all_ok = false;
                continue;
            }
            ok_res = expected == r.value;
            ok_step = (s.status == 0) && expected == s.value;
            if (!ok_res) {
                std::printf("resident MISMATCH at limb %zu for n=%llu\n",
                            first_difference(r.value, expected),
                            (unsigned long long)n);
            }
            if (!ok_step) {
                std::printf("step ABI MISMATCH at limb %zu for n=%llu\n",
                            s.status == 0
                                ? first_difference(s.value, expected)
                                : (size_t)-1,
                            (unsigned long long)n);
            }
        } else {
            ok_step = (s.status == 0) && s.value == r.value;
        }
        all_ok = all_ok && ok_res && ok_step;
        if (csv) {
            std::printf("validation,%llu,%llu,%zu,%.3f,%.3f,%s,%s\n",
                        (unsigned long long)n,
                        (unsigned long long)limb_bits(r.value),
                        r.value.size(), r.wall_ms, s.wall_ms,
                        ok_res ? "yes" : "no", ok_step ? "yes" : "no");
        } else {
            std::printf("| %llu | %llu | %zu | %.3f | %.3f | %s | %s |\n",
                        (unsigned long long)n,
                        (unsigned long long)limb_bits(r.value),
                        r.value.size(), r.wall_ms, s.wall_ms,
                        ok_res ? "yes" : "no", ok_step ? "yes" : "no");
        }
        std::fflush(stdout);
    }
    std::printf("%s\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}

int mode_bench(int argc, char** argv) {
    std::vector<u64> sizes = {kMib, 8 * kMib, 32 * kMib};
    int runs = 3;
    int warmup = 1;
    bool csv = false;
    bool hostloop = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sizes" && i + 1 < argc) sizes = parse_sizes(argv[++i]);
        else if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--oracle" && i + 1 < argc) g_oracle = argv[++i];
        else if (a == "--csv") csv = true;
        else if (a == "--hostloop") hostloop = true;
    }
    if (runs < 1) runs = 1;

    burn_gpu();

    struct Row {
        u64 n = 0;
        u64 bits = 0;
        size_t limbs = 0;
        double med = 0, best = 0, device = 0, host = 0;
        u64 transforms = 0, stages = 0, max_ntt = 0, launches = 0;
        bool checked = false, exact = true;
        size_t diff = 0;
    };
    std::vector<Row> rows;

    for (u64 n : sizes) {
        for (int i = 0; i < warmup; ++i) {
            RunResult w = hostloop ? run_hostloop(n) : run_resident(n);
            if (w.status != 0) {
                std::fprintf(stderr, "warmup n=%llu failed %d\n",
                             (unsigned long long)n, w.status);
                return 1;
            }
        }
        std::vector<double> samples;
        Row row;
        row.n = n;
        RunResult last;
        for (int i = 0; i < runs; ++i) {
            last = hostloop ? run_hostloop(n) : run_resident(n);
            if (last.status != 0) {
                std::fprintf(stderr, "n=%llu failed %d\n",
                             (unsigned long long)n, last.status);
                return 1;
            }
            samples.push_back(last.wall_ms);
            row.device += last.stats.gpu_ms;
            row.host += last.stats.host_ms;
        }
        row.med = median(samples);
        row.best = *std::min_element(samples.begin(), samples.end());
        row.device /= (double)runs;
        row.host /= (double)runs;
        row.bits = limb_bits(last.value);
        row.limbs = last.value.size();
        row.transforms = last.stats.transforms;
        row.stages = last.stats.stage_kernels;
        row.launches = last.stats.launches;
        row.max_ntt = last.stats.max_ntt_length;
        if (!g_oracle.empty()) {
            std::vector<u32> expected;
            double oms = 0;
            row.checked = true;
            if (!oracle_fib(n, expected, &oms)) {
                std::fprintf(stderr, "oracle failed\n");
                return 1;
            }
            row.exact = expected == last.value;
            if (!row.exact) row.diff = first_difference(last.value, expected);
        }
        rows.push_back(row);
        std::fflush(stdout);
    }

    if (csv) {
        std::printf("kind,n,f_bits,limbs,median_ms,min_ms,device_ms,host_ms,"
                    "transforms,stage_kernels,launches,max_ntt,exact\n");
        for (const Row& r : rows) {
            std::printf("bench,%llu,%llu,%zu,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,"
                        "%llu,%s\n",
                        (unsigned long long)r.n,
                        (unsigned long long)r.bits, r.limbs, r.med, r.best,
                        r.device, r.host,
                        (unsigned long long)r.transforms,
                        (unsigned long long)r.stages,
                        (unsigned long long)r.launches,
                        (unsigned long long)r.max_ntt,
                        r.checked ? (r.exact ? "yes" : "no") : "not-checked");
        }
        return 0;
    }

    std::printf("\nPath: %s\n",
                hostloop ? "host loop over fused steps"
                         : "resident loop (one call)");
    std::printf("| n | F bits | limbs | median ms | min ms | device ms | "
                "host ms | transforms | stage kernels | launches | max NTT | "
                "exact |\n");
    std::printf("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | :--- |\n");
    for (const Row& r : rows) {
        std::printf("| %llu | %llu | %zu | %.3f | %.3f | %.3f | %.3f | %llu | "
                    "%llu | %llu | %llu | %s |\n",
                    (unsigned long long)r.n, (unsigned long long)r.bits,
                    r.limbs, r.med, r.best, r.device, r.host,
                    (unsigned long long)r.transforms,
                    (unsigned long long)r.stages,
                    (unsigned long long)r.launches,
                    (unsigned long long)r.max_ntt,
                    r.checked ? (r.exact ? "yes" : "no") : "not-checked");
        if (r.checked && !r.exact) {
            std::printf("| | | | | | | | | | | first difference at limb | %zu |\n",
                        r.diff);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    std::string mode = argv[1];
    int rc = 2;
    if (mode == "step") rc = mode_step(argc, argv);
    else if (mode == "fib") rc = mode_fib(argc, argv);
    else if (mode == "validate") rc = mode_validate(argc, argv);
    else if (mode == "bench") rc = mode_bench(argc, argv);
    else if (mode == "--help" || mode == "-h") {
        print_usage();
        rc = 0;
    } else {
        print_usage();
    }
    gpu_fib_shutdown();
    return rc;
}
