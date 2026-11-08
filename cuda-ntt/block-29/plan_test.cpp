// Host-only planner check.  No CUDA, no GPU access.
//
//   g++ -O2 -std=c++17 plan_test.cpp -o build\plan_test.exe
//   .\build\plan_test.exe 61865563170

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "plan_model.h"

using namespace blkplan;

static void show(uint64_t n) {
    size_t L = operand_limb_bound(n);
    size_t product = 2 * L;
    Plan p = plan_for(L, L);
    MemEstimate e = estimate(p, L, L);
    std::printf(
        "n=%llu  operand<=%zu limbs (%.2f GiB each)  product<=%zu  "
        "lg=%d B=2^%d na=%d nb=%d K=%d block=%d\n",
        (unsigned long long)n, L, L * 4.0 / (1 << 30), product, p.lg,
        (int)(p.B ? __builtin_ctzll((unsigned long long)p.B) : 0), p.na, p.nb,
        p.K, (int)p.block_mode);
    std::printf(
        "    device: arrays %.2f GiB + stage %.2f + tw %.3f + ctx %.2f = "
        "%.2f GiB (GPU 23.89 GiB)\n",
        gib(e.device_arrays), gib(e.device_stage), gib(e.device_twiddles),
        gib(e.device_context),
        gib(e.device_arrays + e.device_stage + e.device_twiddles +
            e.device_context));
    std::printf(
        "    host:   operands %.2f + seg0 %.2f + working %.2f + one product "
        "%.2f = %.2f GiB\n",
        gib(e.host_operands), gib(e.host_seg0), gib(e.host_working),
        gib(e.host_output),
        gib(e.host_operands + e.host_seg0 + e.host_working + e.host_output));
    std::printf("    supported: %s (max %d blocks)\n",
                plan_supported(p) ? "yes" : "NO", 6);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) show(std::strtoull(argv[i], nullptr, 10));
        return 0;
    }
    const uint64_t sizes[] = {
        1048576ull,
        33554432ull,
        1024000000ull,
        2048000000ull,
        4096000000ull,
        6186556317ull,
        12373112634ull,
        20000000000ull,
        40000000000ull,
        61865563170ull,
    };
    for (uint64_t n : sizes) show(n);
    return 0;
}
