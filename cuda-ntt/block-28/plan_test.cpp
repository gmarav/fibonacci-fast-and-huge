// Planner and memory-model table (no CUDA, no GPU allocation).

#include <cinttypes>
#include <cstdio>

#include "plan_model.h"

using namespace blk28plan;

static void row(uint64_t n) {
    size_t la = operand_limb_bound(n);
    Plan p = plan_for(la, la + 1);
    MemEstimate e = estimate(p, la, la + 1);
    std::printf(
        "n=%14" PRIu64 "  la<=%zu  lg=%2d B=2^%d  na=%d  K=%d  "
        "dev=%.2f GiB (arr %.2f + stg %.2f + tw %.3f + ctx %.2f)  "
        "host=%.2f GiB (op %.2f + dl %.2f + seg %.2f + out %.2f)\n",
        n, la, p.lg, p.lg - 1, p.na, p.K, gib(device_total(e)),
        gib(e.device_arrays), gib(e.device_staging), gib(e.device_twiddles),
        gib(e.device_context), gib(host_total(e)), gib(e.host_operands),
        gib(e.host_download), gib(e.host_segwork), gib(e.host_output));
}

int main() {
    const int max_blocks = 16;
    std::printf("=== 2^28 option A memory model ===\n");
    const uint64_t ns[] = {1000000ull,      1024000000ull,  2048000000ull,
                           4096000000ull,  6186556317ull,  61865563170ull,
                           73628010691ull, 90611789663ull, 100000000000ull,
                           110000000000ull, 120000000000ull, 130000000000ull,
                           140000000000ull};
    for (uint64_t n : ns) row(n);

    const size_t device_budget = (size_t)(22.9 * 1024 * 1024 * 1024);
    const size_t host_budget_45gib = (size_t)(45.0 * 1024 * 1024 * 1024);
    const size_t host_budget_49gb = (size_t)49ull * 1000 * 1000 * 1000;
    std::printf("\n=== ceilings ===\n");
    std::printf("device budget %.2f GiB, max_blocks=%d -> n <= %" PRIu64 "\n",
                gib(device_budget), max_blocks,
                max_index_for_device(device_budget, max_blocks));
    std::printf("host budget %.2f GiB -> n <= %" PRIu64 "\n",
                gib(host_budget_45gib),
                max_index_for_host(host_budget_45gib, max_blocks));
    std::printf("host budget %.2f GiB -> n <= %" PRIu64 "\n",
                gib(host_budget_49gb),
                max_index_for_host(host_budget_49gb, max_blocks));
    std::printf("coefficient bound (la+lb)*(2^32-1)^2 < P -> n <= %" PRIu64
                "\n", max_index_for_coefficient_bound());
    return 0;
}
