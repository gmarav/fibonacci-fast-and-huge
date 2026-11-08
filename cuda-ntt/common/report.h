// Common human-readable report lines for the CUDA engine binaries.
//
// The machine-readable report is `--stats-json`; these helpers only keep the
// console output consistent.  Engine-specific counters go into `engine:`.
#pragma once

#include <cstdint>
#include <cstdio>

namespace fibcommon {

inline void print_fib_header(const char* engine, uint64_t n, int iteration,
                             int iterations, size_t limbs, uint64_t bits) {
    std::printf(
        "fib engine=%s n=%llu iter=%d/%d limbs=%zu bits=%llu\n", engine,
        (unsigned long long)n, iteration, iterations, limbs,
        (unsigned long long)bits);
}

inline void print_timing(double call_ms, double total_ms, double stream_ms) {
    std::printf("  timing: call_ms=%.3f total_ms=%.3f stream_ms=%.3f\n",
                call_ms, total_ms, stream_ms);
}

inline void print_memory(double vram_mib, double host_mib) {
    std::printf("  memory: vram_mib=");
    if (vram_mib > 0.0) {
        std::printf("%.1f", vram_mib);
    } else {
        std::printf("-");
    }
    std::printf(" host_mib=");
    if (host_mib > 0.0) {
        std::printf("%.1f", host_mib);
    } else {
        std::printf("-");
    }
    std::putchar('\n');
}

inline void print_engine(const char* extras) {
    if (extras && *extras) std::printf("  engine: %s\n", extras);
}

inline void print_written(const char* path, long long bytes,
                          long long expected) {
    std::printf("  wrote: %s (%lld bytes, expected %lld)\n", path, bytes,
                expected);
}

inline void print_sha256(const char* hash) {
    std::printf("  sha256: %s\n", hash);
}

}  // namespace fibcommon
