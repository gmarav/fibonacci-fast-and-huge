// Shared JSON statistics for the CUDA engine CLIs.
//
// Every engine runs its own iteration loop inside one process. The Rust CLI
// starts the engine once, reads this file, and never spawns a process per
// iteration.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

inline double stats_number(double value) {
    return (value == value) ? value : 0.0;
}

inline void stats_array(FILE* file, const char* name,
                        const std::vector<double>& values) {
    std::fprintf(file, "\"%s\":[", name);
    for (size_t i = 0; i < values.size(); ++i) {
        std::fprintf(file, "%s%.6f", i ? "," : "", stats_number(values[i]));
    }
    std::fputc(']', file);
}

inline bool write_stats_json(const std::string& path, const char* engine,
                             unsigned long long n,
                             const std::vector<double>& samples_ms,
                             const std::vector<double>& stream_ms,
                             const std::vector<double>& call_ms,
                             const std::vector<double>& total_ms,
                             double peak_vram_mib, double host_peak_mib) {
    if (path.empty()) return true;
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    std::fprintf(file,
                 "{\"engine\":\"%s\",\"n\":%llu,\"iterations\":%zu,",
                 engine, n, samples_ms.size());
    stats_array(file, "samples_ms", samples_ms);
    std::fputc(',', file);
    stats_array(file, "stream_ms", stream_ms);
    std::fputc(',', file);
    stats_array(file, "call_ms", call_ms);
    std::fputc(',', file);
    stats_array(file, "total_ms", total_ms);
    std::fprintf(file,
                 ",\"peak_vram_mib\":%.6f,\"host_peak_mib\":%.6f}\n",
                 stats_number(peak_vram_mib), stats_number(host_peak_mib));
    const bool ok = std::ferror(file) == 0;
    std::fclose(file);
    return ok;
}
