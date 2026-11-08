// Common `fib` run options shared by all engine drivers.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace fibcommon {

struct RunOptions {
    uint64_t n = 0;
    bool have_n = false;
    std::string out_path;
    int iterations = 1;
    std::string stats_path;
    std::string progress_path;
};

/// Overwrite the small text file the Rust CLI polls for the second progress
/// row. Format: `iteration iterations [detail...]`.
inline void write_progress(const std::string& path, int iteration,
                           int iterations, const char* detail) {
    if (path.empty()) return;
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return;
    std::fprintf(file, "%d %d %s\n", iteration, iterations,
                 detail ? detail : "");
    std::fclose(file);
}

/// Parse one of `--n`, `--out`, `--iterations`/`--repeat`, `--stats-json`.
/// Returns true when the argument (and its value) was consumed.
inline bool parse_run_option(const std::string& arg, int& index, int argc,
                             char** argv, RunOptions& options) {
    auto value = [&]() -> const char* {
        if (index + 1 >= argc) return nullptr;
        return argv[++index];
    };
    if (arg == "--n") {
        const char* text = value();
        if (!text) return false;
        options.n = std::strtoull(text, nullptr, 10);
        options.have_n = true;
        return true;
    }
    if (arg == "--out") {
        const char* text = value();
        if (!text) return false;
        options.out_path = text;
        return true;
    }
    if (arg == "--iterations" || arg == "--repeat") {
        const char* text = value();
        if (!text) return false;
        options.iterations = std::atoi(text);
        return true;
    }
    if (arg == "--stats-json") {
        const char* text = value();
        if (!text) return false;
        options.stats_path = text;
        return true;
    }
    if (arg == "--progress-file") {
        const char* text = value();
        if (!text) return false;
        options.progress_path = text;
        return true;
    }
    return false;
}

}  // namespace fibcommon
