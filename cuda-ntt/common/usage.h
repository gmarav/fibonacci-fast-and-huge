// Common usage layout for the CUDA engine binaries.
#pragma once

#include <cstddef>
#include <cstdio>

namespace fibcommon {

struct UsageCommand {
    const char* synopsis;
    const char* description;
};

inline void print_usage(const char* engine, const char* description,
                        const UsageCommand* commands, size_t count,
                        const char* engine_options) {
    std::printf("%s\n  %s\n\nCommands:\n", engine, description);
    for (size_t i = 0; i < count; ++i) {
        std::printf("  %s\n      %s\n", commands[i].synopsis,
                    commands[i].description);
    }
    std::printf(
        "\nShared options:\n"
        "  --n N  --out FILE  --iterations R  --stats-json PATH\n"
        "  --progress-file PATH\n");
    if (engine_options && *engine_options) {
        std::printf("\nEngine options:\n%s", engine_options);
    }
}

}  // namespace fibcommon
