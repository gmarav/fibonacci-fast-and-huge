// Monotonic wall-clock helper shared by all engine drivers.
#pragma once

#include <chrono>

namespace fibcommon {

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace fibcommon
