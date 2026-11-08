// Host memory helpers shared by the engine drivers.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace fibcommon {

#if defined(_WIN32)
inline size_t host_peak_mib() {
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<size_t>(counters.PeakWorkingSetSize / (1024 * 1024));
    }
    return 0;
}

inline uint64_t host_avail_bytes() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0;
}
#else
inline size_t host_peak_mib() { return 0; }
inline uint64_t host_avail_bytes() { return 0; }
#endif

}  // namespace fibcommon
