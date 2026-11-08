// Canonical limb-file I/O and small helpers shared by all CUDA engines.
//
// Format: u64 limb_count (little-endian) followed by limb_count little-endian
// u32 limbs. Zero is a zero-length value; trailing zero limbs are not stored.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fibcommon {

using Limbs = std::vector<uint32_t>;

inline void trim(Limbs& value) {
    while (!value.empty() && value.back() == 0) value.pop_back();
}

inline bool read_limbs(const std::string& path, Limbs& value) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    uint64_t count = 0;
    if (std::fread(&count, 8, 1, file) != 1) {
        std::fclose(file);
        return false;
    }
    value.resize(static_cast<size_t>(count));
    if (count && std::fread(value.data(), 4, static_cast<size_t>(count), file) !=
                     static_cast<size_t>(count)) {
        std::fclose(file);
        return false;
    }
    std::fclose(file);
    trim(value);
    return true;
}

inline bool write_limbs(const std::string& path, const Limbs& value) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const uint64_t count = value.size();
    std::fwrite(&count, 8, 1, file);
    if (count) std::fwrite(value.data(), 4, static_cast<size_t>(count), file);
    const bool ok = std::ferror(file) == 0;
    std::fclose(file);
    return ok;
}

inline uint64_t limb_bits(const Limbs& value) {
    size_t count = value.size();
    while (count && value[count - 1] == 0) --count;
    if (!count) return 0;
    const uint32_t top = value[count - 1];
    uint64_t bits = 0;
    uint32_t probe = top;
    while (probe) {
        ++bits;
        probe >>= 1;
    }
    return static_cast<uint64_t>(count - 1) * 32 + bits;
}

/// First differing limb, or `SIZE_MAX` when the values are identical.
inline size_t first_diff(const Limbs& a, const Limbs& b) {
    const size_t common = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < common; ++i) {
        if (a[i] != b[i]) return i;
    }
    if (a.size() != b.size()) return common;
    return static_cast<size_t>(-1);
}

/// Backward-compatible name used by the resident driver.
inline size_t first_difference(const Limbs& a, const Limbs& b) {
    return first_diff(a, b);
}

}  // namespace fibcommon
