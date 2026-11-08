// Cold host-side base-2^32 limb arithmetic shared by the engine drivers.
//
// These helpers are deliberately integer-only (no __int128) so the drivers can
// be compiled by nvcc/MSVC with the same code.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "limb_io.h"

namespace fibcommon {

inline Limbs add_limbs(const Limbs& a, const Limbs& b) {
    size_t n = std::max(a.size(), b.size());
    Limbs r(n + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        r[i] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
    r[n] = static_cast<uint32_t>(carry);
    trim(r);
    return r;
}

inline void add_inplace(Limbs& a, const Limbs& b) {
    if (b.size() > a.size()) a.resize(b.size(), 0);
    uint64_t carry = 0;
    size_t i = 0;
    for (; i < a.size(); ++i) {
        uint64_t s = static_cast<uint64_t>(a[i]) + carry +
                     (i < b.size() ? b[i] : 0);
        a[i] = static_cast<uint32_t>(s);
        carry = s >> 32;
        if (carry == 0 && i >= b.size()) break;
    }
    if (carry) a.push_back(static_cast<uint32_t>(carry));
    trim(a);
}

inline Limbs sub_limbs(const Limbs& a, const Limbs& b) {
    Limbs r(a.size(), 0);
    uint64_t borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t av = a[i];
        uint64_t bv = (i < b.size() ? b[i] : 0) + borrow;
        if (av < bv) {
            r[i] = static_cast<uint32_t>(av + (uint64_t{1} << 32) - bv);
            borrow = 1;
        } else {
            r[i] = static_cast<uint32_t>(av - bv);
            borrow = 0;
        }
    }
    trim(r);
    return r;
}

inline Limbs shl1_limbs(const Limbs& a) {
    Limbs r(a.size() + 1, 0);
    uint32_t carry = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = (a[i] << 1) | carry;
        carry = a[i] >> 31;
    }
    r[a.size()] = carry;
    trim(r);
    return r;
}

inline Limbs mul_small(const Limbs& a, const Limbs& b) {
    if (a.empty() || b.empty()) return {};
    Limbs r(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < b.size(); ++j) {
            uint64_t cur =
                static_cast<uint64_t>(a[i]) * b[j] + r[i + j] + carry;
            r[i + j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        size_t p = i + b.size();
        while (carry) {
            uint64_t cur = static_cast<uint64_t>(r[p]) + carry;
            r[p] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
            ++p;
        }
    }
    trim(r);
    return r;
}

}  // namespace fibcommon
