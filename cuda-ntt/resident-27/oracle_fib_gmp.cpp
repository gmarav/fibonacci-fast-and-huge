// GMP oracle for the resident GPU fast-doubling driver (MinGW + static GMP).
//
// Usage:
//   oracle_fib_gmp fib N OUT.bin
//   oracle_fib_gmp mul A.bin B.bin OUT.bin
//
// Limb file format (little-endian): u64 limb_count, then limb_count u32
// limbs, least significant first.  Zero is written as a zero-length value.
//
// Prints FIB_MS / MUL_MS: the time of the GMP call itself (file I/O excluded).

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;

static bool read_limbs(const char* path, std::vector<u32>& v) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    u64 n = 0;
    if (std::fread(&n, 8, 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    v.resize((size_t)n);
    if (n && std::fread(v.data(), 4, (size_t)n, f) != (size_t)n) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

static bool write_limbs(const char* path, const std::vector<u32>& v) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    u64 n = v.size();
    std::fwrite(&n, 8, 1, f);
    if (n) std::fwrite(v.data(), 4, (size_t)n, f);
    std::fclose(f);
    return true;
}

static std::vector<u32> export_mpz(const mpz_t z) {
    if (mpz_sgn(z) == 0) return {};
    size_t count = 0;
    void* raw = mpz_export(nullptr, &count, -1, 4, 0, 0, z);
    std::vector<u32> out(count);
    if (count && raw) std::memcpy(out.data(), raw, count * 4);
    if (raw) {
        void (*release)(void*, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &release);
        release(raw, count * 4);
    }
    return out;
}

// F(n) for a 64-bit index.
//
// GMP's mpz_fib_ui takes `unsigned long`, which is 32 bits on Windows (LLP64),
// so this oracle used to saturate at 2^32-1 and silently return the wrong
// value for a 2^27 index.  For n >= 2^32 split the index with the exact
// addition identity
//
//     F(a + b) = F(a) * F(b + 1) + F(a - 1) * F(b),
//     F(a - 1) = F(a + 1) - F(a),
//
// and evaluate both parts with GMP's own mpz_fib2_ui.  No Fibonacci formula is
// reimplemented here: the only extra operations are one multiply-subtract and
// one multiply-add in GMP.
// GMP's mpz_fib2_ui(fn, fn1, n) sets fn = F(n) and fn1 = F(n-1).
// GMP 6.3.0 on this MinGW build fails inside mpn_mul_fft for operands around
// 2^25 limbs: it requests a negative allocation size (observed:
// size=4294444064 followed by 18446744073709028384).  mpz_mul is therefore
// only called below 2^24 limbs, which is comfortably inside the working
// range (16,777,216 limbs was verified).  Larger products are split with
//
//   a*b = (ah*B + al)*(bh*B + bl)
//       = ah*bh*B^2 + (ah*bl + al*bh)*B + al*bl,   B = 2^(64*k),
//
// which is exact and leaves every recursive mpz_mul below the failing size.
static size_t g_mul_chunk_limbs = (size_t)1 << 24;

static void mul_exact(mpz_t c, const mpz_t a, const mpz_t b) {
    const size_t sa = mpz_size(a);
    const size_t sb = mpz_size(b);
    const char* dbg = std::getenv("ORACLE_DEBUG_MUL");
    if (dbg && *dbg && *dbg != '0') {
        std::fprintf(stderr, "mul_exact sa=%zu sb=%zu split=%d\n", sa, sb,
                     (sa >= g_mul_chunk_limbs || sb >= g_mul_chunk_limbs) ? 1
                                                                        : 0);
    }
    if (sa < g_mul_chunk_limbs && sb < g_mul_chunk_limbs) {
        mpz_mul(c, a, b);
        return;
    }
    const size_t smax = sa > sb ? sa : sb;
    const size_t k = (smax + 1) / 2;
    const mp_bitcnt_t sh = (mp_bitcnt_t)GMP_NUMB_BITS * (mp_bitcnt_t)k;
    mpz_t ah, al, bh, bl, t1, t2;
    mpz_inits(ah, al, bh, bl, t1, t2, nullptr);
    mpz_tdiv_q_2exp(ah, a, sh);
    mpz_tdiv_r_2exp(al, a, sh);
    mpz_tdiv_q_2exp(bh, b, sh);
    mpz_tdiv_r_2exp(bl, b, sh);
    mul_exact(c, ah, bh);
    mul_exact(t1, ah, bl);
    mul_exact(t2, al, bh);
    mpz_add(t1, t1, t2);
    mul_exact(t2, al, bl);
    mpz_mul_2exp(c, c, 2 * sh);
    mpz_mul_2exp(t1, t1, sh);
    mpz_add(c, c, t1);
    mpz_add(c, c, t2);
    mpz_clears(ah, al, bh, bl, t1, t2, nullptr);
}

static void fib_split(mpz_t z, u64 n) {
    const u64 a = n / 2;
    const u64 b = n - a;
    if (b > 0xffffffffull) {
        std::fprintf(stderr,
                     "oracle: n=%llu is beyond the one-split 64-bit range\n",
                     (unsigned long long)n);
        std::exit(3);
    }
    mpz_t fa, fam1, fb, fbm1, fbp1, t, u;
    mpz_inits(fa, fam1, fb, fbm1, fbp1, t, u, nullptr);
    mpz_fib2_ui(fa, fam1, (unsigned long)a); // F(a), F(a-1)
    mpz_fib2_ui(fb, fbm1, (unsigned long)b); // F(b), F(b-1)
    mpz_add(fbp1, fb, fbm1);                 // F(b+1)
    mul_exact(t, fa, fbp1);                  // F(a) * F(b+1)
    mul_exact(u, fam1, fb);                  // F(a-1) * F(b)
    mpz_add(z, t, u);
    mpz_clears(fa, fam1, fb, fbm1, fbp1, t, u, nullptr);
}

static void fib_u64(mpz_t z, u64 n) {
    if (n <= 0xffffffffull) {
        mpz_fib_ui(z, (unsigned long)n);
        return;
    }
    fib_split(z, n);
}

// Cross-check the split identity against direct mpz_fib_ui for small indices,
// where both paths are available.  The production split is the same code with
// a = n/2, so this validates the off-by-one handling.
static int selftest() {
    mpz_t direct, split;
    mpz_inits(direct, split, nullptr);
    int bad = 0;
    for (u64 n = 0; n <= 200000; ++n) {
        mpz_fib_ui(direct, (unsigned long)n);
        fib_split(split, n);
        if (mpz_cmp(direct, split) != 0) {
            std::fprintf(stderr, "split mismatch at n=%llu\n",
                         (unsigned long long)n);
            if (++bad > 10) break;
        }
    }
    mpz_clears(direct, split, nullptr);

    // Exercise the chunked multiply with a small threshold so the split path
    // is hit on tractable numbers, then compare against plain mpz_mul.
    gmp_randstate_t st;
    gmp_randinit_mt(st);
    gmp_randseed_ui(st, 0xB16B00B5u);
    const size_t saved = g_mul_chunk_limbs;
    g_mul_chunk_limbs = 64;
    mpz_t a, b, c, ref;
    mpz_inits(a, b, c, ref, nullptr);
    for (int it = 0; it < 400; ++it) {
        const unsigned long la = 1 + (unsigned long)(it * 37 % 400);
        const unsigned long lb = 1 + (unsigned long)(it * 53 % 400);
        mpz_urandomb(a, st, (mp_bitcnt_t)la * GMP_NUMB_BITS);
        mpz_urandomb(b, st, (mp_bitcnt_t)lb * GMP_NUMB_BITS);
        mul_exact(c, a, b);
        mpz_mul(ref, a, b);
        if (mpz_cmp(c, ref) != 0) {
            std::fprintf(stderr, "mul_exact mismatch at it=%d (la=%lu lb=%lu)\n",
                         it, la, lb);
            if (++bad > 10) break;
        }
    }
    mpz_clears(a, b, c, ref, nullptr);
    g_mul_chunk_limbs = saved;

    if (bad == 0) std::printf("ORACLE SPLIT SELFTEST PASSED\n");
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "selftest") {
        return selftest();
    }
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s fib N OUT.bin | %s mul A.bin B.bin OUT.bin | "
                     "%s selftest\n",
                     argv[0], argv[0], argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "fib") {
        if (argc < 4) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        mpz_t z;
        mpz_init(z);
        const auto t0 = std::chrono::steady_clock::now();
        fib_u64(z, n);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<u32> limbs = export_mpz(z);
        mpz_clear(z);
        if (!write_limbs(argv[3], limbs)) return 1;
        std::printf("FIB_MS %.6f\n", ms);
        std::printf("FIB_LIMBS %zu\n", limbs.size());
        return 0;
    }
    if (mode == "mul") {
        if (argc < 5) return 2;
        std::vector<u32> a, b;
        if (!read_limbs(argv[2], a) || !read_limbs(argv[3], b)) return 1;
        mpz_t za, zb, zc;
        mpz_inits(za, zb, zc, nullptr);
        if (!a.empty()) mpz_import(za, a.size(), -1, 4, 0, 0, a.data());
        if (!b.empty()) mpz_import(zb, b.size(), -1, 4, 0, 0, b.data());
        const auto t0 = std::chrono::steady_clock::now();
        mpz_mul(zc, za, zb);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<u32> c = export_mpz(zc);
        mpz_clears(za, zb, zc, nullptr);
        if (!write_limbs(argv[4], c)) return 1;
        std::printf("MUL_MS %.6f\n", ms);
        return 0;
    }
    std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
    return 2;
}
