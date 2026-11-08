// Shared GMP oracle for the block-27 and block-28 engines: arbitrary-size
// exact multiply, exact Fibonacci for 64-bit indices, and an anchored
// Fibonacci chain.
//
//   oracle_gmp fib N OUT.bin
//   oracle_gmp mul A.bin B.bin OUT.bin
//   oracle_gmp fib-anchor N M OUT.bin [--hash-only]
//   oracle_gmp fib-anchor-mpz N M OUT.bin [--hash-only]
//   oracle_gmp anchor-selftest
//   oracle_gmp selftest
//   oracle_gmp bigmultest LIMBS     (limb-array multiply benchmark)
//   oracle_gmp rawmul LIMBS         (plain mpz_mul probe)
//   oracle_gmp bigtest LIMBS        (mpz capacity probe)
//   oracle_gmp multest LIMBS        (chunked multiply probe)
//   oracle_gmp shifttest LIMBS BITS (mpz_mul_2exp probe)
//
// IMPORTANT LIMIT OF THIS GMP BUILD.  gmp.h defines mp_bitcnt_t as
// `unsigned long`, which is 32 bits on Windows LLP64.  This build therefore
// cannot represent or reallocate an mpz wider than 2^32 bits (512 MiB);
// mpz_mul_2exp, mpz_limbs_write and mpn_mul_fft hit a 32-bit size overflow
// above that, and even plain mpz_mul fails for operands of 2^25 limbs
// (4 GiB FFT scratch request).  F(6.19e10) needs 42.95e9 bits, so the chain is
// carried out on plain 64-bit limb arrays with the mpn primitives; only block
// products of at most 33,000,000 limbs (well below the limit) use mpz_mul.
//
// The chunked mpz_mul path below (mul_exact) is used for indices whose values
// fit the limit (up to F(6,186,556,317), which is 4,294,966,696 bits).

#include <gmp.h>
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

struct Sha256 {
    u32 h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    u64 total = 0;
    unsigned char buf[64];
    size_t used = 0;

    static u32 ror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const unsigned char* p) {
        static const u32 K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        u32 w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((u32)p[4 * i] << 24) | ((u32)p[4 * i + 1] << 16) |
                   ((u32)p[4 * i + 2] << 8) | (u32)p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            u32 s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            u32 s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
            g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = hh + S1 + ch + K[i] + w[i];
            u32 S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            u32 mj = (a & b) ^ (a & c) ^ (b & c);
            u32 t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(const void* data, size_t len) {
        const unsigned char* p = (const unsigned char*)data;
        total += len;
        while (len) {
            size_t take = std::min(len, sizeof(buf) - used);
            std::memcpy(buf + used, p, take);
            used += take;
            p += take;
            len -= take;
            if (used == sizeof(buf)) {
                block(buf);
                used = 0;
            }
        }
    }

    std::string hex() {
        unsigned char pad[72];
        size_t padlen = 0;
        u64 bits = total * 8;
        pad[padlen++] = 0x80;
        while ((total + padlen) % 64 != 56) pad[padlen++] = 0;
        for (int i = 7; i >= 0; --i)
            pad[padlen++] = (unsigned char)(bits >> (8 * i));
        update(pad, padlen);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08X", h[i]);
        out[64] = 0;
        return std::string(out);
    }
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static int g_trace = 0;
static int g_depth = 0;
static size_t g_mul_chunk_limbs = (size_t)1 << 24;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static size_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (size_t)pmc.PeakWorkingSetSize;
    return 0;
}

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
    bool ok = std::fwrite(&n, 8, 1, f) == 1;
    if (ok && n) ok = std::fwrite(v.data(), 4, (size_t)n, f) == (size_t)n;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) std::fprintf(stderr, "oracle: write failed: %s\n", path);
    return ok;
}

// Stream the canonical limb file (u64 count + little-endian u32 limbs) from a
// GMP integer without materialising a second multi-gigabyte copy, and hash the
// exact bytes written.
static bool write_mpz_limbs(const char* path, const mpz_t z, bool hash_only,
                            std::string* hash_out, u64* limbs_out) {
    const size_t n64 = mpz_size(z);
    u32 top_hi = 0;
    if (n64) {
        mp_limb_t top = mpz_getlimbn(z, (mp_limb_t)(n64 - 1));
        top_hi = (u32)(top >> 32);
    }
    u64 n32 = n64 ? (u64)n64 * 2 - (top_hi == 0 ? 1 : 0) : 0;
    if (limbs_out) *limbs_out = n32;

    Sha256 sha;
    const u64 hdr = n32;
    sha.update(&hdr, 8);

    FILE* f = nullptr;
    if (!hash_only) {
        f = std::fopen(path, "wb");
        if (!f) {
            std::fprintf(stderr, "oracle: cannot create %s\n", path);
            return false;
        }
        if (std::fwrite(&hdr, 8, 1, f) != 1) {
            std::fclose(f);
            return false;
        }
    }

    constexpr size_t kChunk = 1 << 20; // u32 words per write
    std::vector<u32> buf(kChunk);
    const mp_limb_t* limbs = n64 ? mpz_limbs_read(z) : nullptr;
    u64 written = 0;
    while (written < n32) {
        size_t take = (size_t)std::min<u64>(kChunk, n32 - written);
        for (size_t i = 0; i < take; ++i) {
            const u64 idx = written + i;
            const mp_limb_t limb = limbs[idx >> 1];
            buf[i] = (idx & 1) ? (u32)(limb >> 32) : (u32)limb;
        }
        sha.update(buf.data(), take * 4);
        if (f && std::fwrite(buf.data(), 4, take, f) != take) {
            std::fclose(f);
            std::fprintf(stderr, "oracle: write failed at %llu\n",
                         (unsigned long long)written);
            return false;
        }
        written += take;
    }
    if (f && std::fclose(f) != 0) return false;
    if (hash_out) *hash_out = sha.hex();
    return true;
}

// ---------------------------------------------------------------------------
// Exact multiplication with the GMP workaround
//
// mp_bitcnt_t is 32-bit in this MinGW build, so shifts are applied in <=2^30
// bit chunks.
// ---------------------------------------------------------------------------

static void mul_2exp_big(mpz_t x, u64 bits) {
    while (bits) {
        u64 step = std::min<u64>(bits, (u64)1 << 30);
        mpz_mul_2exp(x, x, (mp_bitcnt_t)step);
        bits -= step;
    }
}

static void q_2exp_big(mpz_t q, const mpz_t a, u64 bits) {
    mpz_set(q, a);
    while (bits) {
        u64 step = std::min<u64>(bits, (u64)1 << 30);
        mpz_tdiv_q_2exp(q, q, (mp_bitcnt_t)step);
        bits -= step;
    }
}

static void mul_exact(mpz_t c, const mpz_t a, const mpz_t b) {
    const size_t sa = mpz_size(a);
    const size_t sb = mpz_size(b);
    if (sa < g_mul_chunk_limbs && sb < g_mul_chunk_limbs) {
        mpz_mul(c, a, b);
        return;
    }
    const bool top = (g_depth == 0);
    ++g_depth;
    if (g_trace && top)
        std::fprintf(stderr, "mul_exact top sa=%zu sb=%zu k=%zu\n", sa, sb,
                     (sa > sb ? sa : sb) / 2);
    const size_t smax = sa > sb ? sa : sb;
    const size_t k = (smax + 1) / 2;
    const u64 sh = (u64)GMP_NUMB_BITS * (u64)k;
    mpz_t ah, al, bh, bl, ap, bp, p1, p2, p3, shl;
    mpz_inits(ah, al, bh, bl, ap, bp, p1, p2, p3, shl, nullptr);
    q_2exp_big(ah, a, sh);
    q_2exp_big(bh, b, sh);
    mpz_set(shl, ah);
    mul_2exp_big(shl, sh);
    mpz_sub(al, a, shl);
    mpz_set(shl, bh);
    mul_2exp_big(shl, sh);
    mpz_sub(bl, b, shl);
    mpz_clear(shl);
    if (g_trace && top) std::fprintf(stderr, "  split done sh=%llu\n",
                                     (unsigned long long)sh);

    // Karatsuba: three recursive products instead of four.
    mul_exact(p1, ah, bh);
    if (g_trace && top) std::fprintf(stderr, "  p1 done\n");
    mul_exact(p3, al, bl);
    if (g_trace && top) std::fprintf(stderr, "  p3 done\n");
    mpz_add(ap, ah, al);
    mpz_add(bp, bh, bl);
    mpz_clears(ah, al, bh, bl, nullptr);
    mul_exact(p2, ap, bp);
    if (g_trace && top) std::fprintf(stderr, "  p2 done\n");
    mpz_clears(ap, bp, nullptr);
    mpz_sub(p2, p2, p1);
    mpz_sub(p2, p2, p3);

    mul_2exp_big(p1, 2 * sh);
    if (g_trace && top) std::fprintf(stderr, "  shift p1 done\n");
    mul_2exp_big(p2, sh);
    if (g_trace && top) std::fprintf(stderr, "  shift p2 done\n");
    mpz_add(p1, p1, p2);
    mpz_clear(p2);
    mpz_add(c, p1, p3);
    if (g_trace && top) std::fprintf(stderr, "  combine done\n");
    mpz_clears(p1, p3, nullptr);
    --g_depth;
}
// F(2k) only (the last anchored doubling does not need F(2k+1)).
static void double_first(mpz_t o, const mpz_t a, const mpz_t b) {
    mpz_t t;
    mpz_init(t);
    mpz_mul_2exp(t, b, 1);
    mpz_sub(t, t, a);
    mul_exact(o, a, t);
    mpz_clear(t);
}

// Square via a distinct second object: GMP's mpn_sqr/FFT path is the one that
// fails on this build, while mpn_mul of two distinct objects works at the same
// sizes.
static void mul_square(mpz_t c, const mpz_t x) {
    mpz_t y;
    mpz_init_set(y, x);
    mul_exact(c, x, y);
    mpz_clear(y);
}

// ---------------------------------------------------------------------------
// Fibonacci
// ---------------------------------------------------------------------------

// (F(n), F(n+1)) by exact fast doubling.
static void fib_pair(mpz_t f, mpz_t f1, u64 n) {
    if (n == 0) {
        mpz_set_ui(f, 0);
        mpz_set_ui(f1, 1);
        return;
    }
    // GMP's internal doubling multiplies numbers of about F(n/2); n <= 2^31
    // keeps every internal operand near 11.6M limbs, below the failing size.
    if (n <= (1ull << 31)) {
        mpz_t g;
        mpz_init(g);
        mpz_fib2_ui(f, g, (unsigned long)n); // f=F(n), g=F(n-1)
        mpz_add(f1, f, g);
        mpz_clear(g);
        return;
    }
    mpz_t a, b, c, d, t;
    mpz_inits(a, b, c, d, t, nullptr);
    fib_pair(a, b, n >> 1);
    if (g_trace)
        std::fprintf(stderr, "  doubling n=%llu a=%zu b=%zu\n",
                     (unsigned long long)n, mpz_size(a), mpz_size(b));
    // c = a*(2b-a) = F(2k), d = a*a + b*b = F(2k+1)
    mpz_mul_2exp(t, b, 1);
    mpz_sub(t, t, a);
    mul_exact(c, a, t);
    mul_square(d, a);
    mul_square(t, b);
    mpz_add(d, d, t);
    if (n & 1ull) {
        mpz_add(t, c, d); // F(2k+2)
        mpz_set(f, d);
        mpz_set(f1, t);
    } else {
        mpz_set(f, c);
        mpz_set(f1, d);
    }
    mpz_clears(a, b, c, d, t, nullptr);
}

static void fib_u64(mpz_t f, u64 n) {
    mpz_t f1;
    mpz_init(f1);
    fib_pair(f, f1, n);
    mpz_clear(f1);
}

// (F(2k), F(2k+1)) from (F(k), F(k+1)).
static void double_pair(mpz_t o0, mpz_t o1, const mpz_t a, const mpz_t b) {
    mpz_t c, d, t;
    mpz_inits(c, d, t, nullptr);
    mpz_mul_2exp(t, b, 1);
    mpz_sub(t, t, a);
    mul_exact(c, a, t);
    mul_square(d, a);
    mul_square(t, b);
    mpz_add(d, d, t);
    mpz_swap(o0, c);
    mpz_swap(o1, d);
    mpz_clears(c, d, t, nullptr);
}

// (F(a+b), F(a+b+1)) from (F(a),F(a+1)) and (F(b),F(b+1)):
//   F(a+b)   = F(a) F(b+1) + (F(a+1) - F(a)) F(b)
//   F(a+b+1) = F(a+1) F(b+1) + F(a) F(b)
static void add_pair(mpz_t o0, mpz_t o1, const mpz_t fa, const mpz_t fa1,
                     const mpz_t fb, const mpz_t fb1) {
    mpz_t am1, p, q, r, tmp;
    mpz_inits(am1, p, q, r, tmp, nullptr);
    mpz_sub(am1, fa1, fa); // F(a-1)
    mul_exact(p, fa, fb1);
    mul_exact(q, am1, fb);
    mpz_add(r, p, q);
    mul_exact(p, fa1, fb1);
    mul_exact(q, fa, fb);
    mpz_add(tmp, p, q);
    mpz_swap(o0, r);
    mpz_swap(o1, tmp);
    mpz_clears(am1, p, q, r, tmp, nullptr);
}

// ---------------------------------------------------------------------------
// Big limb-array layer
//
// This GMP 6.3.0 build has a 32-bit mp_bitcnt_t, so mpz cannot represent
// integers wider than 2^32 bits (512 MiB): F(6.19e10) needs 5.37 GiB.  The
// anchored chain is therefore carried out on plain little-endian 64-bit limb
// arrays with the mpn primitives, and only block products (each well below
// the 2^32-bit limit) go through mpz_mul.  mpn_size arguments are 32-bit on
// Windows too, but every array here is below 2^31 limbs.
// ---------------------------------------------------------------------------

struct Big {
    std::vector<mp_limb_t> d; // canonical little-endian limbs
};

static size_t g_block_limbs = 33000000;
static int g_threads = 8;

static void big_trim(Big& x) {
    while (!x.d.empty() && x.d.back() == 0) x.d.pop_back();
}

static Big big_from_mpz(const mpz_t z) {
    Big r;
    const size_t n = mpz_size(z);
    if (n) r.d.assign(mpz_limbs_read(z), mpz_limbs_read(z) + n);
    big_trim(r);
    return r;
}

static Big big_add(const Big& a, const Big& b) {
    if (a.d.empty()) return b;
    if (b.d.empty()) return a;
    if (a.d.size() < b.d.size()) return big_add(b, a);
    Big r;
    r.d.assign(a.d.size() + 1, 0);
    const mp_limb_t carry =
        mpn_add(r.d.data(), a.d.data(), (mp_size_t)a.d.size(), b.d.data(),
                (mp_size_t)b.d.size());
    r.d[a.d.size()] = carry;
    big_trim(r);
    return r;
}

// a - b, requires a >= b.
static Big big_sub(const Big& a, const Big& b) {
    Big r;
    if (a.d.empty() || b.d.empty()) return a;
    r.d.assign(a.d.size(), 0);
    mpn_sub(r.d.data(), a.d.data(), (mp_size_t)a.d.size(), b.d.data(),
            (mp_size_t)b.d.size());
    big_trim(r);
    return r;
}

// Direct mpz product of two arrays that fit the 2^32-bit limit.
static Big big_mul_direct(const mp_limb_t* ap, size_t an, const mp_limb_t* bp,
                          size_t bn) {
    Big r;
    if (an == 0 || bn == 0) return r;
    mpz_t za, zb, zp;
    mpz_roinit_n(za, ap, (mp_size_t)an);
    mpz_roinit_n(zb, bp, (mp_size_t)bn);
    mpz_init(zp);
    mpz_mul(zp, za, zb);
    r.d.assign(mpz_limbs_read(zp), mpz_limbs_read(zp) + mpz_size(zp));
    mpz_clear(zp);
    big_trim(r);
    return r;
}

struct MulTask {
    size_t off;
    const mp_limb_t* ap;
    size_t an;
    const mp_limb_t* bp;
    size_t bn;
};

static Big big_mul(const Big& A, const Big& B) {
    Big C;
    if (A.d.empty() || B.d.empty()) return C;
    const size_t an = A.d.size();
    const size_t bn = B.d.size();
    if (an < g_block_limbs && bn < g_block_limbs) {
        return big_mul_direct(A.d.data(), an, B.d.data(), bn);
    }
    C.d.assign(an + bn + 2, 0);

    std::vector<MulTask> tasks;
    for (size_t i = 0; i < an; i += g_block_limbs) {
        size_t na = std::min(g_block_limbs, an - i);
        while (na > 1 && A.d[i + na - 1] == 0) --na;
        if (na == 1 && A.d[i] == 0) continue;
        for (size_t j = 0; j < bn; j += g_block_limbs) {
            size_t nb = std::min(g_block_limbs, bn - j);
            while (nb > 1 && B.d[j + nb - 1] == 0) --nb;
            if (nb == 1 && B.d[j] == 0) continue;
            MulTask t;
            t.off = i + j;
            t.ap = A.d.data() + i;
            t.an = na;
            t.bp = B.d.data() + j;
            t.bn = nb;
            tasks.push_back(t);
        }
    }

    std::atomic<size_t> next{0};
    std::mutex add_mutex;
    auto worker = [&]() {
        mpz_t zp;
        mpz_init(zp);
        for (;;) {
            const size_t k = next.fetch_add(1);
            if (k >= tasks.size()) break;
            const MulTask& t = tasks[k];
            mpz_t za, zb;
            mpz_roinit_n(za, t.ap, (mp_size_t)t.an);
            mpz_roinit_n(zb, t.bp, (mp_size_t)t.bn);
            mpz_mul(zp, za, zb);
            const size_t pn = mpz_size(zp);
            if (pn) {
                const mp_limb_t* pp = mpz_limbs_read(zp);
                std::lock_guard<std::mutex> lk(add_mutex);
                const mp_size_t n1 = (mp_size_t)(C.d.size() - t.off);
                const mp_limb_t carry =
                    mpn_add(C.d.data() + t.off, C.d.data() + t.off, n1, pp,
                            (mp_size_t)pn);
                if (carry) {
                    std::fprintf(stderr, "big_mul: carry out of result\n");
                    std::abort();
                }
            }
        }
        mpz_clear(zp);
    };

    int nt = g_threads;
    if (nt < 1) nt = 1;
    if ((size_t)nt > tasks.size()) nt = (int)tasks.size();
    std::vector<std::thread> pool;
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    big_trim(C);
    return C;
}

// Stream a Big value in the canonical limb-file format (u64 count + LE u32).
static bool write_big_limbs(const char* path, const Big& v, bool hash_only,
                            std::string* hash_out, u64* limbs_out) {
    const size_t n64 = v.d.size();
    u32 top_hi = 0;
    if (n64) top_hi = (u32)(v.d[n64 - 1] >> 32);
    u64 n32 = n64 ? (u64)n64 * 2 - (top_hi == 0 ? 1 : 0) : 0;
    if (limbs_out) *limbs_out = n32;

    Sha256 sha;
    const u64 hdr = n32;
    sha.update(&hdr, 8);

    FILE* f = nullptr;
    if (!hash_only) {
        f = std::fopen(path, "wb");
        if (!f) {
            std::fprintf(stderr, "oracle: cannot create %s\n", path);
            return false;
        }
        if (std::fwrite(&hdr, 8, 1, f) != 1) {
            std::fclose(f);
            return false;
        }
    }
    constexpr size_t kChunk = 1 << 20;
    std::vector<u32> buf(kChunk);
    u64 written = 0;
    while (written < n32) {
        size_t take = (size_t)std::min<u64>(kChunk, n32 - written);
        for (size_t i = 0; i < take; ++i) {
            const u64 idx = written + i;
            const mp_limb_t limb = v.d[idx >> 1];
            buf[i] = (idx & 1) ? (u32)(limb >> 32) : (u32)limb;
        }
        sha.update(buf.data(), take * 4);
        if (f && std::fwrite(buf.data(), 4, take, f) != take) {
            std::fclose(f);
            std::fprintf(stderr, "oracle: write failed at %llu\n",
                         (unsigned long long)written);
            return false;
        }
        written += take;
    }
    if (f && std::fclose(f) != 0) return false;
    if (hash_out) *hash_out = sha.hex();
    return true;
}

// ---------------------------------------------------------------------------
// block28 additions: a canonical-file <-> Big bridge and chain primitives.
// Every mode below is exact GMP/mpn arithmetic; large products go through
// big_mul (33M-limb blocks), so operands may exceed the 2^32-bit mpz limit.
// ---------------------------------------------------------------------------

static Big big_from_u32(const std::vector<u32>& v) {
    Big r;
    r.d.assign((v.size() + 1) / 2, 0);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i & 1)
            r.d[i >> 1] |= (mp_limb_t)v[i] << 32;
        else
            r.d[i >> 1] = v[i];
    }
    big_trim(r);
    return r;
}

static bool read_big(const char* path, Big& b) {
    std::vector<u32> v;
    if (!read_limbs(path, v)) return false;
    b = big_from_u32(v);
    return true;
}

static bool write_big_file(const char* path, const Big& b, std::string* hash,
                           u64* limbs, bool hash_only) {
    return write_big_limbs(path, b, hash_only, hash, limbs);
}


static bool anchored_fib10_big(u64 M, std::string* hash, bool hash_only,
                               const char* out_path, u64* limbs_out,
                               double* anchor_ms, double* chain_ms) {
    mpz_t m0, m1;
    mpz_inits(m0, m1, nullptr);
    double t0 = now_ms();
    fib_pair(m0, m1, M);
    *anchor_ms = now_ms() - t0;
    if (g_trace)
        std::fprintf(stderr, "anchor F(%llu) limbs=%zu\n",
                     (unsigned long long)M, mpz_size(m0));

    t0 = now_ms();
    Big fm = big_from_mpz(m0);
    Big fm1 = big_from_mpz(m1);
    mpz_clears(m0, m1, nullptr);
    if (g_trace)
        std::fprintf(stderr, "anchor big limbs=%zu %zu\n", fm.d.size(),
                     fm1.d.size());

    // 2M
    Big t = big_add(fm1, big_sub(fm1, fm)); // 2 F(M+1) - F(M)
    Big a0 = big_mul(fm, t);
    Big a1 = big_add(big_mul(fm, fm), big_mul(fm1, fm1));
    t.d.clear();
    t.d.shrink_to_fit();
    if (g_trace)
        std::fprintf(stderr, "2M limbs=%zu %zu\n", a0.d.size(), a1.d.size());

    // 4M
    Big t2 = big_add(a1, big_sub(a1, a0));
    Big b0 = big_mul(a0, t2);
    Big b1 = big_add(big_mul(a0, a0), big_mul(a1, a1));
    t2.d.clear();
    t2.d.shrink_to_fit();
    a0.d.clear();
    a0.d.shrink_to_fit();
    a1.d.clear();
    a1.d.shrink_to_fit();
    if (g_trace)
        std::fprintf(stderr, "4M limbs=%zu %zu\n", b0.d.size(), b1.d.size());

    // 5M = 4M + M
    Big am1 = big_sub(b1, b0); // F(4M-1)
    Big p = big_mul(b0, fm1);
    Big q = big_mul(am1, fm);
    Big c0 = big_add(p, q);
    p = big_mul(b1, fm1);
    q = big_mul(b0, fm);
    Big c1 = big_add(p, q);
    p = Big();
    q = Big();
    am1.d.clear();
    am1.d.shrink_to_fit();
    b0.d.clear();
    b0.d.shrink_to_fit();
    b1.d.clear();
    b1.d.shrink_to_fit();
    fm.d.clear();
    fm.d.shrink_to_fit();
    fm1.d.clear();
    fm1.d.shrink_to_fit();
    if (g_trace)
        std::fprintf(stderr, "5M limbs=%zu %zu\n", c0.d.size(), c1.d.size());

    // 10M: only F(10M) is needed.
    Big s = big_add(c1, big_sub(c1, c0)); // 2 F(5M+1) - F(5M)
    Big d0 = big_mul(c0, s);
    s.d.clear();
    s.d.shrink_to_fit();
    c0.d.clear();
    c0.d.shrink_to_fit();
    c1.d.clear();
    c1.d.shrink_to_fit();
    if (g_trace) std::fprintf(stderr, "10M limbs=%zu\n", d0.d.size());
    *chain_ms = now_ms() - t0;

    bool ok = write_big_limbs(out_path, d0, hash_only, hash, limbs_out);
    d0.d.clear();
    d0.d.shrink_to_fit();
    return ok;
}

// Exact fast doubling for any 64-bit index, on the Big limb layer.
static void fib_pair_big(u64 n, Big& f, Big& f1) {
    if (n == 0) {
        f = Big();
        f1 = Big();
        f1.d.push_back(1);
        return;
    }
    if (n <= (1ull << 31)) {
        mpz_t z, z1;
        mpz_inits(z, z1, nullptr);
        fib_pair(z, z1, n);
        f = big_from_mpz(z);
        f1 = big_from_mpz(z1);
        mpz_clears(z, z1, nullptr);
        return;
    }
    Big a, b;
    fib_pair_big(n >> 1, a, b);
    Big t = big_add(b, big_sub(b, a)); // 2b - a
    Big c = big_mul(a, t);             // F(2k)
    Big a2 = big_mul(a, a);
    Big b2 = big_mul(b, b);
    Big d = big_add(a2, b2); // F(2k+1)
    a2 = Big();
    b2 = Big();
    t = Big();
    if (n & 1ull) {
        f = d;
        f1 = big_add(c, d);
    } else {
        f = c;
        f1 = d;
    }
}

// Independent second chain for the same target: F(10M) = F(8M + 2M).
// Shares only the anchor, the multiply primitive and the identities; the
// products and the addition order are completely different.
static bool anchored_fib10_alt(u64 M, std::string* hash, bool hash_only,
                               const char* out_path, u64* limbs_out,
                               double* anchor_ms, double* chain_ms) {
    mpz_t m0, m1;
    mpz_inits(m0, m1, nullptr);
    double t0 = now_ms();
    fib_pair(m0, m1, M);
    *anchor_ms = now_ms() - t0;
    t0 = now_ms();
    Big fm = big_from_mpz(m0);
    Big fm1 = big_from_mpz(m1);
    mpz_clears(m0, m1, nullptr);

    // 2M
    Big t = big_add(fm1, big_sub(fm1, fm));
    Big p2 = big_mul(fm, t);
    Big q2 = big_add(big_mul(fm, fm), big_mul(fm1, fm1));
    t = Big();
    fm = Big();
    fm1 = Big();

    // 4M
    Big t2 = big_add(q2, big_sub(q2, p2));
    Big p4 = big_mul(p2, t2);
    Big q4 = big_add(big_mul(p2, p2), big_mul(q2, q2));
    t2 = Big();
    if (g_trace) std::fprintf(stderr, "alt 4M\n");

    // 8M
    Big t3 = big_add(q4, big_sub(q4, p4));
    Big p8 = big_mul(p4, t3);
    Big q8 = big_add(big_mul(p4, p4), big_mul(q4, q4));
    t3 = Big();
    p4 = Big();
    q4 = Big();
    if (g_trace) std::fprintf(stderr, "alt 8M\n");

    // 10M = 8M + 2M: F(8M+2M) = F(8M) F(2M+1) + F(8M-1) F(2M)
    Big am1 = big_sub(q8, p8);
    Big c0 = big_add(big_mul(p8, q2), big_mul(am1, p2));
    am1 = Big();
    p8 = Big();
    q8 = Big();
    p2 = Big();
    q2 = Big();
    if (g_trace) std::fprintf(stderr, "alt 10M\n");
    *chain_ms = now_ms() - t0;

    bool ok = write_big_limbs(out_path, c0, hash_only, hash, limbs_out);
    c0 = Big();
    return ok;
}

// F(10*M) anchored at an exact (F(M), F(M+1)); fills n32/hash.
static bool anchored_fib10(u64 M, std::string* hash, bool hash_only,
                           const char* out_path, u64* limbs_out,
                           double* anchor_ms, double* chain_ms) {
    mpz_t m0, m1; // F(M), F(M+1)
    mpz_t a0, a1; // 2M
    mpz_t b0, b1; // 4M
    mpz_t c0, c1; // 5M
    mpz_t d0, d1; // 10M
    mpz_inits(m0, m1, a0, a1, b0, b1, c0, c1, d0, d1, nullptr);

    double t0 = now_ms();
    fib_pair(m0, m1, M);
    *anchor_ms = now_ms() - t0;
    if (g_trace)
        std::fprintf(stderr, "anchor F(%llu) limbs=%zu\n",
                     (unsigned long long)M, mpz_size(m0));

    t0 = now_ms();
    double_pair(a0, a1, m0, m1); // 2M
    if (g_trace) std::fprintf(stderr, "2M done limbs=%zu\n", mpz_size(a0));
    double_pair(b0, b1, a0, a1); // 4M
    if (g_trace) std::fprintf(stderr, "4M done limbs=%zu\n", mpz_size(b0));
    mpz_clears(a0, a1, nullptr); // 2M no longer needed
    add_pair(c0, c1, b0, b1, m0, m1); // 5M
    if (g_trace) std::fprintf(stderr, "5M done limbs=%zu\n", mpz_size(c0));
    mpz_clears(b0, b1, m0, m1, nullptr); // 4M and anchor no longer needed
    double_first(d0, c0, c1); // 10M: only F(10M) is the target value
    if (g_trace) std::fprintf(stderr, "10M done limbs=%zu\n", mpz_size(d0));
    mpz_clears(c0, c1, nullptr);
    *chain_ms = now_ms() - t0;

    bool ok = write_mpz_limbs(out_path, d0, hash_only, hash, limbs_out);
    mpz_clears(d0, d1, nullptr);
    return ok;
}

// ---------------------------------------------------------------------------
// Self-tests
// ---------------------------------------------------------------------------

static int selftest() {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    int bad = 0;
    mpz_t direct, got;
    mpz_inits(direct, got, nullptr);
    for (u64 n = 0; n <= 200000; ++n) {
        mpz_fib_ui(direct, (unsigned long)n);
        fib_u64(got, n);
        if (mpz_cmp(direct, got) != 0) {
            std::fprintf(stderr, "fib mismatch at n=%llu\n",
                         (unsigned long long)n);
            if (++bad > 5) break;
        }
    }
    gmp_randstate_t st;
    gmp_randinit_mt(st);
    gmp_randseed_ui(st, 0xC0FFEEu);
    const size_t saved = g_mul_chunk_limbs;
    g_mul_chunk_limbs = 64;
    mpz_t a, b, c, ref;
    mpz_inits(a, b, c, ref, nullptr);
    for (int it = 0; it < 300 && bad == 0; ++it) {
        const unsigned long la = 1 + (unsigned long)(it * 29 % 500);
        const unsigned long lb = 1 + (unsigned long)(it * 61 % 500);
        mpz_urandomb(a, st, (mp_bitcnt_t)la * GMP_NUMB_BITS);
        mpz_urandomb(b, st, (mp_bitcnt_t)lb * GMP_NUMB_BITS);
        mul_exact(c, a, b);
        mpz_mul(ref, a, b);
        if (mpz_cmp(c, ref) != 0) {
            std::fprintf(stderr, "mul_exact mismatch it=%d\n", it);
            ++bad;
        }
    }
    mpz_clears(a, b, c, ref, nullptr);
    g_mul_chunk_limbs = saved;
    mpz_clears(direct, got, nullptr);
    gmp_randclear(st);
    if (bad == 0) std::printf("EXT ORACLE SELFTEST PASSED\n");
    return bad == 0 ? 0 : 1;
}

// Verify the anchored chain at small sizes against direct fast doubling.
static int anchor_selftest() {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const u64 ms[] = {0, 1, 2, 3, 10, 100, 1000, 12345, 100000, 1000000};
    int bad = 0;
    for (u64 M : ms) {
        mpz_t ref;
        mpz_init(ref);
        fib_u64(ref, 10 * M);
        std::string refh;
        u64 reflimbs = 0;
        write_mpz_limbs("", ref, true, &refh, &reflimbs);
        mpz_clear(ref);

        for (int variant = 0; variant < 3; ++variant) {
            std::string h;
            u64 limbs = 0;
            double t0 = 0, t1 = 0;
            const size_t saved_block = g_block_limbs;
            const int saved_threads = g_threads;
            if (variant == 0) {
                g_block_limbs = 64; // force the blocked Big path
                g_threads = 4;
            } else if (variant == 1) {
                g_threads = 1; // blocked Big path, single worker
            }
            bool ok = (variant == 2)
                          ? anchored_fib10(M, &h, true, "", &limbs, &t0, &t1)
                          : anchored_fib10_big(M, &h, true, "", &limbs, &t0, &t1);
            g_block_limbs = saved_block;
            g_threads = saved_threads;
            if (!ok) {
                std::fprintf(stderr, "anchor variant %d failed at M=%llu\n",
                             variant, (unsigned long long)M);
                ++bad;
                continue;
            }
            if (h != refh || limbs != reflimbs) {
                std::fprintf(stderr,
                             "anchor variant %d mismatch M=%llu (limbs %llu vs "
                             "%llu)\n",
                             variant, (unsigned long long)M,
                             (unsigned long long)limbs,
                             (unsigned long long)reflimbs);
                ++bad;
            }
        }
    }
    if (bad == 0) std::printf("ANCHOR SELFTEST PASSED\n");
    return bad == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* tr = std::getenv("ORACLE_TRACE");
    g_trace = (tr && *tr && *tr != '0') ? 1 : 0;
    const char* ch = std::getenv("ORACLE_CHUNK_LOG2");
    if (ch && *ch) {
        int lg = std::atoi(ch);
        if (lg >= 10 && lg <= 26) g_mul_chunk_limbs = (size_t)1 << lg;
    }
    const char* bl = std::getenv("ORACLE_BLOCK_LIMBS");
    if (bl && *bl) {
        size_t v = (size_t)std::strtoull(bl, nullptr, 0);
        if (v >= 1) g_block_limbs = v;
    }
    const char* thr = std::getenv("ORACLE_THREADS");
    if (thr && *thr) {
        int v = std::atoi(thr);
        if (v >= 1 && v <= 64) g_threads = v;
    }
    if (argc >= 2 && std::string(argv[1]) == "selftest") return selftest();
    if (argc >= 2 && std::string(argv[1]) == "anchor-selftest")
        return anchor_selftest();
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s fib N OUT.bin | %s mul A.bin B.bin OUT.bin | "
                     "%s fib-anchor N M OUT.bin [--hash-only] | "
                     "%s fib-anchor-mpz N M OUT.bin | %s selftest | "
                     "%s anchor-selftest\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "fib") {
        if (argc < 4) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        mpz_t z;
        mpz_init(z);
        const double t0 = now_ms();
        fib_u64(z, n);
        const double ms = now_ms() - t0;
        std::string hash;
        u64 limbs = 0;
        if (!write_mpz_limbs(argv[3], z, false, &hash, &limbs)) return 1;
        std::printf("FIB_MS %.6f\nFIB_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                    ms, (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        mpz_clear(z);
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
        mul_exact(zc, za, zb);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<u32> c;
        // Canonical 32-bit export for comparison with the GPU.
        Sha256 sha;
        u64 hdr = 0;
        {
            size_t count = 0;
            void* raw = mpz_export(nullptr, &count, -1, 4, 0, 0, zc);
            hdr = count;
            c.resize(count);
            if (count && raw) std::memcpy(c.data(), raw, count * 4);
            if (raw) {
                void (*release)(void*, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &release);
                release(raw, count * 4);
            }
        }
        sha.update(&hdr, 8);
        sha.update(c.data(), c.size() * 4);
        if (!write_limbs(argv[4], c)) return 1;
        std::printf("MUL_MS %.6f\nMUL_LIMBS %zu\nSHA256 %s\n", ms, c.size(),
                    sha.hex().c_str());
        mpz_clears(za, zb, zc, nullptr);
        return 0;
    }
    if (mode == "bigmul" || mode == "bigadd" || mode == "bigsub" ||
        mode == "bigshl1") {
        const bool hash_only = (argc > 5 &&
                                std::string(argv[argc - 1]) == "--hash-only");
        if (mode == "bigshl1") {
            if (argc < 4) return 2;
            Big a;
            if (!read_big(argv[2], a)) return 1;
            Big r;
            r.d.assign(a.d.size() + 1, 0);
            mp_limb_t carry = a.d.empty()
                                  ? 0
                                  : mpn_lshift(r.d.data(), a.d.data(),
                                               (mp_size_t)a.d.size(), 1);
            r.d[a.d.size()] = carry;
            big_trim(r);
            std::string hash;
            u64 limbs = 0;
            if (!write_big_file(argv[3], r, &hash, &limbs, hash_only)) return 1;
            std::printf("BIG_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                        (unsigned long long)limbs, hash.c_str(),
                        (double)peak_rss_bytes() / 1048576.0);
            return 0;
        }
        if (argc < 5) return 2;
        Big a, b;
        double t0 = now_ms();
        if (!read_big(argv[2], a) || !read_big(argv[3], b)) return 1;
        const double read_ms = now_ms() - t0;
        Big r;
        t0 = now_ms();
        if (mode == "bigmul")
            r = big_mul(a, b);
        else if (mode == "bigadd")
            r = big_add(a, b);
        else
            r = big_sub(a, b);
        const double op_ms = now_ms() - t0;
        std::string hash;
        u64 limbs = 0;
        const char* out = argv[4];
        if (!write_big_file(out, r, &hash, &limbs, hash_only)) return 1;
        std::printf("READ_MS %.3f\nOP_MS %.3f\nBIG_LIMBS %llu\nSHA256 %s\n"
                    "PEAK_RSS_MB %.1f\n",
                    read_ms, op_ms, (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "fibpair") {
        if (argc < 5) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        Big f, f1;
        const double t0 = now_ms();
        fib_pair_big(n, f, f1);
        const double ms = now_ms() - t0;
        std::string h0, h1;
        u64 l0 = 0, l1 = 0;
        if (!write_big_file(argv[3], f, &h0, &l0, false)) return 1;
        if (!write_big_file(argv[4], f1, &h1, &l1, false)) return 1;
        std::printf("PAIR_MS %.3f\nF_LIMBS %llu\nF1_LIMBS %llu\nSHA256_F %s\n"
                    "SHA256_F1 %s\nPEAK_RSS_MB %.1f\n",
                    ms, (unsigned long long)l0, (unsigned long long)l1,
                    h0.c_str(), h1.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "fib-anchor") {
        if (argc < 5) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        const u64 m = std::strtoull(argv[3], nullptr, 0);
        if (n != 10 * m) {
            std::fprintf(stderr, "fib-anchor requires N == 10*M (got %llu, %llu)\n",
                         (unsigned long long)n, (unsigned long long)m);
            return 2;
        }
        bool hash_only = false;
        for (int i = 5; i < argc; ++i)
            if (std::string(argv[i]) == "--hash-only") hash_only = true;
        std::string hash;
        u64 limbs = 0;
        double anchor_ms = 0, chain_ms = 0;
        if (!anchored_fib10_big(m, &hash, hash_only, argv[4], &limbs, &anchor_ms,
                                &chain_ms))
            return 1;
        std::printf("ANCHOR_M %llu\nANCHOR_MS %.6f\nCHAIN_MS %.6f\n", 
                    (unsigned long long)m, anchor_ms, chain_ms);
        std::printf("FIB_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                    (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "fib-anchor-alt") {
        if (argc < 5) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        const u64 m = std::strtoull(argv[3], nullptr, 0);
        if (n != 10 * m) return 2;
        bool hash_only = false;
        for (int i = 5; i < argc; ++i)
            if (std::string(argv[i]) == "--hash-only") hash_only = true;
        std::string hash;
        u64 limbs = 0;
        double anchor_ms = 0, chain_ms = 0;
        if (!anchored_fib10_alt(m, &hash, hash_only, argv[4], &limbs, &anchor_ms,
                                &chain_ms))
            return 1;
        std::printf("ANCHOR_M %llu\nANCHOR_MS %.6f\nCHAIN_MS %.6f\n",
                    (unsigned long long)m, anchor_ms, chain_ms);
        std::printf("FIB_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                    (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "fib-anchor-mpz") {
        if (argc < 5) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        const u64 m = std::strtoull(argv[3], nullptr, 0);
        if (n != 10 * m) return 2;
        bool hash_only = false;
        for (int i = 5; i < argc; ++i)
            if (std::string(argv[i]) == "--hash-only") hash_only = true;
        std::string hash;
        u64 limbs = 0;
        double anchor_ms = 0, chain_ms = 0;
        if (!anchored_fib10(m, &hash, hash_only, argv[4], &limbs, &anchor_ms,
                            &chain_ms))
            return 1;
        std::printf("ANCHOR_M %llu\nANCHOR_MS %.6f\nCHAIN_MS %.6f\n",
                    (unsigned long long)m, anchor_ms, chain_ms);
        std::printf("FIB_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                    (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "multest") {
        // usage: multest LIMBS [ROUNDS]
        if (argc < 3) return 2;
        const size_t limbs = (size_t)std::strtoull(argv[2], nullptr, 0);
        gmp_randstate_t st;
        gmp_randinit_mt(st);
        gmp_randseed_ui(st, 0x1234ABCDu);
        mpz_t a, b, c;
        mpz_inits(a, b, c, nullptr);
        mpz_urandomb(a, st, (mp_bitcnt_t)(limbs * 64));
        mpz_urandomb(b, st, (mp_bitcnt_t)(limbs * 64));
        std::printf("multest limbs=%zu\n", limbs);
        const double t0 = now_ms();
        mul_exact(c, a, b);
        std::printf("MUL_MS %.3f\nresult_limbs=%zu\n", now_ms() - t0,
                    mpz_size(c));
        mpz_clears(a, b, c, nullptr);
        gmp_randclear(st);
        return 0;
    }
    if (mode == "bigtest") {
        // usage: bigtest LIMBS
        const size_t limbs =
            argc > 2 ? (size_t)std::strtoull(argv[2], nullptr, 0) : 671088640ull;
        mpz_t a, c;
        mpz_inits(a, c, nullptr);
        std::printf("create %zu limbs (%.2f GB)\n", limbs,
                    (double)limbs * 8 / 1e9);
        std::fflush(stdout);
        mp_limb_t* p = mpz_limbs_write(a, (mp_size_t)limbs);
        for (size_t i = 0; i < limbs; ++i) p[i] = (mp_limb_t)(i * 2654435761u);
        mpz_limbs_finish(a, (mp_size_t)limbs);
        std::printf("a size=%zu\n", mpz_size(a));
        std::fflush(stdout);
        mpz_add(c, a, a);
        std::printf("add ok size=%zu\n", mpz_size(c));
        std::fflush(stdout);
        mpz_sub(c, c, a);
        std::printf("sub ok size=%zu\n", mpz_size(c));
        std::fflush(stdout);
        mpz_mul_2exp(c, a, 1);
        std::printf("mul2exp ok size=%zu\n", mpz_size(c));
        std::fflush(stdout);
        mpz_clears(a, c, nullptr);
        return 0;
    }
    if (mode == "fib-big") {
        if (argc < 4) return 2;
        const u64 n = std::strtoull(argv[2], nullptr, 0);
        const double t0 = now_ms();
        Big f, f1;
        fib_pair_big(n, f, f1);
        const double ms = now_ms() - t0;
        f1 = Big();
        std::string hash;
        u64 limbs = 0;
        if (!write_big_limbs(argv[3], f, false, &hash, &limbs)) return 1;
        std::printf("FIB_MS %.6f\nFIB_LIMBS %llu\nSHA256 %s\nPEAK_RSS_MB %.1f\n",
                    ms, (unsigned long long)limbs, hash.c_str(),
                    (double)peak_rss_bytes() / 1048576.0);
        return 0;
    }
    if (mode == "bigmultest") {
        // usage: bigmultest LIMBS
        const size_t limbs =
            argc > 2 ? (size_t)std::strtoull(argv[2], nullptr, 0) : 100000000ull;
        gmp_randstate_t st;
        gmp_randinit_mt(st);
        gmp_randseed_ui(st, 0xB16B00B5u);
        Big A, B;
        A.d.resize(limbs);
        B.d.resize(limbs);
        {
            u64 s0 = 0x123456789abcdefull, s1 = 0xfedcba987654321ull;
            for (size_t i = 0; i < limbs; ++i) {
                s0 ^= s0 << 13;
                s0 ^= s0 >> 7;
                s0 ^= s0 << 17;
                s1 ^= s1 << 13;
                s1 ^= s1 >> 7;
                s1 ^= s1 << 17;
                A.d[i] = s0;
                B.d[i] = s1;
            }
        }
        big_trim(A);
        big_trim(B);
        std::printf("bigmultest limbs=%zu threads=%d block=%zu\n", limbs,
                    g_threads, g_block_limbs);
        std::fflush(stdout);
        const double t0 = now_ms();
        Big C = big_mul(A, B);
        const double ms = now_ms() - t0;
        Sha256 sha;
        u64 n32 = 0;
        std::string h;
        write_big_limbs("", C, true, &h, &n32);
        std::printf("MUL_MS %.3f result_limbs=%zu u32limbs=%llu sha=%s\n", ms,
                    C.d.size(), (unsigned long long)n32, h.c_str());
        gmp_randclear(st);
        return 0;
    }
    if (mode == "rawmul") {
        // usage: rawmul LIMBS
        if (argc < 3) return 2;
        const size_t limbs = (size_t)std::strtoull(argv[2], nullptr, 0);
        gmp_randstate_t st;
        gmp_randinit_mt(st);
        gmp_randseed_ui(st, 0x55AA55AAu);
        mpz_t a, b, c;
        mpz_inits(a, b, c, nullptr);
        mpz_urandomb(a, st, (mp_bitcnt_t)(limbs * 64));
        mpz_urandomb(b, st, (mp_bitcnt_t)(limbs * 64));
        std::printf("rawmul limbs=%zu\n", limbs);
        const double t0 = now_ms();
        mpz_mul(c, a, b);
        std::printf("MUL_MS %.3f result_limbs=%zu\n", now_ms() - t0,
                    mpz_size(c));
        mpz_clears(a, b, c, nullptr);
        gmp_randclear(st);
        return 0;
    }
    if (mode == "shifttest") {
        // usage: shifttest LIMBS BITS
        if (argc < 4) return 2;
        const size_t limbs = (size_t)std::strtoull(argv[2], nullptr, 0);
        const u64 bits = std::strtoull(argv[3], nullptr, 0);
        gmp_randstate_t st;
        gmp_randinit_mt(st);
        gmp_randseed_ui(st, 7u);
        mpz_t a;
        mpz_init(a);
        mpz_urandomb(a, st, (mp_bitcnt_t)(limbs * 64));
        std::printf("before limbs=%zu bits=%llu\n", mpz_size(a),
                    (unsigned long long)bits);
        const double t0 = now_ms();
        mul_2exp_big(a, bits);
        std::printf("SHIFT_MS %.3f after limbs=%zu\n", now_ms() - t0,
                    mpz_size(a));
        mpz_clear(a);
        gmp_randclear(st);
        return 0;
    }
    std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
    return 2;
}
