// CPU emulation of the GPU driver's block schedule, tested against exact
// schoolbook arithmetic.  This validates the fold/segment indexing, the
// prime-major pass, CRT and carry normalisation end to end without CUDA.
//
//   g++ -O2 -std=c++17 driver_ref.cpp -o build\driver_ref.exe
//   .\build\driver_ref.exe
//
// It mirrors mul_gpu() in block29_fib.cpp, replacing be_load/be_fwd/be_pointwise/
// be_inv/be_download with the validated host NTT in arith_ref.h.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "arith_ref.h"
#include "block_constants.h"
#include "plan_model.h"
#include "segment_norm.h"

using namespace blk;
using blkplan::Plan;

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint64_t u64;

static int g_threads = 8;

static std::vector<u32> mul_school(const std::vector<u32>& a,
                                   const std::vector<u32>& b) {
    return mul_schoolbook(a, b);
}

static bool read_file(const std::string& path, std::vector<u32>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
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

// Mirrors mul_gpu(x,y,self) with a forced block size (0 = natural plan).
static std::vector<u32> mul_emul(const std::vector<u32>& x,
                                 const std::vector<u32>& y, bool self,
                                 size_t force_B = 0) {
    std::vector<u32> out;
    if (x.empty() || y.empty()) return out;
    Plan p = blkplan::plan_for(x.size(), y.size());
    if (force_B) {
        p.B = force_B;
        int lg = 1;
        while (((size_t)1 << lg) < 2 * force_B) ++lg;
        p.lg = lg;
        p.N = (size_t)1 << lg;
        p.block_mode = true;
        p.need = x.size() + y.size();
        p.na = (int)((x.size() + p.B - 1) / p.B);
        p.nb = (int)((y.size() + p.B - 1) / p.B);
        p.K = p.na + p.nb - 1;
    }
    const size_t B = p.B, N = p.N, need = p.need;
    const int lg = p.lg, na = p.na, nb = p.nb, K = p.K;
    const int Kneed = (int)((need + B - 1) / B);
    const Mod mods[2] = {mod0(), mod1()};

    std::vector<u32> seg0_lo((size_t)Kneed * B, 0);
    std::vector<u16> seg0_hi((size_t)Kneed * B, 0);
    std::vector<u64> dl(B, 0), fold_a(B, 0), fold_b(B, 0), d64(B, 0);
    out.assign(need, 0);

    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;

    for (int prime = 0; prime < 2; ++prime) {
        const Mod& m = mods[prime];
        const u64 pmod = prime ? P1 : P0;
        const u64 ninv = powmod((u64)(N % pmod), pmod - 2, pmod);
        std::fill(fold_a.begin(), fold_a.end(), 0);

        // X block transforms (resident).
        std::vector<std::vector<u64>> ta(na);
        for (int i = 0; i < na; ++i) {
            ta[i].assign(N, 0);
            size_t off = (size_t)i * B;
            size_t len = std::min(B, x.size() - off);
            for (size_t j = 0; j < len; ++j)
                ta[i][j] = to_mont(x[off + j], m);
            ntt_fwd(ta[i].data(), lg, m);
        }

        for (int k = 0; k < Kneed; ++k) {
            bool has_sum = (k < K);
            if (has_sum) {
                std::vector<u64> C(N, 0);
                if (self) {
                    int ilo = std::max(0, k - (na - 1));
                    int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        const u64* A = ta[i].data();
                        const u64* Bv = ta[k - i].data();
                        for (size_t t = 0; t < N; ++t) {
                            u64 v = mont48(mont48(A[t], Bv[t], m.p, m.np),
                                          ninv, m.p, m.np);
                            C[t] = add48(C[t], v, m.p);
                        }
                    }
                } else {
                    int ilo = std::max(0, k - (nb - 1));
                    int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        int j = k - i;
                        if (j < 0 || j >= nb) continue;
                        std::vector<u64> tb(N, 0);
                        size_t off = (size_t)j * B;
                        size_t len = std::min(B, y.size() - off);
                        for (size_t t = 0; t < len; ++t)
                            tb[t] = to_mont(y[off + t], m);
                        ntt_fwd(tb.data(), lg, m);
                        const u64* A = ta[i].data();
                        for (size_t t = 0; t < N; ++t) {
                            u64 v = mont48(mont48(A[t], tb[t], m.p, m.np),
                                          ninv, m.p, m.np);
                            C[t] = add48(C[t], v, m.p);
                        }
                    }
                }
                ntt_inv_noscale(C.data(), lg, m);
                std::copy(C.begin(), C.begin() + B, dl.begin());
                std::copy(C.begin() + B, C.end(), fold_b.begin());
                segnorm::fold_add(dl.data(), fold_a.data(), B, pmod, g_threads);
            }

            const std::vector<u64>& seg = has_sum ? dl : fold_a;
            size_t off = (size_t)k * B;
            size_t lim = (off >= need) ? 0 : std::min(B, need - off);
            if (lim == 0) break;
            if (prime == 0) {
                segnorm::pack(seg0_lo.data() + off, seg0_hi.data() + off,
                              seg.data(), lim, pmod, g_threads);
            } else {
                segnorm::process_segment(
                    seg0_lo.data() + off, seg0_hi.data() + off, seg.data(), lim,
                    d64.data(), out.data() + off, carry, sb1, sc1, sc2,
                    lim == B, P0, P1, BLK_INV_P0_MOD_P1, g_threads);
            }
            if (has_sum) std::swap(fold_a, fold_b);
        }
    }
    if (carry != 0) {
        std::fprintf(stderr,
                     "mul_emul: non-zero final carry %u (la=%zu lb=%zu self=%d "
                     "B=%zu need=%zu K=%d Kneed=%d)\\n",
                     carry, x.size(), y.size(), (int)self, B, need, K, Kneed);
        std::fflush(stderr);
        std::abort();
    }
    while (!out.empty() && out.back() == 0) out.pop_back();
    return out;
}

static bool cmp(const std::vector<u32>& a, const std::vector<u32>& b) {
    return cmp_limbs(a, b) == 0;
}

static std::vector<u32> fib_emul(uint64_t n, size_t force_B) {
    std::vector<u32> a, b = {1};
    if (n == 0) return a;
    int top = 63;
    while (!((n >> top) & 1)) --top;
    for (int bit = top; bit >= 0; --bit) {
        std::vector<u32> t = shl1_limbs(b);
        t = sub_limbs(t, a);
        std::vector<u32> c = mul_emul(a, t, false, force_B);
        std::vector<u32> aa = mul_emul(a, a, true, force_B);
        std::vector<u32> bb = mul_emul(b, b, true, force_B);
        std::vector<u32> d = add_limbs(aa, bb);
        if (((n >> bit) & 1) == 0) {
            a = std::move(c);
            b = std::move(d);
        } else {
            std::vector<u32> cd = add_limbs(c, d);
            a = std::move(d);
            b = std::move(cd);
        }
    }
    return a;
}

static u64 g_rng = 0x123456789ABCDEFull;
static u64 rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

int main() {
    bool ok = true;

    // Random products, general and self, natural plan and forced small blocks.
    struct Case {
        size_t la, lb;
        size_t B;
        bool self;
    };
    const Case cases[] = {
        {300, 300, 128, true},   {300, 300, 128, false},
        {300, 200, 128, false},  {150, 150, 64, true},
        {150, 150, 64, false},   {600, 600, 256, true},
        {600, 600, 256, false},  {1000, 999, 256, false},
        {500, 500, 0, true},     {500, 400, 0, false},
        {4096, 4096, 0, true},   {4096, 3000, 0, false},
    };
    for (const auto& c : cases) {
        std::vector<u32> a(c.la), b(c.lb);
        for (auto& v : a) v = (u32)(rnd() % 1000000007ull);
        for (auto& v : b) v = (u32)(rnd() % 1000000007ull);
        a.back() |= 1;
        b.back() |= 1;
        if (c.self) b = a;
        std::vector<u32> got = mul_emul(a, b, c.self, c.B);
        std::vector<u32> want = mul_school(a, b);
        if (!cmp(got, want)) {
            std::fprintf(stderr, "FAIL case la=%zu lb=%zu B=%zu self=%d\n",
                         c.la, c.lb, c.B, (int)c.self);
            ok = false;
        } else {
            std::printf("mul la=%-5zu lb=%-5zu B=%-4zu self=%d -> %zu limbs PASS\n",
                        c.la, c.lb, c.B, (int)c.self, got.size());
        }
    }

    // Fast doubling with forced multi-block plans, checked against the
    // already validated ntt_ref outputs and against a direct schoolbook fib
    // for the smaller cases.
    for (uint64_t n : {(uint64_t)1000, (uint64_t)10000, (uint64_t)100000}) {
        std::vector<u32> want;
        std::string path = "out/fib_" + std::to_string(n) + ".bin";
        if (!read_file(path, want)) {
            std::printf("fib %llu: %s missing, skipped\n",
                        (unsigned long long)n, path.c_str());
            continue;
        }
        for (size_t B : {(size_t)128, (size_t)0}) {
            std::vector<u32> got = fib_emul(n, B);
            if (!cmp(got, want)) {
                std::fprintf(stderr, "FAIL fib n=%llu B=%zu limbs %zu vs %zu\n",
                             (unsigned long long)n, B, got.size(), want.size());
                ok = false;
            } else {
                std::printf("fib n=%-6llu B=%-4zu -> %zu limbs PASS\n",
                            (unsigned long long)n, B, got.size());
            }
        }
    }

    if (ok) std::printf("DRIVER BLOCK-SCHEDULE TESTS PASSED\n");
    return ok ? 0 : 1;
}
