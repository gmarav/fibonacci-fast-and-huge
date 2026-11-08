// CPU emulation of the GPU block schedule (structure test, no CUDA).
//
// This file reproduces the exact control flow of the CUDA driver:
//
//   * per output segment k, per prime p in {0,1}:
//       - zero the frequency accumulator C,
//       - for every block pair (i,j) with i+j=k:
//           transform A_i, transform B_j (re-transformed per pair, because the
//           option-A engine has only three transform arrays),
//           C += n^-1 * A_i * B_j   (Montgomery pointwise, accumulate),
//       - inverse transform of C with no final scale,
//       - keep C[0..B) as the low half and C[B..2B) as the previous-high half
//         for the next segment,
//   * fold C_k[0..B) + C_{k-1}[B..2B) in the residue domain,
//   * CRT + base-2^32 carry normalisation per segment (segment_norm.h).
//
// It is compared against schoolbook multiplication with forced multi-block
// plans, and the full fast-doubling loop is compared against exact Python via
// verify_python.py.  All arithmetic is exact integer arithmetic.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "arith_ref.h"
#include "plan_model.h"
#include "segment_norm.h"

using namespace blk28;

static u64 g_rng = 0xD81B28ull;
static u64 rnd64() {
    u64 z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void trim(std::vector<u32>& v) {
    while (!v.empty() && v.back() == 0) v.pop_back();
}

// ---------------------------------------------------------------------------
// Transform a block of `limbs` limbs starting at `off` into A (Montgomery).
// ---------------------------------------------------------------------------
static void build_block(const std::vector<u32>& src, size_t off, size_t limbs,
                        int lg, const Mod& m, std::vector<u64>& out) {
    const size_t N = (size_t)1 << lg;
    out.assign(N, 0);
    for (size_t i = 0; i < limbs; ++i) out[i] = to_mont(src[off + i], m);
    ntt_fwd(out.data(), lg, m);
}

// Exact product with the device schedule.  `self` uses symmetric block pairs
// (i <= j, weight 2 for i < j), mirroring mul_gpu(x, x).
static std::vector<u32> mul_sched(const std::vector<u32>& a,
                                  const std::vector<u32>& b, bool self,
                                  int force_bl, bool verbose) {
    if (a.empty() || b.empty()) return {};
    blk28plan::Plan p;
    if (force_bl > 0) {
        p.lg = force_bl + 1;
        p.B = (size_t)1 << force_bl;
        p.N = (size_t)1 << p.lg;
        p.block_mode = false;
        p.na = (int)((a.size() + p.B - 1) / p.B);
        p.nb = self ? p.na : (int)((b.size() + p.B - 1) / p.B);
    } else {
        p = blk28plan::plan_for(a.size(), b.size());
    }
    const size_t B = p.B, N = p.N;
    const int lg = p.lg;
    const int na = p.na, nb = p.nb;
    const int K = na + nb - 1;
    const size_t need = a.size() + b.size();
    const int Kneed = (int)((need + B - 1) / B);
    if (verbose)
        std::printf("  plan lg=%d B=2^%d na=%d nb=%d K=%d Kneed=%d need=%zu\n",
                    lg, p.lg - 1, na, nb, K, Kneed, need);

    const Mod m[2] = {mod0(), mod1()};
    u64 ninv[2] = {powmod((u64)(N % m[0].p), m[0].p - 2, m[0].p),
                   powmod((u64)(N % m[1].p), m[1].p - 2, m[1].p)};

    std::vector<std::vector<u64>> low(2, std::vector<u64>(B, 0));
    std::vector<std::vector<u64>> prev(2, std::vector<u64>(B, 0));
    std::vector<u64> A(N), Bt(N), C(N);
    std::vector<u32> out(need, 0);
    std::vector<u64> d64(B, 0);
    u32 carry = 0, sb1 = 0, sc1 = 0, sc2 = 0;

    for (int k = 0; k < Kneed; ++k) {
        const bool has_sum = (k < K);
        if (has_sum) {
            for (int pi = 0; pi < 2; ++pi) {
                std::fill(C.begin(), C.end(), 0);
                if (!self) {
                    const int ilo = std::max(0, k - (nb - 1));
                    const int ihi = std::min(na - 1, k);
                    for (int i = ilo; i <= ihi; ++i) {
                        const int j = k - i;
                        if (j < 0 || j >= nb) continue;
                        build_block(a, (size_t)i * B,
                                    std::min(B, a.size() - (size_t)i * B),
                                    lg, m[pi], A);
                        build_block(b, (size_t)j * B,
                                    std::min(B, b.size() - (size_t)j * B),
                                    lg, m[pi], Bt);
                        for (size_t t = 0; t < N; ++t)
                            C[t] = add48(C[t],
                                         mont48(mont48(A[t], Bt[t], m[pi].p,
                                                       m[pi].np),
                                                ninv[pi], m[pi].p, m[pi].np),
                                         m[pi].p);
                    }
                } else {
                    // Symmetric pairs i <= j at this diagonal.
                    const int ilo = std::max(0, k - (na - 1));
                    const int ihi = k / 2;
                    for (int i = ilo; i <= ihi; ++i) {
                        const int j = k - i;
                        if (j >= na) continue;
                        build_block(a, (size_t)i * B,
                                    std::min(B, a.size() - (size_t)i * B),
                                    lg, m[pi], A);
                        if (i == j) {
                            for (size_t t = 0; t < N; ++t)
                                C[t] = add48(C[t],
                                             mont48(mont48(A[t], A[t], m[pi].p,
                                                           m[pi].np),
                                                    ninv[pi], m[pi].p,
                                                    m[pi].np),
                                             m[pi].p);
                        } else {
                            build_block(a, (size_t)j * B,
                                        std::min(B, a.size() - (size_t)j * B),
                                        lg, m[pi], Bt);
                            for (size_t t = 0; t < N; ++t) {
                                u64 v = mont48(mont48(A[t], Bt[t], m[pi].p,
                                                      m[pi].np),
                                               ninv[pi], m[pi].p, m[pi].np);
                                v = add48(v, v, m[pi].p);   // weight 2
                                C[t] = add48(C[t], v, m[pi].p);
                            }
                        }
                    }
                }
                ntt_inv_noscale(C.data(), lg, m[pi]);
                for (size_t t = 0; t < B; ++t) low[pi][t] = C[t];
                // Fold C_k[0..B) + C_{k-1}[B..2B) in the residue domain.
                segnorm::fold_add(low[pi].data(), prev[pi].data(), B, m[pi].p,
                                  4);
                // The old prev is dead now; mimic the async high download.
                for (size_t t = 0; t < B; ++t) prev[pi][t] = C[B + t];
            }
        } else {
            // Last segment is the high half of the final block sum.
            for (int pi = 0; pi < 2; ++pi) std::swap(low[pi], prev[pi]);
        }
        const size_t off = (size_t)k * B;
        if (off >= need) break;
        const size_t lim = std::min(B, need - off);
        segnorm::process_segment(low[0].data(), low[1].data(), lim, d64.data(),
                                 out.data() + off, carry, sb1, sc1, sc2,
                                 lim == B, P0, P1, BLK28_INV_P0_MOD_P1, 4);
        if (!has_sum) break;   // nothing beyond the last high half
    }
    if (carry != 0) {
        std::printf("  mul_sched: non-zero final carry %u\n", carry);
        return {};
    }
    trim(out);
    return out;
}

static std::vector<u32> mul_ref(const std::vector<u32>& a,
                                const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    return mul_schoolbook(a, b);
}

static int test_products(int cases) {
    int bad = 0;
    for (int it = 0; it < cases; ++it) {
        const int bl = 4 + (int)(rnd64() % 4);          // 16..128 limbs
        const size_t B = (size_t)1 << bl;
        const size_t la = 1 + (size_t)(rnd64() % (3 * B));
        const size_t lb = 1 + (size_t)(rnd64() % (3 * B));
        std::vector<u32> a(la), b(lb);
        for (auto& v : a) v = (u32)rnd64();
        if (it % 3 == 0) {
            b = a;                                       // self product
        } else {
            for (auto& v : b) v = (u32)rnd64();
        }
        const bool self = (it % 3 == 0);
        std::vector<u32> want = mul_ref(a, b);
        std::vector<u32> got = mul_sched(a, b, self, bl, false);
        if (got != want) {
            std::printf("product it=%d bl=%d la=%zu lb=%zu self=%d FAIL "
                        "(%zu vs %zu limbs)\n", it, bl, la, lb, self,
                        got.size(), want.size());
            ++bad;
        }
    }
    std::printf("driver products: %d cases, %s\n", cases,
                bad ? "FAIL" : "PASS");
    return bad;
}

// Small-block forced products with near-maximal coefficients.
static int test_adversarial() {
    int bad = 0;
    for (int bl : {4, 5, 6}) {
        const size_t B = (size_t)1 << bl;
        for (size_t k : {(size_t)1, (size_t)2, (size_t)3, (size_t)4}) {
            std::vector<u32> a(k * B), b(k * B);
            for (auto& v : a) v = 0xFFFFFFFFu;
            for (auto& v : b) v = 0xFFFFFFFFu;
            std::vector<u32> want = mul_ref(a, b);
            std::vector<u32> got = mul_sched(a, b, false, bl, false);
            if (got != want) {
                std::printf("adversarial bl=%d k=%zu FAIL (%zu vs %zu)\n", bl,
                            k, got.size(), want.size());
                ++bad;
            }
        }
    }
    std::printf("driver adversarial all-ones: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

// Full fast-doubling loop through the schedule (natural plans).
static std::vector<u32> fib_sched(uint64_t n, bool verbose) {
    std::vector<u32> a, b{1};
    if (n == 0) return a;
    int top = 63;
    while (!((n >> top) & 1)) --top;
    for (int bit = top; bit >= 0; --bit) {
        std::vector<u32> t2 = shl1_limbs(b);
        t2 = sub_limbs(t2, a);
        std::vector<u32> c = mul_sched(a, t2, false, 0, verbose);
        std::vector<u32> d;
        if (!(bit == 0 && ((n >> bit) & 1) == 0)) {
            std::vector<u32> aa = mul_sched(a, a, true, 0, verbose);
            std::vector<u32> bb = mul_sched(b, b, true, 0, verbose);
            d = add_limbs(aa, bb);
        }
        if (((n >> bit) & 1) == 0) {
            a = std::move(c);
            b = std::move(d);
        } else {
            std::vector<u32> s = add_limbs(c, d);
            a = std::move(d);
            b = std::move(s);
        }
    }
    trim(a);
    return a;
}

static bool write_limbs(const std::string& path, const std::vector<u32>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    u64 n = v.size();
    std::fwrite(&n, 8, 1, f);
    if (n) std::fwrite(v.data(), 4, (size_t)n, f);
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "all";
    const std::string dir = argc > 2 ? argv[2] : "out";
    int bad = 0;

    if (cmd == "all" || cmd == "products") bad += test_products(60);
    if (cmd == "all" || cmd == "adversarial") bad += test_adversarial();

    if (cmd == "all" || cmd == "fib") {
        for (uint64_t n : {(uint64_t)1000, (uint64_t)10000,
                           (uint64_t)100000, (uint64_t)1000000}) {
            std::vector<u32> got = fib_sched(n, false);
            std::string path = dir + "/fibsched_" + std::to_string(n) + ".bin";
            if (!write_limbs(path, got)) return 1;
            std::printf("fib-sched n=%" PRIu64 " limbs=%zu written\n", n,
                        got.size());
        }
        std::printf("driver fibonacci schedule written (verified by Python)\n");
    }
    if (bad) {
        std::printf("DRIVER REF FAIL\n");
        return 1;
    }
    std::printf("DRIVER REF PASS\n");
    return 0;
}
