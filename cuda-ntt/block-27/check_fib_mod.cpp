// Cross-check the anchored Fibonacci limb file modulo several moduli.
// Reads the canonical limb file (u64 count + LE u32 limbs) in chunks and
// compares the residue with an independent fast-doubling computation.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned __int128 u128;

struct ModInfo {
    const char* name;
    u64 mod; // 0 means 2^64 (natural wraparound)
};

// (F(n), F(n+1)) mod m, m odd prime or 0 for 2^64.
static void fib_pair_mod(u64 n, u64 m, u64& f, u64& f1) {
    u64 a = 0, b = (m ? 1 % m : 1);
    if (n == 0) { f = a; f1 = b; return; }
    for (int i = 63; i >= 0; --i) {
        u64 c, d;
        if (m == 0) {
            c = (u64)((u128)a * (2 * b - a));
            d = (u64)((u128)a * a + (u128)b * b);
        } else {
            c = (u128)a * ((2 * b % m + m - a) % m) % m;
            d = ((u128)a * a + (u128)b * b) % m;
        }
        if ((n >> i) & 1ull) {
            a = d;
            b = m ? (c + d) % m : c + d;
        } else {
            a = c;
            b = d;
        }
    }
    f = a;
    f1 = b;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s FILE N\n", argv[0]);
        return 2;
    }
    const u64 n = std::strtoull(argv[2], nullptr, 0);
    FILE* fp = std::fopen(argv[1], "rb");
    if (!fp) { std::perror("open"); return 1; }
    u64 count = 0;
    if (std::fread(&count, 8, 1, fp) != 1) return 1;

    const ModInfo mods[] = {
        {"2^64", 0},
        {"1000000007", 1000000007ull},
        {"998244353", 998244353ull},
        {"2147483647", 2147483647ull},
        {"2305843009213693951", 2305843009213693951ull}, // 2^61-1
        {"1000000000000000003", 1000000000000000003ull},
    };
    const int nm = sizeof(mods) / sizeof(mods[0]);
    std::vector<u64> acc(nm, 0), pw(nm, 1);

    const size_t chunk = 1 << 20;
    std::vector<u32> buf(chunk);
    u64 done = 0;
    u32 top_limb = 0;
    while (done < count) {
        const size_t take = (size_t)std::min<u64>(chunk, count - done);
        if (std::fread(buf.data(), 4, take, fp) != take) return 1;
        for (int k = 0; k < nm; ++k) {
            const u64 m = mods[k].mod;
            u64 a = acc[k];
            u64 p = pw[k];
            if (m == 0) {
                for (size_t i = 0; i < take; ++i) {
                    a += (u64)buf[i] * p;
                    p *= 4294967296ull;
                }
            } else {
                for (size_t i = 0; i < take; ++i) {
                    a += (u128)buf[i] * p % m;
                    if (a >= m) a -= m;
                    p = (u128)p * 4294967296ull % m;
                }
            }
            acc[k] = a;
            pw[k] = p;
        }
        top_limb = buf[take - 1];
        done += take;
    }
    std::fclose(fp);

    int bad = 0;
    for (int k = 0; k < nm; ++k) {
        u64 want = 0, dummy = 0;
        fib_pair_mod(n, mods[k].mod, want, dummy);
        const bool ok = (acc[k] == want);
        std::printf("%-20s file=%20llu  want=%20llu  %s\n", mods[k].name,
                    (unsigned long long)acc[k], (unsigned long long)want,
                    ok ? "OK" : "MISMATCH");
        if (!ok) ++bad;
    }
    std::printf("count=%llu top_limb=%08x\n", (unsigned long long)count,
                top_limb);
    return bad == 0 ? 0 : 1;
}
