# Verification

## Canonical file format

Every large result file is:

```text
u64 limb_count (little-endian)
limb_count * u32 limbs (little-endian)
```

A file is exact only when its full bytes match an independent computation.
Bit length alone is not sufficient.

## Verified target: F(100,000,000,000)

```text
bits       69,424,191,362
bytes      8,678,023,932
limbs      2,169,505,981
SHA-256    9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0
```

Independent paths:

1. GPU block28 (2^28, 2x48, option A) output.
2. Anchored GMP/limb-array reference:
   - anchor pair F(6,186,556,317), F(6,186,556,318) from GMP;
   - four doublings M -> 2M -> 4M -> 8M -> 16M;
   - `r = 1,015,098,928`;
   - `F(16M+r) = F(16M) F(r+1) + F(16M-1) F(r)`.
3. `check_fib_mod` over the full file for moduli
   `2^64`, `1,000,000,007`, `998,244,353`, `2^31-1`, `2^61-1`,
   `1,000,000,000,000,000,003`.

The full values are not committed because each is 8.68 GB. Generate a local
copy with `--gpu-out` and compare its SHA-256 with the canonical value above.


Verify on Windows:

```powershell
Get-FileHash path\to\gpu-block28-F-100000000000.bin -Algorithm SHA256
Get-FileHash path\to\gmp-block28-F-100000000000.bin -Algorithm SHA256
```

The full limb files are not committed; their size and SHA-256 are recorded in
[`HASHES.md`](HASHES.md).

## Other verified points

`resident`, `block27`, and `block29` through `F(6,186,556,317)` were compared
byte-for-byte with GMP. Hashes are listed in
[LARGE_RESULTS.md](LARGE_RESULTS.md) and [`HASHES.md`](HASHES.md).

## Limit points

The block27 `80e9`/`86.6e9`, block29 `73.628e9`, and block28
`134.5e9`..`184e9` points are successful runs with recorded SHA-256 values,
but they were not independently reproduced with GMP. They are labelled
"hash only" everywhere. The block28 `1e11` point is the fully verified target.

## What "exact" means here

- `exact` — full byte comparison against an independent exact computation.
- `hash` — a complete run and SHA-256, but no independent reference comparison.
- `preflight reject` — the engine refused before allocating, because the
  requested index did not fit the memory model.
