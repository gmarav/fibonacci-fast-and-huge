# Backend limits on this machine

Reference configuration:

```text
CPU     Intel Core Ultra 9 275HX, 24 physical / 24 logical cores
RAM     63.4 GB total (~49 GB free during the measurements)
GPU     NVIDIA GeForce RTX 5090 Laptop, 24,462.6 MiB (23.89 GiB)
Driver  610.47
CUDA    13.0, compute capability 12.0 (sm_120)
OS      Windows, WDDM
```

A backend can be limited by the NTT length, by the block planner, by device
VRAM, or by host RAM. The table below shows the measured/verified points and
the current practical ceiling.

| Backend | Verified exact max | Measured max | Hard / planner max | Device at verified/max | Host at verified/max | Limiter |
|---|---:|---:|---:|---|---|---|
| `gpu` resident 2^27, 3x32 | 6,186,556,317 | 6,186,556,317 | 6,186,556,317 | 15,157.6 MiB | ~0.5 GiB | NTT 2^27 |
| `gpu-block27` 2^27, 3x32 blockwise | 61,865,563,170 | 86,600,000,000 | none (VRAM-bound) | 19.81 GiB / 22.81 GiB | ~14 / ~19 GiB | device VRAM |
| `gpu-block29` 2^29, 2x48 | 61,865,563,170 | 73,628,010,691 | 73,628,010,691 | 22.30 GiB | 27.08 GiB/product | 3+3 block planner |
| `gpu-block28` 2^28, 2x48, option A | **100,000,000,000** | **184,000,000,000** | **184,070,026,875** | 16,673.6 MiB / 22,817.6 MiB | 25,941 / 43,319 MiB | device VRAM (k=15 boundary) |

Notes:

- `184,070,026,875` is the conservative block28 planner boundary for `k=15`;
  `185,000,000,000` is `k=16` and is rejected by the engine's preflight.
- `200,000,000,000` is `k=17` and cannot fit the 24 GB card.
- block27's practical ceiling on this machine is the VRAM model
  `12.31 GiB + 0.75 GiB * k`; `86.6e9` is the measured `k=14` point. It is
  hash-only: the exact/GMP-verified maximum remains `61,865,563,170`.
- block29 is not preferred for any index that block27 can run: it is slower,
  uses 22.30 GiB at every n, and stops at 73.628e9. It remains useful as an
  independent 48-bit engine for cross-checking.

## Automatic recommendation order

```text
resident   : n <= 6,186,556,317
block27    : 6,186,556,317 < n <= practical VRAM limit (~86.6e9)
block28    : above block27's practical limit, up to 184,070,026,875
block29    : only if block27/block28 do not fit and n <= 73,628,010,691
reject     : n > 184,070,026,875
```

The CLI checks free VRAM and available host RAM before starting an engine and
prints one `GPU preflight` table with a row per selected engine; only skipped
or forced engines get an additional note. `--force-memory` / `--force-index`
override the check explicitly.

## Memory models used by the CLI

Resident:

```text
device = 108 B/element * 2^lg + ~1.30 GiB context
```

block27:

```text
device = 12.31 GiB + 0.75 GiB * k + context
host   = 3 block buffers + results + staging + margin
```

block29:

```text
device = 5 * 2^29 * 8 + staging + twiddles + ~1.30 GiB context = 22.30 GiB
```

block28 option A:

```text
device = 3 * 2^28 * 8 (NTT) + 2 * k * 2^27 * 4 (block buffers) + context/staging
host   = operands + result + CRT/carry workspace
```

Representative block28 values:

| n | k | device (measured) | host (measured) |
|---:|---:|---:|---:|
| 100,000,000,000 | 9 | **16.28 GiB** | **25.33 GiB** |
| 134,500,000,000 | 11 | **18.28 GiB** | **32.30 GiB** |
| 164,500,000,000 | 13 | **21.28 GiB** | **38.36 GiB** |
| 175,000,000,000 | 15 | **22.28 GiB** | **40.49 GiB** |
| 184,000,000,000 | 15 | **22.28 GiB** | **42.30 GiB** |

The block28 plan model gives 16.31 GiB device / 25.37 GiB host at 1e11 and
22.31 GiB / 42.49 GiB at 184e9; the measured values above are close and are
what the current `--show-limits` reports.

All numbers are for the configuration above. On a card with less VRAM or a
machine with less RAM, the practical maxima are lower; the preflight check is
authoritative for the current free resources.
