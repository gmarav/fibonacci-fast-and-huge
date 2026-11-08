# Benchmarks

All GPU measurements are on the reference RTX 5090 Laptop. Clock and thermal
state vary on a laptop, so treat close differences as noise. Definitions:

- **stream/device** — CUDA-event time: kernels plus H2D/D2H copies on the engine
  stream; excludes host CRT/carry and file I/O.
- **call wall** — the engine/library call, including host CRT/carry.
- **total wall** — the process, including output file writing and, where noted,
  SHA-256.

## Current process protocol (one process per method)

The current tree starts each selected engine once per method and runs all
`--iterations` inside that process. The reported **median is the engine call
median**, not process wall time: process startup, preflight, and the one-time
output-file path are excluded. `total` below is still the engine-reported total
for the run that produced stream/call, so it is not the same quantity as the
process totals in the historical tables.

The measurements below were taken on 2026-11-08 from the current tree, after
the shared-kernel and shared-limb-math refactors. Each point is the median of
three independent runs, and each run used `--iterations 5`. The median column
is the median of the three run medians; stream and total are medians across the
three runs. The exact command was:

```text
.\run.ps1 --backend gpu-all --method doubling --iterations 5 --no-print --no-banner --no-hwinfo --no-color --no-progress --json --gpu-sha256 <expected> <n>
```

| engine | n | median | stream | call | total | peak VRAM | peak host |
|---|---:|---:|---:|---:|---:|---:|---:|
| resident | 1,000,000 | 0.900 ms | 0.844 ms | 0.900 ms | 0.900 ms | n/a | n/a |
| block-27 | 1,000,000 | 3.353 ms | 1.812 ms | 3.353 ms | 3.353 ms | 1,315.6 MiB | n/a |
| block-28 | 1,000,000 | 7.781 ms | 2.184 ms | 7.781 ms | 8.039 ms | 1,313.6 MiB | 125.0 MiB |
| block-29 | 1,000,000 | 9.164 ms | 2.982 ms | 9.164 ms | 9.446 ms | 1,313.6 MiB | n/a |
| resident | 1,000,000,000 | 258.359 ms | 222.249 ms | 258.359 ms | 258.359 ms | n/a | n/a |
| block-27 | 1,000,000,000 | 722.370 ms | 257.198 ms | 722.370 ms | 722.370 ms | 4,787.6 MiB | n/a |
| block-28 | 1,000,000,000 | 1,664.471 ms | 464.485 ms | 1,664.471 ms | 1,727.349 ms | 2,209.6 MiB | 1,058.0 MiB |
| block-29 | 1,000,000,000 | 3,112.818 ms | 759.464 ms | 3,112.818 ms | 3,194.599 ms | 2,657.6 MiB | n/a |

Notes:

- `total` is reported by the engine and, in this protocol, is close to `call`;
  the historical process totals below include startup and SHA-256.
- `n/a` means the engine does not report that peak (the resident engine reports
  neither peak).
- All runs produced the expected SHA-256 for all four engines:
  `9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6` at
  `n=1,000,000` (86,792 output bytes, 694,241 bits) and
  `D57068140E4974AB4FE1F8847DA629A798C8DAC02AEF517E8C0711AFAF884B14` at
  `n=1,000,000,000` (86,780,248 output bytes, 694,241,913 bits).
- Run-to-run medians varied by up to about 10% at `n=1e9` (for example,
  block-27 produced 662/722/726 ms). This is laptop clock and thermal behavior,
  not a change in the algorithm.
- These points are far below every backend's measured maximum, so they do not
  change the limits in `LIMITS.md`.

## Current process protocol: large block-28 points

Both large points were re-run on the current build on 2026-11-08 with one
iteration per process and the recorded SHA-256. The GPU hash matches the
canonical value; this re-run did not repeat the independent GMP comparison.

| n | median | stream | call | total | peak VRAM | peak host | SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100,000,000,000 | 249,346.422 ms | 159,450.695 ms | 249,346.422 ms | 257,882.027 ms | 16,673.6 MiB | 25,938 MiB | `9117AD93...9CA0` |
| 184,000,000,000 | 474,392.602 ms | 344,142.011 ms | 474,392.602 ms | 489,670.145 ms | 22,817.6 MiB | 43,318 MiB | `CDD416B4...C758` |

The `1e11` point is the exactness anchor (GPU output equals the anchored
GMP/limb-array reference). The `184e9` point is the measured maximum and remains
hash-only; the re-run confirms that the current build reproduces its recorded
SHA-256.

## Portable target: RTX 3080 Laptop 16 GB / Ryzen 9 5900HX

Hardware and software:

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 5900HX (Zen 3), 8 cores / 16 threads |
| SIMD | AVX2, selected at runtime (Zen 3 has no AVX-512) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 3080 Laptop, 16 GB |
| Driver | CUDA 13.0-capable NVIDIA driver |
| OS | Windows 11, 10.0.26200 |
| PowerShell | 7.6.9 |
| Samples | 3 iterations per point |

The engines were built for `sm_86`, `sm_89`, `sm_90`, and `sm_120`; the RTX
3080 Laptop uses `sm_86`. GPU execution succeeded, which confirms that the
installed driver supports CUDA 13.0. The exact driver version was not captured
by this run.

CPU and small GPU points (median milliseconds, lower is better):

| n | CPU/SIMD | num-bigint | GMP | GPU resident | block-27 | block-28 | block-29 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 19.138 | 7.837 | 2.732 | 1.102 | 5.913 | 11.774 | 15.842 |
| 10,000,000 | 325.608 | 97.023 | 25.692 | 2.682 | 8.766 | 53.720 | 95.028 |
| 100,000,000 | 11,598.866 | 2,823.108 | 408.019 | 37.879 | 78.109 | 266.835 | 439.737 |
| 1,000,000,000 | - | - | - | 359.182 | 667.684 | 1,764.910 | 2,951.878 |

Large GPU points on the same machine:

| n | engine | stream | call | total | peak VRAM | host peak |
|---:|---|---:|---:|---:|---:|---:|
| 6,186,556,317 | resident | 2,446.567 ms | 2,540.561 ms | 2,540.561 ms | n/a | n/a |
| 6,186,556,317 | block-27 | 2,008.392 ms | 3,134.576 ms | 3,134.576 ms | 14,966.5 MiB | n/a |
| 6,186,556,317 | block-28 | 4,955.279 ms | 12,062.220 ms | 12,587.009 ms | 8,286.5 MiB | 5,242 MiB |
| 6,186,556,317 | block-29 | 5,576.519 ms | 14,836.846 ms | 15,402.554 ms | 11,870.5 MiB | n/a |
| 61,865,563,170 | block-28 | 91,403.128 ms | 138,249.522 ms | 142,496.026 ms | 13,408.7 MiB | 23,163 MiB |

Verified hashes at every point above:

| n | SHA-256 |
|---:|---|
| 1,000,000 | `9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6` |
| 10,000,000 | `667F558207A611BEFE401972E369E378391206EBB25BE1BBEA4D52B35E0919AC` |
| 100,000,000 | `9D69F2E524D1C955D39D0AFDC8EA3F1EAC09FA0AFAE5C0256FF6935FB0387CC9` |
| 1,000,000,000 | `D57068140E4974AB4FE1F8847DA629A798C8DAC02AEF517E8C0711AFAF884B14` |
| 6,186,556,317 | `E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750` |
| 61,865,563,170 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` |

All four GPU engines produced the same hash at each shared point. The
`61.865e9` block-28 point also matches the canonical GPU == GMP anchor recorded
in `HASHES.md`. The 16 GB card cannot run `1e11` or `184e9`: block-28 needs more
than 16 GiB of VRAM at those indices, so those points remain measured on the
24 GB reference machine.

## GPU model comparison: RTX 3080 Laptop vs RTX 5090 Laptop

Both cards are mobile GPUs; the numbers below are pure **stream/device** time
so they do not include host CRT/carry. The 3080 runs on the Ryzen 9 5900HX
target; the 5090 runs on the Intel Core Ultra 9 275HX reference machine. The
different host CPUs make `call`/`total` unsuitable for a pure GPU comparison.
The ratio is `3080 stream / 5090 stream`; values above 1 mean the 5090 is
faster.

| n | resident | block-27 | block-28 | block-29 |
|---:|---|---|---|---|
| 1e6 | 0.873 / 0.844 ms (1.03x) | 2.653 / 1.812 ms (1.46x) | 2.864 / 2.184 ms (1.31x) | 4.668 / 2.982 ms (1.57x) |
| 1e7 | 2.310 / 1.260 ms (1.83x) | 4.678 / 2.950 ms (1.59x) | 12.405 / 10.060 ms (1.23x) | 17.298 / 12.200 ms (1.42x) |
| 1e8 | 35.161 / 19.490 ms (1.80x) | 47.487 / 19.210 ms (2.47x) | 102.432 / 57.290 ms (1.79x) | 152.850 / 111.100 ms (1.38x) |
| 1e9 | 348.630 / 222.249 ms (1.57x) | 426.949 / 257.198 ms (1.66x) | 790.293 / 464.485 ms (1.70x) | 1,200.282 / 759.464 ms (1.58x) |
| 6.186e9 | 2.447 / 1.663 s (1.47x) | 2.008 / 1.316 s (1.53x) | 4.955 / 3.320 s (1.49x) | 5.577 / 3.663 s (1.52x) |
| 61.865e9 | - | - | 91.403 / 66.465 s (1.38x) | - |

The 5090 is roughly 1.3-2.5x faster in GPU time. The small `1e6` resident point
is overhead-dominated and nearly equal. The `61.865e9` block-28 point is
host-heavy: its end-to-end call time was `138.250 s` on the 3080 and
`128.421 s` on the 5090, only about 1.08x, even though the GPU stream time
differs by 1.38x.

Measurement caveat: the 3080 points use 3 samples. The 5090 `1e6`/`1e9`
points use three runs of 5 samples; `1e7`/`1e8` use one 5-sample run; the
`6.186e9` point uses 3 samples; `61.865e9` uses 1 sample.

## Historical measurements (earlier transport)

> The GPU tables below were recorded with the earlier in-process resident
> transport (one process per block iteration). They are retained unchanged as
> historical data. Under that transport the table median was the per-process
> block median, so startup and per-process overhead were included.

## Resident 2^27 (3x32), verified points

| n | device | wall | max NTT |
|---:|---:|---:|---:|
| 1,048,576 | 3.559 ms | 112.7 ms | 2^15 |
| 33,554,432 | 7.569 ms | 179.4 ms | 2^20 |
| 1,024,000,000 | 244.1 ms | 1.567 s | 2^25 |
| 2,048,000,000 | 514.8 ms | 3.114 s | 2^26 |
| 4,096,000,000 | 1.100 s | 6.302 s | 2^27 |
| 6,186,556,317 | 1.712 s | 7.319 s | 2^27 |

## block27 (2^27, 3x32)

| n | stream | call | total | peak VRAM | output bytes | exactness |
|---:|---:|---:|---:|---:|---:|---|
| 6,186,556,317 | 1.317 s | 16.671 s | 20.697 s | 15,157.6 MiB | 536,870,848 | exact |
| 61,865,563,170 | 66.059 s | 97.512 s | 134.412 s | 20,289.6 MiB | 5,368,708,380 | exact |
| 80,000,000,000 | 112.178 s | 145.915 s | 195.493 s | 22,593.6 MiB | 6,942,419,148 | hash only |
| 86,600,000,000 | 121.281 s | 156.760 s | 205.560 s | 23,361.6 MiB | 7,515,168,724 | hash only |

## block29 (2^29, 2x48)

| n | stream | call | total | peak VRAM | output bytes | exactness |
|---:|---:|---:|---:|---:|---:|---|
| 6,186,556,317 | 3.766 s | 25.527 s | 26.316 s | 12,065.6 MiB | 536,870,848 | exact |
| 61,865,563,170 | 82.436 s | 243.458 s | 251.270 s | 22,817.6 MiB | 5,368,708,380 | exact |
| 73,628,010,691 | 82.182 s | 268.368 s | 276.788 s | 22,817.6 MiB | 6,389,456,388 | hash only |

## block28 (2^28, 2x48, option A)

Verified points:

| n | stream | call | total | peak VRAM | host peak | exactness |
|---:|---:|---:|---:|---:|---:|---|
| 1,048,576 | 0.0039 s | 0.0181 s | 0.0198 s | 1,313.6 MiB | 127 MiB | GMP exact |
| 33,554,432 | 0.0423 s | 0.1638 s | 0.1654 s | 1,341.6 MiB | 157 MiB | GMP exact |
| 1,024,000,000 | 0.4634 s | 1.6740 s | 1.7329 s | 2,209.6 MiB | 981 MiB | GMP exact |
| 2,048,000,000 | 1.0634 s | 3.6370 s | 3.8285 s | 3,105.6 MiB | 1,833 MiB | GMP exact |
| 4,096,000,000 | 2.2252 s | 8.6682 s | 9.1504 s | 4,897.6 MiB | 3,537 MiB | GMP exact |
| 6,186,556,317 | 3.4722 s | 17.3328 s | 17.9285 s | 8,481.6 MiB | 4,738 MiB | GMP exact |
| 61,865,563,170 | 66.165 s | 138.127 s | 144.782 s | 13,601.6 MiB | 18,050 MiB | GMP exact |
| **100,000,000,000** | **169.071 s** | **272.266 s** | **281.457 s** | **16,673.6 MiB** | **25,941 MiB** | **exact (GPU == GMP)** |

Limit points:

| n | stream | call | total | peak VRAM | host peak | output bytes | exactness |
|---:|---:|---:|---:|---:|---:|---:|---|
| 134,500,000,000 | 210.465 s | 346.087 s | 361.326 s | 18,721.6 MiB | 33,077 MiB | 11,671,942,184 | hash only |
| 150,000,000,000 | 275.546 s | 417.054 s | 431.021 s | 20,769.6 MiB | 36,284 MiB | 13,017,035,892 | hash only |
| 164,500,000,000 | 298.742 s | 458.166 s | 474.416 s | 21,793.6 MiB | 39,284 MiB | 14,275,349,360 | hash only |
| 175,000,000,000 | 338.524 s | 494.383 s | 510.759 s | 22,817.6 MiB | 41,457 MiB | 15,186,541,872 | hash only |
| **184,000,000,000** | **335.925 s** | **496.770 s** | **515.033 s** | **22,817.6 MiB** | **43,319 MiB** | **15,967,564,024** | **measured max, hash only** |
| 185,000,000,000 | — | — | — | — | — | — | preflight reject (k=16) |

The block28 1e11 target is the exactness anchor; its GPU SHA-256 and the
anchored GMP SHA-256 are identical (`9117AD93...9CA0`). The 1e11 timing is
included above; the limit points are the additional measured points.

## Head-to-head at 61,865,563,170

| metric | block27 | block29 | winner |
|---|---:|---:|---|
| stream | 66.06 s | 82.44 s | block27 (1.25x) |
| call wall | 97.51 s | 243.46 s | block27 (2.50x) |
| total | 134.41 s | 251.27 s | block27 (1.87x) |
| peak VRAM | 20,289.6 MiB | 22,817.6 MiB | block27 (2.5 GiB less) |

At the measured limits:

| n | backend | stream | call | total | peak VRAM |
|---:|---|---:|---:|---:|---:|
| 86.6e9 | block27 | 121.28 s | 156.76 s | 205.56 s | 23,361.6 MiB |
| 73.628e9 | block29 | 82.18 s | 268.37 s | 276.79 s | 22,817.6 MiB |
| 184e9 | block28 | 335.92 s | 496.77 s | 515.03 s | 22,817.6 MiB |

The block27 and block29 limit points are not independently verified beyond the
recorded SHA-256. The block28 `1e11` point is the verified exact target.

## Verification and repeatability

- `1e11`: GPU and anchored GMP/limb-array reference produced the same
  `8,678,023,932` bytes and the same SHA-256.
- block27/block29 intermediate points through 6.186e9 were compared
  byte-for-byte with GMP.
- The 184e9 block28 run used `k=15`, `22,817.6 MiB` peak VRAM, and
  `43,319 MiB` host peak; `185e9` was rejected by the device preflight.
- On 2026-11-08 the current build reproduced the recorded GPU SHA-256 at both
  `1e11` and `184e9`; see the current large-point table above. This was a
  GPU-only re-run, not a repeated GMP comparison.
