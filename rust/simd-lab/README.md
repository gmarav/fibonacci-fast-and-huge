# fib-simd-lab

`fib-simd-lab` is the experimental CLI that runs the project-local SIMD CPU
algorithms, the original `fib-rust` num-bigint/GMP implementations, and four
optional CUDA backends side by side. Results are always exact integers; no
floating-point value is used to calculate Fibonacci numbers.

```powershell
fib-simd-lab --backend gpu-block27 --method doubling --no-print --json 61865563170
fib-simd-lab --backend auto --method doubling 1000000
fib-simd-lab --show-limits
```

## Backends

| `--backend` | Engine | Implementation | Maximum index |
| --- | --- | --- | --- |
| `gpu` | `GPU/CUDA resident` | `cuda-ntt/resident-27/build/resident27_fib.exe`, process-backed, 2^27 / 3x32 | 6,186,556,317 (hard) |
| `gpu-block27` | `GPU/CUDA block-27` | `cuda-ntt/block-27/build/block27_fib.exe`, process-backed | no planner cap; verified 61,865,563,170, VRAM-bound above |
| `gpu-block28` | `GPU/CUDA block-28` | `cuda-ntt/block-28/build/block28_fib.exe`, process-backed | verified 100,000,000,000; measured 184,000,000,000; planner 184,070,026,875 |
| `gpu-block29` | `GPU/CUDA block-29` | `cuda-ntt/block-29/build/block29_fib.exe`, process-backed | 73,628,010,691 (3 x 2^28 limbs per operand) |
| `gpu-block` | all three block engines | one table row each, run sequentially | per engine |
| `gpu-all` | all four GPU engines | resident + block27 + block28 + block29, one table row each, sequentially | per engine |

The resident `gpu` backend is unchanged. All CPU backends (`cpu`,
`num-bigint`, `gmp`) and the existing combinations (`gmp-gpu`, `both`, `all`)
keep their previous behavior. `auto` selects one CUDA engine from the requested
index and the memory preflight:

1. resident for `n <= 6,186,556,317`;
2. block-27 above that while its VRAM preflight fits;
3. block-28 above block-27's practical maximum, up to `n = 184,070,026,875`;
4. block-29 only as a fallback when block-27 and block-28 do not fit and
   `n <= 73,628,010,691`;
5. reject indices above `184,070,026,875` with a concrete recommendation.

The CLI prints why `auto` chose an engine. It never silently selects an
over-limit engine.

## Backend limits and memory models

`--show-limits` (or the `limits` command) prints a single table. The models are
also used by the startup preflight, `auto`, and JSON output.

### resident 2^27 / 3x32

* Hard limit: `n = 6,186,556,317`; `6,186,556,318` is rejected before the engine
  is started.
* Device model: `108 B/element * required power-of-two transform + 1.30 GiB`
  context. At the cap this is 14.80 GiB, matching the measured 15,157.6 MiB.
* Host model: result limbs plus 256 MiB.
* Limiter: the 2^27 transform and the engine's 32-bit limb-count limit.
* Measured at the cap: 1.712 s device / 7.319 s wall.

### block-27 (A)

* Verified target: `n = 61,865,563,170`, SHA-256
  `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580`.
* No planner/hard cap. The binding resource is **device VRAM**:
  `12.31 GiB fixed + 0.75 GiB * k`, where `k = ceil(limbs / 2^26)` is the number
  of operand blocks. The fixed part includes 12 NTT sets, 6 twiddle tables, the
  accumulator, `limb`/`hi`, and the measured ~1.31 GiB overhead/context.
* Measured points on the 24 GB RTX 5090 Laptop:
  * `k=10` (61.865e9): 19.81 GiB model, 20,289.6 MiB measured;
  * `k=13` (80e9): 22.06 GiB model, 22,593.6 MiB measured;
  * `k=14` (86.6e9): 22.81 GiB model, 23,361.6 MiB measured.
  `k=20` (2x target, 123.73e9) needs roughly 27.3 GiB and cannot fit a 24 GB
  card; this point is a projection, not a run.
* Host model: `3*k*2^26*4 + 2*result + 512 MiB pinned + 1 GiB margin`.
* Limiter: VRAM (0.75 GiB per extra block). The practical maximum is about
  80-87e9 depending on free VRAM; the exact safe point is whatever the
  preflight accepts.

### block-28 (option A)

* Verified anchor: `n = 100,000,000,000`, SHA-256
  `9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0`
  (byte-for-byte GMP reference). The largest measured success is
  `n = 184,000,000,000` (hash only); the planner boundary is
  `n = 184,070,026,875`.
* Device model: 3 u64 arrays of `2^28` (6.00 GiB) + 2 staging buffers of `k`
  `2^27`-limb blocks (1 GiB each) + twiddles + 1.31 GiB context. At 1e11
  (`k=9`) that is 16.31 GiB; at 184e9 (`k=15`) 22.31 GiB. 185e9 needs `k=16`
  (23.31 GiB) and is rejected by the engine preflight.
* Host model: 25.37 GiB at 1e11 (25.33 GiB measured) and 42.49 GiB at 184e9
  (42.30 GiB measured).
* Limiter: device VRAM at 15 x `2^27` limbs per operand. Measured points:
  134.5e9 (18,721.6 MiB), 150e9 (20,769.6 MiB), 164.5e9 (21,793.6 MiB),
  175e9/184e9 (22,817.6 MiB).
* At 61,865,563,170 the engine measured 66.16 s stream / 138.13 s call /
  144.78 s total, matching block-27's stream time but with a larger host-bound
  call wall.

### block-29 (B)

* Planner maximum: `n = 73,628,010,691`; the next index needs 4 x 2^28 limbs
  per operand and is rejected.
* Device model: 5 arrays of `2^29` u64 (20.00 GiB) + 1 GiB staging + twiddles +
  1.30 GiB context = **22.30 GiB**. Measured at the target and at the planner
  maximum: 22,817.6 MiB.
* Host model at the target: **27.08 GiB per product** (`operands 5.04 + packed
  prime-0 segments 9.00 + working 8.00 + one product 5.04`). At the planner
  maximum it is 29.00 GiB. The true peak can be higher because previously
  finished products stay live; the model is the published per-product figure.
* Limiter: the `plan_model.h` 3-block-per-operand planner cap.
* Measured at the target: 72.63 s stream / 172.74 s call / 177.12 s total in
  the published run. The integration rerun measured 82.44 s / 243.46 s /
  251.27 s on the same machine.

## Startup preflight

Before starting a CUDA engine or allocating device memory,
the CLI:

1. prints one compact `GPU preflight` table for the requested `n` with the
   resources in the header and one row per selected engine;
2. estimates required device VRAM and host RAM for every selected engine;
3. queries free VRAM with `nvidia-smi` (no CUDA context is created) and
   available host RAM with `GlobalMemoryStatusEx`;
4. subtracts a 0.5 GiB device and host safety margin;
5. prints a `recommendation:` line only for `--backend auto`, and a `skipped:`
   line only for an infeasible engine.

Example:

```text
GPU preflight n=1,000,000,000 | device 23.57 GiB free of 23.89 GiB | host 46.91 GiB available of 63.43 GiB
BACKEND                   DEVICE NEEDED / FREE          HOST NEEDED / FREE          MAX INDEX  STATUS
GPU/CUDA resident         4.67 GiB / 23.57 GiB        0.29 GiB / 46.91 GiB      6,186,556,317  ✓
GPU/CUDA block-27        13.06 GiB / 23.57 GiB        2.33 GiB / 46.91 GiB     none (~86.6e9)  ✓
GPU/CUDA block-28         2.19 GiB / 23.57 GiB        0.83 GiB / 46.91 GiB    184,070,026,875  ✓
GPU/CUDA block-29         2.61 GiB / 23.57 GiB        0.85 GiB / 46.91 GiB     73,628,010,691  ✓
```

A `✗` in `STATUS` means the engine is skipped (`index`, `vram`, or `host`);
`✓ unverified` marks a run above the verified maximum.

`--show-limits` uses the same tabular layout for verified maxima, hard/planner
maxima, limiter notes, measured speed, and measured limit points.

`--force-memory` overrides the memory check. `--force-index` overrides a
hard/planner index cap. They are separate because forcing an impossible index
still cannot make the engine produce a value.

If `nvidia-smi` or the Win32 call fails, the preflight cannot verify that
resource; it proceeds and says so instead of inventing a number.

## Exactness

* For `n <= 6,186,556,317`, the block engine's output file is compared **limb by
  limb** with a CPU doubling reference. The CLI reuses an already measured CPU
  doubling row when one exists; otherwise it runs GMP doubling once. This is a
  full value comparison, not just a hash.
* Above the resident cap, a CPU reference may not fit (this GMP build has a
  32-bit `mp_bitcnt_t` limit). The CLI reports the output size and SHA-256 and
  leaves `exact` as `null` unless `--gpu-sha256` or `--gpu-reference` is supplied.
* `--gpu-sha256 HEX` compares the full-file SHA-256. Known target values are
  `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` for
  block-27/block-29 at 61,865,563,170, and
  `9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0` for
  block-28 at 100,000,000,000. The block-28 184e9 point has a measured hash
  (`CDD416B4...`) but is not marked exact because no full reference ran.
* `--gpu-reference PATH` compares the full output file byte for byte with
  another canonical limb file (the same `u64 count + little-endian u32 limbs`
  format used by the engines and the GMP reference).
* A huge result is **never** marked exact unless one of those comparisons
  actually ran.

## Output files

All block engines write `u64 limb_count` followed by `limb_count` little-endian
`u32` limbs. The 61,865,563,170 target file is 5,368,708,380 bytes; the block-28
1e11 file is 8,678,023,932 bytes.

* By default a unique temporary file is created, hashed, compared, and deleted.
* `--gpu-out PATH` retains the output at `PATH`.
* `--keep-gpu-output` retains the temporary file instead of deleting it.
* With `gpu-block` (all three engines), a path that is an existing directory or
  ends with `/` or `\` receives `gpu-F-<n>-block-27.bin`,
  `...-block-28.bin`, and `...-block-29.bin`; otherwise `-block-27`,
  `-block-28`, or `-block-29` is appended. Multiple iterations append `-run2`,
  `-run3`, and so on so an existing file is never overwritten.

The full target outputs were kept under `D:\FibReference` during validation.

## CLI flags

| Flag | Behavior |
| --- | --- |
| `--backend` | `cpu`, `num-bigint`, `gmp`, `gpu`, `gpu-block27`, `gpu-block28`, `gpu-block29`, `gpu-block`, `gpu-all`, `gmp-gpu`, `both`, `all`, `auto` |
| `--method` | `direct`, `matrix`, `doubling`, `compare`; CUDA rows only support doubling/compare |
| `--gpu-resident-bin` | resident-27 engine executable |
| `--gpu-block27-bin` | block-27 `block27_fib.exe` (env `FIB_GPU_BLOCK27`) |
| `--gpu-block28-bin` | block-28 `block28_fib.exe` (env `FIB_GPU_BLOCK28`) |
| `--gpu-block29-bin` | block-29 `block29_fib.exe` (env `FIB_GPU_BLOCK29`) |
| `--gpu-out` | retain the full block output |
| `--keep-gpu-output` | keep the temporary block output |
| `--gpu-sha256` | expected full-file SHA-256 |
| `--gpu-reference` | expected full binary file |
| `--force-memory` | ignore the memory preflight |
| `--force-index` | ignore a hard/planner index cap |
| `--show-limits` | print the limits table and exit |
| `--json` | one JSON object on stdout; includes `gpu_preflight` and `skipped` |
| `--no-progress` | disable the live elapsed/ETA line |

Block engines are single-index only; `--minute` remains supported for the
resident GPU and the CPU families. GPU rows require `--method doubling` or
`--method compare`.

## JSON additions

Top-level `gpu_preflight` contains:

```json
{
  "requested_backend": "gpu-block29",
  "auto": {"selected": "gpu-block29", "reason": "..."} ,
  "selected": ["gpu-block29"],
  "skipped": [],
  "engines": [
    {"engine": "gpu-block29", "n": 73628010691,
     "verified_max": 61865563170, "hard_max": 73628010691,
     "required_device_bytes": 23944697107, "required_host_bytes": 31138512896,
     "available_device_bytes": 25309478912, "available_host_bytes": 53776388096,
     "index_allowed": true, "fits_device": true, "fits_host": true,
     "fits": true, "over_verified": true, "skip_reason": null, "note": "..." }
  ],
  "resources": { "device_total_bytes": 25651314688, "device_free_bytes": 25309478912,
                 "host_total_bytes": 68112736256, "host_available_bytes": 53776388096 }
}
```

Block-engine measurements add `engine`, `gpu_stream_ms`, `call_wall_ms`,
`total_wall_ms` when printed, `process_wall_ms`, `peak_vram_mib`,
`host_peak_mib` when printed (block-28), `output_sha256`, `output_bytes`,
`output_kept`, and `reference`. Existing fields and resident `gpu` JSON keep
their previous shape.

## Build and test

From the repository root:

```powershell
.\build.ps1 -Check
```

This builds the CUDA engines, the Rust CLI, and `dist/SHA256SUMS.txt`. For a
CPU-only build use `.\build.ps1 -SkipCuda`. See `docs/BUILD_WINDOWS.md` for
toolchain prerequisites and manual commands.

## Validation results

All runs below were on one RTX 5090 Laptop (24,463 MiB, CUDA 13.0, sm_120) with
one GPU job at a time.

| back end | n | engine stream | call wall | total wall | peak VRAM | output bytes | SHA-256 | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | :---: |
| resident | 6,186,556,317 | 1.712 s device | 2.037 s CLI | — | 15,157.6 MiB | — | known `E2AC...750` | not compared in this run |
| block-27 | 6,186,556,317 | 1.317 s | 16.671 s | 20.697 s | 15,157.6 MiB | 536,870,848 | `E2AC2BC7...750` | yes (CPU/GMP) |
| block-29 | 6,186,556,317 | 3.766 s | 25.527 s | 26.316 s | 12,065.6 MiB | 536,870,848 | `E2AC2BC7...750` | yes (CPU/GMP) |
| block-27 | 61,865,563,170 | 66.059 s | 97.512 s | 134.412 s | 20,289.6 MiB | 5,368,708,380 | `4F5D93B4...E580` | yes (`--gpu-sha256`) |
| block-28 | 61,865,563,170 | 66.16 s | 138.13 s | 144.78 s | 13,601.6 MiB | 5,368,708,380 | `4F5D93B4...E580` | yes (engine GMP reference) |
| block-29 | 61,865,563,170 | 82.436 s | 243.458 s | 251.270 s | 22,817.6 MiB | 5,368,708,380 | `4F5D93B4...E580` | yes (`--gpu-sha256`) |
| block-28 | 100,000,000,000 | 169.07 s | 272.27 s | 281.46 s | 16,673.6 MiB | 8,678,023,932 | `9117AD93...CA0` | yes (GMP reference) |
| block-27 | 80,000,000,000 | 112.178 s | 145.915 s | 195.493 s | 22,593.6 MiB | 6,942,419,148 | `d52c6389...eff` | unverified limit point |
| block-27 | 86,600,000,000 | 121.281 s | 156.760 s | 205.560 s | 23,361.6 MiB | 7,515,168,724 | `f335c793...1da` | unverified limit point |
| block-28 | 184,000,000,000 | 335.92 s | 496.77 s | 515.03 s | 22,817.6 MiB | 15,967,564,024 | `CDD416B4...758` | hash only |
| block-29 | 73,628,010,691 | 82.182 s | 268.368 s | 276.788 s | 22,817.6 MiB | 6,389,456,388 | `3cbaa6b2...f6f` | unverified limit point |

The block-28 rows are the engine's published measurements; the CLI smoke runs
reproduced the 1M and 6,186,556,317 exactness checks. The 80e9/86.6e9/73.628e9
rows are limit points, not independently reproduced values; they are reported
as `exact: null`. The 2x block-27 point (123,731,126,340) was **not** run: its
model is ~27.3 GiB of device memory, above the 24 GB card. No single run
exceeded the requested ~15 minute bound.

## Process-adapter limitations

* The block engines print different fields. block-27 prints a final SHA-256 but
  no total wall time; the adapter uses the CLI's own child-process wall time for
  `total_wall`. block-28 and block-29 print `total_wall_ms`, `gpu_stream_ms`,
  and `peak_vram_used_mib`; block-28 also prints `host_peak_mib`. Neither prints
  a SHA-256, so the adapter always hashes the file itself.
* block-27's output is parsed with simple token matching, not a formal
  machine-readable protocol. A format change requires a parser update. The
  block-28/block-29 space-separated reports ignore unknown keys so new engine
  counters do not break parsing.
* `--iterations > 1` reruns the whole child process and keeps only the last
  value for comparison; the table median is the median child-process wall time.
* The adapter hashes every output file in this process. At the 5.37 GB target
  this adds roughly 35-50 s; it is reported separately from the engine's own
  timing and is excluded from `median_ns` only in the sense that the engine's
  own call time is captured separately in JSON.
* The preflight uses `nvidia-smi`, so its free-VRAM number includes other
  processes but not a CUDA context this process has not created yet. The models
  include the measured engine context overhead.
* Block-29's host model is the published per-product figure; the engine can
  briefly retain earlier products, so its true host peak can be higher. The
  target runs succeeded with roughly 49 GiB available.
