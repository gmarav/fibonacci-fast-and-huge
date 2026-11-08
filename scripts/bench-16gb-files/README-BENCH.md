# Portable Fibonacci benchmark

Target machine: **RTX 3080 Laptop 16 GB, AMD Ryzen 9 5900HX, 32 GB RAM,
Windows 11 x64**.

The CUDA engines in this folder are built for `sm_86`, `sm_89`, `sm_90`, and
`sm_120`, so the RTX 3080 Laptop (`sm_86`) is covered. No CUDA toolkit is
required on the target machine, but the NVIDIA driver must support **CUDA
13.0** (driver branch 580 or newer). The benchmark reads the CUDA version from
`nvidia-smi` and skips the GPU points with an English warning if the driver
reports CUDA 12.x or older; use `-ForceGpu` only for a matching custom build.

## Quick start

Double-click or run:

```bat
run-bench.cmd
```

`run-bench.cmd` prefers **PowerShell 7** (`pwsh.exe`) and falls back to Windows
PowerShell 5.1 only if PowerShell 7 is not installed. To call PowerShell
directly:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\bench.ps1
```

Default profile:

- 3 samples for every CPU and GPU point;
- CPU/SIMD, num-bigint, and GMP at `1e6`, `1e7`, and `1e8`;
- all four GPU engines at `1e6`, `1e7`, `1e8`, and `1e9`;
- resident and block engines at `6.186e9` when the free VRAM allows;
- block-28 at `61.865e9` with the canonical SHA-256 check.

The full default run takes roughly 15-25 minutes on the target machine. Use
`-SkipLarge` or `-LargeIterations 1` when a shorter run is enough.

Useful variants:

```powershell
.\bench.ps1 -SkipLarge              # only 1e6..1e9, faster
.\bench.ps1 -Iterations 2           # fewer samples
.\bench.ps1 -SkipGpu                # CPU-only
.\bench.ps1 -SkipCpu                # GPU-only
.\bench.ps1 -LargeIterations 1      # one sample for large GPU points
```

## What fits in 16 GB VRAM

| point | fit | note |
|---|---|---|
| all four engines at `1e6`..`1e9` | yes | small footprint |
| block engines at `6.186e9` | yes | block-27/28/29 fit |
| resident at `6.186e9` | usually | needs about 14.8 GiB; the CLI may skip it if Windows is using VRAM |
| block-28 at `61.865e9` | yes | about 13.3 GiB VRAM and 18 GiB host RAM |
| block-27 at `61.865e9` | no | about 19.8 GiB VRAM |
| block-29 at `61.865e9` | no | about 22.3 GiB VRAM |
| `1e11` / `184e9` | no | needs more than 16 GB VRAM |

The CLI preflight is authoritative. A skipped engine is recorded in
`summary.csv` and shown as a dash in `report.md`; it is not a crash.

## CPU/SIMD on Ryzen 9 5900HX

The Rust CLI selects the SIMD kernel at runtime. Zen 3 supports AVX2 but not
AVX-512, so the expected `kernel` value is `avx2`. The binary does not require
AVX-512. `CPU/SIMD` in the report is the project-local integer; `num-bigint`
and `GMP` are the two CPU reference families.

CPU/SIMD at `1e9` is deliberately not part of this profile: it is much slower
than the big-integer libraries and would dominate the run time. `1e8` is the
CPU ceiling; `1e9` and above are measured on the GPU.

## Results

The script creates `bench-results\<timestamp>\` next to itself:

- `report.md` — combined CPU/SIMD/GPU table with speedups;
- `summary.csv` — all raw measurements;
- `summary.md` — per-point detailed table;
- `environment.txt` — CPU, GPU, RAM, OS, and binary SHA-256 values;
- one JSON and one stderr file per run.

Copy the whole `bench-results\<timestamp>` folder back for comparison.

## SHA-256 anchors

```text
1e6      9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6
1e9      D57068140E4974AB4FE1F8847DA629A798C8DAC02AEF517E8C0711AFAF884B14
6.186e9  E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750
61.865e9 4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580
```

A non-zero exit code with a hash mismatch means the engine or the driver
produced a wrong result; keep the JSON/stderr files.
