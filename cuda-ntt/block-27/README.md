# block-27: 2^27 blockwise engine

A process-backed CUDA engine for indices above the resident 2^27 limit. It
splits the operands into 2^26-limb blocks and reuses the 2^27 three-prime NTT.

- Output: `build/block27_fib.exe`.
- Tested target: `sm_120`, CUDA 13.0.
- Verified exact: `61,865,563,170`.
- Measured maximum: `86,600,000,000` (hash only).
- No planner cap; device VRAM is the limit. Device model:
  `12.31 GiB + 0.75 GiB * k`, where `k` is the number of 2^26-limb blocks.
- Peak device memory: 19.81 GiB at 61.865e9, 22.81 GiB at 86.6e9.

Build from the repository root with `.\build.ps1`, or locally:

```powershell
.\build.ps1
```

The Rust CLI runs it through `--backend gpu-block27` or `gpu-all`.
