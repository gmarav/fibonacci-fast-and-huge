# resident-27: 2^27 GPU-resident fast doubling

A CUDA fast-doubling engine that keeps the full state on the device. It uses a
2^27 three-prime NTT with 3x32-bit limbs (`p < 2^32`) and five transforms per
prime per doubling step.

- Output: `build/resident27_fib.exe` (process-backed engine).
- Tested target: `sm_120`, CUDA 13.0.
- Exact range: `n <= 6,186,556,317`; `n = 6,186,556,318` is rejected.
- Peak device memory at the maximum: 14.80 GiB.
- The Rust CLI starts the executable through `--backend gpu`.

Build from the repository root with `.\build.ps1`, or locally:

```powershell
.\build.ps1
```

The engine was verified against GMP limb-for-limb through `F(6,186,556,317)`.
See the top-level README and `docs/HASHES.md` for hashes.
