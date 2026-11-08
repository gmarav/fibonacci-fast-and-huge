# block-29: 2^29 two-prime 48-bit engine

An independent large-index engine used mainly for cross-checking block-27 and
block-28. It uses a 2^29 two-prime NTT with 48-bit limbs and at most three
blocks per operand.

- Output: `build/block29_fib.exe` (process-backed engine).
- Tested target: `sm_120`, CUDA 13.0.
- Verified exact: `61,865,563,170`.
- Planner maximum: `73,628,010,691`.
- Fixed device footprint: 22.30 GiB plus context/staging; peak measured VRAM
  22.28 GiB.
- Slower than block-27 at the same index; useful as an independent 48-bit
  implementation.

Build from the repository root with `.\build.ps1`, or locally:

```powershell
.\build.ps1
```

The Rust CLI runs it through `--backend gpu-block29`, `gpu-block`, `gpu-all`,
or `auto`.
