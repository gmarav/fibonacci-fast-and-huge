# block-28: 2^28 option A

The preferred large-index engine. It uses a 2^28 two-prime NTT with 48-bit
limbs and `2^27`-limb blocks (option A).

- Output: `build/block28_fib.exe`.
- Tested target: `sm_120`, CUDA 13.0.
- Verified exact: `100,000,000,000` (GPU and anchored GMP SHA-256 identical).
- Measured maximum: `184,000,000,000` (hash only).
- Planner boundary: `184,070,026,875` (`k=15`). `185e9` needs `k=16` and is
  rejected by the engine preflight.
- Peak device memory: 16.28 GiB at 1e11, 22.28 GiB at 184e9.
- Host peak: 25,941 MiB at 1e11, 43,319 MiB at 184e9.

Build from the repository root with `.\build.ps1`, or locally:

```powershell
.\build.ps1
```

The Rust CLI runs it through `--backend gpu-block28`, `gpu-block`, `gpu-all`,
or `auto`. The canonical `1e11` SHA-256 is
`9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0`.
