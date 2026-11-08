# Building on Windows

## Prerequisites

- Windows 10/11 x64.
- Rust GNU toolchain `1.99.1-x86_64-pc-windows-gnu`.
- MSYS2/MinGW `gcc`, `make`, `m4`; `gmp-mpfr-sys` builds GMP from source.
- CUDA 13.0 with `nvcc`, and Visual Studio Build Tools with the C++ workload
  (`vcvars64.bat`).
- NVIDIA GPU matching the build architecture. The scripts default to `sm_120`
  (RTX 50 series); pass `-CudaArch sm_89` (or another value) for a different
  target.

The build script looks for MSYS2 under `MSYS2_ROOT`, `C:\msys64`, or
`.tools\msys64` in the repository. Put `gcc`/`make`/`m4` on `PATH` if they live
elsewhere.

## One-command build

From the repository root:

```powershell
.\build.ps1
```

Options:

```powershell
.\build.ps1 -SkipCuda     # CPU/SIMD, num-bigint, GMP only
.\build.ps1 -SkipRust     # CUDA engines only
.\build.ps1 -Check        # fmt + clippy + tests before the release build
.\build.ps1 -CudaArch sm_89
```

Artifacts:

```text
dist\fib-simd-lab.exe
cuda-ntt\resident-27\build\resident27_fib.exe
cuda-ntt\block-27\build\block27_fib.exe
cuda-ntt\block-28\build\block28_fib.exe
cuda-ntt\block-29\build\block29_fib.exe
dist\SHA256SUMS.txt
```

`dist\` and `cuda-ntt\*\build\` are local build outputs and are ignored by git.

## Manual Rust build

```powershell
$env:CARGO_TARGET_DIR = "build\rust"
rustup run 1.99.1-x86_64-pc-windows-gnu cargo build --release `
  --manifest-path rust\simd-lab\Cargo.toml --locked
```

The executable is `build\rust\release\fib-simd-lab.exe`. It links GMP
statically and imports only Windows system DLLs.

With tests and lints:

```powershell
rustup run 1.99.1-x86_64-pc-windows-gnu cargo fmt --manifest-path rust\simd-lab\Cargo.toml -- --check
rustup run 1.99.1-x86_64-pc-windows-gnu cargo clippy --release --all-targets `
  --manifest-path rust\simd-lab\Cargo.toml --locked -- -D warnings
rustup run 1.99.1-x86_64-pc-windows-gnu cargo test --release `
  --manifest-path rust\simd-lab\Cargo.toml --locked
```

## Manual CUDA build

Each engine has its own `build.ps1`, but the top-level script invokes the same
`nvcc` commands directly so the GMP oracles are not required:

```powershell
# resident-27
Push-Location cuda-ntt\resident-27
cmd /c "`"$vcvars`" >nul && nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler resident27.cu resident27_fib.cpp -o build\resident27_fib.exe"

# block-27
Push-Location cuda-ntt\block-27
cmd /c "`"$vcvars`" >nul && nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler block27_engine.cu block27_fib.cpp -o build\block27_fib.exe"

# block-28
Push-Location cuda-ntt\block-28
cmd /c "`"$vcvars`" >nul && nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler block28_engine.cu block28_fib.cpp -o build\block28_fib.exe psapi.lib"

# block-29
Push-Location cuda-ntt\block-29
cmd /c "`"$vcvars`" >nul && nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler block29_engine.cu block29_fib.cpp -o build\block29_fib.exe"
```

The CLI discovers the artifacts relative to the repository root. Override:

```text
--gpu-resident-bin PATH
--gpu-block27-bin PATH
--gpu-block28-bin PATH
--gpu-block29-bin PATH
```

## Quick run

```powershell
.\run.ps1 --show-limits
.\run.ps1 --backend cpu --method doubling --iterations 3 --no-print 1000000000
.\run.ps1 --backend gpu-all --method doubling --iterations 1 --no-print 1000000000
```

`run.ps1` changes to the repository root so relative engine discovery works.

## Local checks

```powershell
.\scripts\lint.ps1                       # Rust, host C++, and CUDA analysis
.\scripts\lint.ps1 -SkipRust -SkipCuda   # host C++ only
.\scripts\host-tests.ps1                 # CPU reference tests, no GPU
.\scripts\engine-smoke.ps1               # GPU smoke test at n=1e6
```

`build.ps1 -Check` runs `lint.ps1` (host C++, plus CUDA unless `-SkipCuda`) and
the Rust formatting, clippy, and test checks before building. The CUDA analysis
step needs `nvcc` and `vcvars64.bat`; it is skipped with a message when either
is missing.

## Portable benchmark package

Build engines for more than one GPU architecture when the package must run on a
different machine:

```powershell
.\build.ps1 -CudaArch "sm_86,sm_89,sm_90,sm_120"
.\scripts\package-bench-16gb.ps1
```

The package lands in `dist\bench-16gb\` and `dist\fib-bench-16gb.zip`. It
contains the CLI, all four engines, `bench.ps1`, `run-bench.cmd`, and the
target notes. The default profile covers CPU/SIMD, num-bigint, and GMP through
`1e8`, all four GPUs through `1e9`, the block engines at `6.186e9`, and
block-28 at `61.865e9`; the CLI preflight skips anything that does not fit the
target card.

For a 16 GB Ampere card such as the RTX 3080 Laptop (`sm_86`), omit `sm_120`
only if build time matters. `sm_86` is the required entry for that GPU.

The engines are CUDA 13.0 binaries, so the target driver must support CUDA
13.0 (driver branch 580 or newer). A driver that reports CUDA 12.x in
`nvidia-smi` cannot initialise the CUDA 13 runtime; `bench.ps1` detects that
case and skips the GPU points with an English warning. Update the NVIDIA
driver, or rebuild the engines with the CUDA 12.x toolkit used by the target.
