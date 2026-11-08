# Build the exact Fibonacci suite: Rust CLI (CPU SIMD + GPU FFI) and CUDA engines.
#
#   .\build.ps1                 # CUDA engines + Rust CLI + dist/SHA256SUMS.txt
#   .\build.ps1 -SkipCuda       # CPU/SIMD and num-bigint/GMP only
#   .\build.ps1 -SkipRust       # CUDA engines only
#   .\build.ps1 -Check          # additionally run fmt, clippy, tests, and C++/CUDA lint
#   .\build.ps1 -CudaArch "sm_86,sm_89,sm_120"   # portable multi-architecture engines
#
# See docs/BUILD_WINDOWS.md for prerequisites.

[CmdletBinding()]
param(
    [string]$Toolchain = '1.99.1-x86_64-pc-windows-gnu',
    [string]$CudaArch = 'sm_120',
    [switch]$SkipCuda,
    [switch]$SkipRust,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$dist = Join-Path $root 'dist'
$buildRoot = Join-Path $root 'build'
New-Item -ItemType Directory -Force -Path $dist, $buildRoot | Out-Null

function Find-VcVars {
    $candidates = @(
        'C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Auxiliary\Build\vcvars64.bat' |
            Select-Object -First 1
        if ($path) { return $path.Trim() }
    }
    throw 'vcvars64.bat not found. Install Visual Studio Build Tools with the C++ workload.'
}

function Get-NvccArchFlags {
    $parts = @($CudaArch -split '[,; ]+' | Where-Object { $_ })
    if ($parts.Count -le 1) { return "-arch=$($parts[0])" }
    $flags = foreach ($part in $parts) {
        if ($part -match '^sm_(\d+)$') {
            "-gencode arch=compute_$($matches[1]),code=sm_$($matches[1])"
        } elseif ($part -match '^compute_(\d+)$') {
            "-gencode arch=compute_$($matches[1]),code=compute_$($matches[1])"
        } else {
            "-arch=$part"
        }
    }
    return ($flags -join ' ')
}

$script:archFlags = Get-NvccArchFlags

function Invoke-Nvcc {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$Arguments
    )
    Push-Location $Directory
    try {
        cmd /c "`"$script:vcvars`" >nul && nvcc -O3 -std=c++17 $script:archFlags -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler $Arguments"
        if ($LASTEXITCODE -ne 0) {
            throw "nvcc failed in $Directory"
        }
    } finally {
        Pop-Location
    }
}

function Add-MsysPath {
    $env:MSYSTEM = 'MINGW64'
    if (Get-Command make.exe -ErrorAction SilentlyContinue) { return }
    $candidates = @()
    if ($env:MSYS2_ROOT) { $candidates += (Join-Path $env:MSYS2_ROOT 'usr\bin') }
    $candidates += @(
        'C:\msys64\usr\bin',
        (Join-Path $root '.tools\msys64\usr\bin')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'make.exe')) {
            $env:PATH = "$candidate;$env:PATH"
            return
        }
    }
}

if ($Check) {
    Write-Host 'running C++/CUDA lint'
    & (Join-Path $root 'scripts\lint.ps1') -SkipRust -SkipCuda:$SkipCuda
    if ($LASTEXITCODE -ne 0) { throw 'C++/CUDA lint failed' }
}

if (-not $SkipCuda) {
    $script:vcvars = Find-VcVars
    Write-Host "CUDA: $CudaArch, host $script:vcvars"

    $resident = Join-Path $root 'cuda-ntt\resident-27'
    $block27 = Join-Path $root 'cuda-ntt\block-27'
    $block28 = Join-Path $root 'cuda-ntt\block-28'
    $block29 = Join-Path $root 'cuda-ntt\block-29'
    foreach ($dir in $resident, $block27, $block28, $block29) {
        New-Item -ItemType Directory -Force -Path (Join-Path $dir 'build') | Out-Null
    }

    Write-Host 'building resident-27 (2^27, 3x32) engine'
    Invoke-Nvcc $resident 'resident27.cu resident27_fib.cpp -o build\resident27_fib.exe'

    Write-Host 'building block-27 (2^27, 3x32) engine'
    Invoke-Nvcc $block27 'block27_engine.cu block27_fib.cpp -o build\block27_fib.exe'

    Write-Host 'building block-28 (2^28, 2x48) engine'
    Invoke-Nvcc $block28 'block28_engine.cu block28_fib.cpp -o build\block28_fib.exe psapi.lib'

    Write-Host 'building block-29 (2^29, 2x48) engine'
    Invoke-Nvcc $block29 'block29_engine.cu block29_fib.cpp -o build\block29_fib.exe'
}

if (-not $SkipRust) {
    $rustup = Get-Command rustup.exe -ErrorAction SilentlyContinue
    if (-not $rustup) { throw 'rustup.exe not found. Install Rust from https://rustup.rs.' }
    Add-MsysPath
    foreach ($tool in 'gcc.exe', 'make.exe', 'm4.exe') {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool not found. Install MSYS2/MinGW and add its usr\bin to PATH."
        }
    }
    $manifest = Join-Path $root 'rust\simd-lab\Cargo.toml'
    $env:CARGO_TARGET_DIR = Join-Path $buildRoot 'rust'
    if (-not $env:GMP_MPFR_SYS_CACHE) {
        $env:GMP_MPFR_SYS_CACHE = Join-Path $buildRoot 'gmp-cache'
    }

    if ($Check) {
        Write-Host 'checking Rust formatting'
        & rustup run $Toolchain cargo fmt --manifest-path $manifest -- --check
        if ($LASTEXITCODE -ne 0) { throw 'cargo fmt --check failed' }
        Write-Host 'running clippy'
        & rustup run $Toolchain cargo clippy --release --all-targets --manifest-path $manifest --locked -- -D warnings
        if ($LASTEXITCODE -ne 0) { throw 'cargo clippy failed' }
        Write-Host 'running tests'
        & rustup run $Toolchain cargo test --release --manifest-path $manifest --locked
        if ($LASTEXITCODE -ne 0) { throw 'cargo test failed' }
    }

    Write-Host "building Rust CLI with $Toolchain"
    & rustup run $Toolchain cargo build --release --manifest-path $manifest --locked
    if ($LASTEXITCODE -ne 0) { throw 'Rust release build failed' }
    Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR 'release\fib-simd-lab.exe') -Destination (Join-Path $dist 'fib-simd-lab.exe') -Force
}

$artifacts = @(
    'dist\fib-simd-lab.exe',
    'cuda-ntt\resident-27\build\resident27_fib.exe',
    'cuda-ntt\block-27\build\block27_fib.exe',
    'cuda-ntt\block-28\build\block28_fib.exe',
    'cuda-ntt\block-29\build\block29_fib.exe'
)
$sums = Join-Path $dist 'SHA256SUMS.txt'
Set-Content -LiteralPath $sums -Value '# SHA-256 of local build artifacts (relative to the repository root)' -Encoding utf8
foreach ($relative in $artifacts) {
    $full = Join-Path $root $relative
    if (Test-Path -LiteralPath $full) {
        $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
        Add-Content -LiteralPath $sums -Value "$hash  $relative" -Encoding utf8
    }
}

Write-Host ''
Write-Host 'Build completed.'
Write-Host "  CLI      $dist\fib-simd-lab.exe"
Write-Host "  engines  cuda-ntt\*\build\"
Write-Host "  hashes   $sums"
