# build.ps1 — block-27 host tools
#
#   powershell -ExecutionPolicy Bypass -File .\build.ps1          # build
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Test    # CPU selftests
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Clean
#
# Produces under cuda-ntt/block-27/build:
#   block27_fib.exe        GPU CLI: fib / mul / selftest / sqtest / bench / info
#   oracle_ext_gmp.exe   anchored GMP oracle (fib, mul, fib-anchor, probes)
#   check_fib_mod.exe    residue cross-check for a limb file
#
# The GPU binary needs a working CUDA context; the GMP tools are CPU-only.

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Test,
    [switch]$NoCuda
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$build = Join-Path $root "build"
Push-Location $root

try {
    if ($Clean -and (Test-Path $build)) {
        Remove-Item -Recurse -Force $build
        Write-Host "cleaned $build"
        if (-not $Test) { return }
    }
    if (-not (Test-Path $build)) {
        New-Item -ItemType Directory -Path $build -Force | Out-Null
    }

    if (-not $NoCuda) {
        $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
        if (-not (Test-Path $vcvars)) { throw "MSVC vcvars64.bat not found: $vcvars" }
        $nvcc = "nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler"
        Write-Host "building block27_fib.exe"
        cmd /c "`"$vcvars`" >nul && $nvcc block27_engine.cu block27_fib.cpp -o build\block27_fib.exe"
        if ($LASTEXITCODE -ne 0) { throw "nvcc build failed" }
    }

    $gmpCandidates = @()
    if ($env:GMP_ROOT) { $gmpCandidates += $env:GMP_ROOT }
    foreach ($cand in (Get-ChildItem (Join-Path $root '..\..\build') -Recurse -Filter 'libgmp.a' -ErrorAction SilentlyContinue)) {
        $gmpCandidates += (Split-Path (Split-Path $cand.FullName -Parent) -Parent)
    }
    $gmpRoot = $null
    foreach ($cand in $gmpCandidates) {
        if ((Test-Path (Join-Path $cand 'include\gmp.h')) -and
            (Test-Path (Join-Path $cand 'lib\libgmp.a'))) {
            $gmpRoot = $cand
            break
        }
    }
    $oracle = Join-Path $build 'oracle_ext_gmp.exe'
    if ($null -eq $gmpRoot) {
        Write-Warning 'static libgmp.a not found; skipping oracle_ext_gmp.exe'
    } else {
        Write-Host "building oracle_ext_gmp.exe with $gmpRoot"
        & g++ -O2 -std=c++17 "..\common\oracle_gmp.cpp" "-I$gmpRoot\include" `
            "$gmpRoot\lib\libgmp.a" -lpsapi -o $oracle
        if ($LASTEXITCODE -ne 0) { throw "g++ oracle build failed" }
    }

    Write-Host "building check_fib_mod.exe"
    & g++ -O2 -std=c++17 "check_fib_mod.cpp" -o (Join-Path $build 'check_fib_mod.exe')
    if ($LASTEXITCODE -ne 0) { throw "g++ check build failed" }

    if ($Test -and (Test-Path $oracle)) {
        & $oracle selftest
        if ($LASTEXITCODE -ne 0) { throw "oracle selftest failed" }
        & $oracle anchor-selftest
        if ($LASTEXITCODE -ne 0) { throw "anchor selftest failed" }
    }
    Write-Host ""
    Write-Host "artifacts under $build"
} finally {
    Pop-Location
}
