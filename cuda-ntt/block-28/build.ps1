# Build the 2^28 two-prime 48-bit option-A block engine (CUDA 13, sm_120).
#
#   powershell -ExecutionPolicy Bypass -File .\build.ps1
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -HostRef
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -NoCuda -HostRef
#
# -HostRef also builds the CPU-only reference tests with MinGW g++.  No GPU
# kernel is executed by this script.

param(
    [switch]$HostRef,
    [switch]$Clean,
    [switch]$NoCuda,
    [switch]$Gmp
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path build, out | Out-Null
if ($Clean) { Remove-Item -Force -ErrorAction SilentlyContinue build\*.exe, build\*.obj }

if (-not $NoCuda) {
    $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        $vcvars = 'C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    }
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found" }

    $nvcc = 'nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler'
    Write-Host '== building block28_fib.exe =='
    cmd /c "`"$vcvars`" >nul && $nvcc block28_engine.cu block28_fib.cpp -o build\block28_fib.exe psapi.lib"
    if ($LASTEXITCODE -ne 0) { throw 'nvcc engine/fib failed' }
}

if ($HostRef) {
    Write-Host '== building CPU reference (MinGW g++) =='
    foreach ($pair in @(
            @('ntt_ref.cpp', 'ntt_ref.exe'),
            @('plan_test.cpp', 'plan_test.exe'),
            @('segment_test.cpp', 'segment_test.exe'),
            @('twiddle_test.cpp', 'twiddle_test.exe'),
            @('driver_ref.cpp', 'driver_ref.exe'))) {
        g++ -O2 -std=c++17 $pair[0] -o "build\$($pair[1])"
        if ($LASTEXITCODE -ne 0) { throw "g++ $($pair[0]) failed" }
    }
}

if ($Gmp) {
    Write-Host '== building exact GMP chain oracle (MinGW g++) =='
    $gmpRoot = $null
    $candidates = @()
    if ($env:GMP_ROOT) { $candidates += $env:GMP_ROOT }
    foreach ($cand in (Get-ChildItem (Join-Path $PSScriptRoot '..\..\build') -Recurse -Filter 'libgmp.a' -ErrorAction SilentlyContinue)) {
        $candidates += (Split-Path (Split-Path $cand.FullName -Parent) -Parent)
    }
    foreach ($candRoot in $candidates) {
        if ((Test-Path (Join-Path $candRoot 'include\gmp.h')) -and
            (Test-Path (Join-Path $candRoot 'lib\libgmp.a'))) { $gmpRoot = $candRoot; break }
    }
    if ($null -eq $gmpRoot) { throw 'static libgmp.a not found; set GMP_ROOT or build the Rust CLI first' }
    Write-Host "   using $gmpRoot"
    & g++ -O2 -std=c++17 "..\common\oracle_gmp.cpp" "-I$gmpRoot\include" `
        "$gmpRoot\lib\libgmp.a" -lpsapi -o build\oracle_big_gmp.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ oracle_big_gmp failed' }
}

Write-Host 'build complete (no GPU kernel was executed)'
