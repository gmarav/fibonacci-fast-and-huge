# Build the 2^29 two-prime 48-bit block engine (CUDA 13, sm_120).
#
#   powershell -ExecutionPolicy Bypass -File .\build.ps1
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -HostRef
#
# -HostRef also builds the CPU-only reference tests (ntt_ref.exe, plan_test.exe)
# with MinGW g++.  No GPU kernel is executed by this script.

param(
    [switch]$HostRef,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path build, out | Out-Null
if ($Clean) { Remove-Item -Force -ErrorAction SilentlyContinue build\*.exe, build\*.obj }

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    $vcvars = 'C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
}
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found" }

$nvcc = 'nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler'

Write-Host '== building block29_fib.exe =='
cmd /c "`"$vcvars`" >nul && $nvcc block29_engine.cu block29_fib.cpp -o build\block29_fib.exe"
if ($LASTEXITCODE -ne 0) { throw 'nvcc block29_fib failed' }

if ($HostRef) {
    Write-Host '== building CPU reference (MinGW g++) =='
    g++ -O2 -std=c++17 ntt_ref.cpp -o build\ntt_ref.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ ntt_ref failed' }
    g++ -O2 -std=c++17 plan_test.cpp -o build\plan_test.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ plan_test failed' }
    g++ -O2 -std=c++17 segment_test.cpp -o build\segment_test.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ segment_test failed' }
    g++ -O2 -std=c++17 twiddle_test.cpp -o build\twiddle_test.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ twiddle_test failed' }
    g++ -O2 -std=c++17 driver_ref.cpp -o build\driver_ref.exe
    if ($LASTEXITCODE -ne 0) { throw 'g++ driver_ref failed' }
}

Write-Host 'build complete (no GPU kernel was executed)'
