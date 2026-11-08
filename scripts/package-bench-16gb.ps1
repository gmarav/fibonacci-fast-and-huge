# Build the portable 16 GB benchmark package from the local release build.
#
#   .\scripts\package-bench-16gb.ps1
#
# The CUDA engines must already contain the target architectures.  Rebuild with
#   .\build.ps1 -CudaArch "sm_86,sm_89,sm_90,sm_120"
# if the target GPU is not covered.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $root 'dist\bench-16gb'
$engines = Join-Path $pkg 'engines'
$zip = Join-Path $root 'dist\fib-bench-16gb.zip'

$required = @(
    (Join-Path $root 'dist\fib-simd-lab.exe'),
    (Join-Path $root 'cuda-ntt\resident-27\build\resident27_fib.exe'),
    (Join-Path $root 'cuda-ntt\block-27\build\block27_fib.exe'),
    (Join-Path $root 'cuda-ntt\block-28\build\block28_fib.exe'),
    (Join-Path $root 'cuda-ntt\block-29\build\block29_fib.exe')
)
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing build artifact: $file. Run .\build.ps1 first."
    }
}

if (Test-Path -LiteralPath $pkg) { Remove-Item -LiteralPath $pkg -Recurse -Force }
New-Item -ItemType Directory -Force -Path $engines | Out-Null

Copy-Item -LiteralPath (Join-Path $root 'dist\fib-simd-lab.exe') -Destination $pkg -Force
Copy-Item -LiteralPath (Join-Path $root 'cuda-ntt\resident-27\build\resident27_fib.exe') -Destination $engines -Force
Copy-Item -LiteralPath (Join-Path $root 'cuda-ntt\block-27\build\block27_fib.exe') -Destination $engines -Force
Copy-Item -LiteralPath (Join-Path $root 'cuda-ntt\block-28\build\block28_fib.exe') -Destination $engines -Force
Copy-Item -LiteralPath (Join-Path $root 'cuda-ntt\block-29\build\block29_fib.exe') -Destination $engines -Force
Copy-Item -LiteralPath (Join-Path $root 'scripts\bench-16gb.ps1') -Destination (Join-Path $pkg 'bench.ps1') -Force
$templates = Join-Path $PSScriptRoot 'bench-16gb-files'
foreach ($name in @('README-BENCH.md', 'run-bench.cmd')) {
    $source = Join-Path $templates $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Package template missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $pkg -Force
}

$cuobjdump = Get-Command cuobjdump.exe -ErrorAction SilentlyContinue
if ($cuobjdump) {
    foreach ($name in @('resident27_fib.exe', 'block27_fib.exe', 'block28_fib.exe', 'block29_fib.exe')) {
        $elf = & $cuobjdump.Source --list-elf (Join-Path $engines $name) 2>$null
        if (-not (($elf -join "`n") -match 'sm_86')) {
            throw "$name does not contain an sm_86 cubin; rebuild with -CudaArch `"sm_86,sm_89,sm_90,sm_120`"."
        }
    }
    Write-Host 'Verified sm_86 cubins in all engines.'
} else {
    Write-Host 'WARNING: cuobjdump not found; skipped the sm_86 cubin check.' -ForegroundColor Yellow
}

$sums = Join-Path $pkg 'SHA256SUMS.txt'
$lines = @('# SHA-256 of the portable benchmark package')
foreach ($file in (Get-ChildItem -LiteralPath $pkg -Recurse -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object FullName)) {
    $relative = $file.FullName.Substring($pkg.Length + 1).Replace('\', '/')
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    $lines += "$hash  $relative"
}
$lines | Set-Content -LiteralPath $sums -Encoding utf8

if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $pkg '*') -DestinationPath $zip -Force

Write-Host "Package: $pkg"
Write-Host "Archive: $zip"
