# Lint the Rust CLI, the host C++ drivers/reference tests, and the CUDA engines.
#
#   .\scripts\lint.ps1                       # everything available locally
#   .\scripts\lint.ps1 -SkipRust -SkipCuda   # host C++ only (used by CI)
#   .\scripts\lint.ps1 -SkipCpp              # Rust and CUDA only
#
# Rust:      cargo fmt --check, clippy -D warnings, cargo test --locked.
# Host C++:  g++ -Wall -Wextra for every driver and CPU reference test, plus
#            MSVC /analyze for the four production host drivers.
# CUDA:      nvcc -Xcompiler /analyze for the four engine translation units
#            when nvcc and the MSVC host toolchain are available.
#
# Diagnostics from MSVC system headers are reported but do not fail the run:
# /analyze is known to warn inside the standard library and CUDA headers.

[CmdletBinding()]
param(
    [string]$Toolchain = '1.99.1-x86_64-pc-windows-gnu',
    [string]$CudaArch = 'sm_120',
    [switch]$SkipRust,
    [switch]$SkipCpp,
    [switch]$SkipCuda
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cudaNtt = Join-Path $root 'cuda-ntt'
$script:failed = $false

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "== $Message =="
}

function Note-Failure([string]$Message) {
    $script:failed = $true
    Write-Host "FAILED: $Message" -ForegroundColor Red
}

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
        $path = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -find 'VC\Auxiliary\Build\vcvars64.bat' | Select-Object -First 1
        if ($path) { return $path.Trim() }
    }
    return $null
}

function Find-Gxx {
    $onPath = Get-Command g++.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $candidates = @()
    if ($env:MSYS2_ROOT) { $candidates += (Join-Path $env:MSYS2_ROOT 'mingw64\bin\g++.exe') }
    $candidates += @(
        'C:\msys64\mingw64\bin\g++.exe',
        (Join-Path $root '.tools\msys64\mingw64\bin\g++.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Invoke-RustLint {
    $rustup = Get-Command rustup.exe -ErrorAction SilentlyContinue
    $cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
    if (-not $rustup -and -not $cargo) {
        Note-Failure 'neither rustup nor cargo is on PATH'
        return
    }
    $manifest = Join-Path $root 'rust\simd-lab\Cargo.toml'
    $parentManifest = Join-Path $root 'rust\Cargo.toml'
    $basetarget = Join-Path $root 'build\rust'
    New-Item -ItemType Directory -Force -Path $basetarget | Out-Null
    if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = $basetarget }
    if (-not $env:GMP_MPFR_SYS_CACHE) {
        $env:GMP_MPFR_SYS_CACHE = Join-Path $root 'build\gmp-cache'
    }

    function Invoke-Cargo([string[]]$Arguments) {
        if ($rustup) {
            & rustup run $Toolchain cargo @Arguments
        } else {
            & cargo @Arguments
        }
    }

    Push-Location $root
    try {
        Write-Step 'Rust formatting'
        Invoke-Cargo @('fmt', '--manifest-path', $manifest, '--', '--check')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo fmt --check (simd-lab)' }
        Invoke-Cargo @('fmt', '--manifest-path', $parentManifest, '--', '--check')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo fmt --check (fib-rust)' }
        if (-not $script:failed) { Write-Host 'rustfmt: ok' }

        Write-Step 'Rust clippy'
        Invoke-Cargo @('clippy', '--release', '--all-targets', '--manifest-path', $manifest,
            '--locked', '--', '-D', 'warnings')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo clippy (simd-lab)' }
        Invoke-Cargo @('clippy', '--release', '--all-targets', '--manifest-path', $parentManifest,
            '--locked', '--', '-D', 'warnings')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo clippy (fib-rust)' }
        if (-not $script:failed) { Write-Host 'clippy: ok' }

        Write-Step 'Rust tests'
        Invoke-Cargo @('test', '--release', '--manifest-path', $manifest, '--locked')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo test (simd-lab)' }
        Invoke-Cargo @('test', '--release', '--manifest-path', $parentManifest, '--locked')
        if ($LASTEXITCODE -ne 0) { Note-Failure 'cargo test (fib-rust)' }
        if (-not $script:failed) { Write-Host 'tests: ok' }
    } finally {
        Pop-Location
    }
}

$hostDrivers = @(
    'block-27\block27_fib.cpp',
    'block-28\block28_fib.cpp',
    'block-29\block29_fib.cpp',
    'resident-27\resident27_fib.cpp'
)

$hostReferences = @(
    'block-27\check_fib_mod.cpp',
    'block-28\driver_ref.cpp',
    'block-28\ntt_ref.cpp',
    'block-28\plan_test.cpp',
    'block-28\segment_test.cpp',
    'block-28\twiddle_test.cpp',
    'block-29\driver_ref.cpp',
    'block-29\ntt_ref.cpp',
    'block-29\plan_test.cpp',
    'block-29\segment_test.cpp',
    'block-29\twiddle_test.cpp',
    'resident-27\arith_selftest.cpp'
)

function Test-IsSystemDiagnostic([string]$Line) {
    return ($Line -match '\\MSVC\\[^\\]+\\include\\') -or
           ($Line -match '\\Windows Kits\\') -or
           ($Line -match '\\CUDA\\v[^\\]+\\include\\')
}

function Invoke-GxxLint {
    $gxx = Find-Gxx
    if (-not $gxx) {
        Note-Failure 'g++ not found; install MSYS2/MinGW or put g++ on PATH'
        return
    }
    Write-Step "Host C++ warnings ($gxx)"
    $temp = Join-Path $env:TEMP 'fib-lint'
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    foreach ($relative in ($hostDrivers + $hostReferences)) {
        $path = Join-Path $cudaNtt $relative
        $obj = Join-Path $temp (([IO.Path]::GetFileNameWithoutExtension($relative)) + '.o')
        $out = & $gxx -O2 -std=c++17 -Wall -Wextra -c `
            -fdiagnostics-color=never $path -o $obj 2>&1
        if ($LASTEXITCODE -ne 0) {
            Note-Failure "g++ $relative"
            $out | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
        } elseif ($out) {
            $out | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
            if (($out | Where-Object { $_ -match ': warning:' }).Count -gt 0) {
                Note-Failure "g++ warnings in $relative"
            }
        }
    }
    if (-not $script:failed) { Write-Host 'g++ -Wall -Wextra: clean' }
}

function Invoke-MsvcHostAnalyze {
    param([string]$VcVars)
    Write-Step 'MSVC /analyze on host drivers'
    $temp = Join-Path $env:TEMP 'fib-lint'
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    foreach ($relative in $hostDrivers) {
        $path = Join-Path $cudaNtt $relative
        $dir = Split-Path -Parent $path
        $name = Split-Path -Leaf $path
        $obj = Join-Path $temp (([IO.Path]::GetFileNameWithoutExtension($name)) + '.obj')
        Push-Location $dir
        try {
            $out = cmd /c "`"$VcVars`" >nul && cl /nologo /std:c++17 /EHsc /analyze /c $name /Fo`"$obj`"" 2>&1
            $code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $diagnostics = $out | Where-Object { $_ -match 'warning C\d+|error C\d+' }
        $ours = $diagnostics | Where-Object { -not (Test-IsSystemDiagnostic $_) }
        if ($diagnostics.Count -gt $ours.Count) {
            Write-Host "  ($($diagnostics.Count - $ours.Count) system-header diagnostics ignored)"
        }
        $ours | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
        if ($code -ne 0 -or $ours.Count -gt 0) {
            Note-Failure "MSVC /analyze $relative"
        }
    }
    if (-not $script:failed) { Write-Host 'MSVC /analyze (host): clean' }
}

function Invoke-CudaAnalyze {
    param([string]$VcVars)
    Write-Step 'nvcc /analyze on CUDA engines'
    $engines = @(
        'resident-27\resident27.cu',
        'block-27\block27_engine.cu',
        'block-28\block28_engine.cu',
        'block-29\block29_engine.cu'
    )
    $temp = Join-Path $env:TEMP 'fib-lint'
    New-Item -ItemType Directory -Force -Path $temp | Out-Null
    foreach ($relative in $engines) {
        $path = Join-Path $cudaNtt $relative
        $dir = Split-Path -Parent $path
        $name = Split-Path -Leaf $path
        $obj = Join-Path $temp (([IO.Path]::GetFileNameWithoutExtension($name)) + '.cuda.obj')
        Push-Location $dir
        try {
            $out = cmd /c "`"$VcVars`" >nul && nvcc -O3 -std=c++17 -arch=$CudaArch -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler -Xcompiler /analyze -c $name -o `"$obj`"" 2>&1
            $code = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        $diagnostics = $out | Where-Object { $_ -match 'warning C\d+|error C\d+' }
        $ours = $diagnostics | Where-Object { -not (Test-IsSystemDiagnostic $_) }
        if ($diagnostics.Count -gt $ours.Count) {
            Write-Host "  ($($diagnostics.Count - $ours.Count) system-header diagnostics ignored)"
        }
        $ours | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
        if ($code -ne 0 -or $ours.Count -gt 0) {
            Note-Failure "nvcc /analyze $relative"
        }
    }
    if (-not $script:failed) { Write-Host 'nvcc /analyze: clean' }
}

if (-not $SkipRust) {
    Invoke-RustLint
}

if (-not $SkipCpp) {
    Invoke-GxxLint
    $vcvars = Find-VcVars
    if ($vcvars) {
        Invoke-MsvcHostAnalyze $vcvars
    } else {
        Write-Host 'MSVC /analyze skipped: vcvars64.bat not found' -ForegroundColor Yellow
    }
}

if (-not $SkipCuda) {
    $vcvars = Find-VcVars
    $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if (-not $vcvars -or -not $nvcc) {
        Write-Host 'CUDA /analyze skipped: nvcc or vcvars64.bat not found' -ForegroundColor Yellow
    } else {
        Invoke-CudaAnalyze $vcvars
    }
}

if ($script:failed) {
    Write-Host ''
    Write-Host 'Lint failed.' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'All lint checks passed.'
