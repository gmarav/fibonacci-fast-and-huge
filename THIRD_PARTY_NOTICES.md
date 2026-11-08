# Third-party notices

The project source is licensed under the MIT License (see `LICENSE`). The
components below keep their own licenses. This file is a practical inventory,
not legal advice.

## Rust dependencies

Generated from `cargo metadata` for `rust/simd-lab/Cargo.toml` on 2026-11-08.

| Crate | Version | License |
|---|---:|---|
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| az | 1.3.0 | MIT OR Apache-2.0 |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 |
| either | 1.19.0 | MIT OR Apache-2.0 |
| gmp-mpfr-sys | 1.7.1 | LGPL-3.0-or-later |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| libc | 0.2.191 | MIT OR Apache-2.0 |
| libm | 0.2.16 | MIT |
| memchr | 2.8.3 | Unlicense OR MIT |
| num-bigint | 0.5.1 | MIT OR Apache-2.0 |
| num-integer | 0.1.47 | MIT OR Apache-2.0 |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| rayon | 1.12.0 | MIT OR Apache-2.0 |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 |
| rug | 1.30.0 | LGPL-3.0-or-later |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 |
| serde_json | 1.0.154 | MIT OR Apache-2.0 |
| syn | 3.0.9 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| zmij | 1.0.23 | MIT |

Where a crate offers a choice (`MIT OR Apache-2.0`), this project uses it
under the MIT option. `memchr` is used under the MIT option.

`unicode-ident` also carries the Unicode-3.0 license. If you redistribute
binaries, include the license texts and copyright notices shipped by those
crates.

## GMP / MPFR / MPC (LGPL)

`rug` and `gmp-mpfr-sys` are **LGPL-3.0-or-later**. `gmp-mpfr-sys` builds:

- GMP 6.3.0: dual **LGPL-3.0-or-later / GPL-2.0-or-later**;
- MPFR 4.2.2: **LGPL-3.0-or-later**;
- MPC 1.4.1: **LGPL-3.0-or-later**.

The full texts are available in the `gmp-mpfr-sys` crate under
`LICENSE-LGPL.md`, `LICENSE-GPL.md`, and the bundled `COPYING*` files, and at
<https://www.gnu.org/licenses/lgpl-3.0.html>.

### Important for prebuilt binaries

The default Rust build links GMP/MPFR/MPC **statically** through
`gmp-mpfr-sys`. A distributed `fib-simd-lab.exe` is therefore not an
MIT-only binary: it is a combined work with LGPL components.

LGPL-3.0 section 4 requires that recipients can modify the library and relink
the combined work. For binary releases, choose one of these paths:

1. publish the complete corresponding source for GMP/MPFR/MPC, this
   application's source, and the build scripts, plus the LGPL notices and
   installation information; or
2. ship GMP/MPFR/MPC as replaceable shared libraries (DLLs) and document how
   to replace them; or
3. publish source only (this repository) and let each user build locally.

The `cuda-ntt/common/oracle_gmp.cpp`, `cuda-ntt/resident-27/oracle_fib_gmp.cpp`,
and `arith_ref.h` files are optional verification tools that also link GMP;
the same LGPL terms apply if their binaries are
distributed.

## NVIDIA CUDA

The CUDA engines are compiled with NVIDIA CUDA 13.0 and may link or bundle
NVIDIA runtime components. Those components are **not** covered by this
project's MIT license; their use and redistribution are governed by the NVIDIA
CUDA Toolkit EULA and the list of redistributable files for the installed CUDA
version. Publishing only the source tree and `build.ps1` avoids redistributing
NVIDIA binaries; publishing compiled engines or DLLs requires checking the
EULA for that CUDA release.

## Windows system libraries

`psapi.lib`, the Windows API, and the CUDA/Visual Studio toolchain are used
under their own platform licenses. No Microsoft binary is relicensed by this
project.
