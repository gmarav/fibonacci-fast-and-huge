# fib-rust

The `fib-rust` library provides three exact CPU implementations used by
`simd-lab`:

- `direct`: symbolic Binet evaluation in `Z[phi]`, using `phi^2 = phi + 1`;
- `matrix`: 2x2 binary matrix exponentiation;
- `doubling`: fast doubling with exact limb arithmetic.

Big integers come from `num-bigint` and, with the `gmp` feature, from GMP
through `rug` (integer support only). All methods use `F(0) = 0`,
`F(1) = 1`, and never use a floating-point value to compute a result.

`simd-lab` is a separate crate in `simd-lab/` that adds the project-local SIMD
integer, the CUDA engines, the result table, the chart, and the progress line.
Build the command-line tool from the repository root with `.\build.ps1`; see
the top-level README and `docs/BUILD_WINDOWS.md`.
