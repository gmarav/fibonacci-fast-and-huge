# CUDA engines

Four independent CUDA engines are used by the Rust CLI. Only the engine
binaries are required at runtime; the CLI starts every engine as a child
process.

| Directory | Engine | Transform | Limbs | Output |
|---|---|---|---|---|
| `resident-27` | resident fast doubling | 2^27 | 3 x 32-bit, 3 primes | `resident27_fib.exe` |
| `block-27` | blockwise 2^27 | 2^27 | 3 x 32-bit, 3 primes | `block27_fib.exe` |
| `block-28` | blockwise 2^28 option A | 2^28 | 2 x 48-bit, 2 primes | `block28_fib.exe` |
| `block-29` | blockwise 2^29 | 2^29 | 2 x 48-bit, 2 primes | `block29_fib.exe` |

Shared host code lives in `common/`:

- `limb_io.h` — canonical limb-file read/write, trim, bit length, comparison;
- `limb_math.h` — cold base-2^32 add/sub/shift/schoolbook helpers;
- `time_util.h` — monotonic millisecond clock;
- `memory.h` — peak host working set and available physical memory;
- `run_options.h` — shared `--n`, `--out`, `--iterations`, `--stats-json`;
- `usage.h` — common command/help layout;
- `report.h` — shared `fib`/timing/memory report lines;
- `stats_json.h` — the per-run statistics file read by the Rust CLI;
- `mont48_device.cuh` - shared 48-bit Montgomery primitives for block-28/29;
- `mont48_kernels.cuh` - shared live NTT kernels for block-28/29;
- `oracle_gmp.cpp` - shared GMP chain oracle for block-27/block-28 (not used
  at runtime).

Build everything from the repository root with `.\build.ps1`, or build one
engine with its local `build.ps1`. Each directory has a README with the
algorithm, memory model, and measured results.

The engines write canonical little-endian limb files:

```text
u64 limb_count
limb_count * u32 limb
```

The Rust CLI compares full bytes (or SHA-256) when a reference is supplied.
None of the engines uses floating-point arithmetic to produce Fibonacci
numbers.
