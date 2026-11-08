# Large computed Fibonacci results

All files use the canonical binary format:

```text
u64 limb_count (little-endian)
limb_count * u32 limb (little-endian)
```

`bits` is the exact bit length; `bytes` is the on-disk file size.

## Verified points shared by resident and block27 (1M .. 6.186e9)

These were compared byte-for-byte against GMP.

| n | bits | bytes | SHA-256 |
|---:|---:|---:|---|
| 1,048,576 | 727,965 | 91,004 | `4185565F8EAB4B9A708C8622047EFF5CCCF04670FC0F31F92D802A1348E92AFC` |
| 33,554,432 | 23,294,892 | 2,911,872 | `6AB578CA7642AE1F3555A5FFC7D6CF070D54D7414F535ABF4A8C240F987A9761` |
| 1,024,000,000 | 710,903,719 | 88,862,976 | `748369CD88F957E3534F1AD706050176006AEDEA9201C299B081FCF2B2E41C90` |
| 2,048,000,000 | 1,421,807,438 | 177,725,940 | `3EE4AE5CA623A3439FD90AB5A25D43D395B0B96C0F06880B5A0C3A7560E60F17` |
| 4,096,000,000 | 2,843,614,878 | 355,451,868 | `BB4BFD446A873C6580948CF656BBB3DCE3BB6D1165661221969D9637F1F4FF43` |
| 4,294,967,296 | 2,981,746,314 | 372,718,300 | `CCA1750D4E6BFA5AAA21D8062F6278F3696016022752FAD95067AA2F278DB137` |
| 6,186,556,317 | 4,294,966,696 | 536,870,848 | `E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750` |

## block27 (2^27, three 32-bit primes)

| n | bits | bytes | SHA-256 | exactness |
|---:|---:|---:|---|---|
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | exact (GPU == GMP) |
| 80,000,000,000 | 55,539,353,090 | 6,942,419,148 | `d52c638964e1c491cb0f5685c965844031a0443a46e9ffe88bdd98131e618eff` | limit point, hash only |
| 86,600,000,000 | 60,121,349,720 | 7,515,168,724 | `f335c7937c5254646781d93c04bb4efae1fda860bc76be46d05d2bf8a80631da` | limit point, hash only |

## block29 (2^29, two 48-bit primes)

| n | bits | bytes | SHA-256 | exactness |
|---:|---:|---:|---|---|
| 6,186,556,317 | 4,294,966,696 | 536,870,848 | `E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750` | exact (GPU == GMP) |
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | exact (GPU == GMP) |
| 73,628,010,691 | 51,115,651,038 | 6,389,456,388 | `3cbaa6b2dbd73e15d2c8da0168ad24845d9e6fa0e44ac821a64ca0373cb71f6f` | limit point, hash only |

## block28 (2^28, two 48-bit primes, option A)

| n | bits | bytes | SHA-256 | exactness |
|---:|---:|---:|---|---|
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | exact (GPU == GMP); re-verified on RTX 3080 Laptop |
| **100,000,000,000** | **69,424,191,362** | **8,678,023,932** | **`9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0`** | **exact (GPU == GMP)** |
| 134,500,000,000 | 93,375,537,408 | 11,671,942,184 | `7122055B71E74EE0E3AD4302CF2DC3CAB156C59BBDC5811C8F144101810E5FC4` | limit point, hash only |
| 150,000,000,000 | 104,136,287,072 | 13,017,035,892 | `7A75B12A2CE4D4B6EA80B68C08A38079D42AD8367990D49FA733CDA5D785045F` | limit point, hash only |
| 164,500,000,000 | 114,202,794,816 | 14,275,349,360 | `B1D16D7B4463B4D933244AF89F35C7DED3C247C9B2077A2AD6C0CFBAB489CC1E` | limit point, hash only |
| 175,000,000,000 | 121,492,334,912 | 15,186,541,872 | `C77C13EACCC65AAAA3F5EDF8C1DE1B586A4DE20E78BAC3AFFA5177EC8AF4875F` | limit point, hash only |
| **184,000,000,000** | **127,740,512,128** | **15,967,564,024** | **`CDD416B4E41B1BC4D188DCC5B94296473963CE7822E39B2268BE8A74DA4EC758`** | **measured maximum, hash only** |
| 185,000,000,000 | - | - | - | rejected by preflight (k=16) |
| 200,000,000,000 | - | - | - | impossible (k=17) |

The `100e9` and `184e9` GPU hashes were re-verified on the current build on
2026-11-08. This was a GPU-only check; the exactness labels are unchanged.

On the 16 GB RTX 3080 Laptop the verified block-28 ceiling in this run was
`61.865e9`; the `100e9` and larger points need more than 16 GiB of VRAM and
remain results from the 24 GB reference machine.

## Full values

The two full `F(100,000,000,000)` files are not committed because each is
8.68 GB. A local copy can be reproduced with `--gpu-out` and the anchored GMP
chain; only the byte count and SHA-256 are canonical here.

Both have SHA-256
`9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0`.

The hash is recorded in [`HASHES.md`](HASHES.md) together with the other
verified and limit points. The full 8.7 GB limb files are not committed; the
SHA-256 and byte count are the canonical record.
