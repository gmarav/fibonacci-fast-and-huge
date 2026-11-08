# Canonical result hashes

Format: `u64 limb_count` followed by `limb_count` little-endian `u32` limbs.
`GMP` means the full bytes were compared against GMP; `hash` means the point
was measured but not independently compared beyond its SHA-256.

## Shared verified points: resident, block27, block29

| n | bits | bytes | SHA-256 | status |
|---:|---:|---:|---|---|
| 1,048,576 | 727,965 | 91,004 | `4185565F8EAB4B9A708C8622047EFF5CCCF04670FC0F31F92D802A1348E92AFC` | GMP |
| 33,554,432 | 23,294,892 | 2,911,872 | `6AB578CA7642AE1F3555A5FFC7D6CF070D54D7414F535ABF4A8C240F987A9761` | GMP |
| 1,024,000,000 | 710,903,719 | 88,862,976 | `748369CD88F957E3534F1AD706050176006AEDEA9201C299B081FCF2B2E41C90` | GMP |
| 2,048,000,000 | 1,421,807,438 | 177,725,940 | `3EE4AE5CA623A3439FD90AB5A25D43D395B0B96C0F06880B5A0C3A7560E60F17` | GMP |
| 4,096,000,000 | 2,843,614,878 | 355,451,868 | `BB4BFD446A873C6580948CF656BBB3DCE3BB6D1165661221969D9637F1F4FF43` | GMP |
| 4,294,967,296 | 2,981,746,314 | 372,718,300 | `CCA1750D4E6BFA5AAA21D8062F6278F3696016022752FAD95067AA2F278DB137` | GMP |
| 6,186,556,317 | 4,294,966,696 | 536,870,848 | `E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750` | GMP |

## block27 (2^27, 3x32)

| n | bits | bytes | SHA-256 | status |
|---:|---:|---:|---|---|
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | GMP |
| 80,000,000,000 | 55,539,353,090 | 6,942,419,148 | `d52c638964e1c491cb0f5685c965844031a0443a46e9ffe88bdd98131e618eff` | hash |
| 86,600,000,000 | 60,121,349,720 | 7,515,168,724 | `f335c7937c5254646781d93c04bb4efae1fda860bc76be46d05d2bf8a80631da` | hash |

## block29 (2^29, 2x48)

| n | bits | bytes | SHA-256 | status |
|---:|---:|---:|---|---|
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | GMP |
| 73,628,010,691 | 51,115,651,038 | 6,389,456,388 | `3cbaa6b2dbd73e15d2c8da0168ad24845d9e6fa0e44ac821a64ca0373cb71f6f` | hash |

## block28 (2^28, 2x48, option A)

| n | bits | bytes | SHA-256 | status |
|---:|---:|---:|---|---|
| 100,000,000,000 | 69,424,191,362 | 8,678,023,932 | `9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0` | **GPU == GMP** |
| 134,500,000,000 | 93,375,537,408 | 11,671,942,184 | `7122055B71E74EE0E3AD4302CF2DC3CAB156C59BBDC5811C8F144101810E5FC4` | hash |
| 150,000,000,000 | 104,136,287,072 | 13,017,035,892 | `7A75B12A2CE4D4B6EA80B68C08A38079D42AD8367990D49FA733CDA5D785045F` | hash |
| 164,500,000,000 | 114,202,794,816 | 14,275,349,360 | `B1D16D7B4463B4D933244AF89F35C7DED3C247C9B2077A2AD6C0CFBAB489CC1E` | hash |
| 175,000,000,000 | 121,492,334,912 | 15,186,541,872 | `C77C13EACCC65AAAA3F5EDF8C1DE1B586A4DE20E78BAC3AFFA5177EC8AF4875F` | hash |
| 184,000,000,000 | 127,740,512,128 | 15,967,564,024 | `CDD416B4E41B1BC4D188DCC5B94296473963CE7822E39B2268BE8A74DA4EC758` | hash, measured max |
| 185,000,000,000 | - | - | - | rejected: k=16 |
| 200,000,000,000 | - | - | - | not run: k=17 |

`184e9` is the last measured success. The conservative planner boundary with
`k=15` is `184,070,026,875`. The full `1e11` limb file is not committed
(8.68 GB); its byte count and SHA-256 above are canonical.

## Re-verification on the current build (2026-11-08)

The two large block-28 GPU hashes were reproduced on the current tree after the
shared-kernel and shared-limb-math refactors:

| n | backend | SHA-256 | result |
|---:|---|---|---|
| 100,000,000,000 | block-28 | `9117AD9310DA4A1CC4EB760F0C3655760771B270E080F2EFA7170F670EA49CA0` | match |
| 184,000,000,000 | block-28 | `CDD416B4E41B1BC4D188DCC5B94296473963CE7822E39B2268BE8A74DA4EC758` | match |

This re-run was GPU-only and did not repeat the independent GMP comparison.
The exactness labels above are unchanged: `1e11` remains the verified
`GPU == GMP` anchor, and `184e9` remains a measured hash-only maximum.
## Portable target re-verification: RTX 3080 Laptop 16 GB (2026-11-08)

The portable package was run on an RTX 3080 Laptop 16 GB with a Ryzen 9
5900HX. All four GPU engines reproduced the same canonical hash at each point
through `6.186e9`; block-28 also reproduced the `61.865e9` anchor.

| n | bits | bytes | SHA-256 | status |
|---:|---:|---:|---|---|
| 1,000,000 | 694,241 | 86,792 | `9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6` | all GPU engines agree, CPU reference exact |
| 10,000,000 | 6,942,418 | 867,812 | `667F558207A611BEFE401972E369E378391206EBB25BE1BBEA4D52B35E0919AC` | all GPU engines agree, CPU reference exact |
| 100,000,000 | 69,424,191 | 8,678,032 | `9D69F2E524D1C955D39D0AFDC8EA3F1EAC09FA0AFAE5C0256FF6935FB0387CC9` | all GPU engines agree, CPU reference exact |
| 1,000,000,000 | 694,241,913 | 86,780,248 | `D57068140E4974AB4FE1F8847DA629A798C8DAC02AEF517E8C0711AFAF884B14` | all GPU engines agree, CPU reference exact |
| 6,186,556,317 | 4,294,966,696 | 536,870,848 | `E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750` | all GPU engines agree, CPU reference exact |
| 61,865,563,170 | 42,949,666,962 | 5,368,708,380 | `4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580` | block-28 GPU == GMP anchor |

`1e11` and `184e9` were not run on this machine because the 16 GB card cannot
hold the block-28 device buffers at those indices.
