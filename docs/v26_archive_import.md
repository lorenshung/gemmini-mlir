# V26 compiler archive import

This is an unselected, byte-preserving import of the independent OOT publication
`8b8fd3629f4e8e1c3349176241620621048b96c7`, source tree
`47c212a30c94cffa4a4c713d7b78f3cb416f5d41`.
The archive is at `phase1_generated/el4_fire_aligned_v26_20261019T191400Z/`.
No files from its root were merged into this repository's root, and V23, selected
SDKs, existing hardware adapters, and historical qualifications remain unchanged.

Identity uses sorted relative POSIX path, NUL, complete file bytes, NUL:

| Scope | Files / bytes | SHA-256 |
|---|---:|---|
| Complete imported archive | 54 / 22,791,395 | `3e7d314530b00e966af3bbc7c244ae0423721cea98cdfd1e002d061fe1f3fc3d` |
| Published compiler | 14 / 102,715 | `e7fd1d4e3f5d168050ba212d853aa7ff3283bf2f8b7527932bd3704c1f455039` |
| Original compiler, including two omitted reports | 16 | `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24` |

The published compiler cannot inherit the original package identity or certificate.
`experiment/compiler-inventory.json` records all original file hashes and omissions.
`SHA256SUMS` covers the other 53 imported files. All 54 destination files were
compared to the exact Git blobs. The 53 checksums, all 14 published compiler
size/hash entries, 18 result-manifest entries, both rendered launch-config hashes,
three selected recipe file hashes, compiler Python syntax, and usage arithmetic
were independently checked. The archive's integrity verifier can be repeated:

```sh
python3 -B phase1_generated/el4_fire_aligned_v26_20261019T191400Z/reproduce/verify_archive.py
```

This command verifies integrity only. It does not compile, simulate, or qualify.
The audit output and complete blob inventory are retained in this task at
`out/artifacts/verification/v26-archive-import-v1/{audit.json,files.json}`.

## Runtime and consuming interface

The manifest retains five existing CLI operations: parse,
`lower_interface_to_target`, `emit_command_buffer`, `lower_target_to_llvm`, and
`emit_analysis_bundle`. It now selects `python3` through the execution environment
rather than an archived absolute interpreter path. An original compiler-profile
owner must bind that interpreter and all dependencies before execution.
There is no declared `target_ir_consumer` or separate checked target-IR ABI.
`compiler.py` accepts one source file; mixed lowering emits
`mixed_program_plan_v1` with source SHA, tasks, pointer ABI and LLVM task tags.
These artifacts require the original independent source/effects/output checkers;
this import does not grant any new shared-pass/provider or hardware authority.

Recorded public runtime identities were reopened locally and match:

- RTL facts v6: `b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af`.
- Untuned FireSim-aligned functional Spike: `5df9c82ca3e921823da8606284a258b491541defbafee4586fb938639e03c6b9`.
- Its libgemmini: `2dbd8445376cdda0df63abcf217c7a19f974d34a16ada21b549ed82fffd718eb`.
- Observed capability contract: `8e4e6eb5aff7e0f92012b9c306f5cc22978342c02fb8cafafee8788d8c517be4`.

The archived *target-selection* contract is separately
`bfbfac9e98ff028aa184b5b13ea44e7421afeac2c35e1d3bb572fd0cbad5aa55`;
these different contract namespaces must not be equated. The launch also pins
external gSIM, timing profile, MLIR tools, sealed corpus and model inputs.
Unavailable original/private records and binary/model assets are not verified by
this public archive import and were not opened. They must be provisioned and
rebound by their original owners; sanitized result hashes are not executable
receipts. The example numerics file is not the sealed execution configuration.

The recorded hardware is FireSimGemminiRocketConfig, 16x16 mesh, 64 KiB
accumulator, 30 MHz. Published supporting Merlin is
`f8a43c657f11ee417f4c96d8920fa2f7e66f3f5b`; historical execution used
`cd99c8637bf9ed80b95a88b74305be97f4dabcd2` and qualification used
`0e6260da0b71f27d3a4adc8f0ebcd05adcc28cf8`. Clean publication is not a
replacement for those historical source identities.

## Qualification limits

The archive records 153/153 public and 8/8 hidden capsule observations passing,
including the selected trace/structural requirements. Formal certification is
incomplete: nested original-executor setup refused before the required resource
sample was compiled. The sample still requires its original 120-second and
1 MiB limits. No resource pass, complete neural-network validation, or Phase2
formal readiness is inferred. The published source repair does not retroactively
complete this qualification.

There are 159 fresh public gSIM training observations, paired to the same kernel
objects in Spike but different ELF wrappers; one row lacks complete component
timing fields. `timing_ready` remains false. No calibration, prediction error,
physical latency, speedup or new measured run is established here.
