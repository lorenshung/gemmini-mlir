# Task D: exact quantized movement storage codes

Implementation commit: `7e96ca4227011c99d6d83f26e5ad22efb689bead` (Loren Hung).
Baseline: `f3d92d5e1be080046906d5c809cdff3589293b01`. No push, PR, or edits outside wtD.

## Change and proof

The provider proves closed static element selectors between Q/DQ boundaries, compares all source and output qparams using exact scale rationals, and emits strided int8 scratchpad DMA for identical contracts. There is no accumulator access, i32 scratch, signedness packing or runtime requantization. Literals receive their one exact rational quantization at compilation. Different contracts retain the existing exact i32/host path. Native F1 and arithmetic schedules remain separate contracts.

Bindings and kernel metadata record `readout="storage_codes"`, the identity equation, the source Q/DQ ordinals, original `prov.quant.*` attributes, reduced rationals, zero points, ranges and storage dtypes. The source digest and kernel digest authenticate the proof. See [the consumer contract](core_aten_bitwise_movement.md#f3-quantized-movement-storage-codes) for exact fields and Merlin checking requirements. The provider admission API checks scratchpad-only instructions; all 15 fresh captures also pass that API. Merlin itself was not edited.

## Measurements

The same real Phase 0 f32/int8-recipe capsules and input bytes were replayed against the baseline extracted with `git archive f3d92d5` and the candidate. Checks compare every output code against an independent rational interpretation of the source. Both lanes pass all 15 capsules and the independent final-ELF design-rule checker. The candidate has zero accumulator MVOUTs; the baseline has 29 full-i32 MVOUTs.

Cycles below are Spike `mcycle` deltas around kernel calls, including the baseline generated host requantization. Input quantization and final dequantization are evaluated by the independent host oracle before this device-kernel harness. These are neither whole-model cycles nor RTL/FireSim cycles. Gemmini instructions are dynamically retired RoCC commands from the independent checker.

| Capsule | Baseline Spike cycles | Storage-code cycles | Baseline Gemmini commands | Storage-code commands |
| --- | ---: | ---: | ---: | ---: |
| clone | 413 | 27 | 7 | 5 |
| cat | 787 | 33 | 8 | 7 |
| constant_pad_nd | 1547 | 62 | 10 | 21 |
| reflection_pad1d | 418 | 45 | 11 | 13 |
| reflection_pad2d | 1642 | 104 | 31 | 45 |
| reflection_pad3d | 8124 | 475 | 133 | 209 |
| replication_pad2d | 1647 | 114 | 27 | 49 |
| replication_pad3d | 8077 | 432 | 109 | 227 |
| slice_scatter | 685 | 70 | 17 | 25 |
| select_scatter | 427 | 54 | 13 | 17 |
| repeat | 785 | 31 | 8 | 7 |
| _to_copy | 413 | 27 | 7 | 5 |
| full | 30 | 27 | 5 | 5 |
| full_like | 30 | 27 | 5 | 5 |
| fill | 30 | 27 | 5 | 5 |
| **Total** | **25,055** | **1,555** | **396** | **645** |

Total Spike kernel-call cycles decrease 93.8% (16.1x). Gemmini commands increase 62.9%: irregular pads/scatters emit more general byte-selector DMA jobs than the old accumulator gather. Clone, cat, repeat and `_to_copy` reduce command count; fills keep the same five commands. This negative instruction-count evidence is retained rather than described as an instruction reduction. A future semantic DMA job coalescer can address irregular-map command counts.

## Qualification

- 15/15 original Phase 0 captures: bit-exact baseline and candidate on Spike; design rule OK.
- 15/15 fresh model2MLIR captures: f32 inputs, the same supplied int8 recipe, independent tail shapes, reflection/replication pads, nonzero constant padding, both scatters, repeat, clone, cat, fills and f32-to-f64 `_to_copy`; bit-exact on Spike, design rule OK, no accumulator MVIN/MVOUT, no runtime requantization.
- Unit coverage exhausts every legal storage code for signed, asymmetric/narrow and unsigned/narrow schemes; separately changes scale, range endpoints, zero point and storage dtype to verify the exact fallback. Arithmetic identities cannot select the byte rule. Literal rounding/clamping, rational normalization, source rewriting and provider proof checks pass.
- Requested test suite: 316 passed, 12 skipped; 103 subtests passed (with `MERLIN_RTL_FACTS` selecting the supplied Gemmini Rocket facts).
- Focused movement, schema and exact host-requant validation: 84 passed, 1 skipped (including unsupported-storage rejection). The preceding 16 movement checks plus the existing routing regression also pass with explicit RTL facts (17 passed).
- The initial no-facts suite had 298 passed, 12 skipped and one existing routing failure (`ambiguous or underivable device datapath`). Selecting the supplied real RTL facts resolves that regression without a source change.
- The design-rule checker calls the byte-only stream `OK_full_i32` because it has no narrowed reads; its separate counters explicitly show `acc_full=0`, `acc_narrow=0`.
- No Verilator or FireSim run was requested or performed.

## Reproduction and artifacts

Raw artifacts remain in `out/taskD/`: `candidate/`, `control/`, `fresh/`, `fresh-captures/`, `fresh-provider-proofs.json`, test logs and `recipe.json`. Each replay directory retains source, source-bound catalog, rewritten model, shim, ELF, exact-check Spike console and design-rule receipt. The compact [qualification receipt](perf_records/taskD_quantized_movement_20261018.json) commits per-case source/input/ELF hashes and both cost measures.

The supplied `env-gwtD.sh` path was absent; `out/env-taskD.sh` is derived from the existing `env-gD.sh`, with OUT/TMPDIR/MERLIN_OUT_ROOT and device outputs redirected into this worktree. All captures use `/scratch/loren/core-aten-snap/venvs/m2m-dev/bin/python`. The environment enables the existing device runner toolchain paths.

```sh
source out/env-taskD.sh
$M2M_PY tests/core_aten/e2e/capture_quantized_movement.py out/taskD/recipe.json out/taskD/fresh-captures
$M2M_PY tests/core_aten/e2e/quantized_movement.py out/taskD/fresh-captures out/taskD/fresh --design-rule /scratch/loren/core-aten-snap/run4-out/scripts/design_rule.py --expect-storage-codes
$M2M_PY tests/core_aten/e2e/quantized_movement.py /scratch/loren/core-aten-snap/run6-out/lane-pk/run out/taskD/candidate --design-rule /scratch/loren/core-aten-snap/run4-out/scripts/design_rule.py --expect-storage-codes
```

For the control, point `PYTHONPATH` at `out/taskD/baseline` (the archived baseline source) and run the identical replay driver without `--expect-storage-codes`. The harness uses the M-mode `mcycle` CSR so the mandated checker ISA, which omits Zicntr, can execute it. This does not change the device stream.
