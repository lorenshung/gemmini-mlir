# Task D follow-up: bounded model-scale layout compilation

Implementation: `f1311c2210cc1a2967cdd01bd1420885cf255722` (Loren Hung), based on `026f850`. No rebase onto `719bada`.

## Result

All 17 model-size builds complete in 0.17–2.86 seconds at 57–70 MiB peak RSS. The output tensor is `1x256x56x56`; concat uses two `1x128x56x56` inputs. Tests enforce <30 seconds, <384 MiB, <64 retained instruction templates and <600 emitted LLVM operations. Six baseline cases fail under a 768 MiB address-space cap. The mixed-quantization int8 transpose reproduces the reported call path without allowing another uncontrolled memory spike. The user-reported 37 GB full-model build was not rerun uncapped.

## Changes and exactness

- Structural shape/stride proofs replace dense maps for copy, transpose/permutation, concat, broadcast, slices/inserts, static reshapes and fills. Quantized matching handles closed compositions (including the real capture pad/repeat forms) back to DQ leaves. Native data moves as ordered opaque byte groups, preserving float payloads. No model names, provenance selections, captured outputs or benchmark constants choose schedules.
- Pure movement builders no longer call `_dense_linear`. A return anchored to an already bound producer no longer rebuilds and then discards that producer through `match_linear`. This duplicate lowering caused the large transient allocation even after an otherwise successful byte-movement match.
- Affine CPU loop descriptors are retained before instruction encoding. LLVM consumes these loops directly. Full tiles and tails are explicit constant forms; simulation, hazard checks and encode/expand equality stream their execution. Command forms are verified once per template. Retained objects scale with rank/jobs rather than tensor elements or all dynamic commands.
- Identical qparams retain `storage_codes` proofs and scratchpad-only DMA. Differing qparams use exact i8-to-i32 widening, exact rational host RNE/clamp once per source, then DMA in destination code units. Pointwise quantization commutes with closed selectors. Independent scales, unsigned zero points and concat axes 1/3 are tested. No on-device scaled or narrowed accumulator readout is introduced.
- Uniform host readout parameters produce one CPU loop without tensor-sized parameter arrays. Literal fills use a repeated storage pattern. General interpreted selector/arithmetic maps have an explicit 65,536-point fallback budget. Compact DMA schedules have 1,048,576-command and 64 MiB total-argument budgets. Oversized unsupported forms refuse before map/payload allocation. These compiler budgets are explicit limits, not hardware capacity claims.
- The [provider contract](core_aten_bitwise_movement.md#bounded-layout-schedules) records loop semantics, readout ordering and the Merlin consumer requirement. No Merlin or model2MLIR source was edited.

## Before/after measurements

Baseline source is an immutable `git archive 026f850` under `out/taskD2/baseline`. Each row runs in a fresh process with the same generated input IR, one BLAS thread, a 768 MiB address-space limit and a 40-second timeout. RSS is worker `VmHWM`; time covers all of `catalog.build` (including hazard checks and LLVM module emission), excluding Python imports and external object compilation. `OOM` is a caught `MemoryError` under that cap; its duration is time until failure, not a completed build time.

| Layout | Storage/policy | Before s | Before MiB | After s | After MiB | Templates | LLVM ops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transpose | i8/native | 4.15 | 167.8 | 2.80 | 57.4 | 7 | 51 |
| transpose | f32/native | 4.07 | 173.3 | 2.76 | 66.4 | 7 | 54 |
| permute | i8/native | 22.42 OOM | 687.7 | 2.78 | 57.8 | 7 | 51 |
| permute | f32/native | 4.02 | 186.2 | 2.74 | 66.0 | 7 | 54 |
| copy | i8/native | 12.52 | 666.8 | 0.19 | 57.8 | 7 | 30 |
| copy | f32/native | 1.34 | 151.0 | 0.70 | 66.8 | 7 | 30 |
| cat | i8/native | 12.76 | 559.8 | 0.17 | 57.8 | 12 | 47 |
| cat | f32/native | 1.42 | 162.9 | 0.64 | 66.8 | 12 | 47 |
| transpose | i8/same | 3.90 | 166.8 | 2.74 | 57.8 | 9 | 53 |
| transpose | i8/requant | 16.37 OOM | 685.0 | 2.85 | 70.1 | 14 | 224 |
| permute | i8/same | 3.99 | 186.8 | 2.65 | 57.4 | 9 | 53 |
| permute | i8/requant | 16.25 OOM | 686.1 | 2.86 | 69.8 | 14 | 224 |
| copy | i8/same | 0.69 | 151.3 | 0.27 | 57.4 | 9 | 32 |
| copy | i8/requant | 16.27 OOM | 685.3 | 0.44 | 69.8 | 14 | 203 |
| cat | i8/same | 0.86 | 163.4 | 0.19 | 57.4 | 14 | 49 |
| cat | i8/requant | 20.37 OOM | 686.8 | 0.37 | 70.1 | 24 | 388 |
| transpose | i8/mixed quantized module | 15.49 OOM | 683.2 | 2.79 | 57.8 | 7 | 51 |

For example, native int8 copy drops from 12.52 s / 666.8 MiB to 0.19 s / 57.8 MiB. The problematic transpose in a mixed quantized module fails at 15.49 s / 683.2 MiB under the cap; the fixed build completes in 2.79 s / 57.8 MiB with seven templates and 51 LLVM operations. All native f32 and identical-qparam cases preserve bytes/codes; differing-qparam cases retain the full-i32 exact boundary.

These measurements concern compiler resources. They are not kernel, RTL or FireSim performance measurements. Dynamic commands for an innermost-axis gather can remain large; compact loops remove Python/code-generation expansion. The small Spike batches separately record simulator cycles and retired Gemmini instructions in the [receipt](perf_records/taskD2_layout_scalability_20261018.json).

## Validation

- Full requested suite: **355 passed, 12 skipped**, 103 subtests passed, with the supplied real RTL facts selected.
- Final layout/resource suite: **41 passed**, including the 17 isolated model-size builds, no-dense/no-sparse-map guards, exact loop-encoding checks, deterministic descriptors, oversized-input refusal, and independent unsigned concat qparams. The two unsigned-cat cases were added after full-suite collection and are covered here.
- Spike: **18/18 bit-exact** small capsules, with spatial/channel tails, native int8/f32 payloads, identical qparams and exact requantization. The raw batch has **0 accumulator MVOUTs**. The requant batch has **18 full-i32 accumulator MVOUTs, 0 narrowed**. Both independently linked ELFs pass `final_elf_audit` and the mandated design-rule checker, which exits zero.
- Real model2MLIR f32/int8-recipe movement captures from Task D: **15/15** replay bit-exactly on Spike after the change, with design rule OK and no accumulator MVOUTs. All original capture bytes remain unchanged.
- No full paper ResNet-50, Verilator or FireSim execution is claimed.

## Reproduction and retained artifacts

`tests/core_aten/test_layout_scalability.py` contains the generated IR, isolated worker and resource assertions. `tests/core_aten/e2e/layout_dma.py` builds the independent byte/code checks and both final-ELF audits. Baseline/final resource JSON, before/after logs, ELF/object/source/catalog files, Spike logs, audit records and real-capture replay artifacts are retained under `out/taskD2/`. Hashes and the complete measurement records are committed in the receipt.

```sh
PYTHONPATH=/scratch/loren/core-aten-ref/merlin/src MERLIN_RTL_FACTS=/scratch/loren/core-aten-ref/facts/gemmini-rocket/facts.json /scratch/loren/venvs/gca/bin/python -m pytest tests/core_aten --ignore=tests/core_aten/test_real_model2mlir.py
# With the same LLVM/RISC-V/rocc-tests environment used for Task D:
/scratch/loren/core-aten-snap/venvs/m2m-dev/bin/python tests/core_aten/e2e/layout_dma.py out/taskD2/spike-final --design-rule /scratch/loren/core-aten-snap/run4-out/scripts/design_rule.py
```
