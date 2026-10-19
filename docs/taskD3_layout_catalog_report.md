# B6: model-scale movement catalogs and literal layout folding

Implementation: `41652edc974bd4cb4f8a26fb4c40684f3fe953ad` (Loren Hung), based on `8ddbad9`. No rebase, push, Merlin edit or Verilator run. The [receipt](perf_records/taskD3_layout_catalog_20261019.json) records source identities, measurements, Spike results and hashes for the retained artifacts in `out/taskD3/`.

## Findings and changes

The supplied `diag_catalog_wtD.json` is timestamped **October 18, 22:24 PDT**; `f1311c2` was committed at **22:40 PDT**. Re-running the diagnostic on an immutable archive of `8ddbad9` proves that the earlier fix already catches the reported integer movements: **zero calls** to `ops.permute`, `clone`, `cat` or `_dense_linear`, with no diagnostic movement cap triggered. The saved diagnostic's 994 calls / 295 capped calls therefore do not establish failure of the later fix.

This follow-up adds native floating layout matching inside quantized modules (85 additional live layouts on this source), and literal specialization before instruction generation. Constant transposes, copies, concatenations, slices and inserts evaluate the same proven DMA byte rectangles into typed stored constants, with no runtime layout binding. Byte views preserve all integer codes, signed zeros and NaN payloads without floating arithmetic. Literal convolution operands also specialize their internal layout conversions into private constants, with zero conversion instructions; source bindings and signed/unsigned code-packing proofs remain explicit. Existing identical-qparam `storage_codes` proofs and the differing-qparam full-i32/exact-host-requant path remain intact.

Repeated identical schedules now share their complete hazard audit within a catalog build. Packed literal recognition/global emission/printing avoid whole-tensor Python scalar expansion. Original operation ordinals survive folding; dead source operand edges are removed before closed-region proofs. Host rewriting replays and verifies the authenticated fold records. The [contract](taskD3_constant_layout_contract.md) describes the proof and manifest fields.

## Measured compiler resources

The exact real source is `/scratch/loren/core-aten-snap/models-out/resnet50/prepared.mlir`, **5,792,035 bytes**, SHA256 `b613c453912788539cb7b0a0917d5c51728ed9c79121a551d4d2e34fbb3052aa`. It is byte-identical to the prep bundle's `model.mlir`. Both diagnostic runs use the supplied `diag_catalog.py`, `--cap 100000000`, one BLAS thread, the live Merlin infrastructure, and this worktree's provider/code. Outputs and temporary paths are local to this worktree.

| Work | Before (`8ddbad9`) | After (`41652ed`) |
| --- | ---: | ---: |
| Real source: parse + match only | 5.6 + 8.2 = **13.8 s** | 6.6 + 9.8 = **16.4 s** |
| Diagnostic peak RSS | 1,503.6 MiB | 1,503.9 MiB |
| Old movement constructor calls | **0** | **0** |
| Live byte-layout matches | 1,751 | 1,836 |
| Full real `catalog.build` | **>300 s**, timed out | **165.68 s**, completed |
| Full-build peak RSS | 1,605.4 MiB, until timeout | 1,634.0 MiB |
| Synthetic 320-movement `catalog.build` | **>60 s**, timed out | **8.07 s**, completed |
| Synthetic peak RSS | 88.3 MiB, until timeout | 89.8 MiB |

Timeouts are censored completion-time bounds, not successful builds. Diagnostic time includes parsing/matching but excludes LLVM generation; full build includes parsing, matching, complete hazard checks, instruction encoding/form verification and LLVM module generation. Python imports and external object compilation are excluded from the full-build timer. These are compiler measurements, not model/kernel/RTL execution times. One earlier working-copy full-build sample was 229.11 s at essentially the same RSS; shared-machine timing varies. The delivered-commit run above is pinned separately.

The full real catalog has **1,891 bindings, 289 unique kernels, 18 declines and 83,054 LLVM operations**. Declines retain the unproved arithmetic/quantization cases. RSS is essentially unchanged: `8ddbad9` had already removed the per-element movement map blow-up. Whole-catalog arithmetic schedules and their complete instruction audits remain a substantial cost.

The synthetic module keeps all **280 runtime movements** and **40 constant transposes** live. Runtime tensors are `1x256x56x56`, split across int8 and native f32, using transpose and copy forms. Each constant transpose contains **2,359,296 int8 elements**. The result has four runtime kernels, 162 LLVM operations, 1,960 retained command templates and 40 authenticated folds. The test asserts <60 seconds, <384 MiB peak RSS and a 768 MiB address-space limit. Existing single-layout tests also cover model-size permutation, copy and cat, identical qparams, exact requantization and mixed modules. A constant transpose previously required **313,347 dynamic commands** under the already compact DMA path; folding requires **zero runtime commands**. No Spike-cycle reduction for this large constant is inferred from that count.

## Exactness and validation

- Requested suite at the delivered commit: **375 passed, 12 skipped, 103 subtests passed** (170.67 s). Final focused constant/layout suite: **18 passed**, including signed and unsigned literal convolution inputs.
- Spike layout batches: **18/18 bit-exact** (12 native/code movements, six exact-requant cases). Raw batch: zero accumulator MVOUTs. Requant batch: 18 full-i32 stores, zero narrowed stores.
- Spike folded-literal consumers: **2/2 bit-exact** for int8 and native f32; zero accumulator MVOUTs. The runtime kernel consumes the stored folded literal; no transpose kernel is present.
- Spike internal convolution-layout specializations: **3/3 bit-exact**, covering forward, input-gradient and weight-gradient schedules, with zero runtime conversion instructions. Convolution arithmetic stores remain full i32.
- Real model2MLIR f32-input/int8-recipe movement captures: **15/15 bit-exact** on Spike, including pads, scatters, repeat, copies and fills. Every case has `storage_codes` and zero accumulator MVOUTs. Original source/input hashes are retained.
- All **19 final ELFs** pass `final_elf_audit` and the mandated `python3 -I .../design_rule.py <elf> 268435456 <json>` checker. Across the 38 checks, no accumulator store is narrowed. No full ResNet device execution, Verilator or FireSim measurement is claimed.

## Remaining Merlin integration

**The real ResNet source has 109 function arguments and zero dense tensor constants.** The prep's CONSTANT-labelled stored weights enter the provider as arguments, so the backend cannot freeze their values from this source alone. They continue to use bounded raw DMA. Folding actual IR literals, including embedded convolution layouts, is implemented and tested here.

Freezing external stored weights requires Merlin's authenticated immutable-parameter/caller contract before the provider callback. Also, Merlin's current `device_build.apply_region_offload` rejects a catalog with no ordinary bindings even when it has successful `constant_folds`. The [contract section](taskD3_constant_layout_contract.md#merlin-requirements-not-implemented-in-this-worktree) names the exact files, functions and fields required. These changes belong to Merlin and were not duplicated or edited here. Original model accuracy inputs/goldens remain immutable.

## Reproduction

```sh
# Set GMLIR to the desired archive/worktree and PYTHONPATH to it plus live Merlin/src.
OPENBLAS_NUM_THREADS=1 /usr/bin/time -v \
  /scratch/loren/core-aten-snap/venvs/merlin-dev/bin/python \
  /scratch2/loren/gemmini-coverage/workloads/paper_models/diag_catalog.py \
  /scratch/loren/core-aten-snap/models-out/resnet50/prepared.mlir \
  out/taskD3/diagnostic.json --cap 100000000

PYTHONPATH=/scratch/loren/core-aten-ref/merlin/src \
MERLIN_RTL_FACTS=/scratch/loren/core-aten-ref/facts/gemmini-rocket/facts.json \
  /scratch/loren/venvs/gca/bin/python -m pytest tests/core_aten \
  --ignore=tests/core_aten/test_real_model2mlir.py
```

`tests/core_aten/test_constant_dma.py` contains the budget source/isolated worker. `tests/core_aten/e2e/constant_layout_dma.py`, `layout_dma.py` and `quantized_movement.py` contain the Spike experiments. Use the prior Task D LLVM/RISC-V environment with OUT, TMPDIR, MERLIN_OUT_ROOT and GEMMINI_TEST_WORK under this worktree's `out/`. Full-build driver, before/after archives/logs, sources, objects, ELFs, checker results and the original failed/intermediate experiment logs remain in `out/taskD3/` with hashes in the receipt.
