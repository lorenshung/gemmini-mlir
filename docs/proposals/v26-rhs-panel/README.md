# Experimental V26 RHS-panel adapter — unenabled

This proposal is **not enabled** in an ordinary compiler, factory, selected library or baseline. The helper and static tests are retained for review. There is no native/Spike/ELF execution result, timing qualification or full-model speedup for this candidate. Immutable original V26 package29213759 remains unchanged.

The target helper delegates choice and scheduling to the existing shared `dense_planning` and `block_schedule` algorithms. It selects a full-K, one-N-tile stripe and a bounded group of M tiles by source shape, expected requested traffic and available private rows. Multiple accumulator slots preserve complete increasing-K reductions. This removes the prior complete-RHS64-tile policy restriction without adding an asynchronous/K-spill mechanism.

The renderer uses original Builder/encoding helpers, independent load states0/1/2, explicit zero seeds, accumulating writes, fresh PRELOADs, original fences and raw-i32 readout. Raw functions are limited to0/1/2/3/4/6/14. All15 user-forbidden loop/FSM selectors remain forbidden. Unsupported bias/scaled/i8/transpose/resident semantics, missing private-row ownership, insufficient resources and static-budget excess decline without emission. The prospective backend hook grants the private row epoch only when no cached cross-operation RHS is live; original external-buffer and runtime ownership still require independent checking.

The selected resource snapshot is16x16,16384i8 operand rows and1024addressable i32 accumulator rows from exact facts b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af. This is an explicit target compatibility view, not a portable capacity default, alias proof or completion qualification.

Static verification:25tests pass. Original Builder events were decoded by the selected original operand decoder and checked by the existing independent product/state/readout checker. Shared-schedule execution was independently compared with matrix multiplication. Positive witnesses include544, existing public175(64x256x64), public96(64x464x48), public127(81x641x63), deepK and independent tails. Negative controls cover missing/late zeroing, early readout, operand clobber, masked tails, foreign ownership, bad accumulation and unsupported semantic/resource views. These rendered-event tests do not establish linked code or actual hardware execution.

`candidate-summary.json` pins the immutable17-file candidate c6745e1650b6a8f90d6cf40344ac7e2de5e90d519fbf449936e8dfc4ca817911, its inventory, helper/tests and prior test cost (8.120CPU/8.153wall, one sharedCPU24). Only backend.py/contraction.py differ among the original16files; rhs_panel.py is new. `candidate.patch.disabled` preserves those prospective changes as text and is not applied by any production entry point.

`composition-preparer.py.txt` preserves an **untested, disabled draft**. It is outside import/test paths and must not be executed or treated as a safe preparation API. Filesystem/no-overwrite behavior and composition with other proposals remain unvalidated. No composed candidate is qualified.

Focused host tests require explicit `PANEL_FACTS` and `PANEL_CONTRACT` metadata paths, selected shared-library Python roots and original CPU/resource controls. `PANEL_OOT_ROOT` may explicitly select this checkout. Without external selected metadata, those integration checks skip with an explicit reason. No target tools run. Final25-control verification after adding optional metadata-fixture guards also passed:7.828CPU/7.828wall onCPU24; its exact receipt is pinned in the summary.

```text
python -m pytest -q merlin-support/tests/test_v26_rhs_panel.py
```

Original live source/numerical/ELF/ISA/ownership checks and any timing-domain qualification are still required before use. There is no claim that this proposal closes a quoted full-model90x gap.
