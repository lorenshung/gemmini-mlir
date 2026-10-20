# Exploratory EL4 scheduling candidates

These packages fork the preserved generated compiler at
`phase1_generated/el4_fire_aligned_v23_20261018/package`. That baseline remains
byte-identical, with 21 files and tree SHA256
`b847c973da82f408a32c7622d0c49fb6902f0a58d0f716f131d8501e34175338`.
The handwritten repository-root compiler is not substituted.

These are experimental packages, not formally admitted Phase 2 incumbents.
The user stopped manual optimization at 21:18 UTC and requested the existing
automatic sandboxed experiment route. That process was not launched. Its Phase 1
and performance-corpus admission prerequisites remain incomplete. These source
snapshots preserve exploratory work only; they are not an automatic-search result.
The copied `REPORT.md` files describe the historical Phase 1 author report;
they do not qualify the new schedules. Numerical/readout policies are unchanged.

| Package | Mechanism |
| --- | --- |
| `rhs_residency/package` | Retain a whole RHS above live explicit resident handles and reuse it across M tiles when its padded rows fit. |
| `rhs_config_state/package` | Additionally hoist invariant store configuration and unchanged reduction-load configuration. |
| `rhs_config_stream/package` | Additionally let the selected reservation station order direct-schedule dependencies, preserving caller entry/exit fences and all host-packing fences. |
| `rhs_panel_stream/package` | Additionally retain one K-by-array-width RHS panel across M tiles when the whole weight cannot fit. |
| `rhs_panel_pingpong/package` | Additionally alternate two A scratchpad slots in resident schedules. |
| `rhs_acc_pingpong/package` | Additionally alternate two private output accumulator slots. |
| `rhs_load_states/package` | Fork the A-buffer candidate and use three independent load states for seed/panel, A and B. |

`candidates.json` pins each 21-file package and its changed baseline files.
Incomplete batches, timeouts and refusals remain in the task evidence; no
complete cohort or full-model candidate regression was established.

Choices follow contraction shapes, layouts, live storage and selected target
capabilities. No model name, benchmark identity, reference tensor or golden
output selects a schedule. Whole-RHS and panel allocations preserve rows 0–31
for ordinary temporaries and never exceed the preserved target's 16,384 rows.
Insufficient capacity retains the prior direct schedule.

The task's Merlin example driver records the exact candidate/source/input/ELF,
simulator and counter-window identities, complete-output comparisons, paired
repeats, refusals, costs and roofline assumptions. Qualification of the full
models, the native Spike timing engine, and the complete public corpus remains
separate. An observed kernel improvement is not a whole-model speedup.

## Automatic V26 source development

`automatic_v26_single_full_tile_v6` preserves the exact one-file delta produced
by the sandboxed V26 author on 20 October. It starts from the separate immutable
16-file V26 package, not a V23 candidate or the handwritten backend. Its
`experiment.json` records both package digests and the original refusal.

The author reported structural tile-coverage checks. Its native command sandbox
could not reach the broker, so no queued compilation, complete-output comparison
or timing ran in that author attempt. The original checkpoint is refused and
unconsumable. This patch is an unpromoted hypothesis; any independent subsequent
queue evaluation must retain its own identity and results.

A subsequent independent original queue evaluation passed both arms' 544
outputs and 8,704 source products. LLVM differed, but the final ELF and all 125
scoped count features were identical. The performance hypothesis is rejected.
Native timing remains UNKNOWN because flush activity is outside the fixed
calibration support. This evaluation does not complete the earlier author run.

To reproduce the source only, copy the V26 package into a fresh candidate
directory, make that copy writable, then apply the patch with
`patch -d CANDIDATE_PACKAGE -p1 < automatic_v26_single_full_tile_v6/candidate.patch`.
Keep the original package read-only and recheck the resulting package digest
against `experiment.json` before evaluation.

`automatic_v26_rhs_reuse_85f47996` retains a later exact authored source delta
from the same immutable V26 baseline. Its original queued replay produced
different LLVM, then both arms refused before native build because of a
queue-to-leaf timeout mismatch. Feedback and an UNKNOWN decision were published,
with complete process cleanup. No outputs, linked ELF or fresh counts were
produced by this replay. The subsequent host fix has regression coverage but
was not target-reexecuted. This candidate remains unpromoted; see its README
and pinned `experiment.json` for the separate evidence and remaining checks.
