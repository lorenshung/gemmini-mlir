# Bounded complete builds and selected service facts

Merlin commits `601dc1929` and `b5894a6ee` connect the existing streamed
`WeightPayload` transport through the original pointer ABI renderer and preserve
exact parameter-chain references in bounded panels. Commit `a4b60fa84` corrects
staged roofline assumptions and adds generic necessary-service aggregation.
These changes preserve the selected compiler and numerical references.

The original generated compiler remains
`phase1_generated/el4_fire_aligned_v23_20261018/package` (21 files, aggregate
SHA256 `b847c973da82f408a32c7622d0c49fb6902f0a58d0f716f131d8501e34175338`).
No handwritten target renderer/backend replaces it. The new host input transport
is explicitly `execute_candidate_model(..., input_storage="streamed")`; default
inline behavior remains available for the paired comparison.

Retained evidence under
`/work/merlin/tasks/phase2-target-adapters-20261019/out/artifacts/target-adapters/gemmini/`:

- `streamed-complete-pairs-v4/payload-audit.json`: every byte of all three
  independent 1 MiB parameters, writable `.data`, file-backed writable PT_LOAD,
  64-byte alignment, nonoverlap and all 49 caller pointers independently checked.
  Both complete executables are 3,164,416 bytes. Harness source falls from
  15,744,677 to 10,793 bytes. One fresh guarded pair used 6.725 versus 3.665 CPU
  seconds; these are build costs, not target latency or repeated performance evidence.
- The same final pair directory contains `signed-small-{inline,streamed}`:
  both original native Spike executions match all 960 signed output values exactly.
- `streamed-complete-ladder-v1/payload-audit.json`: complete 256 KiB distinct,
  1 MiB distinct and 256 KiB shared-parameter programs linked and every parameter
  byte/pointer checked. Sharing retains all four source uses. The 3 MiB complete
  pair supplies the upper affordable rung. Larger native execution was not requested.
- `parameter16m-reference-supplement-v1/index.json`: separate explicit admission
  evaluates all 16,777,216 products and all 4,096 original live-out elements. The
  original frozen ordinal-4 reference-budget refusal remains unchanged.
- `parameter16m-admission-v1` records the one fresh successful original capture.
  An operator metadata-key mismatch is retained. `parameter16m-admission-v2`
  reuses those exact captured bytes, checks typed source work, all references and
  all 16 MiB parameter bytes, then builds the real parameter and kernel objects.
- `streamed-parameter16m-full-build-v1` retains the precise complete-build refusal:
  `neutral harness exceeds its complete storage budget before input rendering`.
  The original complete-storage ceiling was not increased. No model was truncated,
  reference omitted or numerical tolerance changed to admit it.
- `reduced-graph-admission-v2/matrix.json`: the three preserved depth-reduced
  public sources have 28 operation-family requirement rows. Static witnesses are
  an overapproximation, with 101 unclassified framework-call surfaces retained;
  this is not a complete dynamic operator inventory. All complete graphs remain
  refused pending selected whole-graph normalization, attention/probability,
  nonlinear/readout and carried-state numerical contracts.

`backend/gemmini_roofline_resources.py` reopens the original selected facts and
FIRRTL input-service proof. Its live owner pins the selected files and generic
resource observers; saved JSON cannot construct it. Actual selected rate ceilings
are 256 MAC sites/cycle, 16 bytes/cycle in each DMA direction and at most one
accepted execute input row/cycle. Cutpoint examples derive 4, 8 and 16 rows from
the selected literal/cone, not guessed instruction latency. These are structural
service facts only. The original packet-to-reachable-control-state invocation
mapping is still unavailable, so the artifact cannot authorize an actual input
service floor. Physical packet coverage, bank read/write ports/arbitration,
readout/completion and complete-call host/packing service remain UNKNOWN.

The source-facts artifact is `selected-resource-facts-v1/record.json` under the
same retained evidence root. Generic floor/feedback use is documented in Merlin
`docs/guides/phase2_resource_floors.md`. No predicted attainment, whole-trace to
kernel attribution, target latency improvement, hardware support or roofline
convergence is claimed. No new gSIM, headline-model or private training run occurred.

Reproduction drivers, guards and exact artifact pins are retained in
`build-roofline-handoff-v1/reproduce` and its manifest. They use the original
shared CPU 6/7 pool and the Phase 1 runtime read-only, with the task's own TMPDIR.
Per-stage costs remain separate; parallel walltimes are not summed. Author,
operator and helper token usage is unavailable and is not estimated.
