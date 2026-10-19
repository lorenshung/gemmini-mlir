# Structural resources for the selected timing model

`merlin-support/backend/gemmini_timing_resources.py` derives a limited CPU
resource augmentation and a Gemmini configuration-routing audit from the exact
selected FIRRTL. It reads the original facts/contract/operand selection through
the existing checked intake. It reuses Merlin's FIRRTL reader and evaluates
only the bounded Boolean routing cone required by this target adapter. Missing,
ambiguous, inactive, cyclic or unsupported dependencies refuse extraction.

Run with the task's TMPDIR and shared CPU admission environment:

```sh
python merlin-support/backend/gemmini_timing_resources.py \
  --facts "$SELECTED_FACTS" \
  --observed-contract "$SELECTED_TARGET_CONTRACT" \
  --rocket-core "$SELECTED_ROCKET_CORE_SOURCE" \
  --reservation-station "$SELECTED_RESERVATION_STATION_SOURCE" \
  --execute-controller "$SELECTED_EXECUTE_CONTROLLER_SOURCE" \
  --spike-source "$SELECTED_TIMING_SPIKE_CHECKOUT" \
  --input-model "$ORIGINAL_NATIVE_MODEL" \
  --harts 1 \
  --output "$FRESH_RESOURCE_OUTPUT"
```

`resources.json` retains exact selected inputs, FIRRTL, Scala audit sources,
native consumer sources, implementation and before/after model identities.
All source pins are reopened before publication. `model.json` is a separate
copy: existing memory geometry, timing domain, target model and coefficients
remain unchanged. Conflicting existing bindings or an issue interval other than
one are refused. Multiple harts require a separate resource namespace and are
currently refused.

The selected IBuf instruction and Rocket retire vectors have width one. The
Rocket has one `RocketALU`, one `MulDiv`, and no `PipelinedMultiplier`. The
augmentation therefore supplies separate scalar CPU issue, integer ALU and
shared multiply/divide resources. Integer, bit manipulation, branch and memory
address generation bind to the ALU; integer multiply/divide bind to MulDiv.
The CPU issue interval of one cycle is only the structural admission service
opportunity. Dependency readiness, replay, instruction fetch and data access,
branch recovery and operation latency/occupancy remain separate costs. In
particular the adapter supplies **no CPU execution prices**. FP arithmetic,
atomic, CSR, fence, system and custom-command execution resources stay unmapped.

The configuration routing is independently evaluated in the selected
reservation-station FIRRTL for each subtype with allocation valid:

| Configuration subtype | Dispatch class | Completes when issued |
| --- | --- | --- |
| Execute, 0 | Execute | No |
| Load, 1 | Load | Yes |
| Store, 2 | Store | Yes |
| Normalization, 3 | Store | Yes |

Selected reservation-station entry counts are load 8, execute 16 and store 4,
with one issue port per class. `ReservationStation.scala` routes commands at
lines 313–315, derives `complete_on_issue` at line 383, and releases ordinary
entries on completion. `ExecuteController.scala` accepts execute configuration
only after the active matmul and pending completed ROB IDs drain, at lines
541–581. Source pins, rather than these explanatory line numbers, bind the audit.

These facts cannot directly fill the native event engine's execution capacity
or waiting-queue capacity. Its waiting reservations leave the queue when they
start; the hardware's ordinary reservation-station entries remain owned until
completion. Load/store configuration also completes without allocating a DMA
command tracker. The current provider sends all configuration through one
configuration resource and its preload event lacks an execute-configuration
read dependency. Correct integration needs explicit reservation lifetime,
issue/service/completion stages, execute-state consumers and paired preload/
compute behavior. This adapter changes no accelerator resource mapping.

The input model's memory regime also remains exact. The saved gSIM labels use
the native emulator's `mm_magic_t` response model; the corresponding FireSim
configuration's cache geometry does not turn those labels into FireSim DRAM
bridge measurements. The parent timing workflow must keep those domains and
all simulator identities distinct.

Host regression tests exercise derivation refusals, guarded routing and exact
preservation of unknown costs and existing model fields. Extraction on selected
RTL establishes structural evidence. Neither result passes calibration or
qualifies predictions. The report retains `prediction_authority: NONE` and
`timing_qualification: not_established`; native event completeness and held-out
timing gates remain necessary.

## Staged completion and the PRELOAD fence exception

`gemmini_staged_completion.py` emits a separate
`gemmini.staged_completion_contract.v1` sidecar from the same selected inputs:

```sh
python merlin-support/backend/gemmini_staged_completion.py \
  --facts "$SELECTED_FACTS" \
  --observed-contract "$SELECTED_TARGET_CONTRACT" \
  --source-root "$SELECTED_CHIPYARD_SOURCE_ROOT" \
  --output "$FRESH_STAGED_CONTRACT"
```

It evaluates actual FIRRTL control cones under explicit state cutpoints, rather
than treating Scala text or resource dimensions as executable support. It checks
the reservation busy truth table, propagation through global busy, preload start
with a following valid command and no RAW hazard, and execute-configuration
completion only after the matmul and pending ROB completions drain. The sidecar
retains the selected FIRRTL and controller-source identities, truth tables and
source locators. Synthetic regression cones check refusal when those effects
change; a run against the selected FIRRTL is separate evidence.

The selected reservation station deliberately excludes a single solitary
PRELOAD from its busy signal. With other frontend and scratchpad paths idle,
the global RoCC busy signal can be false while that PRELOAD remains outstanding.
A Rocket fence may therefore return before its retirement. A following valid
command allows the execute controller to start the preload. The current native
observer's unconditional event-engine fence drain does not implement this
exception. It must retain the pending preload's reservation, descriptor and
execute-configuration dependencies. Adding a read of execute configuration to
the preload event alone does not fix the reservation or completion semantics.

Load/store configuration has another important stage boundary: its reservation
entry is released when accepted by the controller issue path, while its
persistent configuration becomes visible when consumed in controller FIFO
order. It does not allocate a DMA command tracker. Normal DMA reservations
remain owned through actual completion. A load followed by a fence must wait
for that load's busy work; the solitary-PRELOAD exception does not authorize
skipping unfinished loads.

The sidecar records these native event requirements and contrasting examples.
It supplies no prices and does not implement or qualify the native engine.
Standalone PRELOAD synchronous completion is unavailable, so the current
`merlin.dense_schedule.v1` adapter remains disabled. A future checked composite
must describe the complete preload/compute/completion effects explicitly.

## Original count-model correspondence

`gemmini_count_hardware.py` checks the original count model against the selected
facts and retained functional receiver sources. Its separate sidecar binds array
dimensions, signed element and accumulator widths, store rows and row widths,
local address width, custom opcode and mask, selected full-width readout and
pixel-repeater structures, and the store controller's declared reset value.
The functional runtime's deployment record binds its receiver sources to the
unchanged `libgemmini.so`; model library selection and CPU ISA must match that
record. Every model hardware field must equal the derived selection.

```sh
python merlin-support/backend/gemmini_count_hardware.py \
  --facts "$SELECTED_FACTS" \
  --observed-contract "$SELECTED_TARGET_CONTRACT" \
  --model "$ORIGINAL_COUNT_MODEL" \
  --runtime "$SELECTED_FUNCTIONAL_RUNTIME" \
  --receiver "$RETAINED_FUNCTIONAL_RECEIVER_SOURCE" \
  --functional-spike-source "$RETAINED_FUNCTIONAL_SPIKE_SOURCE" \
  --output "$FRESH_COUNT_CORRESPONDENCE"
```

Operator admission uses `CountHardwareCorrespondenceOwner.bind(...)` with that
sidecar and the original model, facts, observed-contract and functional-extension
paths. The factory reconstructs the source proof and rejects a saved sidecar
that differs. The immutable owner supplies `identity`, `revalidate()`,
`selected_binding()` and `record()` for transport through the reviewed-pass
registry. Revalidation reopens its original pinned inputs; a changed model,
source, target selection or runtime refuses admission. A saved success record
alone is insufficient. Candidate code must not construct or replace this owner.

The five registry bindings identify facts as hardware, the observed contract as
software, the count feature schema as numerical policy, the exact count model as
runtime, and the canonical model library list as library. Here numerical policy
identifies the count projection only; the compiler's original numerical-policy
gate remains separately required. These fields transport exact identities and
do not grant numerical, hardware or timing qualification.

This correspondence establishes the count-feature hardware association and the
original functional build association. It does not qualify physical aliasing,
DMA beat accounting, reservation ownership, completion, event scheduling or
timing. In particular, the functional receiver does not establish initialization
of its `acc_act` field; the sidecar binds the selected RTL store reset separately
and makes no full reset-equivalence claim. Hardware execution qualification
remains `not_established`, and native event timing remains unqualified. Empirical
screen qualification is a separate gate with its own source-group splits,
feature domain and original measured labels.
