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
