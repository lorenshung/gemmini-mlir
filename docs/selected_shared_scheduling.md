# Shared scheduling against explicitly selected hardware

`merlin-support/backend/gemmini_selected_scheduling.py` is the trusted intake
boundary for the preserved compiler's selected FireSim configuration. It does
not select the curated backend or replace the generated compiler. Supply the
exact facts and observed target contract; its operand sidecar is resolved only
through the contract's recorded path and SHA256.

```sh
python merlin-support/backend/gemmini_selected_scheduling.py \
  --facts "$SELECTED_FACTS" \
  --observed-contract "$SELECTED_TARGET_CONTRACT" \
  --check-operands \
  --output "$TMPDIR/selected-scheduling-readiness.json"
```

`--check-operands` performs the source rederivation under the existing shared
compute admission controls when configured. Omit it for the cheap inventory
report; a recorded sidecar digest alone is not a fresh source rederivation.

The report binds all three input files, their selected FIRRTL identity,
configuration, observed instruction classes and structural memory dimensions.
Every later use reopens the inputs. A different facts document with the same
configuration name or FIRRTL label cannot replace the original sidecar's exact
facts bytes. Scratchpad and accumulator are separate logical address-space
views; this report does not add physical census memories to their capacities.
Unresolved aliases remain unknown. The structural dataflow observation is
reported independently from executable adapter availability.

The optional `--require configuration_cleanup` (or another report mechanism)
makes the command return 2 when that mechanism is unavailable, retaining the
report and precise refusal. Exit 0 without `--require` means only that a report
was generated. It is not device admission.

The Python `operand_observer()` delegates to Merlin's existing
`bind_original_operand_selection`. That loader verifies the pinned ISA and
controller sources, re-derives selected FIRRTL slices, preserves the original
facts, and binds the existing normalized operand observer. Reserved declaration
slots cannot fill unresolved actual widths. `decode_observation()` accepts an
actual RISC-V instruction word and the register values observed at issue. The
instruction word alone does not contain the scalar operands. The selected
custom opcode/function roster is checked; varying RoCC register-usage bits are
recorded without using them as an instruction identity filter. Decoded fields
establish neither memory effects nor completion or numerical equivalence.

## What remains disabled

The selected observed contract has empty memory/legality declarations, partial
operand roles, and no reviewed `merlin.software_spec.v1` scheduling extension.
The sidecar itself records command acceptance and semantic roles as unproven.
The following shared families consequently stay unavailable:

| Family | Missing executable contract |
| --- | --- |
| Dense blocking and reuse | Typed arithmetic, initialization, exact accumulator continuation, ownership and synchronous completion; actual encoder/projector |
| Temporary allocation and copy removal | Physical backing/alias mapping, byte addressing, reservations and complete instruction memory effects |
| Readout fusion | Selected exact arithmetic, layouts, bounds and actual operand/effect bindings |
| Configuration cleanup | Complete ordered persistent effects, including conditional writes and all other observable effects |
| Device spills | Exact synchronous save/reload and actual byte provenance |
| Complete row reduction | Original numerical algorithm and independently qualified phase primitives |
| Final program checks | Independent final linked-machine decoder, program/address/branch limits and continuation ABI |
| Asynchronous overlap | Dependency/completion effects, queue ownership and a checked asynchronous scheduling transformation |

Library availability, a decoded packet, an accepted schedule or a saved timing
result cannot enable any of these families. The current intake's `require()`
always refuses them. It supplies no adapter with an invented qualification
identity and cannot be used to turn a saved declaration into execution evidence.
The preserved compiler's existing instruction emission remains its original
route; these shared hooks are not inserted into that immutable compiler.

There is also an important realization boundary. Shared
`compile_schedule.ScheduleEncoding` currently handles finite straight-line
programs with one schedule call per decoded instruction and actual operands
reconstructed from final bytes alone. A RISC-V RoCC instruction contains scalar
register indices; the values are computed by surrounding host instructions and
loops. An issue-register observation supplies those values during execution,
but is not a bytes-alone final machine decoder. Serializing requested commands
as an invented packet stream does not check the emitted ELF. The existing
compact dense/source route has a separate C emission and source-conformance
boundary; a real integration must connect that route to its independent target
checker or implement a checked realized-machine bridge. A reviewed software
extension alone cannot erase this distinction.

## Smallest useful next enabling work

Configuration cleanup is a narrow first adapter, but raw register equality is
insufficient. The selected load controller normalizes pixel repeat zero, chooses
one of several load states and may have DMA rows in flight. Execute configuration
has conditional stride-only writes. Store configuration shares activation and
normalization state. Device loop controllers can alter configurations. A reviewed
extension must bind these effects, distinguish barriers and reset assumptions,
and establish visibility/completion through dispatch before the shared cleanup
may remove calls. Unknown commands retain their instructions and invalidate
known state.

This extension can be selected as an additional Phase 2 contract with its own
identity; it need not rewrite the original contract or require new Phase 1
authoring. The copied baseline and candidate must receive the same extension.
A new OOT adapter must lower/project actual operands, connect into the copied
compiler, and survive independent final decoding. Target tests then compare all
signed outputs, inputs/guards, carried state and final configuration across
independent shapes/tails. Configuration, fence, outstanding-memory and
loop-controller adversarial cases are necessary before enablement. Spike can
check functional behavior where supported; qualified timing is a separate gate.

The current tests exercise admission, input mutation, independent structural
dimensions, protocol routing and refusal. They are host tests. A source-rederived
observer diagnostic is also useful evidence about intake, but it is neither a
fresh accelerator execution nor complete-output checking. No hardware speedup
or target adapter qualification follows from these tests.
