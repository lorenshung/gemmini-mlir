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

## Narrow copied-compiler integration

`gemmini_quiescent_load_config.py` now implements an explicit Phase 2-only
source-relative contract and hook, separate from the broad unavailable families
above. It freezes selected input/source pins, reconstructs the preserved C
helper bodies against selected operand encoding, and requires one-hart ownership
with no external configuration writer. It supports only state zero, identity
f32 scale, no shrink and a configuration following an original completion
fence. It uses shared `eliminate_redundant_configuration`; policy compares
actual configuration values rather than model names or known shape identities.

The hook operates on the original direct emitter's bounded unrolled statement
stream. Every other statement stays byte-identical. Unknown controller calls
invalidate known state. Dynamic configuration, other modes and effectful pointer
expressions are barriers. Loops retain the original route. The contract records
source premises independently from hardware qualification; the intake's broad
`configuration_cleanup` preflight remains unavailable.

[The diagnostic driver](../experiments/shared_load_configuration/qualify.py)
copies the original compiler twice and supplies the same library and effect
contract to both copies. Their hook differs only in its enable flag. It reuses
the existing public build/link/reference tools and Fire-aligned Spike runtime.
It checks complete signed outputs against the independent reference, replays
actual linked issue-register observations through the source-pinned observer,
and independently permits only redundant setter omissions. Other accelerator
commands and completion fences must match, with DRAM pointers normalized through
actual linked symbols. Logical DMA spans must fit linked objects; stores must
target declared output objects.

Run this as an infrastructure diagnostic with explicit selected inputs, not as
a resumed performance campaign:

```sh
python experiments/shared_load_configuration/qualify.py \
  --merlin "$MERLIN_CHECKOUT" \
  --facts "$SELECTED_FACTS" \
  --observed-contract "$SELECTED_TARGET_CONTRACT" \
  --baseline-package "$PRESERVED_COMPILER_PACKAGE" \
  --rocket-core "$SELECTED_ROCKET_CORE_SOURCE" \
  --capsule "$PUBLIC_CAPSULE_DIRECTORY" \
  --output "$FRESH_DIAGNOSTIC_OUTPUT"
```

Select the task's existing shared CPU roots and TMPDIR before execution. The
driver creates no campaign state and runs no gSIM or author session. Repeating
`--capsule` adds independent public cases; output directories must be fresh.
Compiler/library/contract, emitted object/ELF, input and trace identities are
retained. Complete memory-guard/input readback, physical beat-level bounds,
hardware completion qualification, qualified timing, automatic author routing
and full-model execution are still separate requirements. Successful diagnostic
issue and output checks do not silently promote the adapter into those domains.

The separate [selected timing-resource adapter](selected_timing_resources.md)
derives limited CPU issue/execution structure and audits accelerator queue and
configuration ownership. Its model augmentation preserves unknown prices and
does not qualify target latency or enable the unavailable shared families.

## Explicit functional owner admission

The copied-emitter hook now requires an adjacent exact
`shared-library-sources.json`. Both copied arms receive that same manifest,
and its bytes participate in the copied package digest. A changed ambient
library refuses before rewriting. This also prevents the exploratory builder's
package cache from silently reusing results across a library selection change.
Historical copied packages and receipts retain their original interpretation.

The operator can recheck a new v2 paired diagnostic with
`experiments/shared_load_configuration/admit.py`. Supply the exact facts,
observed contract, effect contract, staged completion sidecar, symbol inspector,
one or more paired `--report` files and a fresh `--output` receipt. The checker
reopens ELFs, inputs, traces, counter audits and runtime/library identities. It
recomputes all independent outputs, reconstructs actual linked issue operands,
checks logical DMA windows and permits only redundant fenced setter omissions.
At least three independent public sources and two actual applications are
required. Original v1 diagnostics cannot silently acquire this admission.

`gemmini_load_cleanup_capability.LoadCleanupOwner.read(receipt)` exposes the
explicit host-selected owner for the generic reviewed-capability registry:
`identity`, `revalidate()`, `record()` and `selected_binding()`. The registry
transports an original operator selection; it does not issue qualification.
The receipt, checker and all original evidence must remain outside candidate
writes. Missing or changed source, helper, runtime, library or evidence refuses
selection. No target-name discovery or installed-library inference occurs.

The owner's `rewrite(..., execution_domain="functional_spike")` consumes the
same shared configuration pass. Each new realized compiler pair must still
pass `check_actual_pair(...)` on its actual issue-register observations, along
with fresh independent complete-output and memory checks owned by execution.
Other execution domains refuse. The owner qualifies only bounded functional
source effects under the original state-zero, identity-scale, straight-line,
single-hart ownership conditions; unknown commands remain barriers. Its
completion premise concerns loads. A solitary PRELOAD may survive a fence.

This selection supplies no dense adapter, fresh-result model ABI, physical
beat/guard qualification, hardware completion qualification or latency claim.
Its record retains those limits and requires final pair checking. The original
Phase 1 compiler, numerical contracts and broad unavailable mechanisms remain
unchanged.
## Explicit development numerical diagnostics

A fresh source-only development capsule cannot borrow a Phase 1 capsule's
admission. The native whole-program evaluator first requires an explicitly
selected experiment descriptor and its adjacent corpus `MANIFEST.yaml`. The
capsule's required instruction classes must match its coverage declaration, and
the manifest must resolve the original prohibited-role policy. This is an
instruction policy selection, not hardware execution qualification.

For a bounded public integer-contraction diagnostic, make a separate copy of the
current development capsule. Preserve its source, inputs, references, traces and
numeric policy byte for byte. Derive the required classes from the exact selected
software contract's memory, weight-load, matmul and accumulator-readout roles,
and check every class against the selected observed instruction table. Resolve
the original `loop_descriptor` prohibition through Phase 0's
`derive_role_taxonomy` and `resolve_policy` within the selected contract/facts
scopes; an unresolved or vacuous prohibition must refuse. Record all original
and copied file hashes and every policy edit. Do not modify the retained capsule.

Select that new descriptor with `MERLIN_TARGET_EXPERIMENT` and use the ordinary
native evaluator with `simulator="spike"`, `oracle_tier_ceiling="L2"` and
`complete_call_timing=False`. The explicit development policy retains required
tiers L0 through L3, so a successful L2 numerical diagnostic leaves the deeper
gate pending. Reuse unchanged compiled artifacts only when their complete source
and input bindings still match.

This setup must retain false timing/certification claims and an empty performance
objective. It establishes no public-program source basis, sealed capture,
private-validation exclusion, formal Phase 2 admission or hardware-equivalence
receipt. The ordinary automatic public-program selector separately requires its
real source/capture basis; writing a development descriptor cannot provide it.
