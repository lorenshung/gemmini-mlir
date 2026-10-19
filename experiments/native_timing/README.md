# Original native timing selection

`merlin-support/backend/gemmini_native_timing_semantics.py` owns the selected
target-specific correspondence between a frozen native evaluator release and
the current software/RTL selection. It imports no generated or handwritten
compiler. Original Phase 1 compiler and functional-extension bytes stay fixed.

`gemmini_execute_service.derive` additionally reads the selected FIRRTL
`ExecuteController.total_rows` cone and the accepted-row counter. It exhaustively
checks primitive-sized operand extents and explicit control cutpoints, preserving
source locators and exact input pins. The shortened-stream floor and primitive
extent come from the actual expressions; neither 16 rows nor a four-cycle floor
is a default in the proof. Selected evidence gives a 16-row default and a
four-row minimum when the dataflow, garbage operand and transposer predicates
permit shortening. The counter admits at most one input row per cycle.

This is input-row service evidence only. It does not establish reachable states,
mesh propagation, bank stalls, output service, completion or end-to-end latency.
Existing nominal padded-MAC counters keep their interpretation. A future
operand-sensitive evaluator must select this evidence explicitly and obtain its
own source/runtime review and unchanged held-domain validation.

The original host constructs the live owner:

```python
owner = NativeTimingSemanticsOwner.bind(
    original_count_correspondence_owner,
    source_root=selected_chipyard_sources,
    spike_source=reviewed_spike_sources,
    observer_build=frozen_observer_build,
    provider_bundle=closed_provider_sources,
    rebuild_dir=fresh_task_owned_directory,
)
binding = PassCapabilityBinding(**owner.selected_binding())
capability = ReviewedPassCapability("native_timing_semantics", binding, owner)
```

The five binding fields exactly match the original count owner: selected facts,
observed software contract, count-feature schema, model file and ordered runtime
libraries. Neither saved status JSON nor passing timing statistics constructs
this capability. Construction rederives selected reservation/configuration and
PRELOAD completion cutpoints, corroborates cache geometry against selected DTS
and FIRRTL storage declarations, and rebuilds the closed provider under the
shared CPU lease. The resulting library must match the model's actual library
digest. Revalidation reopens every original source, model, runtime and helper pin.

The admitted abstraction uses actual command operands, local-storage effects,
configuration state, reservation ownership and completion. Four stages remain
distinct: admission, issue, controller consumption and final effects. A solitary
PRELOAD survives a fence; its destination tag retires with the later compute.
Load/store configuration completes at controller issue while visibility follows
controller consumption. Unsupported paired mesh/tag overlap remains unavailable.

Whole-command local-store holds are deliberately conservative. Each selected
scratchpad bank has one modeled reservation; all accumulator aliases/ports share
one conflict class. This does not claim one physical accumulator port. Aggregate
DMA/cache resources, LRU/coherence policy and future reservations can reduce
modeled overlap. Requested payload bytes, modeled cache traffic and physical
traffic retain separate meanings. No exact packet arbitration claim is made.

The owner issues no statistical timing or hardware-execution qualification.
The ordinary native queue additionally requires unchanged held ordering/error,
coverage, family and calibration-group gates, and independently checked complete
candidate outputs. Unpriced operations and unsupported domains remain UNKNOWN.
Public archive validation is reused diagnostic validation, not unseen final
generalization; large capacity and attention regimes still require suitable
independent labels.

The retained native-v2 public calibration, source/runtime pins, failed models,
costs and reproduction command are recorded in the
[2026-10-19 handoff](20261019-native-v2-handoff.md).
