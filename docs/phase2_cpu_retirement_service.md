# Scalar retirement service diagnostic

`merlin-support/backend/gemmini_cpu_retirement_service.py` derives a narrow
service inventory for the selected scalar Rocket core. `RetirementServiceContract`
requires the original live `SelectedSchedulingTarget`,
`NativeTimingSemanticsOwner`, and `ArchiveSpikeRuntime`; saved qualification JSON
cannot construct it. It reopens their source/model/runtime pins.

The FIR check follows the scalar CSR retirement interface, the WB instruction's
CSR address/command, and the CSR value through the actual register-file write
data cone. A trace vector or an unused CSR-looking node does not establish this
path. The selected original runtime must retain its 64-bit core-cycle counter
selection. Structural facts do not establish accelerator completion or timing
accuracy.

`contract.evidence(report_path, original_annotated_invocation,
counter_review_path, source_sha256=original_source_digest)` reopens the original
observer process, actual report product, ELF/input/counter scope, model and
selected observer. A preserved byte-identical report copy is allowed only when
its original actual process product reopens successfully; this is reuse, not a
new measurement. The original compile/numerical caller still owns the complete
current source-to-ELF relationship.

The observer count excludes the opening CSR and includes the closing CSR. The
diagnostic also discounts the closing instruction, leaving only strict-interior
retirement opportunities. At most one can retire per selected core clock cycle.
It supplies this necessary demand to Merlin's existing
`compose_service_floor`, using `scope="emitted_schedule"` and `compulsory=False`.
Redundant host initialization can therefore increase this schedule bound, but
cannot raise the fixed workload floor or improve an objective score.

For the preserved fresh public 17×16×32 source program, the original complete
counter window contains 1,773 committed instructions. The source-derived
strict-interior bound is 1,772 core cycles. Source reconstruction and original
process reopening took 9.36 seconds, including a 6.21-second independent native
provider rebuild/queue interval. No simulator was rerun and no price was fitted
for this check. These artifacts are under the task's
`out/artifacts/target-adapters/gemmini/cpu-retirement-service-v1/`.

This partial emitted-schedule bound supplies no measured target cycles,
attainment, full-call floor, timing qualification, performance promotion or
convergence. Packing, dispatch, accelerator service/completion, physical traffic
and shared-resource overlap remain separately bounded or UNKNOWN. Whole-trace
resource counters are not summed into this counter window.
