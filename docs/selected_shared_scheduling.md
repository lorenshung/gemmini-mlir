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
  --compiler-library-selection "$ORIGINAL_SELECTED_LIBRARY_RECORD" \
  --output "$FRESH_DIAGNOSTIC_OUTPUT"
```

Select the task's existing shared CPU roots and TMPDIR before execution. The
driver creates no campaign state and runs no gSIM or author session. Repeating
`--capsule` adds independent public cases; output directories must be fresh.
The optional explicit library selection reopens its original reviewed contract,
then installs the same stdlib-only selected-source startup into both copied
compilers. Exact snapshot namespace stubs and leaves load before compiler code;
already imported foreign leaves refuse. Original host dependencies remain a
separate explicit path scope, with original sandbox/private masks unchanged.
The startup performs source custody only: its JSON is no review or qualification
grant. Original same-release import authority and final checks still apply.
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

## Source-derived arithmetic scopes

`backend/gemmini_arithmetic_scope.py` supplies an original live
`IntegerArithmeticOwner`. Its constructor takes the selected target, explicit
original software specification, selected source tree, exact functional receiver
sources and deployed runtime. It checks every physical PE's multiply/add AST and
active WS path, every tile partial-sum connection, the complete mesh's spatial
transport and the distinct registered accumulator adder. Receiver source hashes
must match the deployment's critical files and runtime artifact hashes.

For the selected elaboration this derives signed8 operands, a20-bit spatial
partial sum through16 rows, and signed32 accumulator storage/addition across
panels. A16-product signed8 panel fits20 bits even when the complete K reduction
exceeds that range. Never apply20 bits to the entire K reduction. The live
`source_rule()` reopens its exact source pins; a saved JSON record cannot create
this owner or grant executable support. Source proof must check every local
panel prefix against20 bits and the complete source reduction against32 bits.

Actual compiler admission remains separate: check actual WS configuration,
panel K≤16, zero incoming partial sum, private storage, continuation addresses,
complete source-to-output binding and selected readout effects. Raw i32 readout
does not authorize replacing integer half-up source arithmetic with hardware
half-even scaling. This additive development sidecar changes no Phase1 file,
numerical policy, hardware qualification or timing model.

## Explicit live cleanup admission

`experiments/shared_load_configuration/admit.py` accepts the same explicit
`--compiler-library-selection` and `--merlin` selection as the paired qualifier.
It requires every loaded shared source to match that exact release, then reopens
the actual linked programs, signed inputs, complete outputs, issued operands,
fences and logical transfer windows. The original host API returns a live
`LiveLoadCleanupOwner` carrying the original target and library owners. The
written receipt and `record()` are evidence, with no JSON constructor for this
live owner. A registry record cannot replace this admission call.

This enables only bounded functional Spike load-configuration source effects;
each newly realized program still requires its independent final pair and
complete numerical checks. Hardware/timing authority and dense-v1 remain
disabled. Ordinary compiler-child selection additionally needs its original
frozen capability/resource projection; loading an SDK does not grant that
projection or enable an unrelated device adapter.

## Actual contraction packet checks

`gemmini_execute_state.py` derives additional EX fields from the selected
elaboration and checks their original controller assignments, including every
scale bit. The explicit supported state is identity WS with unit strides, no
transpose, activation or shift. A missing or different state refuses.
`gemmini_packet_contractions.py` independently follows actual issued load,
preload, compute and store operands. Loads assign private cell provenance;
preloads retain descriptors across fences; compute consumes the current cells
or independently observed retained PE weights. Every output cell must contain
each source K panel once, beginning from independently checked zero storage.
Incomplete work, clobbers, changed owners, unsupported effects and a final lone
preload refuse. Register-packet shape fields and capacities remain selected
source evidence supplied by the original caller.

The generic `merlin.llvmlower.source_contraction_bindings` bridge reconstructs
current rank-two source arithmetic and complete source/ABI names. Those names
alone prove no executed effects. The packet check still needs an original
complete trace, exact source/input/output custody and independent numerical
checks before a live capsule owner can use it. Batched, mixed attention and
nonidentity readout contracts remain disabled unless separately checked.
This staged state induction changes neither the old synchronous dense-v1
contract nor any timing evaluator.

`gemmini_source_execution.SourceExecutionOwner` connects these checks to the
original `checked_native_integer_v1` writer callback. It requires original live
arithmetic/target owners and explicitly selected compiler, native-build and
commit-invocation artifacts. It rechecks the complete source/LLVM/task CFG,
source-entry bytes against the linked ELF, observed writes, independent complete
goldens and both native/trace output streams before checking actual products.
The expected trace command is reconstructed from the exact original native
engine citation and pinned deployment; an ambient Phase0 backend selection
cannot replace it. No JSON reader reconstructs this live owner.

Use the original host `scoped_realization_owner(owner)` API for this specific
source, input and numerical contract. Its successful source correspondence
does not enable a general dense adapter, hardware timing, mixed attention or
arbitrary future input/shape support. Each new realization must pass again.
The opt-in `test_source_execution_owner.py` integration test takes
`MERLIN_SOURCE_EXECUTION_TEST_FIXTURE` selecting the original source/runtime and
completed artifacts; it reruns the checks without simulator execution and
rejects locally mutated output, source, ELF, runtime and compute-state evidence.

## Ordinary copied-compiler cleanup preparation

`gemmini_compiler_passes.restore_registry` is an explicit original-host factory
for Merlin's `OriginalPassFactory`/`CompilerPassSelection` transport. It reruns
the unchanged independent cleanup admission in a fresh interpreter using the
exact selected library. This avoids replacing the host evaluator's imported
SDK namespaces. The returned `OriginalCleanupAdmission` has no saved-record
constructor: its original invocation, outputs, source leaves, selected facts,
software, runtime and library must revalidate. Release a parent compute lease
before admission; the original checker child owns its own shared lease.

The selected `configuration_cleanup` owner prepares separate baseline and
candidate compiler directories **before** their source and edit authority are
frozen. The preserved 21-file compiler remains unchanged. Both copies receive
the same reviewed SDK, startup loader, checked cleanup algorithm and exact
public source leaves. The only initial policy difference is the candidate-owned
`cleanup_policy.py`: a Boolean enable and nonnegative MAC-count threshold chosen
from current M/K/N. Future policy edits remain subject to the ordinary frozen
edit authority and independent complete-program checks.

The copied hook executes at the original direct emitter's actual statement
return. Its public projection contains exact bounded leaves with relative
paths, never a source directory or private qualification evidence. It reopens
their bytes and the selected SDK at every invocation. The cleanup helper names
the SDK's existing exact public IR leaves rather than requesting the broader
`merlin.sched.ir` implementation facade. No SDK algorithm is changed.

Fresh functional integration of the selected copies exercised the ordinary
package integrity scanner and `emit_analysis_bundle` entrypoint. The resulting
LLVM and command buffer exactly matched those linked by the unchanged native
builder. Three signed public geometries (1x16x16, 17x16x16 and 12x48x48) retained
all 864 output values and logical work; independent actual issue-register checks
removed respectively 1, 2 and 9 redundant setters while preserving every other
command, fence and logical DMA window. This is functional command evidence,
not accelerator latency or physical-memory-guard qualification.

Preparation and registry transport do not satisfy the final formal search
gate by themselves. The owner's `check_functional_guard` consumes the original
complete functional regrades, exact frozen compiler copies, complete roster,
clean integrity and certification results. It joins each original source-model
grade to its durable native receipt and emitted LLVM/command buffer. The
`gemmini_cleanup_final_guard` consumer reopens original build/source pins,
checks the exact selected runtime and runs a new bounded issue trace of that
same final ELF. It requires every complete output to reproduce the original
independently graded native values, unchanged source/input/golden identities,
unchanged logical work and only legal redundant-setter omissions. Traces retain
the independent 64MiB ceiling and shared leaf compute admission.

An actual ordinary-copy native pair for freshly sealed source34 passed this
final issue/output consumer (217 complete values per arm). Its zero removals
are an independent geometry control, not an improvement. This diagnostic did
not fabricate a successful formal grade: the original Phase1 certification
prerequisite remains pending. Missing original grades and non-model capsule
families without this exact native receipt route refuse; neither a saved
success nor a preparation record can bypass that requirement. Hardware,
timing, broad dense, asynchronous overlap and attention capabilities remain
disabled.

The opt-in `test_selected_cleanup_child.py` takes
`MERLIN_SELECTED_CLEANUP_TEST_INPUTS` pointing to the explicit original factory
inputs. It reruns original cleanup admission in both parent and fresh sealed
child, restores the exact selected SDK/pass identities, verifies the final
consumer is present and checks the parent's child receipt. Its parent compiler
profile is a neutral transport fixture; passing this test does not certify the
original Phase1 experiment or authorize another capability. Do not hold a
parent CPU lease while waiting for the original checker child.

## Finite staged raw-i32 source route

The explicit SDK v4 release adds the separate generic staged descriptor/consume
API; earlier SDKs and synchronous dense-v1 retain their original meanings.
`gemmini_staged_dense_source.py` projects the preserved direct emitter's actual
finite C statements: complete logical pointer expressions, immutable zero row,
identity load state, pending PRELOAD, following compute and raw-i32 readout.
Unknown loops, calls, flags, scaling, bias or layouts refuse the projection.
Its first physical renderer may omit only independently proved redundant RHS
loads. It preserves every original configuration, compute, store and fence,
including final persistent settings. New placements, added instructions and
broader logical rewrites need separately qualified physical rendering.

`gemmini_staged_dense_admission.prepare_original_prototype` reruns the original
selected cleanup admission, derives arithmetic from the exact selected RTL and
receiver and reruns the selected RTL completion proof. It prepares separate
compiler copies using the same SDK and checked mechanisms. The candidate-owned
`staged_dense_policy.py` chooses from current M/K/N and selected primitive size;
no model or layer identity selects reuse. Both treatments keep configuration
cleanup disabled to isolate the RHS mechanism. Unsupported members retain
their original program and expose a compiler refusal reason.

Prototype preparation grants no executable capability. The original
`restore_registry` requires real sealed-source paired realizations and reruns
the original complete source, packet-product, independent output and CPU-write
checks. Initial scope is one contraction with original immutable external RHS.
Independent actual issued-cell induction must prove every omitted load already
has the exact source bytes in its private cells; all other commands, fences and
settings must match. Qualification requires distinct M-boundary positives and
independent K/N no-gain controls. New executions still repeat the complete final
check. Saved success records cannot issue this owner.

The newly requested positive/control source seal is currently blocked before
capture by original sandbox readiness; its legacy diagnostics remain explicitly
unsealed. The staged registry therefore remains disabled. No source admission,
hardware timing, double buffering, asynchronous overlap, cross-operation leases,
mixed attention or KV capability follows from the source API or prototype.

The same explicit SDK-v4 ordinary compiler copies have now exercised all five
engineering guards. For M=17,31,32 with K=N=16, each candidate removed one actual
RHS transfer (22 to 21 issued commands). Independent K=17 and N=17 controls
retained 34 and 42 commands. All 1,841 complete signed output values per arm
matched; fresh traces reconstructed every required source product, accumulator
continuation and readout and reopened linked immutable input bytes and CPU-write
effects. Five baseline native artifacts were reused only after their command
buffers and LLVM matched the ordinary SDK-v4 baseline emission byte for byte;
all five candidate native executions and all ten traces were fresh. These are
unsealed functional diagnostics, not paired latency measurements or capability
qualification. The original sealed-source checker still refuses each case.

Fresh original admission retains a separate invocation and receipt for each
process. Staged selection identity binds the original stable cleanup owner,
selected sources, arithmetic, completion and contract; it does not bind the new
receipt's output location as a scheduling choice. Each live owner still reopens
its own exact original receipt. This permits original child re-admission without
accepting a saved-success factory or allowing receipt replacement.

## Current source execution in the ordinary diagnostic route

`gemmini_source_execution_registry.restore_registry` is a new explicit original
factory around the SDK-v5 whole-tile route. It first reruns the original cleanup
admission and all required independent sealed-source pairs. Only then can it
add the live `source_execution_diagnostic` capability with the exact same
hardware, software, numerical, runtime and library binding. This extra owner is
not a compiler preparation mechanism and does not expand editable authority.
The factory must be selected by the original host, with its exact public source
leaves available to the existing sealed child. A JSON record cannot select or
construct a successful owner.

`SourceExecutionIssuer.create_source_execution_owner` takes the current native
result, command buffer, lowered program, a fresh work directory and remaining
wall budget. It reopens the current complete native result and original runtime
identity, then runs that exact ELF with the original Spike commit observer. The
leaf process owns its shared CPU lease, has at most sixty seconds and the
unchanged 64 MiB trace ceiling. The caller must release any preceding native
execution lease first. Every call produces a fresh observed invocation; the
issuer has no saved-success or cross-execution trace cache.

The returned value is the existing original `SourceExecutionOwner`. It is not
an admitted capsule or successful numerical certificate. The ordinary source
qualifier still has to derive the fresh sealed source proof and call its live
verifier, checking actual complete products, accumulator state, source inputs,
settings and every output. The diagnostic selector leaves hardware/L3, timing,
formal cohort/privacy and unsupported arithmetic gates unchanged. Missing
original qualification or current artifact correspondence refuses this route.

The SDK-v5 functional run independently checked six source pairs: two-axis
M/N boundaries with K=16 and a K=17 no-gain control. Five pairs issued 42 to 40
commands; the deep-K control retained 66. Both rewrite orders on the M=N=32
case were executable and exact, but residency-before-traversal retained 42
commands. All 3,651 values per baseline/candidate arm and the additional 1,024
reverse-order values matched. These are command/effect and complete-output
checks, not latency evidence. Initial final issuance correctly refused an
original-source versus runtime-copy path mismatch. The corrected custody check
requires the actual original capsule interface as the compiler input and
independently reopens the exact-byte native copy; all original artifacts remain
unchanged. Fresh original factory issuance is still required after that fix.

## Explicit sealed-parent qualification reuse

`gemmini_delegated_source_execution.restore_registry` is a separate opt-in
factory. Its original inputs must select
`parent_delegation_protocol=merlin.original_parent_pass_qualification.v1`.
Ordinary host/CLI loading still performs the full original qualification. A
previously issued live registry can be wrapped with `from_live_registry`;
saved registry JSON cannot be wrapped. `SourceExecutionIssuerV2` composes the
unchanged current-artifact issuer and adds a separate version/source identity.
Its class lives in its own exact source module, so dynamic factory imports do
not create different issuer types between host, CLI and worker.

The generic host must construct `OriginalParentPassDelegation(selection,
exporter=export_qualification, redeemer=redeem_qualification)` and enter
`scoped_original_parent_pass_delegation` around the original capsule runner.
The new sealed-worker protocol authenticates the parent-held sealed descriptor,
exact parent spec, source/factory/pass records and original callback before
issuing a live token. Earlier protocols and unscoped workers retain full replay.
The two new private source leaves must be in the original factory/worker source
closure; this grants no imports or resources to compiler candidate processes.

Only inside that authenticated callback can `token.require(snapshot)` authorize
the target redeemer. The target additionally requires that exact active token,
reopens the original cleanup receipt and observation invocation, rebuilds the
selected SDK/facts/arithmetic/completion/prototype with normal checked owners,
and uses the existing qualified owner's original revalidator on every retained
source, compiler tree/invocation, native artifact/runtime/build source and trace
pin. It requires its exact resulting owner and registry records to match the
live parent's authenticated records. A copied receipt or duck-typed token
outside that callback cannot restore authority.

The worker receipt explicitly says `inherited_original_parent_qualification_revalidated`
and `fresh_six_pair_semantic_replay=false`. This avoids repeating the parent's
six complete source-proof qualification pairs for every unchanged worker. It
does not claim new measurements. Each changed current candidate still needs
the original complete native execution, a fresh bounded current-ELF trace and
the independent source/product/state/full-output proof. Hardware, timing,
numerical, privacy and final-grade gates are unchanged. Real sealed-parent
integration remains required; neutral dispatch and adversarial tests establish
only the plumbing/refusal behavior.

## Conditional finite full-K RHS placement

The separate `gemmini_staged_rhs_registry` factory selects
`full_k_rhs_residency_raw_i32` only through a live
`QualifiedRhsResidencyOwner`. Its constructor requires six independently checked
source/executable pairs: positive M/N/K remainders 0, 1 and D-1 and a single-panel
fallback. Capacity declarations and successful JSON reports cannot construct
this owner. `SourceExecutionIssuerV4` retains the original bounded current-ELF
trace algorithm; inherited historical qualification requires the same original
kernel-authenticated parent protocol, plus the exact complete-program epoch.
The explicit `rhs_allocation_epoch_source` factory input pins that original
allocation observation's native result, compiler artifacts and trace. A new
byte-identical compiler invocation does not replace its original path custody.

The original epoch first proves every issued access in one complete observed
hart-0 functional call, one external immutable RHS, its original A/B windows,
and its sole accumulator stripe. Joint physical placement needs D rows for A
and `ceil(K/D)*D` rows for B. A one-panel load cannot use the extra rows merely
because hardware has capacity. Each new source/program repeats the independent
ownership, source-product, zero/continuation, readout and complete-output checks.
This is conditional functional-call ownership, not a hardware-wide free-memory
or external-writer guarantee.

The generic `staged_rhs_residency.py` proposer is copied identically into both
candidate-owned `passes/` directories under the unchanged explicit SDK-v5
selection. It composes complete-output-tile traversal, source-cell-aware row
placement and proven RHS reload omission. The independent checker reads actual
operands at COMPUTE consumption, including pending PRELOAD state, and permits
only those relocations/omissions. Configurations, fences, arithmetic and outputs
remain checked. The policy function and additional authorized passes can be
edited; independent checkers and source/resource ledgers remain fixed.

Six signed source controls checked 6,369 complete output values per arm. The
five deep-K shapes removed 4, 6, 12, 6 and 9 actual RHS loads; the K=16 control
kept its original schedule. These are original-source/native/issued-effect
diagnostics. Final original compiler qualification and ordinary sealed-worker
execution are separate required gates. Asynchronous overlap, prefetch, connected
residency, attention/KV, hardware timing and latency improvement remain outside
this capability.
