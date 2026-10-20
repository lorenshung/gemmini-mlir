# Generated compiler source diagnostics

`gemmini_generated_source_factory.original_factory(inputs)` selects an original
host-only source diagnostic without requiring a historical compiler or pass
qualification. It supports untouched generated compilers and exposes only
`source_execution_diagnostic`. It enables no compiler transformation.

Select canonical paths explicitly:

| Field | Meaning |
| --- | --- |
| `facts` | Exact RTL facts selected for this experiment. |
| `observed_contract` | Operand-bearing observed **JSON** contract, with its exact original-facts sidecar. This is not the experiment's software-contract YAML. |
| `software_spec` | Current selected software specification. |
| `source_root` | Selected RTL checkout containing the actual Gemmini sources. |
| `receiver_source` | Exact functional receiver source selected by runtime deployment. |
| `runtime` | Deployment directory containing `deployment.json`, `bin/spike` and `lib/libgemmini.so`. |
| `nm` | Selected native ELF symbol inspector. |
| `library_selection` | Explicit same-arm additional-library identity; its presence grants no pass or compiler preparation. |
| `observation_protocol` | Exactly `gemmini.source_observation.v2`, complete committed effects without unused disassembly lines. |

Optional `metadata_verification_protocol` accepts only the existing
`merlin.pass_metadata_operation.v1`. Missing or unknown inputs refuse. This new
factory does not read old cleanup reports, effect JSON or compiler qualification
receipts. Older source factories retain their interpretation.

```python
from gemmini_generated_source_factory import original_factory

factory = original_factory(inputs)
registry = factory.create(work_root=FRESH_DERIVATION_DIRECTORY)
# Retain this original live registry in the ordinary checked-source selection.
```

Construction rederives the original `IntegerArithmeticOwner` and
`StagedCompletionOwner`, reopens receiver/runtime and library bytes, and retains
their original classes and callbacks. It is source-semantic metadata evidence,
not a current executable proof. Every `create_source_execution_owner` invocation
requires current complete native outputs and exact source/CB/LLVM/runtime/ELF
custody, observes a fresh bounded complete trace, and returns the original
`SourceExecutionOwner`. The ordinary qualifier must still call its actual source,
product, state and complete-output checker. No saved success can replace that.

The observation retains the existing 60-second inner process limit and 64MiB
stream ceiling. The operator still owns aggregate CPU/wall/RSS/disk guards and
must not hold a parent lease across a separately leased child. All compilation,
native execution, trace and verification costs remain separate from target cycles.

An explicit `source_binding_protocol="merlin.native_pointer_source.v1"` selects
the separate direct-pointer source checker. It also requires
`execution_contract`, the exact canonical software-contract path selected by the
actual native renderer. The old absent-field whole-program path keeps its
original checker. Direct checks initially support only a closed zero-i32
constant, tensor splat, one signed-i8 rank-two dot and its raw-i32 return.
No `global_program_plan` or emitted `kernel_abi` is manufactured.

The original neutral pointer binder reopens complete source inputs/references
and actual pointer order. The direct owner uses the same packet product/state
engine as the whole-program owner, then independently vetoes CPU result stores
over the actual original caller invocation. It checks all native and trace
outputs against unchanged complete references. The selected execution contract
independently fixes entry/completion symbols; a record cannot select another
entry to evade the host-write check. Unsupported graphs/layouts/initialization
or incomplete traces refuse. Source-semantic metadata preparation alone does
not establish that any direct program passed these checks.

The direct checker explicitly derives selected `MVIN`/`MVIN2`/`MVIN3` load
states from the original operand observer. Each stream retains its own configured
stride; an unconfigured stream refuses. Selected FIRRTL cutpoints also check
that raw pixel-repeat encodings zero and one both normalize to one pixel and
zero remaining repetitions. Identity scale, unshrunk rows and the selected
block stride remain required. The whole-program checker's default state-zero
interpretation is unchanged. This functional correspondence grants no DMA
timing, overlap or physical-transaction claim.

`gemmini_generated_source_native.GeneratedSourceNativeCorrespondenceOwner.bind`
accepts only the exact live generated-source capability and original
`CountHardwareCorrespondenceOwner` capability. It keeps composite software and
raw observed-contract identities distinct. Both source/count owners reopen their
selected inputs, completion proof and runtime; actual five-field compiler/native
bindings must match. Use a freshly derived count owner with the same selected
facts/observed/operand paths, not a rewritten saved sidecar. Wrap the resulting
owner in the existing `compiler_source_native_correspondence` capability using
the **native** binding. This structural bridge grants no calibration or timing
domain; the ordinary independent held gates and current observation remain required.

V26 still has no `target_ir_consumer` or `emit_target_bundle`. This factory does
not supply either interface, qualify hardware execution, complete the original
Phase 1 certificate, or extend native timing domains.
