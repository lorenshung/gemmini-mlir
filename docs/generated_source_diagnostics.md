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
must not hold a parent lease across a separately leased child. All compilation,+native execution, trace and verification costs remain separate from target cycles.

V26 still has no `target_ir_consumer` or `emit_target_bundle`. This factory does
not supply either interface, qualify hardware execution, complete the original
Phase 1 certificate, or extend native timing domains.
