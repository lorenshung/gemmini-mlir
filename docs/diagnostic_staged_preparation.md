# Diagnostic compiler preparation

`gemmini_diagnostic_tile_factory.original_factory(inputs)` delegates to the original
current-source factory and adds exactly one optional capability,
`diagnostic_staged_tile_preparation`. It accepts the exact canonical V1 source
issuer, or the separately selected V2 issuer. The source capability is preserved;
no scheduling, timing, hardware or promotion qualification is added.

The host selects the existing `CompilerPassSelection` and
`CompilerPassPreparation` APIs before freezing compiler edit authority. Both
baseline and candidate seed use the same policy, source factory and selected
library. The generic public diagnostic launcher then copies the immutable parent,
prepares separate arms, and retains distinct raw/prepared identities. It never
replaces a target compiler with the host-only SourceChoices provider.

For the existing staged interface, an explicit disabled seed is:

```python
policy = {
    "enabled": False, "min_m": 0, "min_n": 0,
    "rewrite_order": ["n_before_m", "rhs_residency"],
}
```

Select the existing metadata protocol explicitly in a new factory record when
using repeated selection/preparation getters:
`metadata_verification_protocol="merlin.pass_metadata_operation.v1"`.
Original records remain immutable. Each readonly metadata operation still
checks the exact original source owner before and after it, and directly reopens
the original tile record. Preparation and final emitted-program checks are not
cached. A changed current source, runtime, callback, library or owner refuses.

The host may explicitly select edit authority for
`staged_dense_policy.py:choose_staged_tile_rewrites` and new modules under
`passes/` before freezing. The candidate may then choose legal shape predicates,
checked rewrite implementations and order through the existing public
`StagedDenseRewrite` API. Such a source edit proves no optimization by itself:
ordinary boxed compilation, complete outputs, current source/product/epoch/
packet checks and permanent diagnostic feedback remain required for each arm.
Unsupported asynchronous execution, spills, attention and KV mechanisms remain
unsupported. Selecting or exporting a newer library is a separate explicit
host choice; both arms must select the same release.
