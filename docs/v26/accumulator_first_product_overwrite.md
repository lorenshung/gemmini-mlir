# Explicit first-product accumulator overwrite

The original generated-source factory accepts an optional
`source_accumulator_effects_protocol: gemmini.accumulator_first_product_overwrite.v2`
with either existing direct-source binding. It rejects this option for
`whole_program` and rejects unknown protocols. Omitting the option preserves the
original initialized-accumulator rule and report projection. It does not enable
a compiler pass or change the immutable compiler, numerical contract or runtime.

Before observing a program, the issuer constructs `SelectedAccumulatorOverwrite`
from its exact live `SelectedSchedulingTarget` and `IntegerArithmeticOwner`.
The derivation reopens the selected FIRRTL and the original runtime-deployed
receiver source. This version supports the exact reviewed selected FIRRTL SHA
`b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0`
and receiver SHA
`c468a86deb68cb2e12e3fb1177821075b0eb87e27a37ff835280ee7caf61b13a`;
another implementation requires an explicit review. Source and bound callback
pins, the original selected objects, arithmetic identity and selected inputs are
checked again on use. A saved report, boolean, subclass or unissued object cannot
substitute for the live effect owner.

The source proof checks the selected two-port accumulator write path: the
accepted packet captures address/data/accumulate/masks, and its write-data mux
selects the adder sum when accumulating and the incoming product otherwise.
Both selected accumulator bank modules are checked. ExecuteController's PRELOAD
accumulate field, output-address flag projection and all per-column byte masks
are checked, as are the Scratchpad's selected execute-write flag/address/data/
mask routes. The deployed receiver independently preserves the PRELOAD address
and rows/columns and executes `+= value` or `= value` according to the same bit.
The original arithmetic owner still supplies signed MAC and prefix-width rules.
These are conditional functional cutpoints, not a pipeline reachability,
accepted-cycle, physical overlap or timing proof.

The existing complete packet induction receives that exact effect owner. Only a
**nonaccumulating first source panel** (`k0 == 0`) may replace accumulator cells
without reading their previous values. An accumulating panel still needs an
initialized exact `(source, output row, output column, K prefix)`. A later-K
overwrite refuses. Every product and readout must still be present exactly once;
source operand ownership, retained weights, independent tails, bounds, complete
outputs and the host-answer-store veto are unchanged. The source binder must
independently establish the supported zero-seeded dot expression. This option
does not admit bias, arbitrary initialized reads, transpose/scale effects, alias
inference, unsupported commands or any of the 15 prohibited loop selectors.

A prospective operator uses the same explicitly selected evaluator in both
arms:

```python
inputs = dict(original_generated_source_inputs)
inputs["source_accumulator_effects_protocol"] = (
    "gemmini.accumulator_first_product_overwrite.v2"
)
factory = gemmini_generated_source_factory.original_factory(inputs, sources=sources)
registry = factory.create(work_root=fresh_work_root)
```

The existing source issuer, current compiler profile/library, original native
execution, complete trace producer and source qualification remain mandatory.
The source factory/trace dependency closure changes, so old invocations whose
recorded dependencies differ remain refused; do not rewrite their pins. Fresh
producer output is required when the original invocation verifier demands it.
No historical failure, saved owner or timing result is upgraded by this option.

Neutral tests exercise full product/readout induction for M/N/K tails and deep
reductions, overwrite of stale storage, missing/duplicate work, late overwrite,
uninitialized accumulating reads, aliases, unsupported bias loads, altered RTL
routes and source/callback/object mutations. Separately, original selected-owner
construction has been checked against the actual retained FIRRTL/receiver. Those
host checks supply no new accelerator execution, paired compiler qualification,
measured cycles or promotion authority. An actual fresh paired source/native
proof remains a separate gate.
