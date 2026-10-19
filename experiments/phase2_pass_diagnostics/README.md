# Retained pass infrastructure diagnostics

These scripts preserve an explicit public diagnostic route. They do not admit
the original formal Phase 2 campaign, qualify hardware timing, or promote a
compiler. Both treatments use the exact selected SDK5 and the original Phase 1
package remains immutable.

The completed original run is recorded under the managed task's
`out/artifacts/target-adapters/gemmini/combined-live-rhs-retained-primitive-v4/`.
`status.json` owns final success/refusal; `progress.json` is a sampled live view.
`actual-retained-check.json` proves six RHS-load omissions on the full signed
32×48×32 source, with complete products and 1,024 outputs preserved per arm.
Its source compiler invocation was a host subprocess. The separate
`primitive-current-check.json` independently checks the actual original boxed
17×16×32 pair and its two RHS-load omissions. Original L3 is pending. Unknown
complete host/physical/overlap timing and floor prevent a performance claim.

The later `boxed-retained-rhs-source-v5/` follow-up compiles **both** retained
case13 arms in the actual original compiler-profile sandbox. Its new live
`original_profile_compiler_emission.v2` receipts bind the profile, SDK, compiler,
original input, actual temporary input/output, command buffer and LLVM stdout.
Both emitted pairs match the previous native/trace inputs byte for byte; fresh
original SourceOwners recheck complete source/products/epoch/1,024 outputs per
arm. Native/trace evidence is explicitly reused, not measured again. The same
six RHS loads disappear. This is current boxed source/effect evidence; no new
registry capability, formal grade or timing qualification is issued.

To rerun the original bounded combined diagnostic, choose a fresh output path:

```sh
python3 experiments/phase2_pass_diagnostics/reproduce_retained_rhs.py \
  --output-root /work/merlin/tasks/phase2-pass-integration-20261018/out/artifacts/target-adapters/gemmini/combined-FRESH
```

Use `--kind boxed --output-root .../boxed-FRESH` to repeat the bounded boxed
follow-up with its exact retained operator. No saved live receipt is restored.

The replay verifies the exact retained operator/helper bytes. It changes only
the output path and the location of the byte-identical helper. It retains the
original 600/300/300 aggregate CPU-second qualification/source/primitive stage
guards, 900-second stage wall limits, 1 GiB aggregate RSS and 64 MiB per-file
limits. Each operation uses the original shared CPUs 6/7 admission pool. Reuse
of previous native/trace bytes is labelled as reuse and rechecked independently.

`select_original_pass.py` exposes the normal CLI's exact library/pass/preparation
intake without creating a paid author. The operator must pass the existing
`--compiler-library-*`, `--compiler-pass-*` arguments, with raw selection and
preparation file hashes, a fresh admission root and a disjoint absent stage root.
Selection reruns the original factory; saved selection JSON cannot issue an
owner. Run it under the original admitted resource boundary. A successful
selection alone does not establish compiler, model, timing or formal admission.

Candidate edit authority in the completed run owns
`rhs_residency_policy.py:choose_rhs_rewrites` and new modules under
`passes/learned/`. The edited function selects transformations using current
shapes, layouts and independently assigned rows. The general full-K placement
helper is copied identically into both compilers. Fixed checker/ISA/numerical
sources remain outside candidate authority. The live workspace was mutated
after source freeze; actual compilation used the retained revision.
