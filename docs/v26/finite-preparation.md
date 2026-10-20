# Explicit finite V26 compiler preparation

`gemmini_v26_finite_factory.original_factory(inputs, sources=...)` restores the
original `GeneratedSourceDiagnosticIssuer` and preserves its source capability.
It adds only `diagnostic_v26_finite_preparation`. This is owned source-copy
preparation; it grants no scheduling, hardware, timing or promotion authority.
The original generated-source issuer, arithmetic/completion owners, target
runtime and source/trace algorithms remain unchanged.

Select the exact same original live registry/binding in `CompilerPassSelection`
with `mechanisms=("diagnostic_v26_finite_preparation",)`. The ordinary
`CompilerPassPreparation` must prepare **both seeds** with
`{"enabled": true, "rewrite_policy": "verify_only"}`. Only after original
source/edit freezing may an explicitly retained edit change the candidate's
`finite_staged_policy.py` value to `"n_before_m_rhs"`. `enabled=false` leaves the
original lowering call intact. The original 16-file compiler hash
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`
is preserved and recorded separately from the finite baseline and candidate.

The preparation copies the original package and makes one exact, checked
`compiler.py` insertion immediately before `backend.emit(buffer)`. Whole-program
emission remains outside this hook. The copied finite adapter uses the original
hash-pinned Builder/contraction helpers and only the explicitly selected public
SDK `direct_source`, `source_tokens` and `staged_dense` APIs. The same SDK release
must be granted to both arms. New public imports do not become available from
an old SDK or historical profile. The hook's scalar analysis mapping is an
untrusted proposal: the public typed reader checks the current source and ABI;
only the original private source/packet/full-output check can establish actual
execution correspondence. No capture, reference or evaluator is callable from
the candidate hook.

`check_compiler_binding(package)` reopens all original compiler bytes, the exact
mechanical hook/copy, public contract and SDK record, allowing only the two
bounded literal policy assignments. The literal DIM16/packing compatibility
checks in the existing finite emitter bind this reviewed original emitter;
they do not derive hardware qualification from a model or shape name.

`FiniteV26PreparationOwner.check_actual_pair(...)` accepts original live direct
source owners, source, fresh original proof and capsule directory, supplies the
same issuer's original completion owner, and runs the strict independently
decoded actual-pair verifier. It launches no target process.

`check_source_pair(baseline_workspace=..., candidate_workspace=..., capsule=...,
remaining_s=...)` is the bound queue consumer. It reopens each ordinary
observation, exact graded native/emission products, original source-emission
producer/dependencies, current prepared compiler and unique completed trace.
It freshly re-audits original L0/L1/L2, derives the original current source proof,
constructs the original live `DirectSourceExecutionOwner` objects and checks
the actual pair. A copied successful receipt cannot supply this authority.
Missing or changed grade/source/ELF/trace/compiler data refuses. No trace,
compiler, simulator or reference execution is launched by this consumer.

Only a final `status="PASS"` after those checks may enter the private diagnostic
queue report. This still supplies no formal grade, required L3, physical
overlap/completion qualification, native timing domain or performance promotion.
An actual finite baseline/rewrite pair and independent tail remain required;
host unit tests and shared-pass availability do not satisfy that gate.
