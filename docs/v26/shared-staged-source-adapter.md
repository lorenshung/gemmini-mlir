# Explicit finite staged-source adapter

`merlin-support/backend/gemmini_v26_staged_source.py` connects the existing
Merlin `staged_dense` rewrites to the exact original V26 Builder/contraction
implementation. It issues no target capability and stays disabled by default.
The original 16 compiler files are unchanged. No ordinary compiler route or
existing factory is silently replaced.

The callable interface is:

```python
from gemmini_v26_staged_source import emit_v26_staged_dense

proposal = emit_v26_staged_dense(
    package=original_v26_package,
    source=current_typed_source,
    proof=current_typed_source_record,
    command_buffer=current_command_buffer,
    contract=explicit_staged_contract,
    enabled=explicit_policy,
    rewrite_policy="n_before_m_rhs",  # or "verify_only"
)
```

With `enabled=False`, the function returns `None` without reading its other
arguments. The caller retains the original backend. A successful explicit call
returns `(llvm_text, diagnostic)`; it does not return an admission owner or
qualification receipt. The source record establishes the exact single-dot
binding, not executable truth. A production consumer must separately bind the
original live effects/arithmetic/completion owners, exact library release and
actual current source/compiler artifacts.

The supported source is one closed rank-two signed-i8 dot, seeded by an i32
zero splat and returned completely as raw i32. The complete command buffer must
be the original single-use RES_PACK/MATMUL_RESIDENT/COMMIT/EVICT sequence, with
three disjoint logical pointer names. Bias, transpose, scaling, extra commands,
constant parameters, other source operations and stateful graphs refuse. The
current adapter assigns only the original temporary windows: 16 LHS rows,
16 separate RHS rows, and 16 accumulator rows. It requires the explicit
8/20/32-bit staged arithmetic and ownership premises. Such data declarations do
not qualify those premises or an execution.

Literal 16 and instruction bit patterns in this adapter are compatibility
predicates for the exact hash-pinned original emitter. They are not facts
derived by this adapter, general hardware support or strategy selection by
workload name. Independent selected RTL/software/effect owners remain required.
Its bounded LLVM token inspection reuses Merlin's existing structural lexer;
quoted assembly strings are not interpreted as SSA or control instructions.

The adapter executes the exact hash-pinned original Python Builder and
contraction sources in separate local namespaces. It delegates their pointer,
RoCC packing, configuration, arithmetic and readout generation. Only constant
bounded original loop callbacks are expanded on the host. The hard limits are
4 Mi scalar products and 4,096 primitive tiles; supplied budgets can reduce
them, not raise them. This expansion changes original runtime loop control.
For a clean later comparison, select `enabled=True,rewrite_policy="verify_only"`
for the prepared baseline and `"n_before_m_rhs"` for the treatment. The immutable
original backend remains a separate baseline. No timing advantage is inferred
from either host-code change.

Each complete original output tile is projected into the existing shared
`AccumulatorZero`, `OperandLoad`, `DescriptorIssue`, `ComputeConsume`,
`AccumulatorReadout` and `CompletionFence` types. The adapter calls
`rewrite_staged_dense` with `order_output_tiles_n_before_m` followed by
`eliminate_redundant_rhs_loads`. The existing independent verifier checks the
original and every proposal. State0/1/2 strides are checked separately against
the original emitter's exact configurations. Pending PRELOAD ownership is
consumed only by its matching COMPUTE; a fence does not retire a descriptor.

The V26 leading fence before each zero is retained as physical tile-boundary
metadata. Reduction and readout fences remain in the logical effects as well.
Re-emission moves complete original line chunks, including every leading,
reduction and readout fence/configuration; it deletes only the original RHS
RoCC call corresponding to a checked omission. Pointer calculations for an
omitted call remain. Each moved chunk must use only its own preceding SSA
definitions and the three entry arguments. An exact line-multiset check and
unchanged prologue/suffix prevent other losses or insertions. Package bytes,
typed source, command buffer, proof and contract are checked again before
returning the result. This is a projection check, not a second renderer.

Host tests run the actual original Builder/contraction implementation on
independently derived typed source fixtures. For 32x16x32 the emitted finite
program has two RHS calls instead of four, while retaining four computes and
all 16,384 required products, four zero/readout tiles and all 15 original finite
program fences. Tests cover partial M/K/N, multiple K panels, no-gain cases,
source/package/record mutation, independent load states, descriptor/compute
width correspondence, invalid ownership/layout/effects, hard bounds, unknown
policy, cross-tile SSA and control-flow refusal. These tests invoke no compiler
toolchain, target runtime, native observer, gSIM or model fit.

Before selecting this adapter for an ordinary candidate, an original live
capability must check independently emitted baseline/treatment programs:
complete numerical outputs, current ELF/input/caller/source/SDK custody, full
commit trace, original private-zero proof, actual per-state load configuration,
every required product/readout and host answer-write veto. A new actual-command
pair checker must preserve whole original fence groups and admit only the
proved tile permutation/RHS omissions; old V23 state-zero comparators cannot
be reused as V26 qualification. Current source admission for one baseline is
not this pair evidence. Hardware execution, async overlap, complete-model
composition and timing authority remain absent. Frozen native model/domain
checks and interval decisions are separate and unchanged.

The adapter uses the explicit public
`merlin.compile.scheduling.source_tokens.llvm_tokens` and
`merlin.compile.scheduling.direct_source.bind_direct_source_contraction`
proposal APIs. Export them with `export_shared_scheduling(...,
source_binding=True)` (library manifest V2), then review and select that exact
release equally for both prepared arms. The default V1 library grant does not
include them. The original private source checker independently rebinds the
actual source and execution; this public reader issues no authority. Private
late legalization and source admission modules stay outside the compiler grant.
The target adapter itself still needs an explicitly permitted compiler source
component. A trusted host import does not widen either grant. Ordinary routing
remains disabled in this change.

The callable returns LLVM IR text, matching the original Builder. The original
V26 compiler imports that text with its selected `mlir-translate --import-llvm`
before publishing LLVM-dialect MLIR. Host validation exercised that exact
format importer, then the existing xDSL `Parser(make_llvm_context(), text)` and
`module.verify()` on actual source166 and an independent 31x15x17 tail, for
both finite policies. All four modules passed. This check generated no object
or ELF, ran no target program, and issued no live source/effects capability.
