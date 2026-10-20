# Static reference inspection: transferable requirements

The operator authorized inspection of two local archives on 2026-10-20. No
reference code was executed, imported, supplied to a candidate, or copied into a
compiler. Only general mechanism requirements may cross into development.
Reference kernels, addresses, schedule constants, weights, answers and unmatched
latency labels remain outside the author view and calibration corpus. These
models have been inspected during development and cannot later be called unseen
evaluation workloads.

| Archive | SHA-256 | Actual contents |
| --- | --- | --- |
| `exo-tinyllama-v2-314961814-rocc-source.zip` | `61d5c4e6023cb50862e20e3d25c806713d2047f5a0b724ee547c3d1073f1f52a` | `exo_llm.c`, `exo_llm.h`; seven accelerator kernels |
| `dragon_resnet18_34_50_101_gemmini_baremetal_20261015.zip` | `0022568b38780daa1f5759abdf36d15d408412fb5817bed8769bdc046ae17bbd` | Four ELF files; ResNet50 ELF SHA `44503d99212ba2e5ff78806f36a02b6f6d556bc78e63a155d227d06ef331d9f9` |

The adjacent [static inventory](reference-mechanisms-20261020.json) records
section/function counts and limitations. It is an operator artifact, not a
target execution certificate. The ResNet archive supplies no C source or
scheduling generator. The TinyLlama archive supplies no complete-model driver,
Gemmini include implementation, independent reference, runtime selection or
cycle report. Its principal contraction interfaces use 16 rows, with a separate
one-row vocabulary projection. The supplied sources alone do not establish a
32-token prefill plus 32-step decode comparison. The later user estimate for
ResNet was approximately 24M cycles; neither that estimate nor the approximately
220M TinyLlama estimate is an admitted matched measurement here.

## Findings that change priorities

1. **Efficient raw scheduling is a credible first path.** No hardware-loop
   selector appears in the static ResNet executable sections. The seven
   TinyLlama functions call raw Gemmini command APIs and have one fence call
   site apiece; they call no high-level matmul/conv loop-engine API. This does
   not provide dynamic instruction counts or prove the implementation of
   missing includes. It does show why the mere absence of loop instructions
   cannot assign the current slowdown. The operator explicitly forbids hardware
   loop/FSM instructions and their configuration commands. Raw commands and
   ordinary software loops remain the permitted implementation path.
2. **Transfer and reuse granularity matter.** The TinyLlama source uses blocked
   operand movement, retained scratchpad regions and staged load/compute
   sequences, rather than requiring a host completion between individual tile
   commands. Translate that observation into generic ownership, lifetime,
   blocking, dependence and transfer-granularity requirements. Do not transfer
   its tile constants or addresses.
3. **Packing has a lifetime.** The source consumes packed weight layouts.
   Constant packing once, with an explicit representation and reuse lifetime,
   is a general compiler mechanism. Charge preparation in the declared cold
   boundary or state the amortization across warm invocations. A prepacked
   reference interface does not make our packing free.
4. **Native exits need a numerical contract.** The source exposes both narrowed
   INT8 and full-width INT32 output paths. The ResNet ELF contains device-command
   bodies labelled for residual and pooling operations. These are candidate
   capability requirements, not equivalence proofs for our FP32 host islands.
   Reuse the selected bias, scale, activation order, rounding and clamp policy;
   reject fusion if it changes those semantics.
5. **Batch/layout composition matters beyond individual matrices.** Joint
   projections and attention-shaped kernels motivate general multi-output
   contraction grouping, layout preservation and batching. Our selected GQA,
   precision planes, masking and carried KV state must remain complete. A
   faster reference's different arithmetic cannot replace required source work.

## Least-effort implementation order

First complete the existing automatic retained-candidate queue and the checked
raw-command schedule mechanism. Use public, independently generated controls
for M/N/K tails, short M, deep K, capacity knees, residency invalidation and
connected operation chains. Retain failures and UNKNOWN physical claims.

Then tune legal blocking/loop order, RHS retention, burst-size movement and
configuration lifetime through the current emitter interface. Add
completion-proved fence removal and bounded double buffering as separate
mechanisms, with adverse ownership and overwrite controls. A raw command stream
needs an actual emitted-program consumer; selecting decorative schedule IR is
insufficient.

Expose constant-pack lifetime, layout retention and contraction epilogue
composition in reusable Merlin passes. Keep instruction encoding and selected
target effects in this backend. Policies choose from current semantics, shapes,
layouts and resource proofs, never model names or reference identities.
Existing reference layouts do not become model-specific compiler rules.

Hardware-loop lowering is outside the operator-authorized edit surface. Keep
matmul/conv loop launches and their configuration selectors forbidden, including
optional variants from the selected ISA. Preserve the original final-ELF audit
on both arms. The selected hardware/runtime's loop-engine availability does not
grant permission to emit those instructions. Do not copy a handwritten/reference
provider into the generated compiler.

For the selected custom slot 3, the forbidden function selectors are
`8--13`, `15--21`, `24`, and `25`. Selector 25 is declared in the selected software
header even though it is absent from the observed RTL decode roster. Include it
in the prohibition. Ordinary raw `CONFIG_CMD` selector 0 remains permitted;
configuration cleanup is still an applicable optimization.

Scheduling changes need no rewrite of the immutable Phase 1 output: copy it,
select permitted edits and the same shared release for both arms, and obtain
fresh checks. Changing quantization, admitting a new target semantic interface,
or extending timing coverage is a separate versioned qualification task.
Additional passes alone establish no latency gain.

Finally evaluate a frozen policy with qualified paired timing, then perform
complete-model checks under the original contracts. Keep full-model profiling,
kernel timing, analytical floors and static command counts distinct. These
reference artifacts supply architectural questions; original public calibration
and independent measurements must supply prices and performance evidence.
