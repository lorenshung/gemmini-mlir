# Reported whole-model FireSim costs and Phase 2 priorities

The operator supplied these observations on 2026-10-20. The adjacent
[machine-readable record](firesim-performance-sanity-20261020.json) retains the
numbers, scope and missing provenance. Original ELF, compiler, hardware,
completion and measurement receipts have **not** been reopened here. This is
operator-only validation sanity evidence, excluded from calibration training,
search objectives, author feedback and incumbent promotion. It does not replace
the original Phase 1 qualification.

| Reported complete invocation | Cycles | Complete reference comparison |
| --- | ---: | --- |
| ResNet50, batch 1, 224x224 | 2,179,239,099 | 1,000 elements, zero mismatches |
| TinyLlama, batch 1, 22 layers, 32-token prefill + 32 decode steps | 80,999,302,990 | 24,846,294 elements across 99 arrays, zero mismatches |

Both comparisons report maximum/RMS error 0/0 against their quantized references
and zero runtime errors. Their invocation windows include host computation,
Gemmini commands and DMA inside the model. They exclude startup, input/state
copies, comparisons and logging. Numerical equality to these references does
not establish floating-point model accuracy or a different compiler's contract.

TinyLlama prefill is 6,084,226,968 cycles and all decode steps total
74,915,076,022 cycles. These sum exactly to the invocation total. Decode accounts
for **92.488544%** of the total; mean decode is 2,341,096,125.6875 cycles, agreeing
with the reported rounded 2,341,096,126. The reported range is
2,192,773,326--2,488,755,134. The approximately 99 million comparison memcpy calls
are outside this window. Removing them accelerates validation without reducing
these invocation counts.

## Concrete scheduling questions

The reported first ResNet convolution issues approximately 2.46 million
activation-row loads and 160 thousand fences. The reported 41 kernels use raw
commands rather than loop instructions. This is separate from the 54 offloaded
operation sites (53 convolutions and one final matmul); unique kernels need not
equal call sites. These counts motivate checking general blocking, larger legal
transfers, operand retention and redundant configuration/fence removal. Counts
alone cannot attribute cycle percentages, physical traffic or queue stalls.

The [Gemmini loop-unroller documentation](https://github.com/ucb-bar/gemmini#matmul-and-conv-loop-unrollers)
describes high-level matmul/convolution instructions whose FSMs tile, double
buffer and balance ROB load/store/execute occupancy. The operator explicitly
forbids these hardware-loop instructions and their configuration commands.
Optimize raw explicit commands and ordinary software loops instead. Preserve
the final-ELF zero-loop audit. Blocking, reuse, buffering and overlap still need
storage ownership, dependency, numerical and completion checks; they do not
require selecting the hardware loop engines.

TinyLlama reports all 155 weight-linear sites and 44 attention-matmul sites
offloaded. Head/plane expansion produces 3,675 INT8 contraction calls per
reported invocation. The original call-count window still needs reopening;
do not multiply it by the number of decode steps without that evidence. Check
batching, GQA layout preservation, reuse and dispatch aggregation while retaining
every selected head, precision plane, intermediate and KV state. Reducing call
count by dropping required work is not an optimization.

ResNet retains 16 residual additions, 17 ReLUs and both pools on the FP32 host;
some convolution outputs use INT32 readout before host processing. TinyLlama's
FP32 scaling/encoding audit remains incomplete. These are candidates for exact
fusion only where the selected scale, bias, activation order, rounding, clamp
and output contract permit it. New quantization recipes or approximate native
epilogues need a separate contract decision. They are not scheduling wins under
the frozen Phase 2 objective.

## Roofline and Spike checks

The selected facts `b8ff28ed...` explicitly describe a 16x16 array with one
multiplier per element (256 MAC/cycle optimistic structural capacity) and
16-byte read/write client interfaces. The existing generic
`merlin.perf.capsule_roofline.machine_bounds` derives those values. A client beat
width is not sustained memory bandwidth. The reported FireSim hardware identity
has not yet been joined to those facts.

A complete model floor still needs exact source work, useful versus padded
work, compulsory traffic under checked residency, packing/layout/host feeding,
readout, issue and legal overlap. Full-model floor, attainment and Spike
prediction therefore remain **UNKNOWN**. Do not sum overlapping stage costs or
add a kernel prediction to a full invocation that already includes it. The
current archived calibration domains cannot be extended by inserting these two
validation measurements as training labels.

The quoted approximately 23 million ResNet and 220 million TinyLlama reference
cycles lack source and matched hardware, numerical policy, workload and timing
windows. Keep them as unverified context, not measured speedups or roofline
floors. In particular, establish whether 220 million refers to one decode step
or the complete prefill plus 32-step workload. Compare paired measurements only
after those boundaries and identities match.

The public development controls should independently exercise command density,
reuse/invalidation, M/N/K tails, deep K, logical and physical capacity knees,
readout changes, batched attention and carried state. Preserve the complete
declared roster and refusals. The automated queue must demonstrate emitted
program changes, full outputs, checked effects and correctly scoped feedback
before any performance promotion; shared mechanisms alone are insufficient.
