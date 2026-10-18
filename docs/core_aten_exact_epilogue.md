# Exact integer device work and the retained host epilogue

This extends the `gemmini_core_aten_catalog_v1` numeric contract. It does not
change the exact rational oracle or permit lossy native-dtype quantization.
Merlin owns host packing, exact rational evaluation, dtype semantics and final
requantization. The target owns the following integer schedules and ABI facts.

## Multi-term readout

For output position `p`, let `A_t[p]` be the signed full-width int32 value from
term buffer `t`, after its declared permutation and broadcast. The exact value
in output-code units is

```
v[p] = constant[p] + sum_t factor_t[p] * A_t[p]
if relu: v[p] = max(v[p], 0)
code[p] = clamp(RNE_ties_to_even(v[p]) + zero_point, quant_min, quant_max)
```

`constant` and every factor are exact rationals. A scalar broadcasts to every
output element; arrays have the declared output shape (or equal flattened
extent). A factor is serialized as decimal-string `numerator` and positive
`denominator`; these values, not an informational floating scale, govern the
calculation. Source scales are the fp32-canonical scales recorded at capture.

There is **one RNE after the complete rational sum and optional ReLU**. Terms,
products and constants must not be rounded independently. There is no fp32/f64
approximation of this sum. Integer/rational host intermediates must have proved
capacity or use arbitrary precision. Permutations/broadcasts occur before the
sum. Output zero point follows RNE; clamping follows zero-point addition.
Signed factors and negative half ties follow the same rule.

Each kernel's `requantization` records contain `out`, `terms`, `constant`,
`relu`, `zero_point`, `quant_min`, `quant_max`, `permutations`, `broadcasts`,
`after`, `rounding="half_to_even"`, `execution="host"` and
`device_readout="full_i32"`. A term is `[buffer_name, factor]`. `after` is a
fenced instruction boundary; absent means after the complete device stream.
The host epilogue cannot read a tile before the preceding device store fence.
The target emitter delegates this ABI to Merlin's host readout implementation.

This formula also covers a rational additive scalar that is not integral in
input-code units, multiple differently scaled input tiles, zero-point
compensation and unfused bias. Source-bound seeded integer readout producers
retain their explicit source readout instead of replacing it with this ABI.

## Combining arithmetic on Gemmini

The target derives a positive exact quantum `u[p]` from the coefficients and
input code units. If every term has an integral weight and all intermediate
partial sums are bounded, it computes

```
ACC[p] = seed[p] + sum_i integer_weight_i[p] * signed_byte_input_i[index_i[p]]
v[p] = u[p] * ACC[p] + residual_constant[p]
```

An integral scalar constant in code space becomes an int32 seed loaded on the
device. A nonintegral remainder stays in `residual_constant`. Input zero points
and uint8 XOR-0x80 packing offsets participate in the same exact constant map.
Unsigned source packing happens before device use; the proved device range is
the full signed-byte range `[-128,127]`.

The accumulator proof is
`abs(seed[p]) + 128 * sum_i abs(integer_weight_i[p]) < 2**31`.
It bounds every reordered partial, not just the final sum. Positive unit terms
use accumulating MVINs; signed/general weights use literal integer `c*I` tiles
or K=1 signed-byte products. Weights are split into parts of magnitude at most
127. No weight becomes an fp32 MVIN scale. Dense and proven row broadcasts use
vector tiles; general static index/coefficient maps use the sparse primitive.

For example, equal scales and `a + 2*b` produce one int32 accumulator containing
`a + 2*b - za - 2*zb`. The host applies only the common rational readout factor
and final quantization. Negation uses literal weight -1, preserving `-(-128)`
as int32 128. An integral scalar multiplier runs on the device; a dyadic/rational
multiplier can run its integer numerator there and retain its denominator in
the host factor. A reciprocal with numerator 1, such as division by 3, needs
only widening on the device and exact rational scaling on the host.

Small rational common divisors can also combine noninteger scale ratios. For
unrelated fp32 scales, however, that may require enormous integer weights.
If the int32 bound or the 262,144 instruction-part capacity fails, the target
retains separate bounded int32 coefficient-class sums and the exact multi-term
host formula above. It records
`combination_refusal="common integer units exceed exact i32 or instruction-part budget"`.
It never rounds a ratio into an approximate weight. Within a retained term,
window/overlap sums can still be device arithmetic.

`meta.arithmetic` distinguishes `integer_accumulating_mvin`,
`weighted_integer_compute`, `signed_byte_execute_relu`, `integer_widening`,
and `separate_rational_terms_host_combination`. Bounds, integer weights, constant
placement and activation proofs accompany these records. A COMPUTE containing
only an identity is not by itself evidence that an operation's arithmetic moved
to the device; the source numeric map and these records determine that claim.

## ReLU: full readout bypasses activation

Stock `AccumulatorScale.scala` forwards the original accumulator data through
`full_data`. Store activation applies to the narrow `data` path, not to full
int32 readout. Spike's `gemmini.cc` likewise writes `acc_value` for `full` and
the activated byte for narrow stores. `CONFIG_ST ReLU, scale=1` plus full-i32
MVOUT therefore **does not clamp negative int32 values**.

For a proven signed-byte identity input with zero seed/residual and positive
readout factor, the target instead uses two exact identity products. The first
writes to scratchpad with execute ReLU; `ExecuteController.scala` clips to
signed-byte width before activation. That clipping is an identity because the
product is already in `[-128,127]`. The result `[0,127]` is then widened by an
identity product into ACC and read out at full int32 width. The manifest records
the range and clipping proof in `execute_activation_proof`. Shifts are zero and
all scales are 1. Negative-weight or wider affine accumulators do not meet that
proof and retain the exact host ReLU after the rational sum. No narrowed ACC
readout is introduced to implement ReLU.

## Hazards, liveness and reporting

An explicit fence separates accumulator initialization from accumulating MVINs.
The HW3 lint grants no zero-fill-to-accumulate exception. The golden contraction
catalog also traces the complete ordinary CPU CFG into primitive commands and
runs the same lint. Its per-kernel `fences`, `instructions`, `hazard_lint` and
`hazard_lint_method="complete_static_CPU_CFG_trace"` describe the executed
schedule. Calibrated alternatives are checked from their actual selected
`kernel.gemmini.mlir`, not inferred from a default schedule. An unsafe schedule
declines before its target function is added to the catalog.

Before binding, the target delegates pure tensor DCE to Merlin's
`entry_weight_projection._pure_dce` on a clone. Frozen source bytes, SHA and
original operation ordinals stay unchanged. Target-binding liveness also drops
an initializer/copy whose only consumers are removed source operations and
which no live target call needs as an ABI input. Rewriting delegates remaining
pure cleanup to the same effect-aware helper; runtime assertions and unknown
effects are retained. The frontend's four named Q/DQ value operations are also
pure; an adapter removes their dead instances because they currently parse as
unregistered operations without xDSL effect traits. No generic model optimizer
is duplicated in the target.

Placement alone does not mean arithmetic acceleration. Report movement-only
operators as movement, and qualify affine arithmetic by the actual weights,
constant and activation placement. Nonoverlapping pool-adjoint index maps can
also be movement-only; overlapping maps accumulate on the device. Convolution
backwards on the golden route performs its GEMMs on the device while its
surrounding patch/fold and bias reductions remain explicit host source work.

## Closed reductions and shape epilogues

A closed Q/DQ linear island can include a zero-seeded sum, a literal rational
division and a dense `tensor.reshape`, `tensor.collapse_shape` or
`tensor.expand_shape` before its output quantization. These view operations
preserve flattened element order, element type and static element count. A
reshape additionally requires a literal shape tensor equal to its static
result shape; an unproved dynamic shape operand declines. Expand dimensions
must be static. The same rule applies to nonscalar outputs and longer chains.

The kernel returns full int32 numerator tiles. The host readout factor retains
the divisor exactly: for six values with input scale `sx` and output scale
`sy`, `A = sum(codes)` and `v = A * sx / (6 * sy)`. Neither division nor view
inserts an intermediate rounding. Named and generic zero-seeded reductions
use the same contract. Source-bound `admit_numeric_schedule` proofs use
`abi="gemmini_core_aten_catalog_v1"`, `exact_arithmetic=true`,
`schedule.readout="full_i32"`,
`schedule.requantization="exact_rational_host_epilogue"` and the catalog's
exact `requantization` records. The proof checks the actual integer stream,
accumulator bounds and selected MAC policy. Partial matches do not establish
coverage of a separate live arithmetic producer. Input Q and final DQ remain
host boundary operations; the bound reduction, division, views and output Q
are replaced by the integer kernel plus this exact host readout ABI.

A partial closed-linear layout prelude does not supersede the legacy raw-GEMM
ABI. If total linear coverage fails, admission may independently authenticate
the legacy contraction catalog instead. That proof contains only its actual
source-bound GEMMs and retains layout producers, Q/DQ boundaries and other host
source work. It must still cover every proposed legacy contraction; an unrelated
linear match cannot grant that coverage. If the legacy proof also has no complete
coverage, the original linear refusal is returned. This permits the existing
convolution-backward route without claiming that its host patch/fold work is
device computation.

An empty catalog is an explicit refusal manifest with `status="declined"`,
`reason="no exact device kernel produced"`, no kernels/bindings and no
`compilation` receipt. `core_aten_catalog` and `catalog.merlin_builder` return
`(manifest_path, None)` in that case. `core_aten_offload` returns
`(None, None, None, manifest_path)` and emits no model or shim. Callers must
handle the absent artifacts as a device refusal; a preexisting object in the
work directory does not make it a successful compilation.
