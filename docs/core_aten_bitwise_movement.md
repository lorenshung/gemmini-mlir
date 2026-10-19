# Exact dtype families at the Core ATen provider boundary

This contract extends `gemmini_core_aten_catalog_v1`. Source MLIR and the
source-bound catalog are the interface. Merlin owns admission, host representation
changes, exact recombination and final dtype semantics. The provider owns the
integer device schedules and proves their accumulator bounds. No family grants
permission to replace a native operation with lossy quantization.

## F3: quantized movement storage codes

Closed movement selectors between per-tensor byte DQ and Q boundaries use
`bindings[].numeric.readout="storage_codes"` when **every** input and the output
have identical reduced scale rationals, zero points, quantization ranges and
storage dtypes. This is a semantic selector proof; provenance names never
choose the schedule. It covers the same static selectors as F1, plus lossless
float widening inside a quantized dtype copy. Arithmetic identities, reductions,
data-dependent selectors and lossy conversions do not qualify.

For legal incoming codes `qmin <= c <= qmax`, with positive scale `s` and
integer zero point `z`, `clamp(RNE((c-z)*s/s)+z) = c`. The device therefore
loads and stores byte codes through scratchpad with unit scales. There is no
accumulator access, signed/unsigned repacking, i32 scratch/readout or runtime
requantization. Unsigned byte storage retains its original bits. Literal
elements (pads and constant-only fills) are quantized once during compilation,
using exact rational RNE, zero point and clamp, then loaded as literal bytes.
Constant-only selectors may have an empty input proof. Different qparams retain
the existing full-i32 device schedule and exact rational host requantization.
The original input quantization and final dequantization remain source host work.

The binding numeric record and kernel metadata contain:

- `family="movement"`, `readout="storage_codes"`, `requantization="none"`,
  `exact_arithmetic=true`;
- `identical_qparam_proof.method="closed_static_selectors_identical_qparams"`;
- `identical_qparam_proof.inputs`, indexed by kernel input name, and `.output`.
  Each records `operation_ordinal` of the source DQ/Q, reduced decimal-string
  `scale_numerator`/`scale_denominator`, `zero_point`, `quant_min`, `quant_max`,
  `storage_dtype`, and `source_quant_attributes` containing the original printed
  `prov.quant.*` attributes. Historical IR without rational provenance derives
  its rational from the authenticated constant operand; current captures carry
  the rational attributes. Known schema, operand/provenance consistency and
  placement policies are still checked;
- the identity equation, declared input-code range precondition, and
  `literal_quantization="exact_rational_RNE_once"`.

`source_sha256` authenticates the original bytes; `source_operation_ordinal`
identifies the final selector carrier, `root_ordinal` the Q boundary, and
`replaced_ordinals` the closed selector island and consumed Q/DQ operations.
Shared DQ uses outside the island remain live after rewriting. These proofs
participate in the kernel digest so deduplication cannot substitute another
quantization contract. `kernels[].requantization` is empty. The provider's
`admit_numeric_schedule` returns `device_readout="storage_codes"` and the same
proof in `kernels[].schedule`, after auditing the emitted scratchpad stream.

### Merlin consumer requirement

The current non-contraction region route already accepts these catalog records:
`merlin/targetgen/core_aten_device.py::region_routing_for_bundle` delegates to
the provider, and `merlin/llvmlower/device_build.py::apply_region_offload` checks
nonempty bindings and authenticates the artifacts. No Merlin edit is necessary
to execute this route. To independently enforce the new numeric record, that
region-admission boundary must validate `bindings[].numeric.readout`, re-read
the recorded DQ/Q ordinals from the source with matching `source_sha256`, compare
all six qparam fields (rational scale as a pair), and authenticate the closed
selector and scratchpad-only kernel proof. It must retain input Q/final DQ and
require no runtime requantization for `storage_codes`.

`merlin/targetgen/core_aten_spec_int8.py::provider_coverage` currently requires
`kernels[].schedule.readout="full_i32"` for arithmetic integer readouts. Keep
that check for arithmetic. If this checker is extended to movement islands,
accept `schedule.readout="storage_codes"` only with the identity proof above;
do not treat a byte result type or an `exact_arithmetic` flag as sufficient.
This change does not authorize arithmetic narrowing or floating readout scales.

### Bounded layout schedules

Structural copies, affine permutation/projection generics, transpose, concat,
broadcast, static slice/insert, literal fills and dense shape changes use
`selector_representation="shape_stride_map"`. Quantized matching walks a closed
graph of these primitives to its DQ leaves, including composed pad/repeat forms.
Native movement keeps ordered element-width byte groups. No tensor-sized
coefficient/index arrays or per-element Python objects are constructed on this
path. Provenance names do not affect recognition or scheduling.

`CompactInstructions` retains explicit affine CPU loops before encoding, with
fixed command fields/local addresses and affine DRAM byte offsets. Full tiles
and tails have separate constant forms. LLVM emission consumes that loop tree
directly; it does not first expand a stream and then discover loops. Command
counts still describe dynamic primitive instructions. Hazard lint and the
encoding round-trip check stream the expansion in O(tiles) time without storing
it. Kernel digests authenticate the canonical loop descriptors. Form verification
checks each template because only pointer offsets vary in these loops.

Different qparams use full-i32 widening and the existing exact integer host
readout. For a closed selector graph, pointwise requantization commutes with
selection: each distinct input may be widened and exactly quantized into private
destination-code storage before the DMA graph. It undergoes exactly one RNE and
clamp; repeated selections copy that code. Literals receive their one output
quantization at compilation. `Readout.after` records the primitive-stream
boundary before subsequent DMA reads its converted storage. This changes no
native-dtype arithmetic or numeric policy. Direct movement helpers may also
use scalar readouts on disjoint destination regions; optional `placement`
describes `(count, block, offset, stride)` or a static shape/stride/offset map.
Uniform readout parameters emit one CPU loop without tensor-sized parameter
arrays. These remain target ABI glue around the existing exact readout contract.

Compiler resource limits are explicit: compact DMA kernels admit at most
1,048,576 dynamic primitives and 64 MiB of total declared argument storage.
General interpreted selector/linear fallback maps admit at most 65,536 points;
larger unrecognized forms decline before materialization. These are compiler
budgets, not claims about RTL capacity. Structural model-size layouts bypass
the interpreted-map limit. Pure movement builders never delegate to dense
linear arithmetic; already bound return producers are not rebuilt there.

Merlin region admission must authenticate `full_i32` movement readouts and their
`after` boundaries when checking a composed layout schedule, as well as the
`storage_codes` identity proof above. It must not require every MVOUT to target
the accumulator: scratchpad DMA after exact host conversion is byte movement.
Accumulator MVOUTs still require full-width readout with unit scale. See
`docs/taskD2_layout_scalability.md` for resource measurements and qualification.

## F1: opaque element movement

The provider recognizes closed tensor selectors: copies, concatenation, static
slices/inserts, broadcast/repeat, permutations, static reshapes, constant splats/fills, and
parallel generics containing only static indexing, extract/yield/select and
equal-width bitcasts on data. Arithmetic identities such as `x + 0` are rejected:
they need not preserve floating payloads. Data-dependent predicates/indices and
numeric conversions are not bitwise selectors. Structural carriers may form
several dependent bindings; no provenance operator name chooses a schedule.

Static reshapes preserve flattened element order and element type, with a literal
shape equal to the declared result shape. A rank-zero result has one element;
its zero-length shape descriptor is metadata, not an empty data tensor. This
rule applies at every supported element width and rank. Actual empty payloads
still decline as no device work. The F2/F4 plan has no native-dtype limb-count
limit: each independently bounded component retains its full integer index and
binary exponent, including plans wider than 64 native bits. Host splitting and
recombination remain Merlin's obligations.

Every element is an ordered group of bytes. Storage widths are i1/i8=1,
i16/f16/bf16=2, i32/f32=4, i64/f64/complex<f32>=8, on little-endian RISC-V.
model2mlir may expose complex64 as `tensor<...x2xf32>`; its explicit real/imag
selector maps, rather than a guessed complex layout, govern those 4-byte lanes.
Incoming bytes are opaque, including NaN payloads, subnormals, negative zero and
int64 values outside the floating exact range. Finite literals are encoded once
at their declared IEEE/integer width; infinities use their exact signed IEEE
encoding. NaN literals lacking a payload proof decline with
`NaN movement literal has no preserved payload proof`. Bool literals store 0
or 1 in one ABI byte.

For each F1 binding:

```json
"numeric": {
  "family": "F1",
  "numerics": "bitwise_movement",
  "quantization": "none",
  "element_bytes": 8,
  "operand_element_bytes": {"x0": 8},
  "byte_order": "little",
  "exact_arithmetic": true
}
```

The kernel `meta` repeats these facts and records
`proof="closed_static_element_selectors"`, `readout="scratchpad_bytes"`.
Each `kernels[].args[]` retains the **logical source MLIR** `dtype` and `shape`,
and adds `element_bytes`. Memref offsets and strides remain in logical elements;
the shim checks dense row-major shape/strides and scales the descriptor offset
by `element_bytes` when computing its byte pointer. The C kernel ABI remains one
pointer per listed non-constant argument. Inputs/outputs are disjoint. Intermediate
results receive separate buffers and source ordinal bindings as usual.

F1 emits only unit-scale scratchpad DMA, fences and flush. There is no accumulator
narrowing, quantization, floating data arithmetic or host requantization. Each
call fences before returning. The static selector/source byte-map capacity is
16 MiB per tensor; exceeding it is a reasoned resource decline. Sub-element
column transfers preserve byte order and element boundaries. DMA tiles obey
the existing DIM and scratchpad limits; the regular stream lint checks hazards.

## F2 and F4: exact signed-byte components with full int32 readout

Merlin supplies **already split signed i8 component tensors**. Splitting may use
balanced signed digits or centered unsigned digits with exact host compensation.
The provider does not guess that representation or quantize native inputs.
All offset compensation/carries, dtype wraparound, divisors, signs, ReLU and
rounding remain explicit source host arithmetic after the returned int32 tiles.
Only the integer producer is replaced. A radix-256 plan does not itself prove
that host splitting or recombination implements the original dtype operation.

Attach a JSON `StringAttr` named **`gemmini.limb`** to each integer producer:

```json
{
  "version": 1,
  "family": "F2",
  "radix": 256,
  "inputs": [
    {"operand": 0, "index": 1, "count": 4, "exponent": 0},
    {"operand": 1, "index": 2, "count": 4, "exponent": -16}
  ]
}
```

All shown keys are required; `component` is additionally required for F4. Unknown
keys decline. Input `operand` is the zero-based tensor operand number of this
producer; every actual i8 data input must occur exactly once. `index` is in
`[0,count)`, `count` is positive, and `exponent` is the component tensor's common
dyadic exponent **before** the radix weight. These integers are mathematical
metadata, not device fp32 scales. Inputs always have the full possible signed
byte range [-128,127]; no narrower range is trusted from a label.

An input operand may alternatively be the i32 result of a closed, parallel,
identity-map generic containing just `arith.extsi i8 to i32` and its yield.
The entry then describes its underlying signed-byte tensor. The provider checks
the cast's body/maps and binds the byte tensor directly. Literal i32 coefficient
tensors need no plan entry. Arbitrary runtime i32 inputs are not byte limbs.
Duplicate SSA limb inputs must be represented once, with explicit coefficients.
Quantized `prov.quant.*` and `gemmini.limb` contracts cannot share a producer.

### Bilinear source form

Use static dense `linalg.matmul` or the canonical rank-2/rank-3 integer contraction
generic already accepted by the provider: i8 A/B, zero i32 initializer, signed
extension to i32, multiplication, addition into the accumulator, yield. Rank-3
indexing implements independent batches. The result is i32 with no epilogue.
All extents and indexing/body SSA links are checked. Nonzero/unbounded seeds
are not part of this interface; retain bias in explicit host recombination.

For the example above the returned tile is exactly `A_1 @ B_2`, and its host
coefficient is `2**(8*(1+2) + 0 - 16)`. Convolution and its backwards may use
the frontend's general per-tap/patch GEMM decomposition; each such GEMM is an
independently bound source operation. It must not be selected by model identity.

### Linear source form

Use a closed static `linalg.generic`/sum reduction producing an i32 tensor from
signed i8 tensors: signed extension, integer add/subtract, multiplication by
literal integer coefficients, static indexing/gathers, and constant int32
initialization. Pool/window adjoints, folds, sum/mean numerators and affine maps
are expressed by these actual integer index/coefficient maps. Divisors and
nonlinear/final dtype work follow on the host. Any intermediate unproved
wrapping, data-dependent coefficient/index, division or narrowing declines.
Literal seeds/constants are in the returned component's units and receive its
binary exponent too. Merlin must distribute a native bias correctly across
components or retain it as a separate exact host term; it must not replicate
that bias at every radix position.

All inputs combined in one linear output must have the same
`8*index + exponent`. Otherwise emit separate component producers and combine
their differently weighted outputs on the host. Alternatively the frontend may
rebase the component representation to a common exponent and make integer
alignment explicit in the source body, subject to the bound checks.

### F4 real/imag components

The same source forms and proofs implement F4. `family="F4"` additionally names
`component="rr"|"ii"|"ri"|"ir"` for a bilinear producer, or `"real"|"imag"`
for a linear producer. Merlin emits the four general real/imag products and
retains `(rr-ii) + i*(ri+ir)` as exact source host arithmetic. Component labels
describe that binding; they do not substitute for proving the integer body.

### Returned proof and host boundary

Both `bindings[].numeric` and `kernels[].meta` record `family`, `limb_plan`,
`binary_exponent`, `exact_arithmetic=true`, `readout="full_i32"`,
`recombination="retained_source_host_exact_integer"`, and `bound_proof`.
`bound_proof` records the full signed-byte input range, mesh primitive K and its
maximum absolute partial sum, required signed MAC bits, i32 accumulator bounds,
and the proof method. Bilinear schedules split K into <=16 independent WS dot
products with exact accumulator SRAM addition. Linear schedules use signed-byte
coefficient parts and exact i32 addition, never scaled MVIN or fp32 readout.
Every possible device partial sum is bounded, not merely the final result.
Linear maps currently use K=1 signed products with a fence between reused tiles;
their instruction-part capacity is 262,144 products. Larger maps decline with
`limb linear instruction-part capacity exceeded`. Common exponent alignment is
a frontend obligation. General runtime i32 seeds, division, data-dependent
indices/predicates, nonlinear bodies and unproved wrapping decline. Use separate
bounded producers and retained host terms when an i32 bound fails.

The source digest/ordinal binding authenticates the exact producer and plan.
For F2/F4, `source_operation_ordinal` equals `root_ordinal` and identifies the
marked integer producer; `replaced_ordinals` also lists its proved carriers.
The plan's host representation/dtype interpretation remains a **Merlin
obligation**, independently of this device proof. The provider neither emits
nor removes the explicit host recombination, clamp, modulo wrap, integer
division/promotion, or correctly rounded final float conversion. In particular,
integer true-division may require reproducing eager's compute-dtype casts;
an exact rational quotient is not automatically the same source operation.
Nonfinite values and signed-zero semantics require an explicit frontend proof
or retained host handling; the limb representation alone describes finite
dyadic numbers and does not preserve NaN payloads or a zero's sign. F1 retains
those bits without interpreting them. A native unquantized arithmetic module
without the checked component form declines with
`no exact integer schedule for unquantized data arithmetic or numeric conversion`.

For `linalg.reduce`, serialize the attribute in generic operation syntax; some
xdsl custom `linalg.reduce` printers omit additional attributes. The provider
uses the bytes it receives and never reconstructs a missing plan from provenance.

## Coverage by source form

These are semantic families, not operator-name dispatch rules. Native arithmetic
must first be represented by the explicit component/host boundary above.

| Operations | f16/bf16/i32/i64 | complex64 | bool |
| --- | --- | --- | --- |
| cat, clone, constant_pad_nd, repeat, slice_scatter, select_scatter, full, full_like, fill.Scalar, same-dtype _to_copy | Native F1 | Native F1 | Native F1 |
| reflection_pad1d/2d/3d, replication_pad2d/3d | Native F1 | Native F1 | Eager CPU raises before capture |
| add.Scalar/Tensor, sub.Scalar/Tensor, mul.Scalar, neg | Explicit F2 linear components | Explicit F4 real/imag linear components | No native arithmetic admission here |
| adaptive/ordinary average pools (1d/2d/3d), supported pool adjoints, col2im, mean.dim | Explicit F2 integer numerator/index maps; host divisors | F4 components when native semantics permit | No native arithmetic admission here |
| mm, bmm, addmm, convolution, convolution_backward | Explicit F2 limb-pair GEMMs; host bias | Four F4 real/imag limb-pair families | No native arithmetic admission here |
| div.Scalar, relu | Host semantic step after exact components | Only where native semantics permit | No native arithmetic admission here |

Convolution spatial packing/unpacking is explicit selector IR and per-tap/patch
GEMMs; this interface does not accept an arbitrary unexpanded convolution body.
Numeric dtype-changing `_to_copy` is not an F1 selector; retain it after exact
components or supply a separately proved source conversion. The existing F3
quantized support is independent of this table.
