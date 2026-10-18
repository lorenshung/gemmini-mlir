# Exact dtype families at the Core ATen provider boundary

This contract extends `gemmini_core_aten_catalog_v1`. Source MLIR and the
source-bound catalog are the interface. Merlin owns admission, host representation
changes, exact recombination and final dtype semantics. The provider owns the
integer device schedules and proves their accumulator bounds. No family grants
permission to replace a native operation with lossy quantization.

## F1: opaque element movement

The provider recognizes closed tensor selectors: copies, concatenation, static
slices/inserts, broadcast/repeat, permutations, constant splats/fills, and
parallel generics containing only static indexing, extract/yield/select and
equal-width bitcasts on data. Arithmetic identities such as `x + 0` are rejected:
they need not preserve floating payloads. Data-dependent predicates/indices and
numeric conversions are not bitwise selectors. Structural carriers may form
several dependent bindings; no provenance operator name chooses a schedule.

Every element is an ordered group of bytes. Storage widths are i1/i8=1,
i16/f16/bf16=2, i32/f32=4, i64/f64/complex<f32>=8, on little-endian RISC-V.
model2mlir may expose complex64 as `tensor<...x2xf32>`; its explicit real/imag
selector maps, rather than a guessed complex layout, govern those 4-byte lanes.
Incoming bytes are opaque, including NaN payloads, subnormals, negative zero and
int64 values outside the floating exact range. Finite literals are encoded once
at their declared IEEE/integer width; nonfinite literal payloads lacking a
preservation proof decline. Bool literals store 0 or 1 in one ABI byte.

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

All inputs combined in one linear output must have the same
`8*index + exponent`. Otherwise emit separate component producers and combine
their differently weighted outputs on the host. Integer coefficient alignment
may instead be explicit in the source body, subject to the bound checks.

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

The source digest/ordinal binding authenticates the exact producer and plan.
The plan's host representation/dtype interpretation remains a **Merlin
obligation**, independently of this device proof. The provider neither emits
nor removes the explicit host recombination, clamp, modulo wrap, integer
division/promotion, or correctly rounded final float conversion. In particular,
integer true-division may require reproducing eager's compute-dtype casts;
an exact rational quotient is not automatically the same source operation.
