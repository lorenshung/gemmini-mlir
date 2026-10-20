# Core ATen Gemmini source/catalog ABI

This repository-local note supplements the shared Merlin ABI notes. The interface
is source MLIR plus `gemmini_core_aten_catalog_v1`: `source_sha256` authenticates
the bytes; each binding identifies its `source_operation_ordinal`, `root_ordinal`,
`replaced_ordinals`, operands, result and kernel symbol. Bindings must cover each
live proposed integer producer exactly once. Dead producers are not emitted.
A clean decline carries a reason; a zero-kernel result claims no object path.
Provenance identifies source operations, never selects a numeric strategy.

Merlin owns host representation changes, packing, recombination, final dtype
semantics and generic host code generation. This backend owns target schedules,
resource/bound proofs, instruction encoding, hazard lint and target ABI glue.
The local host requantizers are prototypes for promotion to Merlin, not a second
independent numeric contract.

## Exact int32 readout + host requant

Device arithmetic is integer-exact. Every mesh dot product and every accumulator
prefix, including seeds and zero-point corrections, needs a nonoverflow bound.
Independent WS dot products split K into at most 16 elements; exact i32 SRAM
accumulation combines them. Arithmetic MVOUT uses full-width signed int32, dense
row-major storage. MVIN scales and CONFIG_ST accumulator scales are unity in
these schedules. A nonunit data-path scale would require a recorded exact-product
proof; none of the families below depends on it. Fences establish DMA completion
before host reads, overlapping local writes and buffer reuse.

For output element `p`, the multi-term host epilogue is exactly

```
v[p] = constant[p] + sum_j(coefficient[j,p] * acc[j, index[j,p]])
q[p] = clamp(RNE(v[p]) + zero_point[p], quant_min[p], quant_max[p])
```

All coefficients and constants are exact rationals; broadcast, permutation and
placement maps select elements without rounding. With host ReLU, replace `v` by
`max(0,v)` before RNE. Zero point is added **after** ties-to-even rounding, so an
odd zero point never changes tie parity. There is one quantization. Scales are
the fp32-canonical values recorded by capture, but their products/ratios are
formed exactly. A rational constant (including nonintegral scalar add/sub or
bias) is never prematurely rounded into an i32 seed. Private tiles remain
unchanged until their last host consumer.

Readout parameters carry reduced decimal-string `scale_numerator` and positive
`scale_denominator`, `zero_point`, `quant_min`, `quant_max`, and
`rounding="half_to_even"`. A single term is `v=a*numerator/denominator`;
multi-term `Readout.terms` and `.constant` express the equation above. Exact
scalar conversion accepts Python/NumPy scalars and 0-D arrays via `.item()`;
positive-rank arrays require the explicit broadcast/channel path.

When common integer units fit the proved i32 and instruction budgets, affine
maps combine terms on Gemmini: accumulating MVIN for unit sums, literal signed
integer weight tiles/products for other weights, and an accumulated constant
when integral in those units. Negation uses literal negative weights. Otherwise
separate full-i32 terms retain their exact rational host combination; a
noninteger scale ratio alone does not license approximation into byte weights.

### Device ReLU

The signed-byte identity selector uses an exact identity product into scratchpad
with execute ReLU and zero shift. Execute clips before activation, but the
identity product is already in [-128,127], so clipping changes nothing. A second
identity product widens the activated [0,127] bytes into ACC for full-i32 MVOUT
at unit scale; full-width ACC readout itself bypasses activation. Metadata
records `device_relu` and `execute_activation_proof`.
This rule needs the proved identity map, unit integer weights and zero constant;
it does not move a ReLU across a negative factor or arbitrary host expression.
Other ReLUs remain at their explicit source/host boundary.

### Phase 1 plain exact host readout

The emitter records `host_readout_contract="exact_rational_RNE_v1"` in kernel
metadata and `gemmini.host_readout_contract` on an emitted module with readouts.
It puts each exact rational expression over a common positive denominator.
Host integer widths cover the entire signed-int32 domain of every input tile,
the constant, twice the numerator and all half-integer output boundaries.
A bounded search compares that exact numerator to half-integer boundaries,
including parity at a tie; it implements the equation above without FP or an
intermediate rounding. No multiply-shift approximation, stored transition table,
per-channel algorithm certificate or recorded reachable-range specialization is
used. Channel, broadcast, permutation and placement semantics are unchanged.

Readouts execute serially at their fenced `after` boundary (or after the complete
stream). There are no overlap slots, slot offsets, partial-index parameters or
`readout_pipeline`/`host_requantization_proofs` records. Each complete private i32
tile remains available for its host consumers. Merlin-owned retained-source
readouts are unchanged; this emitter contract applies to provider-owned readouts.

## F1, F2, F4 and quantized movement

**F1** is opaque element movement, not arithmetic: static copies, selectors,
concat, repeat/broadcast, slices/inserts, pads, reshapes and literal fills preserve
ordered byte groups. ABI widths are bool/i8=1, i16/f16/bf16=2, i32/f32=4 and
i64/f64/complex<f32>=8 bytes, little-endian. Explicit real/imag f32 source layouts
retain their own selector maps. Payloads, signed zero and wide integers are not
quantized. Numeric conversions and arithmetic identities are not byte selectors.
Literal encoding must preserve the declared dtype's bits; NaN literals without
payload proof decline.

The binding `numeric` records `family="F1"`, `numerics="bitwise_movement"`,
`quantization="none"`, `element_bytes`, `operand_element_bytes`,
`byte_order="little"`, `exact_arithmetic=true`. Kernel metadata adds
`proof="closed_static_element_selectors"`, `readout="scratchpad_bytes"`.
Arguments retain logical dtype/shape; memref offsets are logical elements and
are multiplied by itemsize for the byte pointer. Dense row-major descriptors
and disjoint buffers are required. Device execution uses scratchpad DMA only.

**F2** (real/integer components) and **F4** (real/imag components) are arithmetic
families, not movement. Merlin supplies signed i8 limb tensors and explicit
source host recombination. A `gemmini.limb` JSON StringAttr on each producer
records `version=1`, `family`, `radix=256`, and `inputs` entries with `operand`,
`index`, `count`, `exponent`. F4 additionally requires a top-level `component`:
`real`/`imag` for linear forms, `rr`/`ii`/`ri`/`ir` for bilinear ones.
Each runtime byte operand
is covered exactly once; closed i8-to-i32 signed widening may be bypassed after
checking its maps/body. No native input is silently quantized or narrowed.

The provider lowers closed linear integer maps and zero-seeded limb-pair
contractions. Each input has the full signed-byte domain. For limb indices i,j
and exponents e,f, a returned pair product has host weight
`2**(8*(i+j)+e+f)`. Bindings record `family`, `limb_plan`, `binary_exponent`,
`bound_proof`, `readout="full_i32"`, `exact_arithmetic=true`,
`recombination="retained_source_host_exact_integer"`. Every device partial sum
is bounded. Host carries, signs, compensation, dtype wraparound, divisors,
nonfinite/signed-zero handling and final conversion remain explicit Merlin
obligations. A limb label alone proves none of those semantics.

Identical-qparam quantized selectors (F3) instead carry
`family="movement"`, `readout="storage_codes"`, `requantization="none"`, plus
`identical_qparam_proof`: all input/output reduced scales, zero points, ranges
and storage dtypes agree. Then Q(DQ(code)) is exactly that code for every legal
input. Scratchpad DMA copies raw bits; literals receive one exact compile-time
quantization. Different qparams retain full-i32 widening and exact host
requantization, followed by movement in destination-code units.

See [the detailed family/binding contract](docs/core_aten_bitwise_movement.md).

## Centered byte planes and integer producers

Closed i32 extension/subtraction/selector graphs can represent asymmetric signed
or unsigned byte contraction operands. Unsigned raw bytes are XOR-0x80 packed:
`extui(raw) = signed(packed) + 128`. The interpreter proves each operand selects
at most one byte with unit coefficient plus a bounded literal integer correction.
That correction is split into literal signed-byte planes. All plane-pair products
are exact GEMMs, combined through full-i32 results/seeds; a magnitude-255 value
is never cast to i8. `centered_byte_contraction` records the rule
`closed_byte_selectors_plus_literal_integer_planes`, selectors' hashes, unsigned
sources, plane counts, operand bounds and a conservative expanded absolute-prefix
bound including the seed. Failure of an intermediate bound declines cleanly.

A catalog may mix raw from-zero integer contractions and readout-backed ones.
Both set `integer_producer=true`, `readout="full_i32"`. Raw producers have
`raw_integer=true`, `integer_readout=false`, `requantization="none"` and return
source i32 directly. Readout-backed producers have `integer_readout=true`,
`raw_integer=false`, `requantization="retained_source_exact_integer"` and source
rational numerator/denominator metadata (signed nonzero numerator, positive
denominator). Only the integer producer is replaced: its exact readout, bias,
channel parameters, reshapes/reindexing and final dtype semantics stay on host.
The consumer must not divide the recorded ratio by output_scale again.

## Layout/DMA schedules

Structural selectors use `selector_representation="shape_stride_map"` and
compact affine CPU command loops, with fixed instruction fields/local addresses
and affine DRAM byte offsets. Full tiles and tails are explicit. LLVM emits the
loop tree directly; hazard lint and encoding checks stream its expansion.
Digests authenticate canonical loop descriptors, proofs and source bindings.
DMA widths/strides, scratchpad capacity, element boundaries and HW3/HW4 fences
remain target legality requirements. Singleton axes may be removed only after
bounds proof; contiguous axes may be coalesced. Literal layout transforms use
the ordinary DMA schedule. There is no `constant_folds` manifest record; a
literal-only movement still has a real device binding and object. Encoding
literal input bytes and unsigned XOR packing remain required ABI operations.

`Readout.after` records the fenced primitive boundary before subsequent DMA
reads host-converted storage. Optional `placement` gives a shape/stride/offset
or count/block/offset/stride destination map. Uniform parameters need no
per-element metadata arrays. Retained integer-readout views remain source host
work; extra native layout kernels are not inserted around an already bound
producer in that route. Direct contraction/conv packing schedules remain valid.

Compact DMA admits at most 1,048,576 dynamic primitives and 64 MiB total declared
argument storage. General interpreted fallback maps admit at most 65,536 points;
other specialized maps retain their documented limits. These are compiler
budgets, not RTL capacity claims. Unsupported forms decline before uncontrolled
materialization. Final-ELF audits must still find unit data-path scales and no
narrowed accumulator readouts; scratchpad byte MVOUT is valid movement.

## Phase 1 placement

This provider has no cost gate or profitability policy. A native family needs
its semantic constraints, source-bound exactness proof and legal target schedule.
No runtime samples or expected outputs establish those proofs. Unsupported or
unsafe constructions decline with a reason. Generic hack detection and native
dtype semantics are Merlin obligations; a provider proof never waives them.
Structural compact DMA loops remain because supported model-size layouts need
them to fit compiler memory/code-generation budgets. Literal folding and the
requant-A performance implementation are preserved only on `gemmini-perf`.

## VCS RTL engine

The backend accepts `vcs` explicitly in `available`, `run_elf` and
`simulator_identity`, with oracle kind `rtl_vcs`. Discovery uses
`MERLIN_GEMMINI_SIMV`, then Chipyard's
`sims/vcs/simv-chipyard.harness-<rtl_sim_config>`. The VCS config override is
`MERLIN_GEMMINI_VCS_CONFIG`; Verilator's override does not select a VCS build.
An explicit VCS request never falls back to another engine.

Execution follows Chipyard's permissive plusarg convention: DRAMSim/ini directory,
cycle cap, seed, `+loadmem=<absolute ELF>`, then `+permissive-off <ELF>`.
Loadmem is unconditional and is the sole image loader, including timing runs;
the positional ELF supplies symbols/HTIF metadata. Cycle counts may be compared
only under this same loading convention, never against TSI-loaded receipts.
`-no_save` avoids VCS save/restore's ASLR re-exec in restricted sandboxes.
The ELF directory is the writable cwd; stdin is `/dev/null`. The caller owns
license environment/access and the wall timeout. DRAM geometry is the selected
RTL model, with the ordinary target linker/arena contract.

Identity binds the executable hash and a sorted recursive `.daidir` file
manifest, its `daidir_sha256`, and a combined `digest`. Missing or incomplete
runtime trees have no complete digest. Native invocation records also include
daidir and DRAM ini files as dependencies. The grader must grant the binary,
its resolved `.daidir`, vendor/runtime shared libraries, ini directory and ELF,
plus writable work/tmp directories. Preserve the site's license environment
and authorized network or file access; the backend supplies no license server.
Verilator and GSIM interfaces remain unchanged.
