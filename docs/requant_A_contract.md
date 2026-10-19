# requant-A-v1: shared Merlin / Gemmini host boundary

The default is `clamp_mul_add_shift`, proved separately for every concrete
parameter class (including each channel and frozen bias). If no candidate passes,
use `threshold_table`. Invalid or tampered constants are rejected, never emitted.
The generic prototype is `mlir_oot/core_aten/requant_a.py`; Merlin owns the eventual
shared implementation. This contract lets both compiler lanes emit the same form
without depending on concurrent, unmerged changes.

All arithmetic parameters and bounds are decimal integer **strings** in JSON;
`S`, zero point, output bounds, counts and flags use JSON integers/booleans.
Rationals have positive denominators and reduced numerator/denominator pairs.
Compiler-side scalar parameters may be Python rational/integer values, NumPy
scalars or proven 0-D arrays. A 0-D array is extracted with exact `.item()`;
Fraction and wide-integer contents never pass through floating point. Arrays
with positive rank belong to the channel/element broadcasting path and are
rejected at a scalar conversion boundary, even when they contain one element.

The canonical semantic record has these exact fields:

```
{
  "schema": "requant-A-v1",
  "algorithm": "clamp_mul_add_shift" | "threshold_table",
  "domain": {"min": "...", "max": "..."},
  "ratio": {"numerator": "...", "denominator": "..."},
  "offset": {"numerator": "...", "denominator": "..."},
  "zero_point": 0, "quant_min": -128, "quant_max": 127,
  "relu": false, "rounding": "half_to_even",
  "constants": {"L": "...", "H": "...", "M": "...", "B": "...",
                "S": 0, "out_clamp": false, "qlo": -128, "qhi": 127},
  "proof": {...}
}
```

For `threshold_table`, omit `constants`; provide `thresholds` (at most 255
nondecreasing decimal strings) and a `refusal` string. Threshold `j` is the first
integer `x` with output at least `quant_min+j+1`, clipped to `[min,max+1]`.
Duplicates are intentional. Evaluate `quant_min + upper_bound(thresholds,x)`.
No per-element rational multiply/divide is needed on the fallback path.

Semantics: `clamp(RNE(x*ratio+offset)+zero_point, quant_min,quant_max)`.
With `relu`, replace the real rational value by `max(0,value)` before RNE.
The fast form is `v=clamp(x,L,H); q=(v*M+B) arithmetic_shift_right S`.
If `out_clamp` is true, finish with `clamp(q,qlo,qhi)` (the reference solver's
second candidate family). Prefer the form without this extra output clamp.
A constant map may use `M=0,S=0,L=H=0,B=code`.

`M>=0`, `0<=S<=62`. `L,H,M,B`, the products at both band endpoints and both
product-plus-bias endpoints must fit signed int64. Check products separately:
cancellation by `B` does not license a wrapping intermediate. Clamp in the
original exact integer width **before** narrowing to int64. Emit no FP, rounding
correction, reciprocal division or second quantization.

Proof: both maps are monotone. Compute every exact RNE output transition with
rational arithmetic; test `t-1,t` inside the domain for all transitions, both
domain endpoints, and in-domain band endpoints. Agreement at both ends of each
constant segment proves agreement everywhere, including every ties-to-even
boundary. The proof record is:

```
{"method":"all_monotone_step_boundaries", "step_count":255,
 "checked_points":0, "checked_points_sha256":"...", "thresholds_sha256":"...",
 "monotone":true, "integer_arithmetic":"signed_i64" | "exact_integer_comparisons",
 "product_min":"...", "product_max":"...", "sum_min":"...", "sum_max":"..."}
```

Omit product/sum fields for threshold records. A hash is SHA-256 of the UTF-8
concatenation `decimal_integer + "\n"` for each value. Checked points are sorted
and deduplicated; thresholds preserve order and duplicates. Reconstruct and
validate the boundary proof when importing a certificate. Never trust a marker.

For multi-term sums, compute an exact integer normal form first. Over a common
denominator `D`, let the variable coefficients be `a[j]`, and `g=gcd(abs(a[j]))`
(or 1 when all are zero). Then `x=sum((a[j]/g)*acc[j])`, `ratio=g/D`,
`offset=constant/D`. Negative weights remain exact integer terms. Frozen bias is
in `offset`, never rounded into an integer seed. Bound each product and partial
sum from the recorded tile intervals; choose a sufficiently wide integer type.
Single-term positive factors reduce to the original accumulator without an
extra runtime multiply. Parameter broadcasting follows output tensor shape;
channel indices and zero points must use the same channel axis as the scales.

Structural DMA layouts can attach a scalar-parameter destination `placement`
to a readout. Its logical ordinal maps to the output and source int32 element
using either `{shape,strides,offset}` or `(count,block,offset,stride)`. For the
shape form, `target = offset + sum(coordinate[j] * strides[j])`, with row-major
coordinates of the logical ordinal. For the block form,
`target = offset + (ordinal // block) * stride + ordinal % block`. These
readouts use the same requant-A certificate; `output_indices` describes the
logical ordinal domain and `output_placement` records this address map. Uniform
scalar parameters retain a single loop and certificate without allocating
tensor-sized parameter or index arrays. Compact structural command loops keep
their serial, fenced exact readout boundary; the overlap planner records this
reason rather than expanding them into individual commands or reusing their
buffers under an unproved placement map.

Gemmini records the semantic record plus `output`, `elements`, `output_indices` (`{start,stop,step}` or `{indices:[...]}`),
`normal_form_width` (signed formation bit width), `normal_form` (ordered tile/decimal-weight pairs) and `accumulator_bounds` under
`kernels[].meta.host_requantization_proofs`. These binding annotations do not
change the semantic record or its hashes. Per-channel classes with identical
parameters may share one proof and emitted body; independent constants never
inherit another channel's proof.


## Gemmini overlap contract

`kernels[].meta.readout_pipeline` records `status=enabled`, `buffers=2`, the
private int32 `source`, `slot_elements`, `original_elements`, and ordered `tiles`.
Each tile has `tile`, `slot` (alternating 0/1), logical min/max, element count,
`ready_after` and `host_after` instruction boundaries (counts, not ordinals).
The static order is:

```
issue tile 0 and full-i32 MVOUT 0; fence
issue tile 1 and full-i32 MVOUT 1
host requant tile 0
fence
issue tile 2 into slot 0
host requant tile 1
fence
...
host requant final tile after its fence
```

The first fence establishes readiness of slot 0. Future commands only write the
other slot; slot reuse follows the previous CPU consumer. The final epilogue is
fenced and drained. RoCC issue asm and fences retain their memory clobbers, so
LLVM cannot move CPU reads before the ready fence or after reuse. Original local
scratchpad/accumulator allocation and internal ordering fences remain intact.
Static hazard lint runs on the transformed stream as well.

Eligibility derives from byte addresses, buffer ownership and dependencies:
one quantized output; a last-produced private identity-layout int32 component;
all other rational components ready before the first tile; no device readers
of the reused buffer; full-width stores partition the exact output; no later
device side effects; and two slots fit within the original scratch extent.
Other components may include immutable input bias or already completed int32
sums, with their existing broadcast/permutation maps. Multi-component cases that
fail these conditions retain serial host execution and record the reason.

Partial `Readout` objects bind global output `indices` to `source_offsets`
(element offsets applied after any layout map). `parameters_compact=true` means
term factors, offset, zero point and quant bounds are scalar or indexed by the
partial index list; otherwise they broadcast to the logical output shape.
`axis` selects the channel dimension for one-dimensional qparam arrays, default
last dimension. Compact parameter slices prevent duplicating full tensor
metadata once per pipeline tile. These fields are execution/binding metadata;
the `requant-A-v1` semantic certificate is unchanged.

## Qualification, 2026-10-19

The incremental implementation is based on `2565fab`. It replaces the previous
reciprocal/remainder path; the production epilogue has only the affine band form
or exact threshold lookup. The solver prefers the smallest certified shift,
including `M=1,B=0,S=0` for unit ratios. Constant choice may differ from the paper
prototype's search order; the exact certificate and validation format above is
common to both consumers.

Spike qualification:

- All 193 frozen Phase 0 cases rebuilt and passed their original grading plus
  exact byte comparison with the frozen outputs. All 38 support ops retained
  the device lane; their new ELFs passed the independent design-rule audit.
- An additional 21 kernel cases passed 65 exact checks; representative interface
  cases passed 36 exact checks. The per-channel multi-term/frozen-bias overlap
  stress passed, including tails, with 81 full-int32 stores and zero narrowed
  stores in its independent final-ELF audit.
- Core-aten: 309 tests plus 103 subtests passed (11 skipped). Pinned real captures:
  127 passed (five skipped). Final proof/native/Spike overlap tests and final
  certificate-type tests each passed 13 tests. Subsequent interface/scalar
  regressions passed 50 tests (one skipped).

The full ResNet-50 frozen capture was rebuilt before and after against the same
Merlin `9acfa92d` and model2MLIR `e05e614` snapshot, unchanged golden and inputs,
and the existing paper driver movement cap of 65,536. Both builds retain 418
source-bound device bindings and 61 kernels. Both full-model outputs are
bit-identical to the immutable golden output (top-1 850).

| Measurement | Before (`2565fab`) | requant-A + overlap |
|---|---:|---:|
| Host epilogue Spike instructions | 197,126,718 | 128,376,115 |
| Eligible kernels using two readout slots | 0 | 34 |
| Private readout bytes, sum over those unique kernels | 4,512,000 | 139,264 |
| Catalog metadata bytes | 277,994,410 | 28,283,909 |

The host-epilogue reduction is **68,750,603 instructions (34.876%)**. Instrumentation
uses machine-mode `minstret` around emitted host epilogues, accumulates counters
per kernel, and sums each kernel's final counter. One start-CSR instruction is
subtracted per window. Both builds use the same instrumentation; pure constant
setup can move outside the windows. These are host instruction measurements,
not whole-model cycles, RTL cycles, FireSim cycles or a hardware overlap speedup.
No Verilator was run.

The new ResNet build records 2,206 tile/parameter certificates, all independently
revalidated after compilation. Every certificate selected the affine form;
the adversarial tests separately qualify exact table fallback. The final model
ELFs pass the target instruction audit, and the emitted new kernel object has
no floating-point instructions. Slots are aligned to one declared Gemmini
int32 row (`DIM` elements). Serial dependency/layout refusals remain explicit.
Retained-source host readouts owned by Merlin remain at their existing ABI
boundary until that compiler lane supplies its own readout plan.

The measurements, per-kernel instruction counters, proof/audit counts, exact
source/artifact hashes, test receipts and negative evidence are preserved in
[the qualification record](perf_records/requant_A_20261019.json). The experiment
scripts are `tools/qualify_requant_A_phase0.py` and
`tools/profile_requant_A_model.py`; outputs and copied frozen sources are under
`out/taskE2/`. Historical Task E receipts are unchanged.
