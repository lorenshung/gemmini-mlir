# Exact integer readout qualification, 2026-10-17

This qualification supersedes the arithmetic/admission conclusions in the earlier
report without editing that historical report or its receipts. The oracle is
exact rational arithmetic from source integer codes and fp32-canonical source
Q/DQ parameters, followed by one round-half-to-even operation and the declared
clamp. Eager float divergence is informational and appears separately in the
[machine receipt](exactness/core_aten_integer_readout_20261017.json).

## Implementation and commits

- `34aac49` — full-width integer results and exact host requantization. Restores
  pooling, mean, add/sub.Tensor, mul/div/neg, concatenation, scatter, convolution,
  all convolution gradients and addmm through integer schedules and exact host
  readout. Adds the Python/C reference interface and exported schema snapshot.
- `a292f72` — strict schema/policy validation on both catalogs; exact-product
  proofs even for negative inputs subsequently clamped by ReLU; bounded integer
  zero-point correction convolutions; shared exact host readout bodies; explicit
  compiler memory ordering around device operations.
- `f61e808` — independent source-graph oracle, scheme/property tests, real pinned
  model2MLIR integration, actual canonical captures and simulator drivers.
- `007ae97` — generates the separate GSIM harness provider contract from the
  selected Merlin target contract and tests byte drift; resolves 14 provider
  discovery failures exposed by the final full run.
- The final qualification commit records these results, reproduction instructions
  and a parity test against Merlin's newly available exact host readout API.
- Concurrent commit `27e2b46` (persistent/resumable e2e results) was preserved.

Prior fixes remain: `6ce53cf` fixes findings **3, 11, 12, 13** (typed descriptor
offsets, empty-shape refusal, generated ceil/divisor coverage, installed frontend
selection); `872b3a3` fixes the target side of **9** with generated Merlin contracts
and a drift test; `872766b` preserves source descriptor ranks. The earlier
`468c431` output-equivalence admission proof is replaced by exact-product proof
and default raw int32 results. Findings **1** and the target side of **5** now use
exact integer computation and exact host readout, with real integration coverage.

Device data scales are unity and accumulator DMA is full signed int32 by default.
Every bounded partial sum is checked before scheduling. No quantization factor is
approximated into integer weights. Uint8 codes are rebased to signed bytes without
rounding; exact zero-point sums and constants are combined on the host. Spatial
padding masks are preserved in convolution corrections. Bias terms with a distinct
scale remain exact rational host terms. Independent tiles are combined before the
single quantize rounding. The LLVM host epilogue uses integer widths derived from
actual rational parameters and the full int32 domain, with eight exact threshold
comparisons for a byte result.

Nonunity hardware scaling needs proof that int32 conversion and the product are
both exact throughout the reachable interval. Final-code equivalence, saturation
and ReLU cannot excuse rounded products. This is a sufficient interval proof and
can conservatively refuse sparse domains. The default path needs no FP32 exception
and is unaffected by the known Spike FP32 scaling bug.

Both `merlin-support` and the separate GSIM harness provider are generated from
Merlin's selected contract; only plugin locations differ. Drift checks pass.

## Integration and ownership

The vendored `merlin-support/quant_schema.json` is the byte-identical schema-v1
export from model2MLIR **a38c319d2f036257153e17d7790060e176b21279**, SHA256
`af385e278adb4f16eed416114b1d3f24e0ab8ea2406e74248ca976790b402214`.
The generator and drift test compare the pinned Git object and current export;
unknown/obsolete keys and host-only policies are refused. A missing owner export
produces a loud warning and skip rather than invented evidence. Exact math uses
`scale_numerator/scale_denominator`; informational `scale` is optional.

Actual `convert(profile=int8_accelerator)` Linear(32,16)+ReLU+Linear(16,8) models,
with and without runtime float bias, each bind **two** integer contractions and
produce callable provider routing. The original exact source readout stays on the
host, including its bias and output conversion. Tests pin the imported frontend
revision and reject modified source. Both complete models passed Spike and RTL,
with bitwise output comparison including float32 storage bits.

[ABI_NOTES.md](../ABI_NOTES.md) defines the temporary target-side reference contract,
Python reference, arbitrary-positive-rational C transition tables and signed-rational
C implementation. Merlin's reusable exact host API is checked against that reference.
The requested parent ABI_NOTES.md is outside this sandbox's writable roots; only the
repository-local copy was written. No Merlin/model2MLIR source files were edited.

## Per-operation device status

All previously refused fixture cases are restored. The table lists all **41 capture
cases** (including shape variants and source decompositions) so none of the requested
support operations is hidden by a single aggregate count. Source provenance may
canonicalize div to mul, fill/full_like to full, or a zero-width pad to clone. Device
means integer work on Gemmini followed by exact CPU requantization; it does not mean
the rational quantizer executes in Gemmini's FP32 readout unit.

| Capture/support operation | Bound source operation | Placement | Spike | Verilator |
|---|---|---|---|---|
| `_adaptive_avg_pool2d` | `aten._adaptive_avg_pool2d.default` | Device | PASS | PASS |
| `_adaptive_avg_pool2d_backward` | `aten._adaptive_avg_pool2d_backward.default` | Device | PASS | PASS |
| `_adaptive_avg_pool3d` | `aten._adaptive_avg_pool3d.default` | Device | PASS | PASS |
| `_to_copy` | `aten._to_copy.default` | Device | PASS | PASS |
| `adaptive_avg_pool1d` | `aten.adaptive_avg_pool1d.default` | Device | PASS | PASS |
| `add_Scalar` | `aten.add.Scalar` | Device | PASS | PASS |
| `add_Tensor` | `aten.add.Tensor` | Device | PASS | PASS |
| `addmm` | `aten.addmm.default` | Device | PASS | PASS |
| `avg_pool1d` | `aten.avg_pool1d.default` | Device | PASS | PASS |
| `avg_pool2d` | `aten.avg_pool2d.default` | Device | PASS | PASS |
| `avg_pool2d_backward` | `aten.avg_pool2d_backward.default` | Device | PASS | PASS |
| `avg_pool3d` | `aten.avg_pool3d.default` | Device | PASS | PASS |
| `bmm` | `aten.bmm.default` | Device | PASS | PASS |
| `cat` | `aten.cat.default` | Device | PASS | PASS |
| `clone` | `aten.clone.default` | Device | PASS | PASS |
| `col2im` | `aten.col2im.default` | Device | PASS | PASS |
| `constant_pad_nd` | `aten.clone.default` | Device | PASS | PASS |
| `constant_pad_nd_p1` | `aten.constant_pad_nd.default` | Device | PASS | PASS |
| `convolution` | `aten.convolution.default` | Device | PASS | PASS |
| `convolution_backward` | `aten.convolution_backward.default` | Device | PASS | PASS |
| `convolution_patch16` | `aten.convolution.default` | Device | PASS | PASS |
| `div_Scalar` | `aten.mul.Scalar` | Device | PASS | PASS |
| `fill_Scalar` | `aten.full.default` | Device | PASS | PASS |
| `full` | `aten.full.default` | Device | PASS | PASS |
| `full_like` | `aten.full.default` | Device | PASS | PASS |
| `mean_dim` | `aten.mean.dim` | Device | PASS | PASS |
| `mm` | `aten.mm.default` | Device | PASS | PASS |
| `mul_Scalar` | `aten.mul.Scalar` | Device | PASS | PASS |
| `neg` | `aten.neg.default` | Device | PASS | PASS |
| `permute` | `aten.permute.default` | Device | PASS | PASS |
| `reflection_pad1d` | `aten.reflection_pad1d.default` | Device | PASS | PASS |
| `reflection_pad2d` | `aten.reflection_pad2d.default` | Device | PASS | PASS |
| `reflection_pad3d` | `aten.reflection_pad3d.default` | Device | PASS | PASS |
| `relu` | `aten.relu.default` | Device | PASS | PASS |
| `repeat` | `aten.repeat.default` | Device | PASS | PASS |
| `replication_pad2d` | `aten.replication_pad2d.default` | Device | PASS | PASS |
| `replication_pad3d` | `aten.replication_pad3d.default` | Device | PASS | PASS |
| `select_scatter` | `aten.select_scatter.default` | Device | PASS | PASS |
| `slice_scatter` | `aten.slice_scatter.default` | Device | PASS | PASS |
| `sub_Scalar` | `aten.sub.Scalar` | Device | PASS | PASS |
| `sub_Tensor` | `aten.sub.Tensor` | Device | PASS | PASS |

These RTL results use fresh canonical owner captures with small tail shapes,
including K=17 contractions and all three convolution gradients. Original archived
fixture shapes separately pass the instruction/source interpreter checks and 43
Spike output checks. Archived pre-schema trace attributes are removed by a test-only
adapter; operation bodies, operands, properties and eager outputs remain unchanged.
Production does not accept legacy aliases. Original and normalized source hashes,
and unchanged canonical capture hashes, are recorded per case in the machine receipt.

Additional signed asymmetric narrow-range and uint8 narrow-range captures cover
neg, mean, add.Tensor, cat, mm, convolution and convolution_backward. Neg enumerates
all 256 storage codes and rounding ties. The supplementary simulator receipt records
**14 captures and 18/18 exact outputs on both Spike and Verilator**.

## Verification and limitations

- Focused backend/provider suite: **170 passed, 103 passing subtests, 1 skipped**.
- Final proof/schema checks: **3 passed**; frontend-pinned real integration:
  **10 passed**; Merlin host-ABI parity: **4 passed**.
- Python/C and generated LLVM host arithmetic properties cover full/narrow signed
  and unsigned schemes, symmetric/asymmetric zero points, random full-int32 values,
  adversarial ties, negative coefficients and extreme fp32 scale exponents. All
  65,536 signed-byte add pairs pass exact comparisons; the corresponding Spike ELF
  and mean-tie case pass (2 output checks). No 1-LSB or percentage gate remains.
- Fresh all-operation batch: **43/43 outputs on Spike and Verilator**.
- Actual Linear models: **2/2 whole-program checks on each simulator**.
- Ceil/divisor extension: **1/1 on each simulator**. Final ELF audits contain no
  forbidden or unknown custom instruction encodings.
- Repository-wide and support-suite results, including individual failure names,
  are retained in the machine receipt; see the final test summary below.

The earlier large archived-shape RTL batch and the 65,536-pair RTL batch were
interrupted after the fresh all-operation RTL batch completed. No completed RTL
result is claimed for those two larger batches. The original-shape whole-graph
shim experiment passed addmm, avg_pool2d and convolution on RTL, then was interrupted
during convolution_backward; no five-case aggregate pass is claimed. These are
coverage limits, not device numeric mismatches. Eager-float divergences in the
receipts are not device failures.

Genuine device refusals remain for unproved int32 seeds/partial-sum overflow,
unsupported/unproved layouts, empty or dynamic geometry, hardware resource limits,
unsupported per-channel Q/DQ matching, invalid/host-only numeric policy and unproved
nonunity hardware products. Those paths remain host work. No qualified table case
is refused, approximated or admitted with a tolerance. Per-channel source integer
readouts can stay on the host; this is separate from matching arbitrary per-channel
Q/DQ graphs.

Reproduction commands and environment selection are in
[the e2e instructions](../tests/core_aten/e2e/README.md). Raw local artifacts and logs
remain under `out/exact-int32`; hashes and simulator summaries are committed in the
machine receipt. Historical model accuracy and byte-preservation gates were not
weakened.


## Repository-wide test results

The full run completed with **1,567 passed, 37 failed, 8 skipped and 131 passing
subtests** (1069.16 seconds). This was followed by targeted corrections and reruns:

- **14 GSIM provider failures fixed:** the minimal harness contract drifted from
  Merlin's selected contract. Generated synchronization and a new drift regression
  now pass **16/16 provider tests**.
- **1 dependency failure resolved in the appropriate environment:** the capture
  Python lacks `jsonschema`; inventory plus contract checks pass **4/4** using
  Merlin's Python environment.
- **22 remaining failures:** 7 native tests require unavailable `clang-23`;
  13 preservation tests reference absent historical Git objects `eefe844`,
  `d1d0e80`, `4e1630b`; 2 existing golden identity-plan tests reject changed
  outlined-source bytes. Their gates remain intact. The earlier full run had
  the same 23 non-GSIM failures before the dependency-specific rerun.

The separate support suite completed with **108 passed, 2 failed, 2 skipped**.
Both failures require the unavailable optional `aet` dependency; its recording
module also prevents collection of the recording suite, which was excluded from
that run. These are recorded as incomplete qualification, not converted to skips
or relaxed assertions. Core backend, schema, numeric, provider and simulator
qualifications described above pass.

No second full-suite aggregate is inferred from subset reruns. All original
failure names, subsequent test summaries and log hashes are in the machine receipt.
