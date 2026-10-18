# Exact Core ATen backend readiness, 2026-10-18

Base: `4e84356172646921ab6eaf031063681cc780ed65`. This qualification uses exact
rational arithmetic on fp32-canonical scales, zero tolerance, full int32 device
readout and a single exact host quantization. No Verilator run was launched.

## Changes and root causes

- **G1:** The seeded producer matcher rejected every rank-3 contraction before
  reaching the already-batched matmul schedule. It now binds independent batches,
  including singleton batches, full per-batch int32 seeds and channel seeds. Full
  seeds have batch-specific byte offsets; reused local tiles have a batch barrier.
  The source readout remains on the host. Proofs retain `readout=full_i32`,
  `requantization=retained_source_exact_integer`, the source rational scale pair,
  exact arithmetic, accumulator bounds and exact source ordinals. Merlin's
  `provider_coverage` accepts the supplied bmm capture. A standalone call with
  `numeric_policy=None` uses the explicitly selected
  `MERLIN_CORE_ATEN_SOFTWARE_SPEC`; without either declaration it returns a
  reasoned decline instead of inferring internal arithmetic.
- **G2:** Zero-point row/column reductions reused scratchpad rows without a
  barrier. In particular a short K left unused rows of a loaded ones vector
  pending when a later reduction overwrote them. Each independent correction
  reduction now begins after a fence. A remaining lint hazard in any catalog
  candidate becomes an `unsafe device schedule: ...` decline with no kernel or
  binding published; direct lint still raises for invalid streams.
- **G3:** Nonzero fill loaded ones with the integer code as an fp32 MVIN scale.
  It now loads a literal int8 code row at unit scale with stride zero. Integer
  coefficient identities likewise use literal coefficient matrices. Production
  kernel finalization refuses non-unit load/store scales and narrowed accumulator
  stores. There are no remaining non-unit scales in qualified production streams,
  so no fp32 product-proof exceptions are needed.
- **G4:** Existing exact rational components already handle non-integral scalar
  constants, pooling divisors, backward windows, folds and pads without rounding
  them into integer seeds. Layout permutations inside convolution schedules had
  used a transposer followed by a narrowed accumulator store. They now use strided
  byte DMA, including columns when the innermost dimension moves. This can require
  more DMA commands; no performance improvement is claimed. The tests cover real
  captures of all requested readiness operations and the f32/f64 code boundaries.
- **G5:** General scalar `tensor.extract` and `tensor.from_elements` interpretation
  now supports full-reduction mean, including keepdim assembly. Rank-zero symbolic
  coefficients and instruction-model output buffers preserve scalar shape.

## Qualification

All artifacts below are retained under `out/codex-dev/` and are intentionally
excluded from Git.

| Check | Result | Receipt |
| --- | --- | --- |
| Core ATen suite, provider execution tests, integer contraction tests, gca venv | 184 passed, 1 RTL-only skip; 103 subtests passed | `pytest-gca-final.log` |
| Full real-model2mlir integration suite, m2m-dev venv, immutable a38c319 checkout | 32 passed | `pytest-real-pinned.log` |
| New bmm/readiness tests against concurrently edited live model2mlir | 22 passed, 10 deselected | `pytest-real-live.log` |
| Standalone captured support set, instruction model + Spike | 41 captures, 43 exact output checks passed | `e2e-support-unit.json` |
| Whole-graph offload rewrite, shim, standard MLIR lowering + Spike | 17 readiness cases passed | `offload-readiness.json` |
| Scalar mean and mean over all dimensions through whole-graph offload | 2 passed | `offload-means.json` |
| Adversarial unsigned bmm and signed/unsigned keepdim means on Spike | 9 exact checks passed | `e2e-adversarial.json` |
| Existing independent bmm sweep | 96 captures, 288 variants, zero mismatches, zero non-unit scales | `bmm-sweep.json` |
| Existing full-reduction keepdim mean sweep | 12 quantization cases, 36 variants, zero mismatches | `mean-sweep.json` |

The independent final-ELF instruction checker reports `OK_full_i32` for the
combined support ELF (3177 retired Gemmini commands, zero narrowed stores), the
adversarial ELF, and Merlin's bmm, full, full_like and f32-to-f64 `_to_copy` ELFs.
These are instruction counts, not cycle measurements. For the standalone harness
the checker needs `--isa rv64gc_zicntr_zihpm`, since its harness reads cycle counters.
The checker's default ISA omits that extension and traps before device execution;
a `no_device` result from that invocation is not qualification.

The gca suite used the existing test-only structlog stub. The pinned frontend
checkout is inside `out/codex-dev/model2mlir-pinned`, at the schema-lock revision
`a38c319d2f036257153e17d7790060e176b21279`. The original live frontend tests first
passed 10/10; once concurrent frontend edits began, their intentional clean-source
guard rejected those bytes. The immutable run and separate live readiness run
keep these two kinds of evidence distinct.

## Integrated lane evidence and outstanding external conditions

`lane/`, `lane38/` and `lane-retry/` contain the real Merlin spec_int8 receipts.
Across these attempts 34 of the 38 operations were observed passing on device.
This is not a 38/38 integrated success claim. Standalone backend qualification
covers all operations, but the live Merlin/model2mlir sources changed during the
run. Initial addmm/full captures failed with:

```
M2MUnavailable: capture implementation changed during cache transaction; retry capture
```

Both have successful device receipts in other attempts. A retry of
`_adaptive_avg_pool2d_backward` also passed on device after the concurrent Merlin
changes. The initial live lane kept these remaining operations on host:

- `avg_pool2d_backward`: `the node's quantization scales/ranges require literal symmetric signed-int8 per-tensor qparams`
  (the shape-only operand had no quantization scale).
- `col2im`: `the window's element count varies or is not static`.
- `convolution_backward`: `captured floating readout lacks a full-domain exact-arithmetic proof`.

Later retries, including `sub.Tensor`, were blocked before this backend by:

```
ValueError: .../merlin/examples/gemmini/target/software-spec.yaml: operation 'host_dequantize_per_tensor' ranks must be positive integers
```

These external conditions require another integrated lane run after the parallel
Merlin/model2mlir work settles. No other repository was edited here. Unsupported
seed broadcasts and unbounded runtime int32 seeds still decline respectively as
`non-channel seed broadcast` and `runtime i32 seed has no nonoverflow range proof`;
the latter is necessary to preserve exact intermediate int32 arithmetic.
