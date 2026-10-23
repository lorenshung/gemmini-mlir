These drivers use actual model2MLIR conversion output. Use the revision in
`merlin-support/quant_schema.lock.json`, with its compiler dependencies installed.
The integration test rejects a different or modified model2MLIR source revision.

Set `PYTHONPATH` to include this repository, `merlin-support`, `tests/core_aten`,
Merlin's `src`, and the pinned model2MLIR checkout. The device runner also needs
`GEMMINI_LLVM_BIN`, `GEMMINI_RISCV_TOOLS`, `GEMMINI_ROCC_TESTS` and
`GEMMINI_TEST_WORK`.
The whole-program driver additionally uses Merlin's compiler environment; see
its command help and `tests/core_aten/device/runner.py` for tool selection.

From the repository root, using the model2MLIR Python environment:

```sh
python tests/core_aten/e2e/capture.py out/canonical --set rtl
python tests/core_aten/e2e/run_e2e.py out/canonical --batch canonical --json out/canonical-results.json
python tests/core_aten/e2e/capture_schemes.py out/schemes
python tests/core_aten/e2e/run_e2e.py out/schemes --batch schemes --json out/scheme-results.json
python tests/core_aten/e2e/run_offload.py --help
```

The `rtl` capture set covers every support operation using small shapes with
tails. It is an experiment selection only; no production matcher or schedule
uses these labels. The scheme set captures signed asymmetric narrow-range and
unsigned narrow-range Q/DQ graphs, including convolution gradients. Negation
exercises all 256 storage codes with rounding ties.

Every device comparison has zero tolerance against an independent interpretation
of the source graph. Eager float divergence is informational. Final ELFs receive
an instruction audit. A successful process exit alone does not establish full
coverage: inspect `cases` for refusals and both simulator result records.

Archived fixtures predate the canonical quant schema. `fixture_source.py` removes
only obsolete provenance from those test bodies; their original files and eager
outputs remain unchanged. Canonical model2MLIR captures pass through unchanged.
Production rejects obsolete or unknown metadata. Qualification receipts record
both original and normalized hashes for archived sources.
