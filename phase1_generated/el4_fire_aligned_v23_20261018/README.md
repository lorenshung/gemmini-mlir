# EL4 FireSim-aligned Phase 1 V23

This directory preserves the current Phase 1 compiler output from Merlin run
`el4-frozen-v23-qualification-20261018-v5`. The original 21 files under `package/`
have source-tree SHA256
`b847c973da82f408a32c7622d0c49fb6902f0a58d0f716f131d8501e34175338`.
See [package-files.json](package-files.json) for each file's digest.

**Qualification: provisional and incomplete.** Read
[QUALIFICATION.md](QUALIFICATION.md) and [checkpoint.json](checkpoint.json)
before using these observations as a baseline. The preserved author report in
`package/REPORT.md` predates the later execution diagnostics; its older capsule
and model claims are historical, not a current whole-model certificate.

## Run the compiler

Use Python 3.12 with the dependencies in `requirements.txt` and a compatible
LLVM installation containing RISC-V-capable `clang` and `mlir-translate`.
The publication checks use xDSL 0.73.0. From this directory:

```sh
python3 -m pip install -r requirements.txt
export MERLIN_MLIR_INSTALL=/path/to/llvm/install
export TMPDIR=/path/to/writable/task-temporary-directory
python3 package/compiler.py --help
python3 package/compiler.py --verify-diagnostics examples/scalar_readout.mlir
python3 package/compiler.py --convert-iface-to-gemmini \
  --emit-command-buffer=commands.json examples/scalar_readout.mlir
python3 package/compiler.py --convert-iface-to-gemmini \
  --emit-target-artifact examples/scalar_readout.mlir > kernel.llvm.mlir
```

The outer [manifest.yaml](manifest.yaml) selects this preserved package through
the same compiler ABI and uses `python3` instead of the original machine's
absolute interpreter path. The original manifest inside `package/` is unchanged.
Set `MERLIN_MLIR_INSTALL` explicitly when moving to another machine.

`examples/block_readout.mlir` records the current expected refusal for different
gains across output-channel blocks. It is not a supported example. Both examples
come from the generated public accelerator corpus; no private model inputs,
goldens, model weights, simulator binaries or frozen builds are included.

The snapshot preserves generated mixed-program helpers for reproducibility.
Promotion of reusable helpers into production remains subject to this
repository's ownership rules: generic host mechanisms belong in Merlin and
target-specific implementation belongs here.
