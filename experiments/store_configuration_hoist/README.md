# Raw-i32 store configuration candidate

This independent candidate moves one invariant `CONFIG_ST` out of the output-tile
loops of V26 `matmul`. It uses the complete original 16-file package with digest
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Only `contraction.py` changes. It does not depend on retained85 or alter its RHS
residency policy. The original package remains immutable.

The raw-i32 branch always configures identical stride `N*4`, identity scale,
no activation and no pooling for each tile. The candidate emits that exact
configuration once, after the existing execute/load setup and before the tile
loops. Every load, zero initialization, preload, compute, readout, CPU layout
operation, completion marker and fence remains. Each new contraction establishes
its own store state, including when the next contraction has a different width.
The i8, per-column scale and pooling paths retain their original behavior.

This is valid only while store state persists across the intervening loads,
compute and fences, with no intervening store/normalization configuration,
unknown call or control-flow escape. The selected StoreController and receiver
sources support this narrow case; prior operations retain their original
completion boundary. This candidate adds no new ownership grant. Original
source, ELF, instruction, complete output, packet, host-write, runtime and
qualification checks must still run on the actual result. All hardware loop
and loop-configuration instructions remain forbidden.

Prepare a fresh candidate using ordinary copies and the fixed source rewrite:

```sh
python3 experiments/store_configuration_hoist/prepare.py \
  --baseline /path/to/exact/original/package \
  --destination /path/to/absent/candidate \
  --report /path/to/absent/preparation.json
```

The preparer checks all 16 names and bytes before and after copying, parses all
Python sources, and refuses changed, missing, extra or linked files. It issues
no compiler/execution authority. `contraction.patch` is the complete zero-context source diff.
The manifest, instruction encoding, numerical recipes and target selection are
unchanged. The source count proof does not claim that an arbitrary edited
compiler preserves the no-intervening-writer condition.

Run static checks under the original shared CPU/resource supervisor:

```sh
GEMMINI_V26_COMPILER_BASELINE=/path/to/exact/original/package \
  python3 -m pytest -q experiments/store_configuration_hoist/test_candidate.py
```

The tests expand generator loops with a concrete test recorder. They verify
identical configuration at every readout and identical other effects across
aligned/tail/deep-K, transpose, bias, resident RHS, and consecutive contractions.
They also check unchanged i8/scale paths and reject altered source inventories
or absent/wrong configuration. This is static generation evidence, not selected
compiler execution, an independent RTL decoder or a hardware proof.

For the existing 17×16×32 public witness, static contraction expansion predicts
four store configurations becoming one: full-kernel configuration commands
8→5, custom instructions 32→29, with four computes, four preloads, twelve loads,
four stores and all 16 full-kernel fences unchanged. The test recorder covers
13 contraction fences; the three other fences remain in unchanged builder,
backend and completion code. These are predictions to check in the next normal
queue. No compilation, native execution, simulator measurement, timing claim or
performance promotion has been performed for this candidate.
