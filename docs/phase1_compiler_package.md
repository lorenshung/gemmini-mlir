# Standalone Phase 1 compiler package

`tools/export_phase1_backend.py DESTINATION` exports the complete compiler at
`mlir_oot.compiler_cli` and the native catalog at `mlir_oot.core_aten.catalog`.
It follows every static local Python import, including imports inside functions,
and copies each complete source file without rewriting. It also copies the
canonical quant schema and pin. A source manifest records every copied hash.
An existing destination refuses, so stale host files cannot leak into a package.

The package declares the four ordinary compiler entrypoints (parse, interface
lowering, command-buffer emission, target-artifact emission) under the existing
experiment ABI with `integrity_exempt: false`. It is independent of Merlin.
Its input is public interface/linalg MLIR, never expected outputs or samples.
No operation names, source IDs or model data select its source closure.

The development checkout is not this submission. Tests, simulation/oracle
adapters, host-side provider registration and optional golden export orchestration
remain host integration code, outside the submitted compiler. They are not copied
and are not treated as integrity-exempt compiler modules. The root `gemmini-opt`
retains its existing optional export commands. Its ordinary Pipeline and the
submitted Pipeline are the same class, factored into `compiler_cli.py`.
The exported native catalog contains the same target matching/recipes/readout
implementation as the host provider. Other legacy/export APIs are not declared
by this standalone package and do not gain permission to import the harness.

Two required generic helpers are pinned locally because submitted code may not
import compiler implementation internals: pure tensor DCE (two functions from
`src/merlin/llvmlower/entry_weight_projection.py`) and loop annotations (the full
`llvm_loop_metadata.py`). Their comments name Merlin commit
`5eaa5aed838d86e4a1a17d56fa957c807e1cb08b`; AST equality tests verify the exact
extractions. The existing host orchestration ownership is unchanged; this is a
frozen, minimal standalone distribution boundary, not another host optimizer.

For the frozen run13 grader, use its unmodified `load_package(DESTINATION)` and
`integrity_scan(package)` over **all** exported sources, with no supplemental
allowlist, compiler-library permission or exemption. Pass the exported directory
as `--reference-backend` to readiness/oracle timing. Scanning the development
checkout instead includes tests and host adapters that are intentionally not
submission code. The integrity scan itself is not changed or vendored here.
