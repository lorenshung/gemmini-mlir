"""Candidate-owned finite proposal; original live checking supplies authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.staged_dense import StagedDenseContract


def emit_if_selected(source, command_buffer):
    # This module is copied into the compiler. It cannot import original owners,
    # references, capture implementations or native/trace checkers.
    from finite_staged_policy import ENABLED, REWRITE_POLICY

    if type(ENABLED) is not bool:
        raise ValueError("finite source selection needs an explicit Boolean")
    if not ENABLED:
        return None
    from finite_staged_emitter import emit_v26_staged_dense

    root, source = Path(__file__).resolve().parent, Path(source)
    declaration = json.loads((root / "finite-staged-public.json").read_bytes())
    if declaration.get("schema") != "gemmini.v26_finite_public.v1":
        raise ValueError("unknown finite source declaration")
    # These are untrusted proposal fields, independently checked by the public
    # typed source reader and by the original current-source owner after emit.
    arguments = command_buffer.get("compiler_pointer_abi", {}).get("arguments", [])
    if len(arguments) != 3:
        raise ValueError("finite source proposal needs its exact three pointers")
    k = command_buffer["tensors"][arguments[0]]["shape"][1]
    proposal = {
        "interface_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "operations": [
            {
                "ordinal": 2,
                "kind": "multiply_accumulate",
                "initial_interval": [0, 0],
                "reduction_extent": k,
                "result_bits": 32,
            }
        ],
    }
    llvm, _diagnostic = emit_v26_staged_dense(
        package=root,
        source=source,
        proof=proposal,
        command_buffer=command_buffer,
        contract=StagedDenseContract(**declaration["contract"]),
        enabled=True,
        rewrite_policy=REWRITE_POLICY,
    )
    return llvm
