"""Copied compiler's finite staged effect hook, under original final checks."""

import json
import sys
from pathlib import Path

from gemmini_staged_dense_source import optimize_statements
from merlin.compile.scheduling.library_sources import shared_library_sources
from merlin.compile.scheduling.staged_dense import StagedDenseContract


def rewrite(statements, *, declarations, lhs, rhs, output, shape):
    from staged_dense_policy import choose_staged_dense

    root = Path(__file__).resolve().parent
    if (
        json.loads((root / "shared-library-sources.json").read_bytes())
        != shared_library_sources()
    ):
        raise ValueError("staged compiler loaded a different reviewed shared library")
    publication = json.loads((root / "staged-dense-public.json").read_bytes())
    if publication.get("schema") != "gemmini.staged_dense_public.v1":
        raise ValueError("staged compiler lacks its original exact public contract")
    if (
        publication.get("library_contract_sha256")
        != json.loads((root / "selected-shared-library.json").read_bytes())[
            "contract_sha256"
        ]
    ):
        raise ValueError("staged public contract selected a different reviewed SDK")
    contract = StagedDenseContract(**publication["contract"])
    enabled = choose_staged_dense(tuple(shape), contract)
    if type(enabled) is not bool:
        raise ValueError("staged policy must return an actual Boolean decision")
    try:
        result, report = optimize_statements(
            statements,
            declarations=declarations,
            lhs=lhs,
            rhs=rhs,
            output=output,
            shape=tuple(shape),
            contract=contract,
            enabled=enabled,
        )
    except ValueError as error:
        # Unsupported members retain their original actual program, with the
        # refusal visible in the ordinary compiler feedback rather than dropped.
        print(
            "STAGED_DENSE "
            + json.dumps({"status": "original_program_retained", "reason": str(error)}),
            file=sys.stderr,
        )
        return statements
    print("STAGED_DENSE " + json.dumps(report, sort_keys=True), file=sys.stderr)
    return result
