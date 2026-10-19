"""Copied compiler's checked whole-output-tile rewrite entry point."""

import json
import sys
from pathlib import Path

from gemmini_staged_tile_source import optimize_statements
from merlin.compile.scheduling.library_sources import shared_library_sources
from merlin.compile.scheduling.staged_dense import StagedDenseContract


def rewrite(statements, *, declarations, lhs, rhs, output, shape):
    from staged_dense_policy import choose_staged_tile_rewrites

    root = Path(__file__).resolve().parent
    if (
        json.loads((root / "shared-library-sources.json").read_bytes())
        != shared_library_sources()
    ):
        raise ValueError("tile compiler loaded a different reviewed shared library")
    publication = json.loads((root / "staged-tile-public.json").read_bytes())
    if (
        publication.get("schema") != "gemmini.staged_tile_public.v1"
        or publication.get("library_contract_sha256")
        != json.loads((root / "selected-shared-library.json").read_bytes())[
            "contract_sha256"
        ]
    ):
        raise ValueError(
            "tile compiler lacks its original exact selected public contract"
        )
    contract = StagedDenseContract(**publication["contract"])
    selected = choose_staged_tile_rewrites(tuple(shape), contract)
    try:
        result, report = optimize_statements(
            statements,
            declarations=declarations,
            lhs=lhs,
            rhs=rhs,
            output=output,
            shape=tuple(shape),
            contract=contract,
            rewrites=selected,
        )
    except ValueError as error:
        print(
            "STAGED_TILE_ORDER "
            + json.dumps({"status": "original_program_retained", "reason": str(error)}),
            file=sys.stderr,
        )
        return statements
    print("STAGED_TILE_ORDER " + json.dumps(report, sort_keys=True), file=sys.stderr)
    return result
