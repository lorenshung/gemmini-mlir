"""Copied compiler hook for one selected closed private contraction epoch."""

import contextlib
import contextvars
import json
import sys
from pathlib import Path

from gemmini_staged_dense_source import project_statements
from gemmini_staged_rhs_placement import assigned_window, optimize_statements

from merlin.compile.scheduling.library_sources import shared_library_sources
from merlin.compile.scheduling.staged_dense import StagedDenseContract, StagedDenseProblem

_EPOCH = contextvars.ContextVar("selected_closed_rhs_program_epoch", default=False)


@contextlib.contextmanager
def scoped_program_epoch(enabled):
    if type(enabled) is not bool:
        raise ValueError("private program epoch needs an explicit source predicate")
    token = _EPOCH.set(enabled)
    try:
        yield
    finally:
        _EPOCH.reset(token)


def rewrite(statements, *, declarations, lhs, rhs, output, shape):
    if not _EPOCH.get():
        return statements
    from rhs_residency_policy import choose_rhs_rewrites

    root = Path(__file__).resolve().parent
    public = json.loads((root / "staged-rhs-window-public.json").read_bytes())
    if (
        public.get("schema") != "gemmini.closed_rhs_program_public.v1"
        or public.get("library_contract_sha256")
        != json.loads((root / "selected-shared-library.json").read_bytes())["contract_sha256"]
        or json.loads((root / "shared-library-sources.json").read_bytes()) != shared_library_sources()
    ):
        raise ValueError("RHS placement loaded a different original resource/library binding")
    contract = StagedDenseContract(**public["original_contract"])
    problem = StagedDenseProblem(*shape)
    try:
        original = project_statements(
            statements, declarations=declarations, lhs=lhs, rhs=rhs, output=output, problem=problem, contract=contract
        )
        assigned, proof = assigned_window(
            original, problem=problem, contract=contract, original_owned_rows=public["maximum_private_physical_rows"]
        )
        rewrites = choose_rhs_rewrites(tuple(shape), assigned)
        result, report = optimize_statements(
            statements,
            declarations=declarations,
            lhs=lhs,
            rhs=rhs,
            output=output,
            shape=tuple(shape),
            contract=assigned,
            rewrites=rewrites,
        )
    except ValueError as error:
        print(
            "STAGED_RHS_WINDOW " + json.dumps(dict(status="original_program_retained", reason=str(error))),
            file=sys.stderr,
        )
        return statements
    print("STAGED_RHS_WINDOW " + json.dumps(dict(**report, window=proof), sort_keys=True), file=sys.stderr)
    return result
