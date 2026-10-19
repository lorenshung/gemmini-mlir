"""Original owned-copy preparation; row capacity alone never selects this hook."""

import json
import shutil
from pathlib import Path

from gemmini_staged_rhs_window import ClosedScratchEpochOwner

from merlin.targetgen.compiler_pass_selection import _pin, _tree, copy_compiler_source


def prepare_compiler(source, destination, *, epoch, generic_source, enabled):
    if type(epoch) is not ClosedScratchEpochOwner or type(enabled) is not bool:
        raise ValueError("RHS preparation needs an original live complete-program epoch and explicit policy")
    epoch.revalidate()
    source, destination, generic_source = map(Path, (source, destination, generic_source))
    parent = _tree(source)
    generic_pin = _pin(generic_source)
    ledgers = {}
    for name in ("selected-shared-library.json", "shared-library-sources.json", "staged-tile-public.json"):
        path = source / name
        _pin(path)
        if path.stat().st_size > 1 << 20:
            raise ValueError("RHS preparation source ledger exceeds its original bound")
        ledgers[name] = path.read_bytes()
    tile = json.loads(ledgers["staged-tile-public.json"])
    if (
        tile.get("schema") != "gemmini.staged_tile_public.v1"
        or tile.get("library_contract_sha256") != json.loads(ledgers["selected-shared-library.json"])["contract_sha256"]
    ):
        raise ValueError("RHS preparation lacks original tile effects")
    for name in (
        "rhs_residency_policy.py",
        "staged-rhs-window-public.json",
        "passes/staged_rhs_residency.py",
        "gemmini_staged_rhs_hook.py",
        "gemmini_staged_rhs_placement.py",
    ):
        if (source / name).exists() or (source / name).is_symlink():
            raise ValueError("RHS preparation requires fresh compiler components")
    direct = (source / "direct_codegen.py").read_text()
    old = "from gemmini_staged_tile_hook import rewrite as _staged_statements"
    if direct.count(old) != 1:
        raise ValueError("RHS preparation lost original unique direct-emitter boundary")
    mixed = (source / "mixed_codegen.py").read_text()
    if mixed.count("def emit_mixed(text):") != 1:
        raise ValueError("RHS preparation lost original source-module entry point")
    mixed += """

# This is a candidate compilation predicate. Original current-program source,
# native/packet/ownership checks remain mandatory outside this compiler.
_rhs_original_emit_mixed = emit_mixed
def emit_mixed(text):
    from gemmini_staged_rhs_hook import scoped_program_epoch
    _, fn = read_module(text)
    contractions = sum(bool(_contraction(op)) for op in fn.body.block.ops if op.name != 'func.return')
    with scoped_program_epoch(contractions == 1):
        return _rhs_original_emit_mixed(text)
"""
    copy_compiler_source(source, destination)
    (destination / "direct_codegen.py").write_text(
        direct.replace(old, "from gemmini_staged_rhs_hook import rewrite as _staged_statements")
    )
    (destination / "mixed_codegen.py").write_text(mixed)
    for name in ("gemmini_staged_rhs_hook.py", "gemmini_staged_rhs_placement.py"):
        shutil.copyfile(Path(__file__).with_name(name), destination / name)
    (destination / "passes").mkdir(exist_ok=True)
    if not (destination / "passes/__init__.py").exists():
        (destination / "passes/__init__.py").write_text("")
    shutil.copyfile(generic_source, destination / "passes/staged_rhs_residency.py")
    basis = epoch.record()
    (destination / "staged-rhs-window-public.json").write_text(
        json.dumps(
            dict(
                schema="gemmini.closed_rhs_program_public.v1",
                original_contract=tile["contract"],
                library_contract_sha256=tile["library_contract_sha256"],
                maximum_private_physical_rows=basis["footprint"]["maximum_private_physical_rows"],
                basis_original_program_epoch=epoch.identity,
                condition="one complete current source/execution epoch; original independent final proof required",
                hardware_execution_qualified=False,
                timing_authority="NONE",
            ),
            sort_keys=True,
        )
    )
    (destination / "rhs_residency_policy.py").write_text(
        f"ENABLED={enabled!r}\n\n"
        "def choose_rhs_rewrites(shape, contract):\n"
        "    from merlin.compile.scheduling.staged_dense import (StagedDenseRewrite, "
        "order_output_tiles_n_before_m, eliminate_redundant_rhs_loads)\n"
        "    from passes.staged_rhs_residency import place_rhs_stripe\n"
        "    m,k,n=shape\n"
        "    required=contract.dim+((k+contract.dim-1)//contract.dim)*contract.dim\n"
        "    if not (ENABLED and m>contract.dim and k>contract.dim and required<=contract.operand_rows): return ()\n"
        '    return (StagedDenseRewrite("n_before_m",order_output_tiles_n_before_m),\n'
        '            StagedDenseRewrite("full_k_rhs",place_rhs_stripe),\n'
        '            StagedDenseRewrite("rhs_omissions",eliminate_redundant_rhs_loads))\n'
    )
    epoch.revalidate()
    if (
        _tree(source) != parent
        or _pin(generic_source) != generic_pin
        or any((destination / name).read_bytes() != data for name, data in ledgers.items())
    ):
        raise ValueError("RHS preparation changed its immutable parent or exact selected SDK")
    return dict(
        schema="gemmini.closed_rhs_program_preparation.v1",
        source=dict(path=str(source), sha256=parent),
        prepared=dict(path=str(destination), sha256=_tree(destination)),
        enabled=enabled,
        generic_candidate_owned_pass=generic_pin,
        original_epoch=epoch.identity,
        hardware_execution_qualified=False,
        timing_authority="NONE",
    )
