"""Owned compiler preparation only; neutral owner fixtures grant no execution."""

import json
from pathlib import Path

import pytest
from gemmini_staged_rhs_hook import rewrite
from gemmini_staged_rhs_preparation import prepare_compiler
from gemmini_staged_rhs_window import ClosedScratchEpochOwner

from merlin.compile.scheduling import staged_rhs_residency
from merlin.targetgen.compiler_pass_selection import _tree


def test_no_current_closed_program_context_cannot_rewrite_even_known_statements():
    source = ["unreviewed();"]
    assert rewrite(source, declarations=[], lhs="a", rhs="b", output="c", shape=(17, 17, 17)) is source


def test_saved_success_or_missing_live_epoch_never_prepares_compiler(tmp_path):
    with pytest.raises(ValueError, match="original live complete-program"):
        prepare_compiler(
            tmp_path / "original",
            tmp_path / "candidate",
            epoch={"status": "qualified"},
            generic_source=Path(staged_rhs_residency.__file__).resolve(),
            enabled=True,
        )
    assert not (tmp_path / "candidate").exists()


def test_both_arms_keep_sdk_and_exact_initial_generic_pass_immutable(tmp_path, monkeypatch):
    # Explicit neutral owner stub tests only owned-copy and namespace plumbing.
    # Actual constructors and native/effect qualification are separate tests.
    epoch = object.__new__(ClosedScratchEpochOwner)
    epoch._identity = "neutral-only"
    monkeypatch.setattr(ClosedScratchEpochOwner, "revalidate", lambda _: None)
    monkeypatch.setattr(
        ClosedScratchEpochOwner, "record", lambda _: dict(footprint=dict(maximum_private_physical_rows=64))
    )
    source = tmp_path / "original"
    source.mkdir()
    ledgers = {
        "selected-shared-library.json": {"contract_sha256": "same-frozen-sdk"},
        "shared-library-sources.json": {"fixture": "exact-original-ledger"},
        "staged-tile-public.json": dict(
            schema="gemmini.staged_tile_public.v1", contract={}, library_contract_sha256="same-frozen-sdk"
        ),
    }
    for name, row in ledgers.items():
        (source / name).write_text(json.dumps(row))
    (source / "direct_codegen.py").write_text("from gemmini_staged_tile_hook import rewrite as _staged_statements\n")
    (source / "mixed_codegen.py").write_text("def emit_mixed(text):\n    return text\n")
    (source / "passes").mkdir()
    (source / "passes/__init__.py").write_text("")
    original = _tree(source)
    generic = Path(staged_rhs_residency.__file__).resolve()
    for enabled in (False, True):
        destination = tmp_path / str(enabled)
        record = prepare_compiler(source, destination, epoch=epoch, generic_source=generic, enabled=enabled)
        assert _tree(source) == original
        assert (destination / "passes/staged_rhs_residency.py").read_bytes() == generic.read_bytes()
        for name in ledgers:
            assert (destination / name).read_bytes() == (source / name).read_bytes()
        assert record["hardware_execution_qualified"] is False
        assert record["timing_authority"] == "NONE"
        compile((destination / "rhs_residency_policy.py").read_text(), "<owned-policy>", "exec")
        compile((destination / "mixed_codegen.py").read_text(), "<owned-mixed>", "exec")
    assert (tmp_path / "False/passes/staged_rhs_residency.py").read_bytes() == (
        tmp_path / "True/passes/staged_rhs_residency.py"
    ).read_bytes()
    with pytest.raises(ValueError, match="fresh compiler components"):
        prepare_compiler(tmp_path / "True", tmp_path / "again", epoch=epoch, generic_source=generic, enabled=True)
