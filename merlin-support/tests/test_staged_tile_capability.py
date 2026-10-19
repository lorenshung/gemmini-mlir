"""Host selection refusals; these fixtures establish no target capability."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_staged_tile_capability")


def test_json_or_missing_original_owners_cannot_construct_tile_authority(
    module, tmp_path
):
    with pytest.raises(ValueError, match="original staged"):
        module.StagedTilePrototype({"status": "qualified"})
    with pytest.raises(ValueError, match="original owner"):
        module.QualifiedStagedTileOwner({"status": "qualified"}, ())
    with pytest.raises(ValueError, match="disabled"):
        module.restore_registry({}, work_root=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"enabled": True, "min_m": 0, "min_n": 0, "rewrite_order": ["rhs_residency"]},
        {
            "enabled": True,
            "min_m": 0,
            "min_n": 0,
            "rewrite_order": ["rhs_residency", "rhs_residency"],
        },
        {
            "enabled": True,
            "min_m": True,
            "min_n": 0,
            "rewrite_order": ["rhs_residency", "n_before_m"],
        },
        {
            "enabled": "yes",
            "min_m": 0,
            "min_n": 0,
            "rewrite_order": ["rhs_residency", "n_before_m"],
        },
    ],
)
def test_invalid_host_policy_refuses_before_copying_or_editing_compiler(
    module, tmp_path, policy
):
    owner = object.__new__(module.StagedTilePrototype)
    owner.revalidate = lambda: None
    with pytest.raises(ValueError, match="explicit Boolean"):
        owner.prepare_compiler(
            tmp_path / "original", tmp_path / "candidate", policy=policy
        )
    assert not (tmp_path / "candidate").exists()
