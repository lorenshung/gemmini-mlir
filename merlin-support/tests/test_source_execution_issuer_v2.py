"""Issuer delegation dispatch controls; neutral owner is no target authority."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def selected(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    module = importlib.import_module("gemmini_source_execution_issuer_v2")
    calls = []

    class NeutralOriginal:
        identity = "neutral-original-current-artifact-verifier"

        def __init__(self, qualification):
            self.qualification = qualification

        def revalidate(self):
            pass

        def source_rule(self):
            return {"neutral": "original-rule"}

        def create_source_execution_owner(self, **kwargs):
            calls.append(kwargs)
            return original_owner

    original_owner = object()
    monkeypatch.setattr(module, "SourceExecutionIssuer", NeutralOriginal)
    issuer = module.SourceExecutionIssuerV2("neutral-live-qualification")
    return module, issuer, calls, original_owner


def test_v2_still_dispatches_current_artifacts_to_original_source_issuer(
    selected, tmp_path
):
    _module, issuer, calls, original = selected
    assert issuer.source_rule() == {"neutral": "original-rule"}
    args = {
        "native_result": tmp_path / "current-result",
        "command_buffer": tmp_path / "current-cb",
        "lowered": tmp_path / "current-lowered",
        "work_root": tmp_path / "fresh-current-trace",
    }
    assert issuer.create_source_execution_owner(**args, remaining_s=10) is original
    assert {key: calls[0][key] for key in args} == args
    assert 0 < calls[0]["remaining_s"] < 10
    assert issuer.identity.startswith("gemmini.original_source_execution_issuer.v2:")


@pytest.mark.parametrize("damage", ["source", "original_verifier"])
def test_changed_delegation_source_or_original_verifier_refuses_before_current_trace(
    selected, tmp_path, damage
):
    _module, issuer, calls, _original = selected
    if damage == "source":
        issuer._source = {**issuer._source, "sha256": "0" * 64}
    else:
        issuer._original = object()
    with pytest.raises(ValueError):
        issuer.create_source_execution_owner(
            native_result=tmp_path / "n",
            command_buffer=tmp_path / "c",
            lowered=tmp_path / "l",
            work_root=tmp_path / "must-not-execute",
            remaining_s=10,
        )
    assert not calls


@pytest.mark.parametrize(
    "remaining", [False, True, 0, -1, float("nan"), float("inf"), "10"]
)
def test_v2_does_not_extend_or_coerce_invalid_native_budget(
    selected, tmp_path, remaining
):
    _module, issuer, calls, _original = selected
    with pytest.raises(ValueError, match="positive remaining"):
        issuer.create_source_execution_owner(
            native_result=tmp_path / "n",
            command_buffer=tmp_path / "c",
            lowered=tmp_path / "l",
            work_root=tmp_path / "must-not-execute",
            remaining_s=remaining,
        )
    assert not calls
