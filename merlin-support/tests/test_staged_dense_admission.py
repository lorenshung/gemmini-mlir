"""Original-owner and missing-evidence controls; synthetic data grants no execution."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_staged_dense_admission")


def test_missing_sealed_realizations_never_create_a_capability_or_launch_admission(
    admission, tmp_path
):
    for declaration in ({}, {"staged_qualification_pairs": []}):
        with pytest.raises(ValueError, match="disabled.*sealed-source"):
            admission.restore_registry(declaration, work_root=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_saved_completion_or_success_record_cannot_construct_live_owners(admission):
    assert not hasattr(admission.StagedCompletionOwner, "read")
    assert not hasattr(admission.QualifiedStagedDenseOwner, "read")
    with pytest.raises(ValueError, match="original exact"):
        admission.StagedCompletionOwner({"status": "pass"}, source_root=Path("unused"))
    with pytest.raises(ValueError, match="original prototype"):
        admission.QualifiedStagedDenseOwner(
            {"status": "pass"}, ({"status": "pass"},) * 5
        )


def test_five_saved_success_flags_still_run_actual_source_packet_checks(
    admission, monkeypatch
):
    prototype = object.__new__(admission.StagedDensePrototype)
    seen = []

    def original_check(**kwargs):
        seen.append(kwargs)
        raise ValueError("original sealed source and actual packet evidence absent")

    monkeypatch.setattr(prototype, "check_pair_artifacts", original_check)
    with pytest.raises(ValueError, match="actual packet"):
        admission.QualifiedStagedDenseOwner(
            prototype,
            ({"before": {"status": "pass"}, "after": {"status": "pass"}},) * 5,
        )
    assert len(seen) == 1
    assert seen[0]["execution_domain"] == "functional_spike"


def test_unqualified_prototype_has_no_automatic_registry_grant(admission):
    assert not hasattr(admission.StagedDensePrototype, "registry")
    with pytest.raises(ValueError, match="original checked"):
        admission.StagedDensePrototype({}, {}, {"status": "pass"})


def test_final_staged_pair_never_uses_an_unqualified_timing_domain(admission):
    prototype = object.__new__(admission.StagedDensePrototype)
    with pytest.raises(ValueError, match="timing domain"):
        prototype.check_pair_artifacts({}, {}, execution_domain="native_timing")
