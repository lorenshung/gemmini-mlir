"""Original-owner and missing-evidence controls; synthetic data grants no execution."""

import importlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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


def _replay_prototype(admission, receipt):
    """Exercise identity and live custody separately; this fixture grants nothing."""
    from merlin.compile.scheduling.staged_dense import StagedDenseContract

    prototype = object.__new__(admission.StagedDensePrototype)
    cleanup_record = {"identity": "exact-original-cleanup", "receipt": receipt}
    arithmetic_record = {"exact-arithmetic": "same-source-pins"}
    completion_record = {"sources": {}, "proof": "same-completion"}
    prototype.cleanup_owner = SimpleNamespace(
        identity="exact-original-cleanup",
        record=lambda: deepcopy(cleanup_record),
        revalidate=lambda: None,
    )
    prototype.arithmetic = SimpleNamespace(
        record=lambda: deepcopy(arithmetic_record), revalidate=lambda: None
    )
    prototype.completion = SimpleNamespace(
        record=lambda: deepcopy(completion_record), revalidate=lambda: None
    )
    prototype._sources = {}
    prototype._cleanup_record = deepcopy(cleanup_record)
    prototype._cleanup_identity = prototype.cleanup_owner.identity
    prototype._arithmetic_record = deepcopy(arithmetic_record)
    prototype._completion = deepcopy(completion_record)
    prototype.contract = StagedDenseContract(
        "facts", "effects", "numerical", "runtime", 16, 32, 16, 8, 20, 32,
        "read_at_compute_consume", "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity", True, True, False,
    )
    prototype._contract = asdict(prototype.contract)
    prototype._identity = prototype._selection_identity()
    return prototype, cleanup_record


def test_fresh_original_receipt_location_does_not_change_selected_capability(admission):
    parent, _ = _replay_prototype(admission, "/original-parent/receipt.json")
    child, _ = _replay_prototype(admission, "/original-child/receipt.json")
    parent.revalidate()
    child.revalidate()
    assert parent.identity == child.identity
    assert parent._cleanup_record != child._cleanup_record


def test_stable_selection_does_not_allow_replacing_a_live_original_receipt(admission):
    owner, record = _replay_prototype(admission, "/original-parent/receipt.json")
    record["receipt"] = "/replacement/success.json"
    with pytest.raises(ValueError, match="original staged"):
        owner.revalidate()


def test_different_original_selected_owner_cannot_share_staged_identity(admission):
    owner, _ = _replay_prototype(admission, "/original-parent/receipt.json")
    before = owner.identity
    owner._cleanup_identity = "different-original-cleanup"
    assert owner._selection_identity() != before.split(":", 1)[1]
    with pytest.raises(ValueError, match="original staged"):
        owner.revalidate()
