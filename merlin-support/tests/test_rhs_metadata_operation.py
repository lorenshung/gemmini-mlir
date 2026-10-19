"""Neutral custody tests; these fixtures issue no target qualification."""

import copy
from types import SimpleNamespace

import gemmini_staged_rhs_capability as capability
import pytest
import yaml


def owner(monkeypatch, checks):
    value = object.__new__(capability.QualifiedRhsResidencyOwner)
    value.prototype = SimpleNamespace(record=lambda: {"neutral": True})
    value._prototype = value.prototype.record()
    value._source = capability._pin(capability.__file__)
    value._pairs, value._emissions, value._checks, value._capsules = [], [], [], []
    value._coverage = {}
    value._identity = value._digest()
    monkeypatch.setattr(capability, "_verify_original_capsules", lambda originals: checks.append(originals))
    return value


def test_only_one_live_metadata_operation_coalesces_duplicate_historical_checks(monkeypatch):
    checks = []
    value = owner(monkeypatch, checks)
    value.revalidate()
    assert len(checks) == 1
    with capability.scoped_original_rhs_metadata(value):
        assert len(checks) == 2
        for _ in range(5):
            value.record()
        with capability.scoped_original_rhs_metadata(value):
            value.revalidate()
        assert len(checks) == 2
    assert len(checks) == 3
    value.revalidate()
    assert len(checks) == 4


def test_foreign_owner_scope_keeps_both_owners_fresh(monkeypatch):
    checks = []
    left = owner(monkeypatch, checks)
    right = owner(monkeypatch, checks)
    with capability.scoped_original_rhs_metadata(left):
        with capability.scoped_original_rhs_metadata(right):
            left.revalidate()  # foreign active owner does not exempt this one
            right.revalidate()
    assert len(checks) == 5


def test_exception_still_runs_full_final_custody_check(monkeypatch):
    checks = []
    value = owner(monkeypatch, checks)
    with pytest.raises(RuntimeError, match="primary"):
        with capability.scoped_original_rhs_metadata(value):
            raise RuntimeError("primary")
    assert len(checks) == 2


def test_changed_artifact_refuses_final_exit_even_without_another_getter(monkeypatch):
    checks = []
    value = owner(monkeypatch, checks)

    def custody(_):
        checks.append(None)
        if len(checks) > 1:
            raise ValueError("original artifact changed")

    monkeypatch.setattr(capability, "_verify_original_capsules", custody)
    with pytest.raises(ValueError, match="artifact changed"):
        with capability.scoped_original_rhs_metadata(value):
            value.record()
    assert len(checks) == 2


@pytest.mark.parametrize("damage", ["identity", "prototype", "record", "callback"])
def test_mutated_original_metadata_cannot_leave_scope(monkeypatch, damage):
    value = owner(monkeypatch, [])
    with pytest.raises(ValueError, match="changed"):
        with capability.scoped_original_rhs_metadata(value):
            if damage == "identity":
                value._identity = "foreign"
            elif damage == "prototype":
                value.prototype = SimpleNamespace(record=lambda: {"neutral": True})
            elif damage == "record":
                value._coverage["changed"] = True
            else:
                monkeypatch.setattr(type(value), "revalidate", lambda _: None)
            value.record()


def test_primary_exception_keeps_failed_final_check_as_evidence(monkeypatch):
    checks = []
    value = owner(monkeypatch, checks)
    with pytest.raises(RuntimeError, match="primary") as caught:
        with capability.scoped_original_rhs_metadata(value):
            value._coverage["changed"] = True
            raise RuntimeError("primary")
    assert any("final metadata/custody" in note for note in caught.value.__notes__)


def capsule(tmp_path, name, attestation, raw):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "capsule.yaml").write_text(
        yaml.safe_dump(
            dict(capture_execution_attestations=[attestation], frontend_trace=dict(capture_mlir_sha256=raw))
        )
    )
    return dict(path=str(directory), files=capability.exact_tree_record(directory))


def test_exact_session_attestation_is_verified_once_per_full_check_not_once_ever(tmp_path, monkeypatch):
    attestation = dict(capture=dict(kind="session", programs=[dict(model_sha256="a"), dict(model_sha256="b")]))
    originals = [capsule(tmp_path, "a", attestation, "a"), capsule(tmp_path, "b", attestation, "b")]
    calls = []
    monkeypatch.setattr(
        capability, "require_verified_execution", lambda document: calls.append(copy.deepcopy(document))
    )
    capability._verify_original_capsules(originals)
    assert calls == [attestation]
    capability._verify_original_capsules(originals)
    assert calls == [attestation, attestation]


def test_each_capsule_keeps_its_own_raw_source_membership_check(tmp_path, monkeypatch):
    attestation = dict(capture=dict(kind="session", programs=[dict(model_sha256="a")]))
    originals = [capsule(tmp_path, "a", attestation, "a"), capsule(tmp_path, "b", attestation, "unattested")]
    calls = []
    monkeypatch.setattr(capability, "require_verified_execution", lambda document: calls.append(document))
    with pytest.raises(ValueError, match="not in its verified complete session"):
        capability._verify_original_capsules(originals)
    assert len(calls) == 1


def test_different_attestation_and_tree_mutation_are_not_exempt(tmp_path, monkeypatch):
    attestation = dict(capture=dict(model_sha256="a"))
    changed = {**attestation, "changed": True}
    originals = [capsule(tmp_path, "a", attestation, "a"), capsule(tmp_path, "b", changed, "a")]
    calls = []

    def verify(document):
        calls.append(document)
        if "changed" in document:
            raise ValueError("changed original attestation")

    monkeypatch.setattr(capability, "require_verified_execution", verify)
    with pytest.raises(ValueError, match="changed original attestation"):
        capability._verify_original_capsules(originals)
    assert len(calls) == 2
    (tmp_path / "a" / "capsule.yaml").write_text("changed")
    with pytest.raises(ValueError, match="publication changed"):
        capability._verify_original_capsules(originals)


def test_json_success_cannot_open_original_metadata_scope():
    with pytest.raises(ValueError, match="original live qualified owner"):
        with capability.scoped_original_rhs_metadata({"status": "qualified"}):
            pytest.fail("unreachable")
