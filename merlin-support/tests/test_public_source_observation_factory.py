"""Explicit mode/identity/transport controls; neutral owners are not target evidence."""

import copy
from dataclasses import replace

import gemmini_public_source_diagnostic as old
import gemmini_public_source_observation as M
import gemmini_public_source_observation_factory as F
import gemmini_source_observation_native as B
import pytest
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability as Cap
from merlin.targetgen.compiler_pass_selection import restore_original_factory
from test_public_source_diagnostic_issuer import original as _original_fixture
from test_source_native_correspondence import owners as _owners_fixture

original = _original_fixture
owners = _owners_fixture


@pytest.fixture
def explicit(original, monkeypatch):
    # Neutral callback dispatch, not an exported target algorithm or authority.
    monkeypatch.setattr(M._RUNTIME, "revalidate_export", lambda: None)
    _, prototype, _, calls, returned = original
    monkeypatch.setattr(M, "StagedDensePrototype", type(prototype))
    monkeypatch.setattr(
        M._RUNTIME,
        "create_source_execution_owner",
        old.SourceExecutionIssuer.create_source_execution_owner,
    )
    issuer = M.PublicSourceObservationIssuer(prototype)
    return prototype, issuer, calls, returned


def test_v2_selects_only_complete_commit_flags(explicit, tmp_path):
    prototype, issuer, calls, returned = explicit
    assert issuer._prototype is prototype
    report = issuer.record()
    assert report["identity"].startswith("gemmini.public_current_source_diagnostic.v2:")
    assert report["source_observation"] == {
        "protocol": "gemmini.source_observation.v2",
        "flags": ["--log-commits"],
    }
    assert report["scheduling_qualification"] == report["timing_authority"] == "NONE"
    assert (
        issuer.create_source_execution_owner(
            native_result=tmp_path / "r",
            command_buffer=tmp_path / "cb",
            lowered=tmp_path / "l",
            work_root=tmp_path / "new",
            remaining_s=10,
        )
        is returned
    )
    assert calls[-1]["observation_protocol"] == M.COMMIT_ONLY_PROTOCOL
    assert 0 < calls[-1]["remaining_s"] <= 10
    assert (
        "source_observation" not in old.PublicSourceDiagnosticIssuer(prototype).record()
    )


@pytest.mark.parametrize(
    "protocol", [None, "gemmini.source_observation.v1", "foreign", True]
)
def test_v2_factory_never_silently_upgrades_inputs(explicit, tmp_path, protocol):
    with pytest.raises(ValueError, match="explicit complete-commit"):
        M.restore_registry(
            {"source_observation_protocol": protocol}, work_root=tmp_path
        )


def test_v2_factory_preserves_canonical_class_and_exact_inputs(
    explicit, tmp_path, monkeypatch
):
    prototype, *_ = explicit
    calls = []

    def prepare(inputs, *, work_root):
        calls.append((inputs, work_root))
        return prototype

    monkeypatch.setattr(M, "prepare_original_prototype", prepare)
    args = {
        "original_field": "unchanged",
        "source_observation_protocol": M.COMMIT_ONLY_PROTOCOL,
    }
    record = F.original_factory(args).record()
    restored = restore_original_factory(record)
    registry = restored.create(work_root=tmp_path / "factory")
    assert type(registry._capabilities[0].owner) is M.PublicSourceObservationIssuer
    assert calls == [({"original_field": "unchanged"}, tmp_path / "factory")]
    assert args["source_observation_protocol"] == M.COMMIT_ONLY_PROTOCOL
    assert restored.record() == record


@pytest.mark.parametrize(
    "fault",
    ["source", "original_identity", "original_callback", "own_callback", "trace"],
)
def test_v2_custody_refuses_mutated_owners_and_callbacks(explicit, monkeypatch, fault):
    prototype, issuer, _, _ = explicit
    state = M._ISSUED[issuer]
    if fault == "source":
        state[3]["gemmini_public_source_observation"]["sha256"] = "0" * 64
    elif fault == "original_identity":
        prototype.identity = "changed"
    elif fault == "original_callback":
        monkeypatch.setattr(state[0], "source_rule", lambda: None)
    elif fault == "own_callback":
        monkeypatch.setattr(issuer, "_state", lambda: state)
    else:
        monkeypatch.setattr(M._RUNTIME, "create_source_execution_owner", lambda: None)
    with pytest.raises(ValueError):
        issuer.revalidate()


def test_unissued_or_foreign_v2_owner_refuses():
    with pytest.raises(ValueError, match="not originally issued"):
        object.__new__(M.PublicSourceObservationIssuer).revalidate()
    with pytest.raises(ValueError, match="foreign"):
        M.from_original_prototype({"status": "pass"})


@pytest.fixture
def joined(owners, monkeypatch):
    _, source_cap, count_cap, original_source, count = owners

    class NeutralV2:
        identity = "neutral-explicit-complete-commit"

        def revalidate(self):
            original_source.revalidate()

    source = NeutralV2()
    monkeypatch.setattr(B, "PublicSourceObservationIssuer", NeutralV2)
    monkeypatch.setattr(B, "CountHardwareCorrespondenceOwner", type(count))
    monkeypatch.setattr(B, "_ORIGINAL_STATE", lambda owner: (original_source,))
    cap = Cap("source_execution_diagnostic", source_cap.binding, source)
    owner = B.SourceObservationNativeCorrespondenceOwner.bind(cap, count_cap)
    return owner, cap, count_cap, source, original_source


def test_v2_bridge_derives_original_raw_composite_relation(joined):
    owner, source, count, *_ = joined
    assert source.binding.software != count.binding.software
    owner.verify_compiler_binding(source.binding, native_binding=count.binding)
    record = owner.record()
    assert record["source_observation_protocol"] == M.COMMIT_ONLY_PROTOCOL
    assert record["timing_qualification"] is False
    assert record["promotion_authority"] == "NONE"


@pytest.mark.parametrize(
    "field", ["hardware", "software", "numerical_policy", "runtime", "library"]
)
def test_v2_bridge_preserves_every_actual_binding_field(joined, field):
    owner, source, count, *_ = joined
    with pytest.raises(ValueError):
        owner.verify_compiler_binding(
            replace(source.binding, **{field: "foreign"}), native_binding=count.binding
        )


@pytest.mark.parametrize("fault", ["owner", "raw_fact", "own_source", "own_callback"])
def test_v2_bridge_stale_selection_refuses(joined, monkeypatch, fault):
    owner, _cap, _, source, original_source = joined
    if fault == "owner":
        monkeypatch.setattr(source, "revalidate", lambda: None)
    elif fault == "raw_fact":
        original_source._prototype.selected.inputs["facts"]["sha256"] = "0" * 64
    elif fault == "own_source":
        B._ISSUED[owner][11]["gemmini_source_observation_native"]["sha256"] = "0" * 64
    else:
        monkeypatch.setattr(owner, "_checked", lambda: None)
    with pytest.raises(ValueError):
        owner.revalidate()


def test_v2_factory_source_mutation_refuses_before_restore():
    record = copy.deepcopy(
        F.original_factory(
            {"source_observation_protocol": M.COMMIT_ONLY_PROTOCOL}
        ).record()
    )
    record["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        restore_original_factory(record)


def test_explicit_export_never_rebinds_the_canonical_owner():
    import gemmini_source_execution as canonical

    runtime = M._RUNTIME
    assert runtime._owner_module is not canonical
    assert runtime.SourceExecutionObservationOwner is not canonical.SourceExecutionOwner
    assert (
        old.SourceExecutionIssuer.create_source_execution_owner.__globals__[
            "SourceExecutionOwner"
        ]
        is canonical.SourceExecutionOwner
    )
    runtime.revalidate_export()


@pytest.mark.parametrize(
    "field",
    ["SourceExecutionOwner", "source_trace_flags", "_native_artifacts", "_limits"],
)
def test_exported_algorithm_global_substitution_refuses(monkeypatch, field):
    monkeypatch.setitem(M._RUNTIME._globals, field, object())
    with pytest.raises(ValueError, match="binding changed"):
        M._RUNTIME.revalidate_export()
