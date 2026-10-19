"""Neutral original-owner transport/custody controls, never accelerator evidence."""

import copy
import gc
import importlib
from pathlib import Path
from types import MethodType

import pytest
from test_public_source_diagnostic_issuer import original as _original_fixture

from merlin.common.cpu_pool import current_thread_owns_cpu_lease
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability
from merlin.targetgen.compiler_pass_selection import restore_original_factory

original = _original_fixture


@pytest.fixture
def selected(original, monkeypatch):
    source, prototype, *_ = original
    module = importlib.import_module("gemmini_diagnostic_tile_preparation")
    tile = importlib.import_module("gemmini_staged_tile_capability")
    factory = importlib.import_module("gemmini_diagnostic_tile_factory")
    monkeypatch.setattr(module, "StagedDensePrototype", type(prototype))
    monkeypatch.setattr(tile, "StagedDensePrototype", type(prototype))
    registry = source.from_original_prototype(prototype)
    joined = module.with_diagnostic_preparation(registry)
    cap = joined._capabilities[-1]
    return module, prototype, registry, joined, cap, factory


def test_real_capability_transport_keeps_source_owner_and_separate_unqualified_preparation(
    selected,
):
    module, _, source, joined, cap, _ = selected
    assert joined._capabilities[0] is source._capabilities[0]
    assert cap.mechanism == module.MECHANISM
    assert cap.binding == source._capabilities[0].binding
    report = cap.owner.record()
    assert report["scheduling_qualification"] == report["timing_authority"] == "NONE"
    assert (
        report["promotion_authority"] == "NONE"
        and not report["hardware_execution_qualified"]
    )
    assert not hasattr(cap.owner, "check_functional_guard")
    assert cap.owner.selected_binding() == report["binding"]
    report["source"]["mechanism"] = "foreign"
    cap.revalidate()


def test_private_factory_restore_delegates_to_canonical_source_issuer(
    selected, monkeypatch, tmp_path
):
    module, prototype, _, _, _, factory = selected
    import gemmini_public_source_diagnostic as source

    calls = []

    def prepare(inputs, *, work_root):
        calls.append((inputs, work_root))
        return prototype

    monkeypatch.setattr(source, "prepare_original_prototype", prepare)
    inputs = {"original_inputs": "unchanged"}
    chosen = factory.original_factory(inputs)
    restored = restore_original_factory(chosen.record())
    registry = restored.create(work_root=tmp_path / "original")
    assert restored.record() == chosen.record()
    assert calls == [(inputs, tmp_path / "original")]
    assert type(registry._capabilities[0].owner) is source.PublicSourceDiagnosticIssuer
    assert (
        type(registry._capabilities[-1].owner) is module.DiagnosticTilePreparationOwner
    )
    closure = {Path(pin["path"]).name for pin in chosen.record()["sources"]}
    assert {
        "gemmini_diagnostic_tile_preparation.py",
        "gemmini_public_source_diagnostic.py",
        "gemmini_source_execution.py",
        "gemmini_source_execution_issuer.py",
    } <= closure


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "registry",
        "tile",
        "binding",
        "sources",
        "prepare_callback",
        "tile_callback",
    ],
)
def test_rebound_owners_binding_sources_and_bound_callbacks_refuse(
    selected, monkeypatch, damage
):
    module, _, source, _, cap, _ = selected
    owner = cap.owner
    if damage == "source":
        original = source._capabilities[0]
        owner.source_capability = ReviewedPassCapability(
            original.mechanism, original.binding, original.owner
        )
    elif damage == "registry":
        owner._source_registry = copy.copy(owner._source_registry)
    elif damage == "tile":
        owner.tile = copy.copy(owner.tile)
    elif damage == "binding":
        owner._binding["library"] = "foreign-library"
    elif damage == "sources":
        owner._sources.clear()
    elif damage == "prepare_callback":
        monkeypatch.setattr(
            owner, "prepare_compiler", MethodType(lambda self, *a, **k: None, owner)
        )
    else:
        monkeypatch.setattr(
            owner.tile,
            "prepare_compiler",
            MethodType(lambda self, *a, **k: None, owner.tile),
        )
    with pytest.raises(ValueError, match="original"):
        cap.revalidate()


def test_current_source_identity_drift_refuses_preparation(selected, tmp_path):
    _, prototype, _, _, cap, _ = selected
    owner = cap.owner
    prototype.identity = "changed-original-source"
    with pytest.raises(ValueError):
        owner.prepare_compiler(tmp_path / "raw", tmp_path / "new", policy={})
    assert not (tmp_path / "new").exists()


def test_json_foreign_duplicate_and_unissued_owners_never_select_preparation(selected):
    module, _, _, joined, _, _ = selected
    with pytest.raises(ValueError, match="actual original"):
        module.with_diagnostic_preparation({"qualified": True})
    with pytest.raises(ValueError, match="no existing"):
        module.with_diagnostic_preparation(joined)
    with pytest.raises(ValueError, match="actual original"):
        module.DiagnosticTilePreparationOwner({"status": "qualified"})
    with pytest.raises(ValueError, match="live issuance"):
        object.__new__(module.DiagnosticTilePreparationOwner).revalidate()


def test_weak_issuance_ledger_does_not_retain_preparation_owner(selected):
    module, _, source, _, _, _ = selected
    owner = module.DiagnosticTilePreparationOwner(source._capabilities[0])
    before = len(module._ISSUED)
    del owner
    gc.collect()
    assert len(module._ISSUED) == before - 1


def test_actual_delegate_runs_under_original_cpu_and_post_source_check(
    selected, monkeypatch, tmp_path
):
    module, prototype, source, _, _, _ = selected
    tile = importlib.import_module("gemmini_staged_tile_capability")
    calls = []

    def prepare(self, origin, destination, *, policy):
        assert current_thread_owns_cpu_lease()
        calls.append((origin, destination, policy))
        prototype.identity = "changed-by-delegate"
        return destination

    monkeypatch.setattr(tile.StagedTilePrototype, "prepare_compiler", prepare)
    owner = module.DiagnosticTilePreparationOwner(source._capabilities[0])
    with pytest.raises(ValueError):
        owner.prepare_compiler(
            tmp_path / "raw", tmp_path / "candidate", policy={"enabled": False}
        )
    assert calls == [(tmp_path / "raw", tmp_path / "candidate", {"enabled": False})]


def test_unknown_observation_protocol_does_not_fall_back(selected):
    _, _, _, _, _, factory = selected
    with pytest.raises(ValueError, match="unknown explicit"):
        factory.original_factory({"source_observation_protocol": "foreign"})


def test_one_readonly_revalidation_freshly_checks_original_source_at_both_boundaries(
    selected, monkeypatch
):
    _, prototype, _, _, cap, _ = selected
    owner = cap.owner
    checks = []
    original_check = prototype.revalidate

    def counted():
        checks.append("fresh-original-source")
        original_check()

    monkeypatch.setattr(prototype, "revalidate", counted)
    owner.revalidate()
    # Two original registry boundaries plus the direct original tile record.
    assert checks == ["fresh-original-source"] * 3
    owner.revalidate()
    assert checks == ["fresh-original-source"] * 6


@pytest.mark.parametrize("raise_body", [False, True])
def test_final_original_source_check_refuses_mutation_and_preserves_body_error(
    selected, monkeypatch, raise_body
):
    module, prototype, source, _, _, _ = selected
    tile = importlib.import_module("gemmini_staged_tile_capability")
    original_record = tile.StagedTilePrototype.record
    mutate = False

    def record(self):
        value = original_record(self)
        if mutate:
            prototype.identity = "changed-at-end-of-metadata-body"
            if raise_body:
                raise RuntimeError("original metadata body failure")
        return value

    monkeypatch.setattr(tile.StagedTilePrototype, "record", record)
    owner = module.DiagnosticTilePreparationOwner(source._capabilities[0])
    mutate = True
    if raise_body:
        with pytest.raises(
            RuntimeError, match="original metadata body failure"
        ) as caught:
            owner.revalidate()
        assert any(
            "final metadata/qualification" in note for note in caught.value.__notes__
        )
    else:
        with pytest.raises(ValueError, match="original"):
            owner.revalidate()
