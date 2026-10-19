"""No target work: private transport must preserve the original issuer type."""

import copy
from pathlib import Path

import gemmini_public_source_diagnostic as canonical
import gemmini_public_source_factory as wrapper
import gemmini_source_native_correspondence as bridge
import pytest
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities
from merlin.targetgen.compiler_pass_selection import restore_original_factory


def test_private_factory_restoration_retains_canonical_callback_and_class(tmp_path):
    extra = tmp_path / "old-original-source.py"
    extra.write_text("# original closure leaf\n")
    record = wrapper.original_factory(
        {"purpose": "neutral-control"}, sources=(extra,)
    ).record()
    source = Path(module_source_path(canonical.__name__))
    assert {row["path"] for row in record["sources"]} == {str(source), str(extra)}
    restored = restore_original_factory(record)
    assert restored.record() == record
    delegated = restored.callback.__globals__["_canonical_restore_registry"]
    assert delegated is canonical.restore_registry
    assert (
        delegated.__globals__["PublicSourceDiagnosticIssuer"]
        is bridge.PublicSourceDiagnosticIssuer
    )
    assert restored.callback.__module__.startswith("merlin_original_pass_factory_")


def test_restored_wrapper_delegates_original_inputs_without_issuing_target_owner(
    tmp_path, monkeypatch
):
    record = wrapper.original_factory({"purpose": "neutral-admission-capture"}).record()
    calls = []

    def neutral(inputs, *, work_root):
        calls.append((inputs, work_root))
        return ReviewedPassCapabilities(())

    monkeypatch.setattr(canonical, "restore_registry", neutral)
    restored = restore_original_factory(record)
    directory = tmp_path / "original-empty-registry"
    registry = restored.create(work_root=directory)
    assert registry.record()["capabilities"] == []
    assert calls == [({"purpose": "neutral-admission-capture"}, directory)]


def test_changed_canonical_callback_source_pin_refuses_before_restoration():
    record = copy.deepcopy(wrapper.original_factory({}).record())
    record["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        restore_original_factory(record)
