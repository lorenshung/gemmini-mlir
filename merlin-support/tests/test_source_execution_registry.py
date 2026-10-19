"""Original live registry composition; neutral owners confer no target claims."""

import importlib
from pathlib import Path

import pytest
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)


class FixtureOwner:
    identity = "neutral-live-fixture.v1"

    def revalidate(self):
        pass


class FixtureIssuer:
    identity = "neutral-live-issuer-fixture.v1"

    def __init__(self, owner):
        self.owner = owner

    def revalidate(self):
        self.owner.revalidate()


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_source_execution_registry")


def test_extra_issuer_preserves_exact_original_compiler_capability_and_binding(
    module, monkeypatch, tmp_path
):
    owner = FixtureOwner()
    binding = PassCapabilityBinding(
        "facts", "software", "numerical", "runtime", "same-release"
    )
    original = ReviewedPassCapability("staged_dense_tile_order_raw_i32", binding, owner)
    calls = []

    def restore(inputs, *, work_root):
        calls.append((inputs, work_root))
        return ReviewedPassCapabilities((original,))

    monkeypatch.setattr(module, "restore_tile_registry", restore)
    monkeypatch.setattr(module, "QualifiedStagedTileOwner", FixtureOwner)
    monkeypatch.setattr(module, "SourceExecutionIssuer", FixtureIssuer)
    inputs = {"original": "fixture-selection"}
    registry = module.restore_registry(
        inputs, work_root=tmp_path / "actual-original-child"
    )
    assert calls == [(inputs, tmp_path / "actual-original-child")]
    assert registry.resolve(original.mechanism, binding) is original
    diagnostic = registry.resolve("source_execution_diagnostic", binding)
    assert diagnostic.binding == original.binding
    assert diagnostic.owner.owner is owner
    assert [row["mechanism"] for row in registry.record()["capabilities"]] == [
        "staged_dense_tile_order_raw_i32",
        "source_execution_diagnostic",
    ]
    assert (
        registry.resolve(
            "source_execution_diagnostic",
            PassCapabilityBinding(
                "facts", "software", "numerical", "runtime", "different-release"
            ),
        )
        is None
    )


@pytest.mark.parametrize("mode", ["absent", "wrong_mechanism", "foreign_owner"])
def test_missing_or_foreign_original_factory_result_cannot_add_diagnostic_authority(
    module, monkeypatch, tmp_path, mode
):
    binding = PassCapabilityBinding(
        "facts", "software", "numerical", "runtime", "same-release"
    )
    capabilities = (
        ()
        if mode == "absent"
        else (
            ReviewedPassCapability(
                "different"
                if mode == "wrong_mechanism"
                else "staged_dense_tile_order_raw_i32",
                binding,
                FixtureOwner(),
            ),
        )
    )
    monkeypatch.setattr(
        module,
        "restore_tile_registry",
        lambda inputs, work_root: ReviewedPassCapabilities(capabilities),
    )
    with pytest.raises(ValueError, match="exact original|saved or foreign"):
        module.restore_registry({}, work_root=tmp_path / "unqualified")
