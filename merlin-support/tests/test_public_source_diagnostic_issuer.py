"""Neutral live-prototype dispatch controls; no target qualification executes."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin.common.cpu_pool import current_thread_owns_cpu_lease


@pytest.fixture
def original(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    module = importlib.import_module("gemmini_public_source_diagnostic")
    calls = []

    class NeutralOriginalPrototype:
        identity = "neutral-unqualified-source-derived-prototype"
        contract = SimpleNamespace(
            hardware="hw",
            software="sw",
            numerical_policy="numeric",
            execution="runtime",
        )
        library = SimpleNamespace(contract=SimpleNamespace(sha256="library"))

        def revalidate(self):
            assert current_thread_owns_cpu_lease()

        def record(self):
            return {"scope": "neutral unqualified prototype"}

    def trace(self, **kwargs):
        assert current_thread_owns_cpu_lease()
        self.revalidate()
        calls.append(kwargs)
        return returned_owner

    returned_owner = object()
    monkeypatch.setattr(module, "StagedDensePrototype", NeutralOriginalPrototype)
    monkeypatch.setattr(
        module.SourceExecutionIssuer, "create_source_execution_owner", trace
    )
    # Constructor is trusted metadata under the original host leaf.
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool(timeout_s=120):
        prototype = NeutralOriginalPrototype()
        issuer = module.PublicSourceDiagnosticIssuer(prototype)
    return module, prototype, issuer, calls, returned_owner


def test_narrow_registry_grants_only_current_source_diagnostics(original, tmp_path):
    module, prototype, issuer, calls, returned_owner = original
    registry = module.from_original_prototype(prototype)
    rows = registry.record()["capabilities"]
    assert len(rows) == 1 and rows[0]["mechanism"] == "source_execution_diagnostic"
    assert not hasattr(issuer, "prepare_compiler") and not hasattr(
        issuer, "check_functional_guard"
    )
    report = issuer.record()
    assert report["scheduling_qualification"] == report["timing_authority"] == "NONE"
    args = {
        "native_result": tmp_path / "native",
        "command_buffer": tmp_path / "cb",
        "lowered": tmp_path / "lowered",
        "work_root": tmp_path / "newtrace",
    }
    assert (
        issuer.create_source_execution_owner(**args, remaining_s=10) is returned_owner
    )
    assert {key: calls[0][key] for key in args} == args
    assert 0 < calls[0]["remaining_s"] <= 10


@pytest.mark.parametrize("value", [{"status": "qualified"}, None, object()])
def test_saved_or_foreign_prototype_cannot_issue_registry(original, value):
    module, *_ = original
    with pytest.raises(ValueError, match="saved or foreign"):
        module.from_original_prototype(value)


@pytest.mark.parametrize("fault", ["identity", "original_trace", "sources"])
def test_current_owner_or_checker_changes_refuse_before_trace(
    original, tmp_path, monkeypatch, fault
):
    module, prototype, issuer, calls, _ = original
    if fault == "identity":
        prototype.identity = "foreign original selection"
    elif fault == "original_trace":
        monkeypatch.setattr(
            module.SourceExecutionIssuer,
            "create_source_execution_owner",
            lambda **kw: None,
        )
    else:
        issuer._sources["gemmini_public_source_diagnostic"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        issuer.create_source_execution_owner(
            native_result=tmp_path / "n",
            command_buffer=tmp_path / "c",
            lowered=tmp_path / "l",
            work_root=tmp_path / "t",
            remaining_s=10,
        )
    assert calls == []


@pytest.mark.parametrize(
    "remaining", [True, False, 0, -1, float("nan"), float("inf"), "10"]
)
def test_invalid_remaining_budget_never_admits_a_trace(original, tmp_path, remaining):
    _, _, issuer, calls, _ = original
    with pytest.raises(ValueError, match="positive remaining"):
        issuer.create_source_execution_owner(
            native_result=tmp_path / "n",
            command_buffer=tmp_path / "c",
            lowered=tmp_path / "l",
            work_root=tmp_path / "t",
            remaining_s=remaining,
        )
    assert calls == []
