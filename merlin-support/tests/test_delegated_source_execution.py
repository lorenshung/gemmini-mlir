"""Delegation boundary controls; neutral tokens are not target qualification."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_delegated_source_execution")


@pytest.fixture
def context(monkeypatch):
    # An explicit test double for dispatch only. Real generic token tests must
    # use the kernel-sealed descriptor and independently restored child scope.
    name = "merlin_experiments.phase1.feedback.compiler_pass_delegation"
    fake = ModuleType(name)
    fake.token = None
    fake.active_original_parent_pass_qualification = lambda: fake.token
    monkeypatch.setitem(sys.modules, name, fake)
    return fake


def test_missing_explicit_protocol_refuses_before_any_original_admission(
    module, tmp_path
):
    with pytest.raises(ValueError, match="explicit original factory opt-in"):
        module.restore_registry({}, work_root=tmp_path / "must-not-execute")
    assert not (tmp_path / "must-not-execute").exists()


def test_normal_host_always_runs_original_admission_even_with_saved_success_input(
    module, context, monkeypatch, tmp_path
):
    import gemmini_source_execution_registry as original

    calls = []
    live, result = object(), object()

    def admission(inputs, *, work_root):
        calls.append((inputs, work_root))
        return live

    monkeypatch.setattr(original, "restore_registry", admission)
    monkeypatch.setattr(
        module,
        "from_live_registry",
        lambda registry: result if registry is live else None,
    )
    inputs = {"parent_delegation_protocol": module.PROTOCOL, "saved_success": True}
    assert (
        module.restore_registry(
            inputs, work_root=tmp_path / "actual-original-admission"
        )
        is result
    )
    assert calls == [(inputs, tmp_path / "actual-original-admission")]


def test_active_parent_dispatch_uses_exact_callback_inputs_and_fixed_decoder(
    module, context, tmp_path
):
    calls = []

    class DispatchToken:
        def redeem(self, callback, inputs, *, decoder, work_root):
            calls.append((callback, inputs, decoder, work_root))
            return "neutral-dispatch-only"

    context.token = DispatchToken()
    inputs = {"parent_delegation_protocol": module.PROTOCOL}
    assert (
        module.restore_registry(inputs, work_root=tmp_path) == "neutral-dispatch-only"
    )
    assert calls == [
        (module.restore_registry, inputs, module.redeem_qualification, tmp_path)
    ]


def test_duck_typed_or_replayed_json_token_cannot_restore_an_original_owner(
    module, context, monkeypatch, tmp_path
):
    class ForgedToken:
        def require(self, snapshot):
            return {"registry": {}, "factory_inputs": {}}

    def forbidden(*args, **kwargs):
        raise AssertionError("unauthenticated token must not construct original owners")

    monkeypatch.setattr(module, "_restore_original_owner", forbidden)
    with pytest.raises(ValueError, match="active original sealed-parent"):
        module.redeem_qualification({}, token=ForgedToken(), work_root=tmp_path)


def test_private_restoration_helper_also_requires_active_authenticated_context(
    module, context
):
    class ForgedToken:
        def require(self, snapshot):
            return {"registry": {}, "factory_inputs": {}}

    with pytest.raises(ValueError, match="authenticated live parent"):
        module._restore_original_owner({}, {}, token=ForgedToken())


@pytest.mark.parametrize("damage", ["snapshot_schema", "roster", "factory_protocol"])
def test_mismatched_parent_snapshot_registry_or_optin_refuses_before_constructor(
    module, context, monkeypatch, tmp_path, damage
):
    snapshot = {
        "schema": module.SNAPSHOT,
        "binding": {},
        "registry": {"identity": "exact-parent"},
        "qualified_owner": {},
        "original_cleanup_admission": {},
    }
    selected = {
        "registry": {"identity": "exact-parent"},
        "factory_inputs": {"parent_delegation_protocol": module.PROTOCOL},
    }
    if damage == "snapshot_schema":
        snapshot["schema"] = "copied-success"
    elif damage == "roster":
        snapshot["registry"] = {"identity": "foreign"}
    else:
        selected["factory_inputs"] = {}

    class Token:
        def require(self, value):
            return selected

    context.token = Token()

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched parent must not construct original owners")

    monkeypatch.setattr(module, "_restore_original_owner", forbidden)
    with pytest.raises(ValueError, match="exact original parent context"):
        module.redeem_qualification(snapshot, token=context.token, work_root=tmp_path)


def test_saved_registry_or_success_cannot_export_original_parent_authority(module):
    for value in ({"status": "qualified"}, {"registry": {"identity": "saved"}}, None):
        with pytest.raises(ValueError, match="original live reviewed"):
            module.export_qualification(value)
        with pytest.raises(ValueError, match="original live reviewed"):
            module.from_live_registry(value)


def test_original_factory_alias_and_canonical_module_share_issuer_type(module):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "neutral_original_factory_alias", module.__file__
    )
    alias = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alias)
    assert alias.restore_registry is not module.restore_registry
    assert alias.SourceExecutionIssuerV2 is module.SourceExecutionIssuerV2
    assert alias.QualifiedStagedTileOwner is module.QualifiedStagedTileOwner
