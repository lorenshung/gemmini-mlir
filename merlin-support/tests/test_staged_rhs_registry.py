"""Original-owner/delegation controls; no fixture issues target qualification."""

import sys
from types import ModuleType, SimpleNamespace

import gemmini_staged_rhs_registry as registry
import pytest
from gemmini_source_execution_issuer_v4 import SourceExecutionIssuerV4
from gemmini_staged_rhs_capability import QualifiedRhsResidencyOwner


@pytest.fixture
def context(monkeypatch):
    name = "merlin_experiments.phase1.feedback.compiler_pass_delegation"
    module = ModuleType(name)
    module.token = None
    module.active_original_parent_pass_qualification = lambda: module.token
    monkeypatch.setitem(sys.modules, name, module)
    return module


def inputs():
    return dict(
        parent_delegation_protocol=registry.PROTOCOL,
        metadata_verification_protocol=registry.METADATA,
        rhs_residency_qualification_pairs=[dict(before="actual", after="actual")] * 6,
        rhs_allocation_epoch_source={},
    )


@pytest.mark.parametrize(
    "field",
    [
        "parent_delegation_protocol",
        "metadata_verification_protocol",
        "rhs_residency_qualification_pairs",
        "rhs_allocation_epoch_source",
    ],
)
def test_missing_original_selection_cannot_start_expensive_admission(tmp_path, field):
    selected = inputs()
    selected.pop(field)
    with pytest.raises(ValueError, match="selection disabled"):
        registry.restore_registry(selected, work_root=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_json_qualification_cannot_construct_current_registry_or_source_issuer():
    with pytest.raises(ValueError, match="original live qualified"):
        registry.from_live_owner({"status": "qualified"})
    with pytest.raises(ValueError, match="original live qualified"):
        SourceExecutionIssuerV4({"status": "qualified"})


def test_source_issuer_pins_original_actual_trace_function_and_code(monkeypatch):
    from gemmini_source_execution_issuer import SourceExecutionIssuer

    # Neutral object fixture tests only implementation custody, never issuance.
    owner = object.__new__(QualifiedRhsResidencyOwner)
    owner._identity = "neutral"
    original = SimpleNamespace(identity="neutral-prototype")
    owner.prototype = SimpleNamespace(seed=SimpleNamespace(tile=SimpleNamespace(prototype=original)))
    monkeypatch.setattr(QualifiedRhsResidencyOwner, "revalidate", lambda _: None)
    issuer = SourceExecutionIssuerV4(owner)
    function = SourceExecutionIssuer.create_source_execution_owner

    def changed(*args, **kwargs):
        raise AssertionError("changed trace must never execute")

    monkeypatch.setattr(SourceExecutionIssuer, "create_source_execution_owner", changed)
    with pytest.raises(ValueError, match="trace implementation changed"):
        issuer.revalidate()
    with pytest.raises(ValueError, match="trace implementation changed"):
        issuer.create_source_execution_owner(
            native_result=None, command_buffer=None, lowered=None, work_root=None, remaining_s=1
        )
    monkeypatch.setattr(SourceExecutionIssuer, "create_source_execution_owner", function)
    with monkeypatch.context() as mutation:
        mutation.setattr(function, "__code__", changed.__code__)
        with pytest.raises(ValueError, match="trace implementation changed"):
            issuer.revalidate()
    issuer.revalidate()


def test_source_issuer_retains_original_full_entry_and_exit_checks_without_a_third_duplicate(monkeypatch):
    from gemmini_source_execution_issuer import SourceExecutionIssuer

    # Neutral call-boundary fixture, not a native execution or target owner.
    checks = []
    owner = object.__new__(QualifiedRhsResidencyOwner)
    owner._identity = "neutral"
    original = SimpleNamespace(identity="neutral-prototype")
    owner.prototype = SimpleNamespace(seed=SimpleNamespace(tile=SimpleNamespace(prototype=original)))
    monkeypatch.setattr(QualifiedRhsResidencyOwner, "revalidate", lambda _: checks.append("full"))

    def original_trace(self, **kwargs):
        self.revalidate()
        self.revalidate()
        return "neutral-boundary-check"

    monkeypatch.setattr(SourceExecutionIssuer, "create_source_execution_owner", original_trace)
    issuer = SourceExecutionIssuerV4(owner)
    checks.clear()
    assert issuer.create_source_execution_owner(
        native_result=None, command_buffer=None, lowered=None, work_root=None, remaining_s=1
    ) == "neutral-boundary-check"
    assert checks == ["full", "full"]


def test_active_child_dispatch_uses_original_factory_and_decoder(context, monkeypatch, tmp_path):
    calls = []

    class Token:
        def redeem(self, callback, selected, *, decoder, work_root):
            calls.append((callback, selected, decoder, work_root))
            return "dispatch-only"

    context.token = Token()
    selected = inputs()
    # Neutral dispatch test; actual native custody is independently tested by
    # the real factory and the fixed artifact-pin boundary.
    monkeypatch.setattr(registry, "_leaf", lambda *args: None)
    result = registry.restore_registry(selected, work_root=tmp_path)
    assert result == "dispatch-only"
    assert calls == [(registry.restore_registry, selected, registry.redeem_qualification, tmp_path)]


def test_unpinned_allocation_cannot_reach_child_or_inherit_a_capacity_claim(context, tmp_path):
    selected = inputs()
    with pytest.raises(ValueError, match="original artifact"):
        registry.restore_registry(selected, work_root=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_json_and_duck_token_cannot_restore_historical_epoch_or_owner(context, tmp_path):
    class Token:
        def require(self, snapshot):
            return dict(registry={}, factory_inputs=inputs())

    token = Token()
    with pytest.raises(ValueError, match="active original sealed-parent"):
        registry.redeem_qualification({}, token=token, work_root=tmp_path)
    with pytest.raises(ValueError, match="active original sealed-parent"):
        registry._restore_epoch({}, None, original_snapshot={}, token=token)


@pytest.mark.parametrize("damage", ["schema", "roster", "protocol", "metadata", "fields", "epoch"])
def test_mismatched_authenticated_context_refuses_before_owner_construction(context, monkeypatch, tmp_path, damage):
    selected = inputs()
    snapshot = dict(
        schema=registry.SNAPSHOT,
        binding={},
        registry={"identity": "parent"},
        qualified_owner={},
        original_cleanup_admission={},
        original_epoch_state={},
    )
    original = dict(registry={"identity": "parent"}, factory_inputs=selected)
    if damage == "schema":
        snapshot["schema"] = "saved-success"
    elif damage == "roster":
        snapshot["registry"] = {"identity": "other"}
    elif damage == "protocol":
        selected["parent_delegation_protocol"] = "other"
    elif damage == "metadata":
        selected.pop("metadata_verification_protocol")
    elif damage == "epoch":
        snapshot.pop("original_epoch_state")
    else:
        snapshot["extra"] = "grant"

    class Token:
        def require(self, payload):
            return original

    context.token = Token()
    monkeypatch.setattr(registry, "_restore_original_owner", lambda *_, **__: pytest.fail("must not construct"))
    with pytest.raises(ValueError, match="exact original parent"):
        registry.redeem_qualification(snapshot, token=context.token, work_root=tmp_path)
