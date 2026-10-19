"""Dispatch/custody controls; these neutral fixtures grant no target support."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import gemmini_zero_initializer_registry as registry
import pytest
from gemmini_source_execution_issuer_v3 import SourceExecutionIssuerV3
from gemmini_zero_initializer_capability import QualifiedZeroInitializerOwner


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
        zero_initializer_qualification_pairs=[{"before": "actual", "after": "actual"}] * 7,
    )


@pytest.mark.parametrize(
    "field", ["parent_delegation_protocol", "metadata_verification_protocol", "zero_initializer_qualification_pairs"]
)
def test_missing_original_selection_refuses_before_child_or_source_admission(tmp_path, field):
    declaration = inputs()
    declaration.pop(field)
    with pytest.raises(ValueError, match="selection disabled"):
        registry.restore_registry(declaration, work_root=tmp_path / "never-started")
    assert not (tmp_path / "never-started").exists()


def test_json_qualification_cannot_construct_live_registry_or_current_source_issuer():
    with pytest.raises(ValueError, match="original live qualified"):
        registry.from_live_owner({"status": "qualified"})
    with pytest.raises(ValueError, match="original live qualified"):
        SourceExecutionIssuerV3({"status": "qualified"})


def test_source_issuer_pins_original_trace_function_and_code(monkeypatch):
    # Deliberately neutral live-object fixture: this test does not execute a
    # trace, issue a registry, or claim the synthetic qualification is real.
    from gemmini_source_execution_issuer import SourceExecutionIssuer

    owner = object.__new__(QualifiedZeroInitializerOwner)
    owner._identity = "neutral-only"
    original = SimpleNamespace(identity="neutral-prototype")
    owner.prototype = SimpleNamespace(tile=SimpleNamespace(prototype=original))
    monkeypatch.setattr(QualifiedZeroInitializerOwner, "revalidate", lambda _: None)
    issuer = SourceExecutionIssuerV3(owner)
    function = SourceExecutionIssuer.create_source_execution_owner

    def changed(*args, **kwargs):
        raise AssertionError("changed trace implementation must never execute")

    monkeypatch.setattr(SourceExecutionIssuer, "create_source_execution_owner", changed)
    with pytest.raises(ValueError, match="trace implementation changed"):
        issuer.revalidate()
    monkeypatch.setattr(SourceExecutionIssuer, "create_source_execution_owner", function)
    with monkeypatch.context() as mutation:
        mutation.setattr(function, "__code__", changed.__code__)
        with pytest.raises(ValueError, match="trace implementation changed"):
            issuer.revalidate()
    issuer.revalidate()


def test_normal_host_always_reissues_original_admission(context, monkeypatch, tmp_path):
    import gemmini_staged_dense_admission as admission
    import gemmini_staged_tile_capability as tile

    calls = []
    original, prepared, prototype, owner, result = (object() for _ in range(5))
    declaration = dict(inputs(), saved_success=True)
    monkeypatch.setattr(
        admission,
        "prepare_original_prototype",
        lambda selected, *, work_root: calls.append((selected, work_root)) or original,
    )
    monkeypatch.setattr(tile, "StagedTilePrototype", lambda selected: prepared if selected is original else None)
    monkeypatch.setattr(
        registry, "ZeroInitializerCompilerPrototype", lambda selected: prototype if selected is prepared else None
    )

    def issue(selected, pairs, *, status_path):
        assert selected is prototype and pairs == tuple(declaration["zero_initializer_qualification_pairs"])
        calls.append((pairs, status_path))
        return owner

    monkeypatch.setattr(registry, "QualifiedZeroInitializerOwner", issue)
    monkeypatch.setattr(registry, "from_live_owner", lambda selected: result if selected is owner else None)
    monkeypatch.setattr(registry, "shared_compute_pool", lambda **_: contextlib.nullcontext())
    assert registry.restore_registry(declaration, work_root=tmp_path) is result
    assert calls[0] == (declaration, tmp_path / "original-admission")
    assert len(calls) == 2


def test_active_child_dispatch_uses_exact_original_callback_and_decoder(context, tmp_path):
    calls = []

    class Token:
        def redeem(self, callback, selected, *, decoder, work_root):
            calls.append((callback, selected, decoder, work_root))
            return "dispatch-only"

    context.token = Token()
    selected = inputs()
    assert registry.restore_registry(selected, work_root=tmp_path) == "dispatch-only"
    assert calls == [(registry.restore_registry, selected, registry.redeem_qualification, tmp_path)]


def test_replayed_json_or_duck_token_cannot_reach_original_owner(context, tmp_path):
    class Token:
        def require(self, snapshot):
            return {"registry": {}, "factory_inputs": inputs()}

    with pytest.raises(ValueError, match="active original sealed-parent"):
        registry.redeem_qualification({}, token=Token(), work_root=tmp_path)


@pytest.mark.parametrize("damage", ["schema", "roster", "protocol", "metadata", "fields"])
def test_mismatched_authenticated_context_refuses_before_original_issuance(context, monkeypatch, tmp_path, damage):
    selected = inputs()
    snapshot = dict(
        schema=registry.SNAPSHOT,
        binding={},
        registry={"identity": "parent"},
        qualified_owner={},
        original_cleanup_admission={},
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
    else:
        snapshot["extra"] = "candidate-grant"

    class Token:
        def require(self, payload):
            return original

    context.token = Token()
    monkeypatch.setattr(registry, "_restore_original_owner", lambda *_, **__: pytest.fail("must not issue"))
    with pytest.raises(ValueError, match="exact original parent"):
        registry.redeem_qualification(snapshot, token=context.token, work_root=tmp_path)
