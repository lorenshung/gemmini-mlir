"""Ancestry lifetime/byte guards; neutral control fixtures issue no target owner."""

import contextlib
import contextvars
from types import MethodType, SimpleNamespace

import gemmini_primitive_current_proof as P
import gemmini_staged_rhs_capability as RHS
import pytest
from gemmini_primitive_context import PrimitiveCompilerContext, _original_methods
from gemmini_primitive_diagnostic import PrimitiveDiagnosticOwner
from gemmini_staged_rhs_capability import (
    QualifiedRhsResidencyOwner,
    RhsResidencyCompilerPrototype,
)
from gemmini_timing_resources import _pin
from gemmini_zero_initializer_capability import ZeroInitializerCompilerPrototype
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _structure(tmp_path):
    # Only pure original type/identity/source structure is exercised. Actual
    # seed.revalidate and historical native qualification are never fabricated.
    leaf = tmp_path / "original.py"
    leaf.write_text("# immutable fixture\n")
    seed = object.__new__(ZeroInitializerCompilerPrototype)
    seed._identity = "seed"
    prototype = object.__new__(RhsResidencyCompilerPrototype)
    prototype.seed = seed
    q = object.__new__(QualifiedRhsResidencyOwner)
    q.prototype = prototype
    q._identity = "ancestor"
    c = object.__new__(PrimitiveCompilerContext)
    c.qualification, c.mode, c.seed = q, "qualified_rhs_seed_only", seed
    c._owner_type, c.binding_owner, c._binding_type = (
        QualifiedRhsResidencyOwner,
        seed,
        ZeroInitializerCompilerPrototype,
    )
    c._callbacks = (
        _original_methods(q, QualifiedRhsResidencyOwner),
        _original_methods(seed, ZeroInitializerCompilerPrototype),
    )
    c._seed_callbacks = _original_methods(seed, ZeroInitializerCompilerPrototype)
    c._origin, c._seed = q.identity, seed.identity
    c._sources = {"fixture": _pin(leaf)}
    c._identity = canonical_sha256([c.mode, c._origin, c._seed, c._sources])
    owner = object.__new__(PrimitiveDiagnosticOwner)
    owner.context = c
    return owner, c, q, seed, leaf


@pytest.mark.parametrize(
    "damage",
    ["source", "seed_identity", "ancestor_identity", "seed_object", "seed_callback"],
)
def test_current_seed_structure_refuses_original_byte_or_callback_drift(
    tmp_path, damage
):
    _owner, c, q, seed, leaf = _structure(tmp_path)
    c._check_current_structure()
    if damage == "source":
        leaf.write_text("# changed\n")
    elif damage == "seed_identity":
        seed._identity = "different"
    elif damage == "ancestor_identity":
        q._identity = "different"
    elif damage == "seed_object":
        q.prototype.seed = object.__new__(ZeroInitializerCompilerPrototype)
    else:
        seed.revalidate = MethodType(lambda self: None, seed)
    with pytest.raises(ValueError, match="primitive|source changed"):
        c._check_current_structure()


def test_current_context_callback_cannot_be_replaced_even_on_exact_type(tmp_path):
    owner, c, _q, _seed, _leaf = _structure(tmp_path)
    P._owner(owner)
    c.revalidate_current_seed = MethodType(lambda self: None, c)
    with pytest.raises(ValueError, match="method/source"):
        P._owner(owner)


def test_view_has_no_serialized_success_constructor():
    with pytest.raises(ValueError, match="cannot be restored"):
        P._CurrentProofView({"status": "pass"})
    with pytest.raises(ValueError, match="original live"):
        P.check_current_pair({"status": "pass"})


def test_failed_initial_revalidation_expires_even_when_traceback_retains_view(
    monkeypatch,
):
    # Only lifetime cleanup is under test; no target owner or proof succeeds.
    captured = []
    owner = SimpleNamespace(
        context=object(),
        identity="fixture",
        selected=object(),
        arithmetic=object(),
        cleanup=object(),
        nm="fixture",
    )
    monkeypatch.setattr(P, "_owner", lambda obj: None)

    def fail(view):
        captured.append(view)
        raise ValueError("fresh initial source drift")

    monkeypatch.setattr(P._CurrentProofView, "revalidate", fail)
    with pytest.raises(ValueError, match="initial source drift"):
        P._CurrentProofView(owner, issuer=P._ISSUER)
    assert len(captured) == 1 and captured[0] not in P._ACTIVE


def test_even_private_key_cannot_issue_subclass_view():
    class ForeignView(P._CurrentProofView):
        pass

    with pytest.raises(ValueError, match="cannot be restored"):
        ForeignView({}, issuer=P._ISSUER)


@pytest.mark.parametrize("outcome", ["success", "current_failure", "ancestor_mutation"])
def test_neutral_control_flow_full_ancestry_checks_both_edges_and_expire_view(
    monkeypatch, outcome
):
    # This fixture deliberately replaces the target body and type selection;
    # it tests exception/lifetime orchestration only, never target admission.
    events = []

    class NeutralOwner:
        context = SimpleNamespace(_origin="neutral_history")
        drift = False

        def revalidate(self):
            events.append("full_history")
            if self.drift:
                raise ValueError("actual historical bytes changed")

    owner = NeutralOwner()

    class NeutralView:
        def __init__(self, origin, *, issuer):
            assert origin is owner
            P._ACTIVE.add(self)

    monkeypatch.setattr(P, "_owner", lambda obj: None)
    monkeypatch.setattr(
        P, "shared_compute_pool", lambda **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(P, "_CurrentProofView", NeutralView)
    monkeypatch.setattr(
        RHS,
        "_METADATA",
        contextvars.ContextVar("neutral_rhs_metadata", default=None),
        raising=False,
    )

    def current(view, **args):
        assert RHS._METADATA.get() is None
        assert view in P._ACTIVE
        events.append("fresh_current_body")
        if outcome == "current_failure":
            raise RuntimeError("current trace failed")
        if outcome == "ancestor_mutation":
            owner.drift = True
        return {"costs": {}, "hardware_execution_qualified": False}

    monkeypatch.setattr(P, "_ORIGINAL_CHECK", current)
    if outcome == "success":
        result = P.check_current_pair(owner)
        assert result["hardware_execution_qualified"] is False
        assert "ancestry only" in result["historical_ancestry"]["scope"]
    else:
        with pytest.raises((ValueError, RuntimeError), match="changed|trace failed"):
            P.check_current_pair(owner)
    assert events == ["full_history", "fresh_current_body", "full_history"]
    assert not P._ACTIVE


def test_even_outer_metadata_scope_cannot_relabel_history_as_fresh(monkeypatch):
    monkeypatch.setattr(P, "_owner", lambda obj: None)
    metadata = contextvars.ContextVar("neutral_metadata", default=None)
    metadata.set(object())
    monkeypatch.setattr(
        RHS,
        "_METADATA",
        metadata,
        raising=False,
    )
    with pytest.raises(ValueError, match="outside historical metadata"):
        P.check_current_pair(SimpleNamespace())


def test_current_proof_leaf_bytes_are_part_of_live_source_custody(
    tmp_path, monkeypatch
):
    owner, _context, _q, _seed, _leaf = _structure(tmp_path)
    source = tmp_path / "current_proof.py"
    source.write_text("# original current proof\n")
    monkeypatch.setattr(P, "__file__", str(source))
    monkeypatch.setattr(P, "_SOURCE", _pin(source))
    P._owner(owner)
    source.write_text("# changed current proof\n")
    with pytest.raises(ValueError, match="proof source changed"):
        P._owner(owner)
