"""Explicit original factory/observation transport, with no target authority."""
# ruff: noqa: F811 -- pytest imported fixtures are intentionally requested by name

import importlib

import pytest
from test_generated_source_diagnostic import issued as generated_issued  # noqa: F401
from test_generated_source_trace import generated  # noqa: F401
from test_source_execution_issuer import fixture, module  # noqa: F401


def test_original_default_record_and_binding_do_not_add_memory_grant(generated_issued):
    M, owner, *_ = generated_issued
    assert "source_memory_effects_protocol" not in owner.record()
    assert M._STATE(owner)["memory_effects"] is None


def test_factory_derives_explicit_live_memory_bundle_before_any_target(
    generated_issued, monkeypatch
):
    M, old, inputs, *_ = generated_issued
    W = importlib.import_module("gemmini_whole_program_memory_effects")
    calls = []

    class NeutralEffects:
        identity = "neutral selected memory effects"

        def __init__(self, selected, arithmetic):
            self.selected = selected
            calls.append(("derive", selected, arithmetic))

        def revalidate(self):
            calls.append("fresh")

        def record(self):
            self.revalidate()
            return {"neutral_only": True}

        def packet_owners(self, selected):
            assert selected is self.selected
            self.revalidate()

    monkeypatch.setattr(W, "WholeProgramMemoryEffects", NeutralEffects)
    owner = M.GeneratedSourceDiagnosticIssuer(
        dict(inputs, source_memory_effects_protocol=W.PROTOCOL)
    )
    assert calls[0][0] == "derive" and "fresh" in calls
    assert owner.record()["source_memory_effects_protocol"] == W.PROTOCOL
    assert owner.selected_binding() != old.selected_binding()
    state = M._STATE(owner)
    assert state["memory_effects_identity"] == NeutralEffects.identity
    state["memory_effects_protocol"] = None
    with pytest.raises(ValueError):
        owner.revalidate()


@pytest.mark.parametrize(
    "protocol", [True, 2, "foreign", "gemmini.source_observation.v2"]
)
def test_factory_cannot_select_unreviewed_memory_protocol(generated_issued, protocol):
    M, _, inputs, *_ = generated_issued
    with pytest.raises(ValueError, match="memory-effects protocol"):
        M.GeneratedSourceDiagnosticIssuer(
            dict(inputs, source_memory_effects_protocol=protocol)
        )


def test_direct_factory_and_direct_constructor_reject_whole_program_memory(
    generated_issued,
):
    M, _, inputs, *_ = generated_issued
    W = importlib.import_module("gemmini_whole_program_memory_effects")
    with pytest.raises(ValueError, match="whole_program"):
        M.GeneratedSourceDiagnosticIssuer(
            dict(
                inputs,
                source_binding_protocol="merlin.native_pointer_source.v1",
                source_memory_effects_protocol=W.PROTOCOL,
            )
        )
    from gemmini_direct_source_execution import DirectSourceExecutionOwner

    with pytest.raises(TypeError, match="whole-program owner"):
        DirectSourceExecutionOwner(
            execution_contract={},
            arithmetic=None,
            selected=None,
            result=None,
            command_buffer=None,
            lowered=None,
            trace_invocation=None,
            nm=None,
            source_memory_effects_protocol=W.PROTOCOL,
        )


def test_fresh_observation_forwards_exact_selected_protocol_and_owner_identity(
    generated,
):
    T, f, issuer, state, calls = generated
    state["memory_effects_protocol"] = "gemmini.whole_program_memory_effects.v1"
    state["memory_effects_identity"] = "neutral independently derived memory"

    def constructor(**kwargs):
        from types import SimpleNamespace

        calls.append(kwargs)
        return SimpleNamespace(
            _memory_effects_identity=state["memory_effects_identity"]
        )

    state["owner_type"] = constructor
    T.observe_current_source(
        issuer,
        native_result=f.result,
        command_buffer=f.cb,
        lowered=f.lowered,
        work_root=f.root / "trace",
        remaining_s=10,
    )
    assert (
        calls[0]["source_memory_effects_protocol"] == state["memory_effects_protocol"]
    )


def test_fresh_observation_refuses_changed_memory_derivation(generated):
    from types import SimpleNamespace

    T, f, issuer, state, _ = generated
    state["memory_effects_protocol"] = "gemmini.whole_program_memory_effects.v1"
    state["memory_effects_identity"] = "original"
    state["owner_type"] = lambda **_: SimpleNamespace(
        _memory_effects_identity="foreign"
    )
    with pytest.raises(ValueError, match="changed selected memory-effects"):
        T.observe_current_source(
            issuer,
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "trace",
            remaining_s=10,
        )


@pytest.mark.parametrize(
    "changed_module", ["gemmini_load_streams", "merlin.targetgen.riscv_memory_effects"]
)
def test_factory_source_closure_is_selected_before_any_privileged_callback(
    tmp_path, monkeypatch, changed_module
):
    F = importlib.import_module("gemmini_generated_source_factory")
    W = importlib.import_module("gemmini_whole_program_memory_effects")
    paths = {}

    def source(name):
        if name not in paths:
            paths[name] = tmp_path / (name + ".py")
            paths[name].write_text("# neutral complete public helper\n")
        return paths[name]

    monkeypatch.setattr(F, "module_source_path", source)
    old = F.original_factory({}).record()
    assert len(old["sources"]) == 3
    selected = F.original_factory({"source_memory_effects_protocol": W.PROTOCOL})
    assert {p["path"] for p in selected.record()["sources"]} == {
        str(source(name))
        for name in (
            "gemmini_generated_source_diagnostic",
            "gemmini_generated_source_trace",
            "merlin.targetgen.riscv_memory_effects",
            "gemmini_whole_program_memory_effects",
            "gemmini_load_streams",
            "gemmini_zero_fill",
            "gemmini_private_zero_storage",
        )
    }
    calls = []
    monkeypatch.setattr(
        F, "_original_restore", lambda *a, **k: calls.append("forbidden")
    )
    source(changed_module).write_text("# changed helper before create\n")
    with pytest.raises(ValueError, match="factory or inputs changed"):
        selected.create(work_root=tmp_path / "new")
    assert calls == []
