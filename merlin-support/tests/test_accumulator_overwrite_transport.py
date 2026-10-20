"""Explicit selector transport under neutral owner fixtures; no target execution."""

import importlib
from pathlib import Path

import pytest
from test_generated_source_diagnostic import issued  # noqa: F401


def direct(issued):  # noqa: F811
    _, _, inputs, *_ = issued
    return dict(
        inputs,
        source_binding_protocol="merlin.native_pointer_source.v1",
        execution_contract=inputs["facts"],
    )


@pytest.mark.parametrize("value", [True, {}, "v2", "gemmini.source_observation.v2"])
def test_unknown_protocol_refuses_before_any_new_owner(issued, value):  # noqa: F811
    M, *_ = issued
    with pytest.raises(ValueError, match="unknown accumulator"):
        M.GeneratedSourceDiagnosticIssuer(
            dict(direct(issued), source_accumulator_effects_protocol=value)
        )


def test_whole_program_default_is_not_widened(issued):  # noqa: F811
    M, _, inputs, *_ = issued
    E = importlib.import_module("gemmini_accumulator_overwrite")
    with pytest.raises(ValueError, match="explicit direct"):
        M.GeneratedSourceDiagnosticIssuer(
            dict(inputs, source_accumulator_effects_protocol=E.PROTOCOL)
        )
    with pytest.raises(ValueError, match="explicit direct"):
        importlib.import_module("gemmini_generated_source_factory").original_factory(
            dict(inputs, source_accumulator_effects_protocol=E.PROTOCOL)
        )


@pytest.fixture
def selected(issued, monkeypatch):  # noqa: F811
    M, _, *_ = issued
    E = importlib.import_module("gemmini_accumulator_overwrite")

    class Effect:
        identity = "neutral explicitly selected write effect"

        def __init__(self, selected, arithmetic):
            self.selected, self.arithmetic = selected, arithmetic

        def revalidate(self):
            pass

        def record(self):
            return {"protocol": E.PROTOCOL, "timing_authority": "NONE"}

    monkeypatch.setattr(E, "SelectedAccumulatorOverwrite", Effect)
    inputs = dict(direct(issued), source_accumulator_effects_protocol=E.PROTOCOL)
    owner = M.GeneratedSourceDiagnosticIssuer(inputs)
    return M, E, owner, inputs


def test_selected_effect_changes_original_binding_and_is_part_of_source_closure(
    selected,
    issued,  # noqa: F811
):
    M, E, owner, inputs = selected
    legacy = M.GeneratedSourceDiagnosticIssuer(direct(issued))
    assert owner.selected_binding().runtime != legacy.selected_binding().runtime
    assert owner.record()["source_accumulator_effects_protocol"] == E.PROTOCOL
    assert "gemmini_accumulator_overwrite" in owner.record()["sources"]
    F = importlib.import_module("gemmini_generated_source_factory")
    factory = F.original_factory(inputs)
    paths = {Path(row["path"]).name for row in factory.record()["sources"]}
    assert {
        "gemmini_accumulator_overwrite.py",
        "gemmini_packet_contractions.py",
        "gemmini_source_packet_artifacts.py",
        "gemmini_direct_source_execution.py",
    } <= paths


@pytest.mark.parametrize(
    "fault", ["object", "identity", "protocol", "selected", "arithmetic"]
)
def test_effect_selection_and_same_owner_transport_cannot_mutate(selected, fault):
    M, _E, owner, _ = selected
    state = M._ISSUED[owner]
    if fault == "object":
        state["overwrite"] = {"status": "PASS"}
    elif fault == "identity":
        state["overwrite"].identity = "changed"
    elif fault == "protocol":
        state["overwrite_protocol"] = "changed"
    elif fault == "selected":
        state["overwrite"].selected = object()
    else:
        state["overwrite"].arithmetic = object()
    with pytest.raises((TypeError, ValueError)):
        owner.revalidate()
