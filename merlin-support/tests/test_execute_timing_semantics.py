"""Synthetic mutation controls issue no hardware or timing qualification."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_execute_timing_semantics")


@pytest.fixture
def selection(adapter):
    original = {
        "libraries": ["a" * 64, "b" * 64],
        "target": {"prices": {"load": 7}, "scheduling": {"fenced": 1}},
        "memory": {"banks": 3},
    }
    candidate = copy.deepcopy(original)
    candidate["libraries"][1] = "c" * 64
    candidate["target"]["execution_service"] = {
        "version": 1,
        "dimension": 8,
        "minimum_rows": 2,
        "reduction_guard_mask": 1 << 24,
        "pipeline_cycles": 5.0,
        "cycles_per_row": 1,
        "source_fact_sha256": "d" * 64,
    }
    control = dict(zip(adapter.CONTROL, (1, 1, 0, 0, 0)))
    proof = {
        "dimension": 8,
        "minimum_reduced_input_rows": 2,
        "reduction_guard_cases": [dict(control, reduced=True)],
    }
    return original, candidate, proof


def test_checked_variant_preserves_original_bytes(adapter, selection):
    before = copy.deepcopy(selection)
    adapter.verify_variant(*selection, fact_sha256="d" * 64)
    assert before == selection


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("dimension", 16),
        ("minimum_rows", 4),
        ("reduction_guard_mask", 1 << 25),
        ("source_fact_sha256", "e" * 64),
        ("cycles_per_row", True),
        ("cycles_per_row", 0.5),
        ("pipeline_cycles", -1),
        ("pipeline_cycles", float("nan")),
        ("pipeline_cycles", float("inf")),
    ],
)
def test_unproven_input_service_refused(adapter, selection, field, value):
    selection[1]["target"]["execution_service"][field] = value
    with pytest.raises(ValueError):
        adapter.verify_variant(*selection, fact_sha256="d" * 64)


@pytest.mark.parametrize(
    "change", ["library", "memory", "scheduling", "price", "extra", "missing"]
)
def test_functional_or_evaluator_change_cannot_hide_in_variant(
    adapter, selection, change
):
    model = selection[1]
    if change == "library":
        model["libraries"][0] = "f" * 64
    elif change == "memory":
        model["memory"]["banks"] += 1
    elif change == "scheduling":
        model["target"]["scheduling"]["fenced"] = 0
    elif change == "price":
        model["target"]["prices"]["load"] = 0
    elif change == "extra":
        model["target"]["execution_service"]["override"] = 1
    else:
        del model["target"]["execution_service"]["minimum_rows"]
    with pytest.raises(ValueError):
        adapter.verify_variant(*selection, fact_sha256="d" * 64)


def test_serialized_verdicts_cannot_construct_original_owner(adapter):
    with pytest.raises(TypeError):
        adapter.ExecuteServiceTimingOwner({"qualified": True})
    with pytest.raises(TypeError, match="original live"):
        adapter.ExecuteServiceTimingOwner.bind(
            object(),
            original_native_owner=object(),
            source_root=None,
            spike_source=None,
            provider_bundle=None,
            source_fact=None,
            rebuild_dir=None,
        )
