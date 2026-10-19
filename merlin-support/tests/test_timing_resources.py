"""Structural timing facts must not fill unknown operation or memory prices."""

import copy
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    spec = importlib.util.spec_from_file_location(
        "timing_resources_under_test", backend / "gemmini_timing_resources.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def module(adapter, body):
    return adapter.F.parse_module(
        "Routing", list(enumerate(("module Routing :\n" + body).splitlines(), 1))
    )


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_boolean_cone_is_derived_from_actual_selected_expression(adapter, value):
    source = module(
        adapter,
        "  node config = eq(funct, UInt<7>(0))\n"
        "  node kind = bits(rs1, 1, 0)\n"
        "  node load = and(config, eq(kind, UInt<2>(1)))\n"
        "  node execute = and(config, eq(kind, UInt<2>(0)))\n"
        "  node store = and(config, or(eq(kind, UInt<2>(2)), eq(kind, UInt<2>(3))))\n"
        "  wire selected : UInt<1>\n"
        "  when valid :\n"
        "    connect selected, or(load, store)\n",
    )
    env = {
        adapter.F.Ref("rs1"): value,
        adapter.F.Ref("funct"): 0,
        adapter.F.Ref("valid"): 1,
    }
    assert adapter._value(source, adapter.F.Ref("selected"), env) == int(value != 0)
    assert adapter._value(source, adapter.F.Ref("execute"), env) == int(value == 0)
    env[adapter.F.Ref("funct")] = 9
    assert adapter._value(source, adapter.F.Ref("selected"), env) == 0


@pytest.mark.parametrize("positive,value", [(True, 0), (False, 1), (True, 2)])
def test_inactive_or_nonboolean_driver_guard_refuses(adapter, positive, value):
    source = adapter.F.ModuleBody("Routing", 1, None)
    source.connects.append(
        adapter.F.Connect(
            adapter.F.Ref("result"),
            adapter.F.Lit("UInt", 1, 1),
            (adapter.F.Guard(adapter.F.Ref("guard"), positive),),
            2,
            None,
        )
    )
    with pytest.raises(ValueError, match="not active"):
        adapter._value(source, adapter.F.Ref("result"), {adapter.F.Ref("guard"): value})


@pytest.mark.parametrize(
    "body,expression",
    [
        ("  node a = b\n  node b = a\n", "a"),
        ("  node a = add(UInt<1>(0), UInt<1>(1))\n", "a"),
        ("  wire a : UInt<1>\n", "a"),
        ("  wire a : UInt<1>\n  connect a, UInt<1>(0)\n  connect a, UInt<1>(1)\n", "a"),
        ("  wire a : UInt<1>\n  when unknown :\n    connect a, UInt<1>(1)\n", "a"),
    ],
)
def test_unproven_routing_never_becomes_a_fact(adapter, body, expression):
    with pytest.raises(ValueError):
        adapter._value(module(adapter, body), adapter.F.Ref(expression), {})


@pytest.fixture
def model():
    return {
        "schema": "merlin.annotated_spike_model.v1",
        "resources": [
            {"name": "accelerator.load", "id": 7, "capacity": 2, "queue_capacity": 6}
        ],
        "parameters": {"memory.l1d_hit_cycles": 3},
        "target": {"prices": {"load.base": 11}, "hardware": {"dim": 16}},
        "domain": {"memory_regime_sha256": "retained-exactly"},
        "memory": {"caches": {"l1d": {"sets": 64}}},
    }


@pytest.fixture
def evidence():
    return {
        "schema": "gemmini.structural_timing_resources.v1",
        "native_safe_resources": [
            {"name": "cpu.issue", "capacity": 1, "queue_capacity": 0},
            {"name": "cpu.integer_alu", "capacity": 1, "queue_capacity": 0},
        ],
        "cpu_execution_resources": {"cpu.load": "cpu.integer_alu"},
    }


def test_augmentation_preserves_original_domains_prices_and_unknowns(
    adapter, model, evidence
):
    original = copy.deepcopy(model)
    result = adapter.augment_model(model, evidence, harts=1)
    assert model == original
    for key in ("schema", "domain", "memory", "target"):
        assert result[key] == model[key]
    assert result["parameters"] == {"memory.l1d_hit_cycles": 3, "cpu.issue_cycles": 1}
    assert result["resources"][0] == model["resources"][0]
    assert [row["id"] for row in result["resources"]] == [7, 8, 9]
    assert result["cpu_execution_resources"] == {"cpu.load": "cpu.integer_alu"}
    assert adapter.augment_model(result, evidence, harts=1) == result


@pytest.mark.parametrize("change", ["resource", "issue", "assignment", "harts"])
def test_conflicting_resource_or_price_is_not_silently_overwritten(
    adapter, model, evidence, change
):
    if change == "resource":
        model["resources"].append(
            {"name": "cpu.issue", "id": 8, "capacity": 2, "queue_capacity": 0}
        )
    elif change == "issue":
        model["parameters"]["cpu.issue_cycles"] = 2
    elif change == "assignment":
        model["cpu_execution_resources"] = {"cpu.load": "other"}
    original = copy.deepcopy(model)
    with pytest.raises(ValueError):
        adapter.augment_model(model, evidence, harts=2 if change == "harts" else 1)
    assert model == original


def test_source_mutation_invalidates_resource_evidence(adapter, tmp_path):
    source = tmp_path / "public-source.scala"
    source.write_text("selected-source")
    sources = {"selected": adapter._pin(source)}
    adapter._revalidate_sources(sources)
    source.write_text("different-source")
    with pytest.raises(ValueError, match="changed"):
        adapter._revalidate_sources(sources)


@pytest.mark.parametrize("mutate_input", [False, True])
def test_cli_preserves_model_file_and_refuses_concurrent_input_change(
    adapter, model, evidence, tmp_path, monkeypatch, mutate_input
):
    from merlin.common import cpu_pool

    original = tmp_path / "model.json"
    original.write_text(json.dumps(model))
    original_bytes = original.read_bytes()
    output = tmp_path / "candidate"
    selected = SimpleNamespace(revalidate=lambda: None)
    monkeypatch.setattr(
        adapter, "SelectedSchedulingTarget", SimpleNamespace(read=lambda *_: selected)
    )
    monkeypatch.setattr(cpu_pool, "shared_compute_pool", nullcontext)
    evidence["sources"] = {}
    monkeypatch.setattr(adapter, "extract", lambda *_, **__: evidence)
    arguments = ["resources"]
    for name in (
        "facts",
        "observed-contract",
        "rocket-core",
        "reservation-station",
        "execute-controller",
        "spike-source",
    ):
        arguments.extend(["--" + name, str(tmp_path / name)])
    arguments.extend(
        ["--input-model", str(original), "--output", str(output), "--harts", "1"]
    )
    monkeypatch.setattr(sys, "argv", arguments)
    if mutate_input:
        augment = adapter.augment_model

        def change_during_augmentation(*args, **kwargs):
            original.write_text("concurrently changed model")
            return augment(*args, **kwargs)

        monkeypatch.setattr(adapter, "augment_model", change_during_augmentation)
        with pytest.raises(ValueError, match="original native model changed"):
            adapter.main()
        assert not output.exists()
    else:
        adapter.main()
        assert original.read_bytes() == original_bytes
        candidate = json.loads((output / "model.json").read_text())
        assert candidate["target"] == model["target"]
        assert candidate["memory"] == model["memory"]
        assert candidate["domain"] == model["domain"]
        report = json.loads((output / "resources.json").read_text())
        assert report["original_model_sha256"] == adapter._pin(original)["sha256"]
