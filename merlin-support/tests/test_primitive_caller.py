"""Original caller correspondence fixtures grant no target execution authority."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from gemmini_primitive_caller import check_original_caller
from gemmini_timing_resources import _pin
from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen import target_registry
from merlin.targetgen.contract.readback_policy import ReadbackPolicy


@pytest.fixture
def caller(tmp_path, monkeypatch):
    facts = {"source": "fixture-only"}
    contract = {
        "name": "gemmini",
        "rtl_source": {"artifact_sha256": {"fir": "fixture-only"}},
        "runner": {"backend": "chipyard_rocc", "chipyard_rocc": {}},
        "harness_abi": {
            "version": 2,
            "kind": "logical_pointer",
            "entry_symbol": "kernel_entry",
            "fence_symbol": "kernel_complete",
            "tensor_alignment": 64,
            "byte_order": "little",
            "main_convention": "void",
            "readback_transport": "out_b64_v1",
        },
    }
    effects = copy.deepcopy(contract)
    effects["additional_operand_source_proof"] = "another role"
    backend = BoundRoCCBackend(
        "gemmini", json.dumps({"contract": contract, "facts": facts}), None
    )
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    paths = {}
    for name, body in (("facts", facts), ("effects", effects), ("transport", contract)):
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(body))
    monkeypatch.setattr(
        target_registry,
        "resolve",
        lambda target: SimpleNamespace(
            capability_contract_path=paths["transport"],
            load_contract=lambda: json.loads(paths["transport"].read_bytes()),
        ),
    )
    selected = SimpleNamespace(
        revalidate=lambda: None,
        facts_path=paths["facts"],
        contract_path=paths["effects"],
        _documents=lambda: (facts, effects, {}),
    )
    cb = {
        "tensors": {
            "W": {"shape": [2, 2], "dtype": "i8", "role": "weight"},
            "A0": {"shape": [1, 2], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [1, 2], "dtype": "i32", "role": "output"},
        },
        "commands": [],
    }
    values = {
        "W": np.array([[-1, 2], [3, -2]], dtype=np.int8),
        "A0": np.array([[-3, 1]], dtype=np.int8),
    }
    harness = tmp_path / "harness.c"
    harness.write_text(
        backend.render_harness(
            cb,
            target="gemmini",
            inputs={k: v.tolist() for k, v in values.items()},
            readback_policy=ReadbackPolicy("out_b64_v1"),
            blobs={},
        )
    )
    arm = {
        "command_buffer": cb,
        "caller_order": list(cb["tensors"]),
        "harness": _pin(harness),
        "build_record": {
            "target": "gemmini",
            "readback_policy": ReadbackPolicy("out_b64_v1").record(),
            "source_pins": [_pin(p) for p in backend.build_source_paths()],
        },
    }
    return selected, arm, values, backend, effects, paths


def test_transport_and_arithmetic_effect_roles_remain_distinct(caller):
    selected, arm, values, _backend, _effects, paths = caller
    row = check_original_caller(selected, arm, values)
    assert row["binding"]["capability_contract"] == _pin(paths["transport"])
    assert row["binding"]["observed_effects"] != row["binding"]["capability_contract"]
    assert row["binding"]["original_pointer_order"] == ["W", "A0", "Y0"]


@pytest.mark.parametrize("key", ["runner", "rtl_source", "harness_abi"])
def test_foreign_runtime_hardware_or_abi_refuses(caller, key):
    selected, arm, values, _backend, effects, _paths = caller
    effects[key] = {"changed": True}
    with pytest.raises(ValueError, match="facts/runtime/ABI"):
        check_original_caller(selected, arm, values)


@pytest.mark.parametrize(
    "damage", ["inputs", "pointer", "harness", "source", "readback"]
)
def test_completed_original_caller_must_match_every_actual_byte_role(caller, damage):
    selected, arm, values, _backend, _effects, _paths = caller
    if damage == "inputs":
        values["A0"][0, 0] += 1
    elif damage == "pointer":
        arm["caller_order"] = ["A0", "W", "Y0"]
    elif damage == "harness":
        with open(arm["harness"]["path"], "a") as stream:
            stream.write("/* changed actual caller */")
    elif damage == "source":
        arm["build_record"]["source_pins"].pop()
    else:
        arm["build_record"]["readback_policy"] = {"transport": "digest"}
    with pytest.raises(ValueError, match="caller"):
        check_original_caller(selected, arm, values)


def test_wrapper_cannot_supply_an_arbitrary_renderer(caller, monkeypatch):
    selected, arm, values, _backend, _effects, _paths = caller
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace())
    with pytest.raises(ValueError, match="original data-bound"):
        check_original_caller(selected, arm, values)


def test_replaced_original_backend_method_refuses(caller, monkeypatch):
    selected, arm, values, _backend, _effects, _paths = caller
    monkeypatch.setattr(BoundRoCCBackend, "render_harness", lambda *args, **kwargs: "")
    with pytest.raises(ValueError, match="method was replaced"):
        check_original_caller(selected, arm, values)


def test_replaced_selected_facts_cannot_relabel_the_runtime(caller):
    selected, arm, values, _backend, _effects, _paths = caller
    selected._documents = lambda: ({"foreign": "facts"}, {}, {})
    with pytest.raises(ValueError, match="facts/runtime/ABI"):
        check_original_caller(selected, arm, values)


def test_original_registry_may_reopen_fresh_identical_bound_data(caller, monkeypatch):
    selected, arm, values, backend, _effects, _paths = caller
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: BoundRoCCBackend(
            backend.target, backend._selection_json, backend.rocc_semantics
        ),
    )
    assert (
        check_original_caller(selected, arm, values)["binding"]["original_harness"]
        == arm["harness"]
    )


def test_fresh_registry_wrapper_cannot_change_the_selected_data(caller, monkeypatch):
    selected, arm, values, backend, _effects, _paths = caller
    calls = 0

    def reopen(target):
        nonlocal calls
        calls += 1
        return backend if calls == 1 else BoundRoCCBackend("gemmini", "{}", None)

    monkeypatch.setattr(base, "get_backend", reopen)
    with pytest.raises(ValueError, match="bytes/selection changed"):
        check_original_caller(selected, arm, values)
