"""Correspondence controls; synthetic source never qualifies hardware/timing."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_count_hardware")


def test_receiver_types_and_literals_are_read_from_source(adapter):
    text = "#define DIM 23\n#define ACC_READ_FULL_WIDTH\ntypedef int16_t elem_t;\n"
    assert adapter._integer_macro(adapter._defines(text), "DIM") == 23
    assert adapter._dtype_bytes(text, "elem_t") == 2
    assert "ACC_READ_FULL_WIDTH" in adapter._defines(text)
    with pytest.raises(ValueError):
        adapter._integer_macro(adapter._defines("#define DIM (16+1)\n"), "DIM")
    with pytest.raises(ValueError):
        adapter._dtype_bytes("typedef uint8_t elem_t;", "elem_t")


def test_only_actual_reset_declaration_supplies_reset_value(adapter, tmp_path):
    fir = tmp_path / "selected.fir"
    fir.write_text(
        "circuit Selected :\n  module StoreController :\n    input clock : Clock\n    input reset : UInt<1>\n    regreset activation : UInt<3>, clock, reset, UInt<3>(0h2)\n"
    )
    assert adapter._reset(fir)[0] == 2
    fir.write_text(
        fir.read_text().replace(
            "regreset activation : UInt<3>, clock, reset, UInt<3>(0h2)",
            "reg activation : UInt<3>, clock",
        )
    )
    with pytest.raises(ValueError, match="reset is absent"):
        adapter._reset(fir)


@pytest.fixture
def binding(adapter, tmp_path):
    paths = {
        name: tmp_path / (name + ".json")
        for name in ("model", "facts", "contract", "runtime")
    }
    for name, path in paths.items():
        path.write_text(json.dumps({"synthetic": name}))
    paths["model"].write_text(
        json.dumps(
            {
                "target": {"hardware": {"dim": 7}},
                "domain": {"feature_schema_sha256": "f" * 64},
                "libraries": ["e" * 64],
            }
        )
    )
    row = {
        "schema": adapter.SCHEMA,
        "implementation": adapter._pin(adapter.__file__),
        "hardware_execution_qualification": "not_established",
        "native_event_timing_qualified": False,
        "hardware": {"dim": 7},
        "pins": {
            "model": adapter._pin(paths["model"]),
            "functional_extension": adapter._pin(paths["runtime"]),
        },
        "selected_inputs": {
            "facts": adapter._pin(paths["facts"]),
            "observed_contract": adapter._pin(paths["contract"]),
        },
    }
    receipt = tmp_path / "binding.json"
    receipt.write_text(json.dumps(row))
    return adapter.CountHardwareBinding.read(receipt), paths, receipt, row


def test_queue_binding_checks_all_original_inputs_and_preserves_model(
    adapter, binding, tmp_path
):
    owner, paths, _, _ = binding
    original = paths["model"].read_bytes()
    assert (
        owner.verify(
            model_path=paths["model"],
            facts_path=paths["facts"],
            observed_contract_path=paths["contract"],
            functional_extension_path=paths["runtime"],
        )
        == owner.identity
    )
    assert paths["model"].read_bytes() == original
    other = tmp_path / "different-facts.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="different count-model/target/runtime"):
        owner.verify(
            model_path=paths["model"],
            facts_path=other,
            observed_contract_path=paths["contract"],
            functional_extension_path=paths["runtime"],
        )


@pytest.mark.parametrize("kind", ["model", "facts", "contract", "runtime"])
def test_changed_original_binding_refuses_instead_of_reusing_screen(
    adapter, binding, kind
):
    owner, paths, _, _ = binding
    paths[kind].write_text("changed selection")
    with pytest.raises(ValueError, match="changed"):
        owner.revalidate()


def test_correspondence_cannot_declare_native_timing_qualification(adapter, binding):
    _, _, receipt, row = binding
    row["native_event_timing_qualified"] = True
    receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="unsupported authority"):
        adapter.CountHardwareBinding.read(receipt)


def test_live_owner_revalidates_original_paths_and_exposes_five_bindings(
    adapter, binding
):
    proof, paths, _, _ = binding
    owner = adapter.CountHardwareCorrespondenceOwner(
        proof, paths["model"], paths["facts"], paths["contract"], paths["runtime"]
    )
    owner.revalidate()
    assert set(owner.selected_binding()) == {
        "hardware",
        "software",
        "numerical_policy",
        "runtime",
        "library",
    }
    assert owner.record()["native_event_timing_qualified"] is False
    paths["runtime"].write_text("changed functional receiver")
    with pytest.raises(ValueError):
        owner.revalidate()


def test_live_owner_factory_does_not_trust_saved_correspondence_status(
    adapter, binding, monkeypatch
):
    _, paths, receipt, row = binding
    row["pins"].update(
        {
            name: adapter._pin(paths["facts"])
            for name in ("deployment", "receiver_gemmini_params.h", "functional_rocc")
        }
    )
    receipt.write_text(json.dumps(row))
    monkeypatch.setattr(adapter.SelectedSchedulingTarget, "read", lambda *_: object())
    monkeypatch.setattr(adapter, "derive", lambda **_: {**row, "hardware": {"dim": 8}})
    with pytest.raises(ValueError, match="fresh source derivation"):
        adapter.CountHardwareCorrespondenceOwner.bind(
            receipt,
            model_path=paths["model"],
            facts_path=paths["facts"],
            observed_contract_path=paths["contract"],
            functional_extension_path=paths["runtime"],
        )
