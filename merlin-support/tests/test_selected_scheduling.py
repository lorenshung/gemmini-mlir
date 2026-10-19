"""Selected-input admission refuses unsupported shared device mechanisms."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def adapter():
    path = (
        Path(__file__).resolve().parents[1] / "backend/gemmini_selected_scheduling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "selected_scheduling_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(path, document):
    path.write_text(json.dumps(document, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, adapter):
    facts = tmp_path / "facts.json"
    contract = tmp_path / "contract.json"
    operand = tmp_path / "operands.json"
    fir = "1" * 64
    facts_digest = write(
        facts,
        {
            "inputs": {"fir_sha256": fir},
            "source_consistency": {"status": "verified", "config": "selected-build"},
            "facts": {
                "target": "gemmini",
                "arrays": [{"name": "mesh", "rows": 3, "cols": 5}],
                "interfaces": [
                    {
                        "name": "funct_decode_table",
                        "custom_opcode": 11,
                        "legal_funct": [9],
                    }
                ],
                "memory_address_spaces": [
                    {"name": "scratchpad", "banks": 3, "depth": 101, "bytes": 1212},
                    {"name": "accumulator", "banks": 2, "depth": 23, "bytes": 920},
                ],
            },
        },
    )
    digest = write(
        operand,
        {
            "target": "gemmini",
            "original_facts_sha256": facts_digest,
            "layouts": {"source_sha256": fir, "word_bits": 64},
            "protocol": {"encoding": {"semantic_class": {"9": "CONFIG"}}},
        },
    )
    write(
        contract,
        {
            "name": "gemmini",
            "rtl_source": {"artifact_sha256": {"fir": fir}, "config": "selected-build"},
            "rocc_operand_selection": {
                "sidecar": {"path": str(operand), "sha256": digest},
                "original_facts": {"path": str(facts), "sha256": facts_digest},
            },
        },
    )
    return adapter.SelectedSchedulingTarget.read(facts, contract)


def test_selected_geometry_preserves_nondefault_dimensions_without_execution_claim(
    selected,
):
    report = selected.report()
    assert report["geometry"]["array_rows"] == 3
    assert report["geometry"]["array_cols"] == 5
    assert report["geometry"]["stores"][0]["total_rows"] == 303
    assert report["geometry"]["stores"][1]["row_bytes"] == 20
    assert report["enabled_mechanisms"] == []
    assert report["execution_qualification"] == "not_established"
    assert report["timing"] == "UNKNOWN"
    assert len(report["mechanisms"]) == 8


@pytest.mark.parametrize(
    "mechanism",
    [
        "dense_blocking_reuse",
        "temporary_storage_copy_elimination",
        "readout_fusion",
        "configuration_cleanup",
        "device_spills",
        "complete_row_reduction",
        "final_program_limits",
        "asynchronous_overlap",
    ],
)
def test_observed_dimensions_and_names_never_enable_effectful_pass(
    selected, adapter, mechanism
):
    with pytest.raises(
        adapter.SchedulingUnavailable, match="independent execution qualification"
    ):
        selected.require(mechanism)


@pytest.mark.parametrize("key", ["facts_path", "contract_path", "operand_path"])
def test_input_mutation_expires_existing_selection(selected, key):
    getattr(selected, key).write_text("{}")
    with pytest.raises(ValueError, match="inputs changed"):
        selected.report()


def test_sidecar_digest_is_checked_before_new_selection(selected, adapter):
    selected.operand_path.write_text("{}")
    with pytest.raises(ValueError, match="sidecar digest differs"):
        adapter.SelectedSchedulingTarget.read(
            selected.facts_path, selected.contract_path
        )


def test_exact_facts_cannot_be_replaced_with_same_firrtl_label(selected, adapter):
    facts = json.loads(selected.facts_path.read_text())
    facts["facts"]["arrays"][0]["rows"] = 99
    write(selected.facts_path, facts)
    with pytest.raises(ValueError, match="original-facts binding differs"):
        adapter.SelectedSchedulingTarget.read(
            selected.facts_path, selected.contract_path
        )


def test_configuration_label_cannot_replace_hardware_identity(selected, adapter):
    contract = json.loads(selected.contract_path.read_text())
    contract["rtl_source"]["artifact_sha256"]["fir"] = "2" * 64
    write(selected.contract_path, contract)
    with pytest.raises(ValueError, match="FIRRTL identities differ"):
        adapter.SelectedSchedulingTarget.read(
            selected.facts_path, selected.contract_path
        )


def test_unknown_mechanism_is_not_silently_added(selected):
    with pytest.raises(ValueError, match="unknown shared scheduling mechanism"):
        selected.require("new_magic_pass")


def test_cli_requirement_refuses_but_retains_machine_readable_report(
    selected, adapter, tmp_path, monkeypatch
):
    output = tmp_path / "readiness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected-scheduling",
            "--facts",
            str(selected.facts_path),
            "--observed-contract",
            str(selected.contract_path),
            "--output",
            str(output),
            "--require",
            "configuration_cleanup",
        ],
    )
    assert adapter.main() == 2
    assert json.loads(output.read_text())["enabled_mechanisms"] == []


def test_source_check_cli_consumes_observer_without_qualifying_device(
    selected, adapter, tmp_path, monkeypatch
):
    output = tmp_path / "source-check.json"
    checked = []
    monkeypatch.setattr(
        type(selected), "operand_observer", lambda self: checked.append(self)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected-scheduling",
            "--facts",
            str(selected.facts_path),
            "--observed-contract",
            str(selected.contract_path),
            "--output",
            str(output),
            "--check-operands",
        ],
    )
    assert adapter.main() == 0 and len(checked) == 1
    report = json.loads(output.read_text())
    assert report["operand_source_check"]["status"] == "source_rederived"
    assert report["execution_qualification"] == "not_established"


@pytest.mark.parametrize("usage", [0, 3, 7])
def test_actual_instruction_usage_bits_are_not_an_identity_filter(
    selected, monkeypatch, usage
):
    class Observer:
        def isa_constants(self, target):
            assert target == "gemmini"
            return {"selected": True}

        def decode_instruction(self, function, rs1, rs2, isa):
            assert (function, rs1, rs2, isa) == (
                9,
                {"raw": 17, "kind": "const"},
                {"raw": 29, "kind": "const"},
                {"selected": True},
            )
            return "CONFIG_LD", {"state_id": 2}

    monkeypatch.setattr(type(selected), "operand_observer", lambda self: Observer())
    result = selected.decode_observation((9 << 25) | (usage << 12) | 11, 17, 29)
    assert result["register_usage"] == usage
    assert result["decoded"] == {"state_id": 2}
    assert result["effects"] == result["completion"] == "UNKNOWN"


@pytest.mark.parametrize(
    "word,rs1,rs2,match",
    [
        ((9 << 25) | 12, 0, 0, "custom opcode"),
        ((8 << 25) | 11, 0, 0, "observed roster"),
        ((9 << 25) | 11, -1, 0, "expected integer"),
        ((9 << 25) | 11, 0, 1 << 64, "selected width"),
        (1 << 32, 0, 0, "architectural width"),
    ],
)
def test_bad_observations_refuse_before_observer_or_execution(
    selected, monkeypatch, word, rs1, rs2, match
):
    monkeypatch.setattr(
        type(selected),
        "operand_observer",
        lambda self: pytest.fail("observer must not run"),
    )
    with pytest.raises(ValueError, match=match):
        selected.decode_observation(word, rs1, rs2)
