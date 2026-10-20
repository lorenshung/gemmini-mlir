"""Synthetic boundary controls only; no target, realization or timing qualification.

The selected-input reader and corpus binding are original implementations.
Demand and writer doubles exercise transport/refusal, never evidence admission.
"""

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(
    os.environ.get(
        "MERLIN_SOURCE_ADMISSION_TEST_ROOT", Path(__file__).resolve().parents[2]
    )
)
sys.path.insert(0, str(ROOT / "experiments/public_source_admission"))
sys.path.insert(0, str(ROOT / "merlin-support/backend"))
INPUTS = importlib.import_module("source_admission_inputs")
P = importlib.import_module("source_admission_publication")
S = importlib.import_module("gemmini_selected_scheduling")


def pin(path):
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return pin(path)["sha256"]


def declaration():
    return {
        "schema": INPUTS.SCHEMA,
        "readiness": {},
        "job_plan": {},
        "source": {key: None for key in INPUTS.SOURCE_FIELDS} | {"ordinal": 0},
        "roles": {"profile": {}, "private_exclusion": {}},
        "compiler": {
            "root": "/unused",
            "inventory": {},
            "sha256": "0" * 64,
            "python": {"launcher": "/unused", "binary": {}, "venv_config": {}},
        },
        "factory_inputs": {},
        "execution_contract": {},
        "purpose": {},
    }


def test_v1_closed_declaration_is_preserved(tmp_path):
    value = declaration()
    path = tmp_path / "declaration.json"
    dump(path, value)
    result, _ = INPUTS.read_declaration(path, pin(path)["sha256"])
    assert result == value
    assert not P.selected(result)


def test_v2_requires_explicit_closed_target_publication(tmp_path):
    value = declaration()
    value.update(
        schema=INPUTS.TARGET_SCHEMA, publication={"scope": "selected_target_v1"}
    )
    path = tmp_path / "declaration.json"
    dump(path, value)
    result, _ = INPUTS.read_declaration(path, pin(path)["sha256"])
    assert result == value and P.selected(result)


@pytest.mark.parametrize(
    "schema,publication",
    [
        (INPUTS.SCHEMA, {"scope": "selected_target_v1"}),
        (INPUTS.TARGET_SCHEMA, None),
        (INPUTS.TARGET_SCHEMA, {"scope": "other"}),
        (INPUTS.TARGET_SCHEMA, {"scope": True}),
        (INPUTS.TARGET_SCHEMA, {"scope": "selected_target_v1", "qualified": True}),
        (INPUTS.TARGET_SCHEMA, ["selected_target_v1"]),
        ("foreign", {"scope": "selected_target_v1"}),
        ([], {"scope": "selected_target_v1"}),
    ],
)
def test_schema_cannot_smuggle_or_imply_publication(tmp_path, schema, publication):
    value = declaration()
    value["schema"] = schema
    if publication is not None:
        value["publication"] = publication
    path = tmp_path / "declaration.json"
    dump(path, value)
    with pytest.raises(ValueError):
        INPUTS.read_declaration(path, pin(path)["sha256"])


@pytest.fixture
def case(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as CS
    from merlin.targetgen.corpus_spec import CorpusBinding

    facts, observed, operands = [
        tmp_path / name for name in ("facts.json", "observed.json", "operands.json")
    ]
    fir = "1" * 64
    facts_sha = dump(
        facts,
        {
            "inputs": {"fir_sha256": fir},
            "source_consistency": {"status": "verified", "config": "neutral-test"},
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
    operand_sha = dump(
        operands,
        {
            "target": "gemmini",
            "original_facts_sha256": facts_sha,
            "layouts": {"source_sha256": fir, "word_bits": 64},
            "protocol": {"encoding": {"semantic_class": {"9": "NEUTRAL_MAC"}}},
        },
    )
    dump(
        observed,
        {
            "name": "gemmini",
            "rtl_source": {"artifact_sha256": {"fir": fir}, "config": "neutral-test"},
            "rocc_operand_selection": {
                "sidecar": {"path": str(operands), "sha256": operand_sha},
                "original_facts": {"path": str(facts), "sha256": facts_sha},
            },
        },
    )
    contract = tmp_path / "execution-contract.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "name": "gemmini",
                "compute_units": [
                    {
                        "name": "neutral",
                        "kind": "systolic",
                        "dtypes": ["int8"],
                        "accumulate": [{"acc": "i32"}],
                    }
                ],
                "encoding": {
                    "corpus_required_classes": ["NEUTRAL_MAC"],
                    "class_family": {"NEUTRAL_MAC": "contraction"},
                    "semantic_class": {"9": "NEUTRAL_MAC"},
                },
            }
        )
    )
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {"target": "gemmini", "hardware_spec": {"target_contract": str(contract)}}
        )
    )
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    program = (
        "module { /* synthetic source transport fixture, not an arithmetic proof */ }\n"
    )
    (diagnostic / "capsule.interface.mlir").write_text(program)
    cap = {
        "name": "neutral_source",
        "kind": "model",
        "inputs": [{"name": "A", "dtype": "i8", "shape": [3, 2]}],
        "operation": {
            "op": "model",
            "attributes": {"arg_order": ["A", "Y0", "Y1"], "outs": ["Y0", "Y1"]},
        },
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "reference_closure": {"golden.json": {"sha256": "2" * 64, "bytes": 10}},
        "public_source_role": {"selected_member": {"role": "functional_guard"}},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "expected": {"instruction_classes": ["NEUTRAL_MAC"]},
        "semantic": {"must_accelerate": True, "semantic_family": "contraction"},
    }
    (diagnostic / "capsule.yaml").write_text(yaml.safe_dump(cap))
    source_binding = CorpusBinding(
        "source_only", None, "int8", "i32", True, ["L0"], "exact_int", 0.0, 0.0
    )
    value = {
        "schema": INPUTS.TARGET_SCHEMA,
        "publication": {"scope": "selected_target_v1"},
        "execution_contract": pin(contract),
    }
    selected = {
        "factory_inputs": {
            "facts": str(facts),
            "observed_contract": str(observed),
            "execution_contract": str(contract),
        }
    }
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract))
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))

    # Explicit neutral transport fixture: no source/realization owner is
    # supplied, no original writer qualification is represented by this double.
    def demand(text, binding):
        assert text == program
        assert binding.target == "gemmini"
        return "contraction", binding.classes_for(op="matmul", output_dtype="i32")

    monkeypatch.setattr(CS, "model_accelerator_demand", demand)
    return dict(
        facts=facts,
        observed=observed,
        operands=operands,
        contract=contract,
        descriptor=descriptor,
        diagnostic=diagnostic,
        cap=cap,
        program=program,
        binding=source_binding,
        declaration=value,
        selected=selected,
        intake=INPUTS.Intake(),
    )


def context(case):
    return P.target_binding(
        case["declaration"],
        case["selected"],
        case["binding"],
        case["diagnostic"],
        case["intake"],
    )


def test_v1_context_keeps_source_binding_and_does_not_derive_target(case, monkeypatch):
    from merlin.targetgen import capsule_source as CS

    case["declaration"] = {"schema": INPUTS.SCHEMA}
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT")
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT")
    monkeypatch.setattr(
        CS,
        "model_accelerator_demand",
        lambda *args: pytest.fail("V1 derived target demand"),
    )
    with context(case) as (binding, evidence):
        assert binding is case["binding"] and evidence is None


def test_v2_derives_original_binding_under_exact_selected_facts(case):
    with context(case) as (binding, evidence):
        assert binding.target == "gemmini" and binding.tile_dim == 3
        assert binding.integer and binding.accum_dtype == "i32"
        assert binding.compare == case["binding"].compare
        assert evidence["expected_instruction_classes"] == ["NEUTRAL_MAC"]
        assert evidence["required_oracle_tiers"] == ["L0", "L1", "L2", "L3"]
        assert evidence["contract_sha256"] == pin(case["contract"])["sha256"]
        assert evidence["qualification_issued"] is False


@pytest.mark.parametrize(
    "kind", ["ambient", "factory", "descriptor", "wrong_target", "stale_pin"]
)
def test_foreign_target_contract_selection_refuses(case, monkeypatch, kind):
    if kind == "ambient":
        monkeypatch.setenv(
            "MERLIN_TARGET_CONTRACT", str(case["contract"].with_name("foreign.yaml"))
        )
    elif kind == "factory":
        case["selected"]["factory_inputs"]["execution_contract"] = "/foreign"
    elif kind in {"descriptor", "wrong_target"}:
        d = yaml.safe_load(case["descriptor"].read_bytes())
        if kind == "descriptor":
            d["hardware_spec"]["target_contract"] = "/foreign"
        else:
            d["target"] = "foreign"
        case["descriptor"].write_text(yaml.safe_dump(d))
    else:
        case["contract"].write_text(case["contract"].read_text() + "# changed\n")
    with pytest.raises(ValueError):
        with context(case):
            pytest.fail("foreign target selection entered publication")


@pytest.mark.parametrize("name", ["facts", "observed", "operands", "contract"])
def test_selected_sources_mutated_during_publication_refuse(case, name):
    with pytest.raises(ValueError, match="changed"):
        with context(case):
            path = case[name]
            path.write_bytes(path.read_bytes() + b"\n")


def test_wrong_facts_cannot_reuse_original_operand_binding(case):
    changed = json.loads(case["facts"].read_text())
    changed["facts"]["arrays"][0]["rows"] = 99
    dump(case["facts"], changed)
    with pytest.raises(ValueError, match="original-facts binding"):
        with context(case):
            pytest.fail("different facts reused original sidecar")


@pytest.mark.parametrize(
    "answer", [(None, []), ("contraction", []), (None, ["NEUTRAL_MAC"])]
)
def test_empty_or_incomplete_derived_demand_is_not_fabricated(
    case, monkeypatch, answer
):
    from merlin.targetgen import capsule_source as CS

    monkeypatch.setattr(CS, "model_accelerator_demand", lambda *args: answer)
    with pytest.raises(ValueError, match="no derived accelerator"):
        with context(case):
            pytest.fail("missing demand entered publication")


@pytest.mark.parametrize("accumulator", ["int8", "f32"])
def test_accumulator_change_refuses_before_source_publication(case, accumulator):
    d = yaml.safe_load(case["contract"].read_text())
    d["compute_units"][0]["accumulate"] = [{"acc": accumulator}]
    case["contract"].write_text(yaml.safe_dump(d))
    case["declaration"]["execution_contract"] = pin(case["contract"])
    with pytest.raises(ValueError, match="accumulator"):
        with context(case):
            pytest.fail("changed accumulator entered publication")


def install_writer(case, monkeypatch, change=None, policy_change=None):
    from merlin.targetgen import native_model_policy as N
    from merlin_experiments.phase0 import writer

    calls = []

    def write(entry, binding, root):
        destination = root / "member"
        shutil.copytree(case["diagnostic"], destination)
        cap = yaml.safe_load((destination / "capsule.yaml").read_bytes())
        cap["semantic"] = {"must_accelerate": True, "semantic_family": "contraction"}
        if change:
            change(cap, destination)
        (destination / "capsule.yaml").write_text(yaml.safe_dump(cap))
        calls.append((entry, binding))
        return destination

    def policy(directory, *, target):
        assert target == "gemmini" and directory.name == "member"
        value = {
            "expected_instruction_classes": ["NEUTRAL_MAC"],
            "required_tiers": ["L0", "L1", "L2", "L3"],
        }
        if policy_change:
            policy_change(value)
        return value

    monkeypatch.setattr(writer, "_write_capsule", write)
    monkeypatch.setattr(N, "_frozen_model_policy", policy)
    return calls


def publish(case, tmp_path):
    return P.write(
        {"neutral_unit_fixture": True},
        case["binding"],
        tmp_path / "published",
        declaration=case["declaration"],
        selected_inputs=case["selected"],
        diagnostic=case["diagnostic"],
        intake=case["intake"],
    )


def test_write_preserves_complete_source_metadata_and_uses_target_binding(
    case, monkeypatch, tmp_path
):
    calls = install_writer(case, monkeypatch)
    original = {p.name: p.read_bytes() for p in case["diagnostic"].iterdir()}
    directory = publish(case, tmp_path)
    assert yaml.safe_load((directory / "capsule.yaml").read_bytes()) == case["cap"]
    assert len(calls) == 1 and calls[0][1].target == "gemmini"
    assert {p.name: p.read_bytes() for p in case["diagnostic"].iterdir()} == original


def test_target_publication_can_strengthen_source_semantics_without_relabelling_source(
    case, monkeypatch, tmp_path
):
    cap = yaml.safe_load((case["diagnostic"] / "capsule.yaml").read_bytes())
    cap["semantic"] = {"must_accelerate": False, "semantic_family": None}
    (case["diagnostic"] / "capsule.yaml").write_text(yaml.safe_dump(cap))
    install_writer(case, monkeypatch)
    directory = publish(case, tmp_path)
    actual = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    assert actual["semantic"] == {
        "must_accelerate": True,
        "semantic_family": "contraction",
    }
    assert yaml.safe_load((case["diagnostic"] / "capsule.yaml").read_bytes()) == cap


@pytest.mark.parametrize(
    "change",
    [
        lambda cap, dst: cap["inputs"][0].update(shape=[1, 6]),
        lambda cap, dst: cap.update(
            numeric_policy={"compare": "tolerance_float", "dtype": "f32"}
        ),
        lambda cap, dst: cap["operation"]["attributes"].update(
            arg_order=["A", "Y1", "Y0"]
        ),
        lambda cap, dst: cap["operation"]["attributes"].update(outs=["Y1", "Y0"]),
        lambda cap, dst: cap["operation"].update(op="foreign"),
        lambda cap, dst: cap.update(reference_closure={}),
        lambda cap, dst: cap.pop("reference_closure"),
        lambda cap, dst: cap.update(
            public_source_role={"selected_member": {"role": "objective"}}
        ),
        lambda cap, dst: cap.update(performance={"member_class": "OBJECTIVE"}),
        lambda cap, dst: (dst / "capsule.interface.mlir").write_text(
            "changed source\n"
        ),
    ],
)
def test_write_refuses_complete_source_abi_numeric_reference_or_role_drift(
    case, monkeypatch, tmp_path, change
):
    install_writer(case, monkeypatch, change=change)
    with pytest.raises(ValueError, match="complete source, ABI or numerical"):
        publish(case, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda policy: policy.update(expected_instruction_classes=[]),
        lambda policy: policy.update(
            expected_instruction_classes=["NEUTRAL_MAC", "invented"]
        ),
        lambda policy: policy.update(required_tiers=["L0", "L1", "L2"]),
        lambda policy: policy.update(required_tiers=["L0", "L1", "L3"]),
    ],
)
def test_write_cannot_drop_or_change_derived_instruction_and_tier_obligations(
    case, monkeypatch, tmp_path, change
):
    install_writer(case, monkeypatch, policy_change=change)
    with pytest.raises(ValueError, match="instruction or oracle obligations"):
        publish(case, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda cap, dst: cap["semantic"].update(must_accelerate=False),
        lambda cap, dst: cap["semantic"].update(must_accelerate=1),
        lambda cap, dst: cap["semantic"].update(semantic_family="movement"),
        lambda cap, dst: cap.pop("semantic"),
    ],
)
def test_required_classes_do_not_substitute_for_selected_acceleration_semantics(
    case, monkeypatch, tmp_path, change
):
    install_writer(case, monkeypatch, change=change)
    with pytest.raises(ValueError, match="obligations|semantic|accelerat"):
        publish(case, tmp_path)


def test_original_body_refusal_survives_final_revalidation_refusal(case):
    with pytest.raises(RuntimeError, match="original body refusal") as raised:
        with context(case):
            case["contract"].write_text(case["contract"].read_text() + "# changed\n")
            raise RuntimeError("original body refusal")
    assert any("changed" in note for note in getattr(raised.value, "__notes__", ()))
