"""Primitive diagnostic admission refuses lost work and stale/missing roles.

These source/custody fixtures grant no target hardware or timing capability.
"""

import copy

import numpy as np
import pytest
from gemmini_primitive_context import _original_methods
from gemmini_primitive_diagnostic import PrimitiveDiagnosticOwner
from gemmini_primitive_grade import _diagnostic_grade
from gemmini_primitive_source import bind_primitive, check_integer_values
from gemmini_zero_initializer_capability import QualifiedZeroInitializerOwner
from merlin.targetgen.contract.interface_emit import emit_interface_mlir


def _source():
    cb = {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "W": {"shape": [4, 5], "dtype": "i8", "role": "weight"},
            "A0": {"shape": [3, 4], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [3, 5], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {
                "opcode": "RES_PACK",
                "operands": {"src": "W", "dst": "wr"},
                "attributes": {"layout": "packed_rhs"},
            },
            {
                "opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": "A0", "rhs": "wr", "dst": "a"},
                "attributes": {},
            },
            {
                "opcode": "COMMIT",
                "operands": {"src": "a", "dst": "Y0"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": "wr"}, "attributes": {}},
        ],
    }
    cap = {
        "semantic": {"must_accelerate": True},
        "expected": {
            "instruction_classes": [
                "LOAD_CMD",
                "PRELOAD_CMD",
                "COMPUTE_AND_FLIP_CMD",
                "STORE_CMD",
            ],
            "modes": {"relu": False, "acc_scale": False, "i8": False},
        },
        "name": "neutral_public_primitive",
        "kind": "layer",
        "label": "public",
        "operation": {
            "op": "matmul",
            "attributes": {
                "lhs": "A0",
                "weight": "W",
                "out": "Y0",
                "epilogue": [],
                "output_dtype": "i32",
            },
        },
        "inputs": [
            dict(name=name, **spec)
            for name, spec in cb["tensors"].items()
            if spec["role"] != "output"
        ],
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
    }
    interface_cb = dict(
        cb,
        tensors={
            name: spec
            for name, spec in cb["tensors"].items()
            if spec["role"] != "output"
        },
    )
    return cap, emit_interface_mlir(interface_cb), cb


def test_source_binder_reconstructs_all_required_products_without_model_attestation():
    cap, source, cb = _source()
    c = bind_primitive(cap, source, cb)
    assert (c.m, c.k, c.n, c.lhs, c.rhs, c.output) == (3, 4, 5, "A0", "W", "Y0")
    assert "global_program_plan" not in cb.get("params", {})


@pytest.mark.parametrize(
    "damage",
    [
        "extra_command",
        "lost_evict",
        "epilogue",
        "role",
        "shape",
        "dtype",
        "extra_output",
    ],
)
def test_current_compiler_cannot_change_source_work_or_numerics(damage):
    cap, source, cb = _source()
    if damage == "extra_command":
        cb["commands"].append({"opcode": "EVICT", "operands": {"handle": "wr"}})
    elif damage == "lost_evict":
        cb["commands"].pop()
    elif damage == "epilogue":
        cb["commands"][2]["attributes"]["epilogue"] = ["relu"]
    elif damage == "extra_output":
        cb["tensors"]["foreign"] = {"shape": [3, 5], "dtype": "i32", "role": "output"}
    else:
        cb["tensors"]["A0"][damage] = {
            "role": "weight",
            "shape": [3, 5],
            "dtype": "u8",
        }[damage]
    with pytest.raises(ValueError, match="primitive compiler"):
        bind_primitive(cap, source, cb)


@pytest.mark.parametrize(
    "damage",
    [
        "model",
        "private",
        "drop_l3",
        "float_policy",
        "source_comment",
        "source_wrong_output_type",
    ],
)
def test_semantic_selector_cannot_invent_original_admission(damage):
    cap, source, cb = _source()
    if damage == "model":
        cap["kind"] = "model"
    elif damage == "private":
        cap["label"] = "hidden"
    elif damage == "drop_l3":
        cap["required_oracle_tiers"].pop()
    elif damage == "float_policy":
        cap["numeric_policy"] = {"compare": "tolerance_float", "dtype": "i32"}
    elif damage == "source_comment":
        source += "// not the complete canonical source\n"
    else:
        source = source.replace("tensor<3x5xi32>", "tensor<3x5xi8>")
    with pytest.raises(ValueError):
        bind_primitive(cap, source, cb)


def _values():
    cap, source, cb = _source()
    c = bind_primitive(cap, source, cb)
    a = np.arange(12, dtype=np.int64).reshape(3, 4) % 7 - 3
    b = np.arange(20, dtype=np.int64).reshape(4, 5) % 5 - 2
    return (
        c,
        {"A0": a, "W": b},
        {"Y0": a @ b},
        {
            "operand_bits": 8,
            "primitive_mac_bits": 20,
            "accumulator_bits": 32,
            "max_primitive_k": 16,
        },
    )


def test_signed_nonconstant_complete_arithmetic_is_independently_recomputed():
    c, values, golden, rule = _values()
    checked = check_integer_values(c, values, golden, rule)
    assert checked["products"] == 60 and checked["output_values"] == 15
    assert checked["signed_inputs"] and checked["nonconstant_inputs"]


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_golden",
        "missing_output",
        "missing_input",
        "float_input",
        "u8_range",
        "wrong_panel_bits",
        "wrong_acc_bits",
    ],
)
def test_arithmetic_refuses_incomplete_or_different_original_values(damage):
    c, values, golden, rule = _values()
    if damage == "wrong_golden":
        golden["Y0"][1, 2] += 1
    elif damage == "missing_output":
        golden = {}
    elif damage == "missing_input":
        values.pop("W")
    elif damage == "float_input":
        values["W"] = values["W"].astype(float)
    elif damage == "u8_range":
        values["W"][0, 0] = 255
    elif damage == "wrong_panel_bits":
        rule["primitive_mac_bits"] = 8
    else:
        rule["accumulator_bits"] = 20
    with pytest.raises(ValueError):
        check_integer_values(c, values, golden, rule)


def _grade():
    cap, _source_text, _cb = _source()
    row = {
        "capsule": cap["name"],
        "kind": "layer",
        "label": "public",
        "status": "incomplete",
        "failure": {
            "plane": "oracle_unavailable",
            "category": "NOT_RUN_IS_NOT_PASS",
            "tier": "L3",
        },
        "numeric": {"status": "pass"},
        "trace_check": {"status": "pass"},
        "tiers": {
            name: {"status": "pass", "mandatory": True, "measured_now": True}
            for name in ("L0", "L1", "L2")
        },
    }
    return cap, row


def test_diagnostic_retains_original_l3_pending_without_converting_grade_to_pass():
    cap, row = _grade()
    before = copy.deepcopy(row)
    _diagnostic_grade(row, cap)
    assert row == before and row["status"] == "incomplete"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_L0",
        "failed_L1",
        "carried_L2",
        "not_fresh",
        "wrong_capsule",
        "trace",
        "budget",
        "missing_other_tier",
    ],
)
def test_original_numerical_roles_cannot_be_waived_by_diagnostic_owner(damage):
    cap, row = _grade()
    if damage == "missing_L0":
        row["tiers"].pop("L0")
    elif damage == "failed_L1":
        row["tiers"]["L1"]["status"] = "fail"
    elif damage == "carried_L2":
        row["tiers"]["L2"]["carried"] = True
    elif damage == "not_fresh":
        row["tiers"]["L2"]["measured_now"] = False
    elif damage == "wrong_capsule":
        row["capsule"] = "another"
    elif damage == "trace":
        row["trace_check"]["status"] = "unavailable"
    elif damage == "budget":
        row["status"] = "budget_exhausted"
    else:
        row["failure"]["tier"] = "L1"
    with pytest.raises(ValueError):
        _diagnostic_grade(row, cap)


def test_saved_success_or_foreign_owner_cannot_construct_primitive_checker():
    with pytest.raises(ValueError, match="original live"):
        PrimitiveDiagnosticOwner({"status": "qualified"})


def test_a_pass_label_cannot_replace_the_original_pending_l3_receipt():
    cap, row = _grade()
    row["status"], row["failure"] = "pass", None
    with pytest.raises(ValueError, match="ordinary grade"):
        _diagnostic_grade(row, cap)


def test_instance_callback_replacement_cannot_impersonate_original_compiler_owner():
    owner = object.__new__(QualifiedZeroInitializerOwner)
    owner.revalidate = lambda: None
    with pytest.raises(ValueError, match="original bound"):
        _original_methods(owner)
