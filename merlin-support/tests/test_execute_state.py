"""EX source field/effect observations refuse drift; no target qualification."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_execute_state")


def source(adapter, *, dataflow=2, sink="config_ex_rs1.dataflow", scale_high=63):
    widths = {
        "dataflow": (dataflow, dataflow),
        "activation": (4, 3),
        "set_only_strides": (7, 7),
        "a_transpose": (8, 8),
        "b_transpose": (9, 9),
        "a_stride": (31, 16),
        "acc_scale": (63, 32),
    }
    words = [
        "module ExecuteController :",
        "  wire rs1s : UInt<64>[3]",
        "  wire rs2s : UInt<64>[3]",
        "  wire preload_cmd_place : UInt<2>",
    ]
    for wire, fields in (
        ("config_ex_rs1", widths),
        (
            "config_ex_rs2",
            {"in_shift": (31, 0), "relu6_shift": (47, 32), "c_stride": (63, 48)},
        ),
    ):
        words.append(
            "  wire "
            + wire
            + " : { "
            + ", ".join(
                name + " : UInt<" + str(high - low + 1) + ">"
                for name, (high, low) in fields.items()
            )
            + " }"
        )
        for name, (high, low) in fields.items():
            words += [
                "  when UInt<1>(1) :",
                f"    connect {wire}.{name}, bits({'rs1' if wire.endswith('rs1') else 'rs2'}s[0], {high}, {low})",
            ]
    for name, rhs, width in (
        ("current_dataflow", sink, 1),
        ("activation", "config_ex_rs1.activation", 2),
        ("in_shift", "config_ex_rs2.in_shift", 32),
        ("a_transpose", "config_ex_rs1.a_transpose", 1),
        ("bd_transpose", "config_ex_rs1.b_transpose", 1),
        ("a_addr_stride", "config_ex_rs1.a_stride", 16),
        ("c_addr_stride", "config_ex_rs2.c_stride", 16),
    ):
        words += [
            f"  wire {name} : UInt<{width}>",
            "  when UInt<1>(1) :",
            f"    connect {name}, {rhs}",
        ]
    words += [
        "  wire acc_scale : { bits : UInt<32> }",
        "  when UInt<1>(1) :",
        f"    connect acc_scale.bits, bits(rs1s[0], {scale_high}, 32)",
        "  node d_rows_default = bits(rs1s[preload_cmd_place],52,48)",
        "  node d_cols_default = bits(rs1s[preload_cmd_place],36,32)",
    ]
    return adapter.F.parse_module("ExecuteController", list(enumerate(words, 1)))


def test_actual_ex_field_position_is_derived(adapter):
    fields = adapter.prove_execute_layout(source(adapter, dataflow=6))
    assert fields["dataflow"]["offset"] == 6
    assert fields["acc_scale"]["width"] == 32
    assert fields["weight_rows"]["offset"] == 48


@pytest.mark.parametrize(
    "kwargs", [{"sink": "config_ex_rs1.a_transpose"}, {"scale_high": 62}]
)
def test_changed_state_or_lost_scale_bit_refuses(adapter, kwargs):
    with pytest.raises(ValueError):
        adapter.prove_execute_layout(source(adapter, **kwargs))
