"""Additional selected EX operand/effect evidence, with no encoding defaults.

This original target adapter chooses named control cutpoints; the generic
FIRRTL observer reconstructs every field from actual wires. Packing does not
establish completion or a numerical algorithm. The narrow admission below
requires a complete identity WS configuration before any observed arithmetic.
"""

from __future__ import annotations

from pathlib import Path

from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_operand_slices as S
from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl.firrtl_bit_transport import BitValue, observe_bit_transport


def prove_execute_layout(module):
    fields = {}
    for operand, wire, names in (
        (
            "rs1",
            "config_ex_rs1",
            (
                "dataflow",
                "activation",
                "set_only_strides",
                "a_transpose",
                "b_transpose",
                "a_stride",
                "acc_scale",
            ),
        ),
        ("rs2", "config_ex_rs2", ("in_shift", "relu6_shift", "c_stride")),
    ):
        for name in names:
            destination = F.parse_expr(wire + "." + name)
            drivers = [row for row in module.connects if row.lhs == destination]
            if len(drivers) != 1:
                raise ValueError(
                    "EX field has missing or ambiguous original wire effects"
                )
            context = {"guards": [guard.text() for guard in drivers[0].guards]}
            if not context["guards"]:
                context = {}
            field = S._field(
                module,
                {
                    "input": operand + "s[0]",
                    "field": wire + "." + name,
                    "packing_context": context,
                },
                64,
                {},
            )
            if field["status"] != "derived":
                raise ValueError("EX field packing remains unavailable")
            fields[name] = {"operand": operand, **field}
    for sink, source in (
        ("current_dataflow", "config_ex_rs1.dataflow"),
        ("activation", "config_ex_rs1.activation"),
        ("in_shift", "config_ex_rs2.in_shift"),
        ("a_transpose", "config_ex_rs1.a_transpose"),
        ("bd_transpose", "config_ex_rs1.b_transpose"),
        ("a_addr_stride", "config_ex_rs1.a_stride"),
        ("c_addr_stride", "config_ex_rs2.c_stride"),
    ):
        rows = [
            row
            for row in module.connects
            if row.lhs == F.Ref(sink) and row.rhs != F.Ref(sink)
        ]
        if len(rows) != 1 or rows[0].rhs != F.parse_expr(source) or not rows[0].guards:
            raise ValueError(
                "EX controller does not assign the selected field to its original state"
            )
    scale = fields["acc_scale"]
    expected = BitValue(
        False,
        tuple(
            ("rs1s[0]", bit)
            for bit in range(scale["offset"], scale["offset"] + scale["width"])
        ),
    )
    if observe_bit_transport(module, "acc_scale.bits", sources=("rs1s[0]",)) != {
        expected
    }:
        raise ValueError(
            "EX scale state does not preserve the actual selected operand bits"
        )
    for name, field in (
        ("weight_rows", "d_rows_default"),
        ("weight_cols", "d_cols_default"),
    ):
        fields[name] = {
            "operand": "rs1",
            **S._field(
                module,
                {
                    "input": "rs1s[preload_cmd_place]",
                    "field": field,
                    "packing_context": {"dynamic_input": True, "node_field": True},
                },
                64,
                {},
            ),
        }
    return fields


class ExecuteStateContract:
    def __init__(self, selected):
        selected.revalidate()
        facts, _, _ = selected._documents()
        fir = Path(facts["inputs"]["fir_path"])
        if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
            raise ValueError("selected EX source changed")
        self._selected = selected
        self._pin = _pin(fir)
        self._implementation = _pin(__file__)
        self._observer = _pin(S.__file__)
        self.fields = prove_execute_layout(
            F.load_modules(fir, {"ExecuteController"})["ExecuteController"]
        )

    def revalidate(self):
        self._selected.revalidate()
        if (
            self._pin != _pin(self._pin["path"])
            or self._implementation != _pin(__file__)
            or self._observer != _pin(S.__file__)
        ):
            raise ValueError("original EX state/field observer changed")

    def identity_ws(self, rs1, rs2):
        self.revalidate()
        values = {}
        for name, field in self.fields.items():
            if name.startswith("weight_"):
                continue
            raw = rs1 if field["operand"] == "rs1" else rs2
            values[name] = raw >> field["offset"] & ((1 << field["width"]) - 1)
        # These are explicit selected target semantics, not generic defaults.
        # WS=1 is the same active PE dataflow branch rechecked by arithmetic_scope.
        required = {
            "dataflow": 1,
            "activation": 0,
            "set_only_strides": 0,
            "a_transpose": 0,
            "b_transpose": 0,
            "a_stride": 1,
            "c_stride": 1,
            "in_shift": 0,
            "relu6_shift": 0,
            "acc_scale": 0x3F800000,
        }
        if values != required:
            raise ValueError(
                "actual EX configuration is outside checked identity WS semantics"
            )
        return values

    def weight_shape(self, rs1):
        return tuple(
            rs1 >> self.fields[name]["offset"] & ((1 << self.fields[name]["width"]) - 1)
            for name in ("weight_rows", "weight_cols")
        )
