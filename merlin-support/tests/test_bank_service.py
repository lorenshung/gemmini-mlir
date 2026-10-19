"""Structural shared service and address slices do not qualify execution."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_bank_service as B


def bank():
    return B.F.parse_module(
        "Bank",
        list(
            enumerate(
                """module Bank :
  output io : {read : {req : {ready : UInt<1>, valid : UInt<1>, bits : {addr : UInt<3>}}, resp : {bits : {data : UInt<32>}}}, write : {ready : UInt<1>, valid : UInt<1>, addr : UInt<3>, data : UInt<32>}}
  wire q_will_be_empty : UInt<1>
  node ren = and(io.read.req.ready, io.read.req.valid)
  connect io.write.ready, UInt<1>(1)
  connect io.read.req.ready, and(q_will_be_empty, not(and(io.write.valid, io.write.ready)))
""".splitlines(),
                1,
            )
        ),
    )


def top():
    lines = ["module Top :"]
    for index in range(2):
        lines += [
            f"  inst spad_mems_{index} of Bank",
            f"  connect spad_mems_{index}.clock, clock",
        ]
        for field in ("write.addr", "write.valid", "read.req.valid"):
            lines += [
                f"  connect spad_mems_{index}.io.{field}, bank_ios[{index}].{field}"
            ]
        lines += [
            f"  connect bank_ios[{index}].write.ready, spad_mems_{index}.io.write.ready",
            f"  node selected{index} = and(and(and(mvin_scale_pixel_repeater.io.resp.valid, eq(mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc, UInt<1>(0))), eq(bits(mvin_scale_pixel_repeater.io.resp.bits.laddr.data, 3, 3), UInt<1>({index}))), bank_ios[{index}].write.ready)",
            f"  when io.srams.write[{index}].valid :",
            f"    connect bank_ios[{index}].write.addr, io.srams.write[{index}].addr",
            "  else :",
            f"    when selected{index} :",
            f"      connect bank_ios[{index}].write.addr, bits(mvin_scale_pixel_repeater.io.resp.bits.laddr.data, 2, 0)",
        ]
    return B.F.parse_module("Top", list(enumerate(lines, 1)))


def test_nonselected_geometry_and_complete_shared_port_boolean_cases():
    owner = B.BankPortCones(bank(), {})
    assert owner.row_bits == 3 and owner.row_bytes == 4
    assert owner.check()["boolean_cut_cases"] == 8
    assert owner.at(response_queue_will_empty=1, read_valid=1, write_valid=1) == {
        "read_accepted": 0,
        "write_accepted": 1,
        "read_ready": 0,
        "write_ready": 1,
    }
    assert (
        owner.at(response_queue_will_empty=0, read_valid=1, write_valid=0)[
            "read_accepted"
        ]
        == 0
    )
    assert (
        owner.at(response_queue_will_empty=1, read_valid=1, write_valid=0)[
            "read_accepted"
        ]
        == 1
    )


def test_dual_port_or_missing_backpressure_cannot_keep_shared_port_claim():
    source = bank()
    row = next(
        row for row in source.connects if row.lhs == B.F.parse_expr("io.read.req.ready")
    )
    source.connects[source.connects.index(row)] = replace(
        row, rhs=B.F.Ref("q_will_be_empty")
    )
    with pytest.raises(ValueError, match="shared-port"):
        B.BankPortCones(source, {}).check()


def test_mapping_uses_contiguous_high_bank_bits_and_preserves_priority():
    mapping = B.derive_address_mapping(top(), ["spad_mems_0", "spad_mems_1"], 3)
    assert mapping["bank_bits"] == [3, 3] and mapping["row_bits"] == [2, 0]
    assert mapping["banks"] == 2 and mapping["rows_per_bank"] == 8
    owner = object.__new__(B.ScratchpadBankService)
    owner.mapping = mapping
    assert owner._local_row(7)["bank"] == 0 and owner._local_row(8)["bank"] == 1
    assert owner._local_row(15)["row"] == 7
    for address in [-1, 16, True]:
        with pytest.raises(ValueError):
            owner._local_row(address)


def test_changed_bank_slice_or_priority_refuses():
    source = top()
    expr, line, locator = source.nodes["selected0"]
    source.nodes["selected0"] = (B.F.Prim("or", expr.args), line, locator)
    with pytest.raises(ValueError, match="bank selection"):
        B.derive_address_mapping(source, ["spad_mems_0", "spad_mems_1"], 3)


@pytest.mark.parametrize("damage", ["bank_slice", "row_slice", "clock"])
def test_address_bit_or_instance_custody_changes_refuse(damage):
    source = top()
    if damage == "bank_slice":
        expr, line, locator = source.nodes["selected0"]

        def change(expression):
            if type(expression) is B.F.Prim:
                if expression.op == "bits":
                    return B.F.Prim(
                        "bits", (expression.args[0], B.F.IntArg(2), B.F.IntArg(2))
                    )
                return B.F.Prim(
                    expression.op, tuple(change(arg) for arg in expression.args)
                )
            return expression

        source.nodes["selected0"] = (change(expr), line, locator)
    else:
        name = (
            "bank_ios[0].write.addr" if damage == "row_slice" else "spad_mems_0.clock"
        )
        row = next(
            row
            for row in source.connects
            if row.lhs == B.F.parse_expr(name)
            and (damage != "row_slice" or len(row.guards) == 2)
        )
        rhs = (
            B.F.parse_expr(
                "bits(mvin_scale_pixel_repeater.io.resp.bits.laddr.data, 3, 1)"
            )
            if damage == "row_slice"
            else B.F.Ref("other_clock")
        )
        source.connects[source.connects.index(row)] = replace(row, rhs=rhs)
    with pytest.raises(ValueError):
        B.derive_address_mapping(source, ["spad_mems_0", "spad_mems_1"], 3)


def test_changed_source_reopens_bytes_and_saved_success_is_not_owner(tmp_path):
    source = tmp_path / "source.fir"
    source.write_text("original")
    owner = object.__new__(B.ScratchpadBankService)
    owner.selected = SimpleNamespace(revalidate=lambda: None)
    owner.sources = {"source": B._pin(source)}
    owner.revalidate()
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        owner.revalidate()
    with pytest.raises(ValueError, match="original selected source owner"):
        B.ScratchpadBankService({"status": "verified"})
