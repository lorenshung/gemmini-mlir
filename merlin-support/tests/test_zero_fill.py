"""Neutral source-route/receiver controls, not hardware qualification."""

import importlib
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_zero_fill")


def modules(M):
    def parse(name, rows):
        return M.F.parse_module(
            name, list(enumerate((f"module {name} :\n" + rows).splitlines(), 1))
        )

    load = parse(
        "Load",
        "  node all_zeros = eq(cmd_q.io.deq.bits.cmd.rs1, UInt<1>(0))\n  connect io.dma.req.bits.all_zeros, all_zeros",
    )
    rows = [
        "  node active = and(io.dma.read.req.valid, io.dma.read.req.bits.all_zeros)",
        "  connect zero_writer.io.req.valid, active",
    ]
    rows += [
        f"  connect zero_writer.io.req.bits.{field}, io.dma.read.req.bits.{field}"
        for field in (
            "laddr.data",
            "laddr.is_acc_addr",
            "laddr.accumulate",
            "cols",
            "block_stride",
        )
    ]
    rows += [
        f"  connect zero_writer_pixel_repeater.io.req.bits.{field}, zero_writer.io.resp.bits.{field}"
        for field in ("laddr.data", "laddr.is_acc_addr", "laddr.accumulate")
    ]
    rows += [
        "  connect zero_writer_pixel_repeater.io.req.bits.pixel_repeats, zero_writer.io.resp.bits.tag.pixel_repeats"
    ]
    return load, parse("Scratch", "\n".join(rows))


def test_actual_predicate_and_extent_routes_are_checked(adapter):
    adapter.prove_zero_route(*modules(adapter))


@pytest.mark.parametrize(
    "fault", ["predicate", "gate", "address", "accumulate", "columns", "repeats"]
)
def test_zero_route_changes_refuse(adapter, fault):
    load, scratch = modules(adapter)
    if fault == "predicate":
        old, n, loc = load.nodes["all_zeros"]
        load.nodes["all_zeros"] = adapter.F.Prim("neq", old.args), n, loc
    else:
        target = {
            "gate": "zero_writer.io.req.valid",
            "address": "zero_writer.io.req.bits.laddr.data",
            "accumulate": "zero_writer.io.req.bits.laddr.accumulate",
            "columns": "zero_writer.io.req.bits.cols",
            "repeats": "zero_writer_pixel_repeater.io.req.bits.pixel_repeats",
        }[fault]
        row = next(
            row for row in scratch.connects if row.lhs == adapter.F.parse_expr(target)
        )
        scratch.connects[scratch.connects.index(row)] = replace(
            row, rhs=adapter.F.Lit("UInt", 1, 1)
        )
    with pytest.raises(ValueError):
        adapter.prove_zero_route(load, scratch)


def receiver():
    return """void gemmini_t::mvin() {
bool is_zeros = dram_addr == 0;
bool const accumulator = (sp_addr >> 31) & 0x1;
bool const accumulate = (sp_addr >> 30) & 0x1;
if (accumulator) {
if (is_zeros) {
              value = 0;
            } else if (!load_shrunk) {}
if (accumulate) {
              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) += value;
            } else {
              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) = value;
}
}
void gemmini_t::mvout() {}
"""


def test_receiver_review_is_exact_and_requires_zero_overwrite(adapter):
    adapter.prove_receiver_zero(receiver())
    for old, new in (
        ("value = 0;", "value = 1;"),
        ("dram_addr == 0", "dram_addr != 0"),
        (" = value;", " += value;"),
        ("sp_addr >> 30", "sp_addr >> 29"),
    ):
        with pytest.raises(ValueError):
            adapter.prove_receiver_zero(receiver().replace(old, new))


@pytest.mark.parametrize(
    "address,is_acc,expected",
    [(0, True, True), (0, False, False), (1, True, False), (False, True, False)],
)
def test_zero_address_is_a_command_effect_not_memory_contents(
    adapter, address, is_acc, expected
):
    owner = object.__new__(adapter.SelectedAccumulatorZeroFill)
    assert (
        owner.accepts({"rs1": {"raw": address}, "decoded": {"is_acc_addr": is_acc}})
        is expected
    )


def test_saved_success_or_foreign_owners_cannot_issue(adapter):
    with pytest.raises(TypeError):
        adapter.SelectedAccumulatorZeroFill({"status": "PASS"}, {})
    forged = object.__new__(adapter.SelectedAccumulatorZeroFill)
    with pytest.raises(ValueError, match="original zero-fill owner changed"):
        forged.revalidate()


def accumulator_module(M):
    rows = [
        "module Scratch :",
        "  input io : {dma : {read : {req : {valid : UInt<1>}}}}",
        "  node zero_guard = and(zero_writer_pixel_repeater.io.resp.valid, zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr)",
        "  node zero_data = SInt<32>(0)",
    ]
    for column in range(2):
        rows += [
            f"  connect acc_mems_0.io.write.bits.data[{column}][0], bank[0].write.bits.data[{column}][0]",
            "  when zero_guard :",
            f"    connect bank[0].write.bits.data[{column}][0], zero_data",
        ]
    return M.F.parse_module("Scratch", list(enumerate(rows, 1)))


def test_all_accumulator_zero_branch_lanes_are_independently_checked(adapter):
    assert (
        adapter.prove_accumulator_zero_data(accumulator_module(adapter), {}, 1, 2) == 2
    )


@pytest.mark.parametrize(
    "fault", ["nonzero_data", "negative_guard", "foreign_response", "missing_lane"]
)
def test_accumulator_zero_data_or_positive_branch_mutations_refuse(adapter, fault):
    source = accumulator_module(adapter)
    if fault == "nonzero_data":
        _, line, locator = source.nodes["zero_data"]
        source.nodes["zero_data"] = adapter.F.Lit("SInt", 32, 1), line, locator
    elif fault == "negative_guard":
        expr, line, locator = source.nodes["zero_guard"]
        source.nodes["zero_guard"] = adapter.F.Prim("not", (expr,)), line, locator
    elif fault == "foreign_response":
        _, line, locator = source.nodes["zero_guard"]
        source.nodes["zero_guard"] = adapter.F.Ref("foreign_response"), line, locator
    else:
        source.connects.pop()
    with pytest.raises(ValueError):
        adapter.prove_accumulator_zero_data(source, {}, 1, 2)
