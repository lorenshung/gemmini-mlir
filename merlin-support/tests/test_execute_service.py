"""Synthetic FIR cones exercise service derivation, never accelerator timing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_execute_service as S


def fixture(*, floor=2):
    text = f"""module ExecuteController :
  reg current_dataflow : UInt<1>, clock
  reg a_fire_counter : UInt<3>, clock
  wire total_rows : UInt<4>
  node mode = eq(current_dataflow, UInt<1>(1))
  node transposed = or(a_should_be_fed_into_transposer, or(b_should_be_fed_into_transposer, d_should_be_fed_into_transposer))
  node reduced = and(and(mode, d_garbage), eq(transposed, UInt<1>(0)))
  node rows_a = mux(a_garbage, UInt<1>(1), a_rows)
  node rows_b = mux(b_garbage, UInt<1>(1), b_rows)
  node larger = mux(gt(rows_a, rows_b), rows_a, rows_b)
  node bounded = mux(gt(larger, UInt<4>({floor})), larger, UInt<4>({floor}))
  connect total_rows, UInt<4>(8)
  when reduced :
    connect total_rows, bounded
  when eq(firing, UInt<1>(0)) :
    connect a_fire_counter, UInt<1>(0)
  else :
    when and(a_fire, mesh_cntl_signals_q.io.enq.ready) :
      connect a_fire_counter, mux(eq(a_fire_counter, sub(total_rows, UInt<1>(1))), UInt<1>(0), add(a_fire_counter, UInt<1>(1)))
"""
    return S.F.parse_module("ExecuteController", list(enumerate(text.splitlines(), 1)))


def test_selected_literal_and_counter_are_derived_without_dim16_or_floor4():
    proof = S.prove(fixture())
    assert proof["dimension"] == 8
    assert proof["minimum_reduced_input_rows"] == 2
    assert proof["shape_cutpoint_checks"] == 32 * 9 * 9 * 4
    assert proof["counter_cutpoint_checks"] == sum(range(1, 9)) * 8
    enabled = [row for row in proof["reduction_guard_cases"] if row["reduced"]]
    assert enabled == [
        {
            "current_dataflow": 1,
            "d_garbage": 1,
            "a_should_be_fed_into_transposer": 0,
            "b_should_be_fed_into_transposer": 0,
            "d_should_be_fed_into_transposer": 0,
            "reduced": True,
        }
    ]
    assert proof["input_rows_per_cycle_maximum"] == 1
    assert "no state reachability" in proof["scope"]


@pytest.mark.parametrize(
    "mutation", ["minimum", "counter", "extra_driver", "zero_floor", "guard", "cycle"]
)
def test_changed_semantics_never_become_an_implicit_service_fact(mutation):
    module = fixture(floor=0 if mutation == "zero_floor" else 2)
    F = S.F
    if mutation == "minimum":
        expression, line, locator = module.nodes["larger"]
        module.nodes["larger"] = (
            F.Prim("mux", (expression.args[0], expression.args[2], expression.args[1])),
            line,
            locator,
        )
    elif mutation == "counter":
        row = module.connects[-1]
        module.connects[-1] = F.Connect(
            row.lhs, F.Lit("UInt", 3, 0), row.guards, row.line, row.locator
        )
    elif mutation == "extra_driver":
        module.connects.append(
            F.Connect(F.Ref("total_rows"), F.Lit("UInt", 4, 8), (), 100, None)
        )
    elif mutation == "guard":
        row = module.connects[1]
        module.connects[1] = F.Connect(
            row.lhs,
            row.rhs,
            (F.Guard(F.Lit("UInt", 2, 2), True),),
            row.line,
            row.locator,
        )
    elif mutation == "cycle":
        module.nodes["bounded"] = (F.Ref("bounded"), 100, None)
    with pytest.raises(ValueError):
        S.prove(module)


def test_unsigned_tail_keeps_actual_expression_width():
    module = fixture()
    value = S.F.parse_expr("tail(add(UInt<3>(7), UInt<3>(1)), 1)")
    assert S._eval(module, value, {}) == 0
