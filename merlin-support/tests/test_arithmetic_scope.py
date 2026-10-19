"""Selected arithmetic scopes refuse wider claims and altered datapaths."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_arithmetic_scope")


def parsed(adapter, name, body):
    return adapter.F.parse_module(name, list(enumerate(body.splitlines(), 1)))


def arithmetic(
    adapter, *, mac_expression=None, ws_input="io.in_b", accumulator_bits=32
):
    mac = parsed(
        adapter,
        "Mac",
        f"""module Mac :
  output io : {{ flip in_a : SInt<8>, flip in_b : SInt<8>, flip in_c : SInt<32>, out_d : SInt<20> }}
  node product = mul(io.in_a, io.in_b)
  node added = add(product, io.in_c)
  connect io.out_d, {mac_expression or "asSInt(tail(added, 1))"}
""",
    )
    pe = parsed(
        adapter,
        "PE",
        f"""module PE :
  output io : {{ flip in_b : SInt<20>, out_b : SInt<20>, flip in_valid : UInt<1>, flip in_control : {{ dataflow : UInt<1>, propagate : UInt<1> }} }}
  when eq(io.in_control.dataflow, UInt<1>(1)) :
    connect mac_unit.io.in_c, {ws_input}
    connect io.out_b, mac_unit.io.out_d
""",
    )
    acc = parsed(
        adapter,
        "Acc",
        f"""module Acc :
  input clock : Clock
  output io : {{ flip op1 : SInt<{accumulator_bits}>, flip op2 : SInt<{accumulator_bits}>, sum : SInt<{accumulator_bits}> }}
  reg saved : SInt<{accumulator_bits}>, clock
  when UInt<1>(1) :
    connect saved, asSInt(tail(add(io.op1, io.op2), 1))
  connect io.sum, saved
""",
    )
    return mac, pe, acc


def test_distinct_mac_and_accumulator_scopes(adapter):
    rule = adapter.prove_local_arithmetic(*arithmetic(adapter))
    assert rule == {"operand_bits": 8, "primitive_mac_bits": 20, "accumulator_bits": 32}
    # Local 16-product panel fits20 even when a full-K result requires >20bits.
    assert 16 * 128 * 128 < 1 << (rule["primitive_mac_bits"] - 1)
    assert 2560 * 128 * 128 > 1 << (rule["primitive_mac_bits"] - 1)
    assert 2560 * 128 * 128 < 1 << (rule["accumulator_bits"] - 1)


@pytest.mark.parametrize(
    "expression",
    [
        "asSInt(tail(product, 1))",
        "asSInt(tail(sub(product, io.in_c), 1))",
        "asSInt(tail(added, 2))",
    ],
)
def test_changed_actual_mac_arithmetic_refuses(adapter, expression):
    with pytest.raises(ValueError, match="exact signed multiply/add"):
        adapter.prove_local_arithmetic(*arithmetic(adapter, mac_expression=expression))


def test_recurrent_pe_state_is_not_spatial_partial_sum(adapter):
    with pytest.raises(ValueError, match="spatial partial sum"):
        adapter.prove_local_arithmetic(
            *arithmetic(adapter, ws_input="saved_accumulator")
        )


def mesh(adapter, *, previous="mesh_0_0.io.out_b[0]", extra=""):
    return parsed(
        adapter,
        "Mesh",
        f"""module Mesh :
  output io : {{ flip in_b : SInt<8>[1][1] }}
  inst mesh_0_0 of Tile
  inst mesh_1_0 of Tile1
  {extra}
  connect mesh_0_0.io.in_b[0], io.in_b[0][0]
  wire delayed : SInt<20>
  connect delayed, {previous}
  connect mesh_1_0.io.in_b[0], delayed
""",
    )


def test_actual_mesh_chain_retains_only_one_partial_sum_per_row(adapter):
    proof = adapter.prove_mesh_scope(mesh(adapter), 2, 1, tile_types(adapter))
    assert proof[-1]["sources"] == ["mesh_0_0.io.out_b[0]"]
    assert proof[-1]["all_signed_bits_preserved"]


def tile_types(adapter):
    tile = parsed(
        adapter,
        "Tile",
        """module Tile :
  output io : { flip in_b : SInt<20>[1], out_b : SInt<20>[1] }
""",
    )
    return {"Tile": tile, "Tile1": tile}


@pytest.mark.parametrize(
    "previous",
    ["mesh_1_0.io.out_b[0]", "io.in_b[0][0]", "add(mesh_0_0.io.out_b[0], SInt<20>(1))"],
)
def test_feedback_bypass_or_extra_arithmetic_refuses(adapter, previous):
    with pytest.raises(ValueError):
        adapter.prove_mesh_scope(
            mesh(adapter, previous=previous), 2, 1, tile_types(adapter)
        )


def test_same_source_with_lost_bits_does_not_establish_transport(adapter):
    with pytest.raises(ValueError, match="every signed bit"):
        adapter.prove_mesh_scope(
            mesh(adapter, previous="bits(mesh_0_0.io.out_b[0],0,0)"),
            2,
            1,
            tile_types(adapter),
        )


def test_unaccounted_physical_mac_instance_refuses(adapter):
    with pytest.raises(ValueError, match="physical MAC tile membership"):
        adapter.prove_mesh_scope(
            mesh(adapter, extra="inst foreign_tile of Tile2"), 2, 1, tile_types(adapter)
        )
