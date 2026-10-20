"""Neutral source-cone and complete-stream controls, never target qualification."""

import copy
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_packet_contractions import command, shape


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_accumulator_overwrite")


def modules(M, dim=2):
    def parse(name, rows):
        return M.F.parse_module(
            name,
            list(
                enumerate(
                    (
                        f"module {name} :\n" + "\n".join("  " + r for r in rows)
                    ).splitlines(),
                    1,
                )
            ),
        )

    memory = [
        "connect pipelined_writes[0].bits, io.write.bits",
        "connect pipelined_writes[1], pipelined_writes[0]",
        "connect mem.io.waddr, pipelined_writes[1].bits.addr",
        "connect mem.io.wen, pipelined_writes[1].valid",
        "node _T = mux(pipelined_writes[1].bits.acc, io.adder.sum, pipelined_writes[1].bits.data)",
        "connect io.adder.valid, and(pipelined_writes[0].valid, pipelined_writes[0].bits.acc)",
        *[f"connect mem.io.wdata[{i}][0], _T[{i}][0]" for i in range(dim)],
        *[
            f"connect mem.io.mask[{i}], pipelined_writes[1].bits.mask[{i}]"
            for i in range(4 * dim)
        ],
    ]
    execute = [
        "connect c_address_rs2.accumulate, bits(_c_address_rs2_WIRE, 30, 30)",
        "connect mesh_cntl_signals_q.io.enq.bits.c_addr.accumulate, c_address_rs2.accumulate",
        "node w_address = mux(dataflow, w_address_result, w_address_result_1)",
        "connect w_address_result, mesh.io.resp.bits.tag.addr",
        "connect w_address_result_1, mesh.io.resp.bits.tag.addr",
        "connect io.acc.write[0].bits.acc, w_address.accumulate",
        "connect io.acc.write[0].bits.addr, w_row",
        *[
            f"node w_mask_{i} = lt(UInt<8>({i}), mesh.io.resp.bits.tag.cols)"
            for i in range(dim)
        ],
        *[
            f"connect io.acc.write[0].bits.mask[{i}], w_mask_{i // 4}"
            for i in range(4 * dim)
        ],
    ]
    scratch = ["inst acc_mems_0 of AccumulatorMem"]
    for field in ("acc", "addr"):
        scratch += [
            f"connect bank_ios_1[0].write.bits.{field}, mux(io.acc.write[0].valid, io.acc.write[0].bits.{field}, dma_{field})",
            f"connect acc_mems_0.io.write.bits.{field}, bank_ios_1[0].write.bits.{field}",
        ]
    for field, count in (("data", dim), ("mask", 4 * dim)):
        for i in range(count):
            tail = f".bits.{field}[{i}]" + ("[0]" if field == "data" else "")
            scratch += [
                "when io.acc.write[0].valid :",
                f"  connect bank_ios_1[0].write{tail}, io.acc.write[0]{tail}",
                f"connect acc_mems_0.io.write{tail}, bank_ios_1[0].write{tail}",
            ]
    return {
        "AccumulatorMem": parse("AccumulatorMem", memory),
        "ExecuteController": parse("ExecuteController", execute),
        "Scratchpad": parse("Scratchpad", scratch),
    }


def receiver():
    return """void gemmini_t::preload() {
gemmini_state.output_sp_addr = static_cast<uint32_t>(c_addr & 0xFFFFFFFF);
gemmini_state.output_cols = (c_addr >> addr_len) & 0xFFFF;
gemmini_state.output_rows = (c_addr >> (addr_len + 16)) & 0xFFFF;
}
void gemmini_t::compute() {
bool const acc = (gemmini_state.output_sp_addr >> 31) & 0x1;
bool const acc_accum = (gemmini_state.output_sp_addr >> 30) & 0x1;
auto const base_sp_addr = gemmini_state.output_sp_addr & 0x1FFFFFFF;
for (size_t i = 0; i < gemmini_state.output_rows; ++i) {
for (size_t j = 0; j < gemmini_state.output_cols; ++j) {
acc_t value = gemmini_state.mode == gemmini_state_t::OS ? gemmini_state.pe_state.at(i).at(j) : results.at(i).at(j);
if (acc_accum) { gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) += value; }
else { gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) = value; }
}}
}
void gemmini_t::next() {}
"""


def prove(M, m, dim=2):
    M.prove_memory_overwrite(m["AccumulatorMem"], dim)
    M.prove_execute_overwrite(
        m["ExecuteController"], m["Scratchpad"], banks=1, dim=dim, accumulate_offset=30
    )


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_complete_write_mask_route_for_alternative_geometries(adapter, dim):
    prove(adapter, modules(adapter, dim), dim)
    adapter.prove_receiver_overwrite(receiver())


@pytest.mark.parametrize(
    "fault",
    [
        "read_old_value",
        "reversed_mux",
        "captured_address",
        "byte_mask",
        "data_lane",
        "captured_subfield",
        "flag_bit",
        "output_flag",
        "tail_mask",
        "scratch_flag",
        "scratch_data",
        "scratch_mask",
    ],
)
def test_modified_source_cones_refuse(adapter, fault):
    m = modules(adapter)
    if fault in {"read_old_value", "reversed_mux"}:
        body = m["AccumulatorMem"]
        old, line, loc = body.nodes["_T"]
        args = (
            (old.args[0], old.args[2], old.args[1])
            if fault == "reversed_mux"
            else (old.args[0], old.args[1], adapter.F.Ref("io.adder.sum"))
        )
        body.nodes["_T"] = adapter.F.Prim("mux", args), line, loc
    elif fault == "captured_subfield":
        body = m["AccumulatorMem"]
        body.connects.append(
            replace(
                body.connects[0],
                lhs=adapter.F.parse_expr("pipelined_writes[1].bits.acc"),
            )
        )
    else:
        name, destination = {
            "captured_address": ("AccumulatorMem", "mem.io.waddr"),
            "byte_mask": ("AccumulatorMem", "mem.io.mask[3]"),
            "data_lane": ("AccumulatorMem", "mem.io.wdata[1][0]"),
            "flag_bit": ("ExecuteController", "c_address_rs2.accumulate"),
            "output_flag": ("ExecuteController", "io.acc.write[0].bits.acc"),
            "tail_mask": ("ExecuteController", "io.acc.write[0].bits.mask[7]"),
            "scratch_flag": ("Scratchpad", "bank_ios_1[0].write.bits.acc"),
            "scratch_data": ("Scratchpad", "bank_ios_1[0].write.bits.data[1][0]"),
            "scratch_mask": ("Scratchpad", "acc_mems_0.io.write.bits.mask[7]"),
        }[fault]
        body = m[name]
        row = next(
            r for r in body.connects if r.lhs == adapter.F.parse_expr(destination)
        )
        body.connects[body.connects.index(row)] = replace(
            row, rhs=adapter.F.Lit("UInt", 1, 0)
        )
    with pytest.raises(ValueError):
        prove(adapter, m)


@pytest.mark.parametrize(
    "old,new",
    [
        ("= value;", "+= value;"),
        (">> 30", ">> 29"),
        ("c_stride * i", "c_stride * j"),
        ("output_cols", "output_rows"),
        ("results.at(i).at(j)", "results.at(j).at(i)"),
    ],
)
def test_receiver_changes_cannot_supply_overwrite(adapter, old, new):
    with pytest.raises(ValueError):
        adapter.prove_receiver_overwrite(receiver().replace(old, new))


def test_comments_strings_and_duplicate_branch_cannot_supply_receiver_proof(adapter):
    for text in (
        "/*" + receiver() + "*/",
        '"' + receiver().replace('"', "") + '"',
        receiver() + receiver(),
    ):
        with pytest.raises(ValueError):
            adapter.prove_receiver_overwrite(text)


@pytest.fixture
def issued(adapter, monkeypatch, tmp_path):
    M = adapter
    m = modules(M)
    fir, cc = tmp_path / "selected.fir", tmp_path / "gemmini.cc"
    fir.write_text("neutral parsed RTL fixture")
    monkeypatch.setattr(M, "_REVIEWED_FIR_SHA256", M._pin(fir)["sha256"])
    cc.write_text(receiver())
    monkeypatch.setattr(M, "_REVIEWED_RECEIVER_SHA256", M._pin(cc)["sha256"])
    inputs = {"facts": {"sha256": "neutral"}}

    class Selected:
        def revalidate(self):
            pass

        def report(self):
            return {
                "selected_inputs": copy.deepcopy(inputs),
                "geometry": {
                    "array_cols": 2,
                    "stores": [{"name": "accumulator", "banks": 1}],
                },
            }

        def _documents(self):
            return (
                None,
                None,
                {
                    "layouts": {
                        "fields": [
                            {
                                "bundle": "PreloadRs",
                                "name": "local_addr",
                                "subfields": [{"field": "accumulate", "offset": 30}],
                            }
                        ]
                    }
                },
            )

    class Arithmetic:
        identity = "neutral live arithmetic"

        def __init__(self, selected):
            self._selected = selected

        def revalidate(self):
            pass

        def record(self):
            return {
                "selected_inputs": copy.deepcopy(inputs),
                "sources": {"firrtl": M._pin(fir), "gemmini.cc": M._pin(cc)},
            }

    monkeypatch.setattr(M, "SelectedSchedulingTarget", Selected)
    monkeypatch.setattr(M, "IntegerArithmeticOwner", Arithmetic)
    monkeypatch.setattr(
        M.F, "load_modules", lambda path, names: {n: m[n] for n in names}
    )
    selected = Selected()
    arithmetic = Arithmetic(selected)
    owner = M.SelectedAccumulatorOverwrite(selected, arithmetic)
    return M, owner, selected, arithmetic, fir, cc, inputs


def test_original_issue_and_detached_record(issued):
    M, owner, *_ = issued
    assert owner.record()["timing_authority"] == "NONE"
    saved = owner.record()
    saved["selected_inputs"]["facts"]["sha256"] = "changed"
    owner.revalidate()
    assert owner.record() != saved
    with pytest.raises(TypeError):
        M.SelectedAccumulatorOverwrite(saved, saved)
    with pytest.raises(ValueError):
        object.__new__(M.SelectedAccumulatorOverwrite).revalidate()


@pytest.mark.parametrize(
    "fault",
    [
        "selected",
        "arithmetic",
        "selected_binding",
        "arithmetic_identity",
        "receiver",
        "firrtl",
        "callback",
        "proof",
        "state",
    ],
)
def test_mutated_live_owner_or_custody_refuses(issued, monkeypatch, fault):
    M, owner, selected, arithmetic, fir, cc, inputs = issued
    if fault == "selected":
        owner.selected = M.SelectedSchedulingTarget()
    elif fault == "arithmetic":
        owner.arithmetic = M.IntegerArithmeticOwner(selected)
    elif fault == "selected_binding":
        inputs["facts"]["sha256"] = "changed"
    elif fault == "arithmetic_identity":
        arithmetic.identity = "changed"
    elif fault in {"receiver", "firrtl"}:
        (cc if fault == "receiver" else fir).write_text("changed")
    elif fault == "callback":
        arithmetic.revalidate = lambda: None
    elif fault == "state":
        M._ISSUED[owner]["evidence"]["accumulate_offset"] = 29
    else:
        monkeypatch.setattr(M, "prove_memory_overwrite", lambda *a: None)
    with pytest.raises(ValueError):
        owner.revalidate()


def stream(m, k, n, *, overwrite=True):
    result = [command("CONFIG_EX")]
    for m0 in range(0, m, 2):
        for n0 in range(0, n, 2):
            rows, cols = min(2, m - m0), min(2, n - n0)
            for k0 in range(0, k, 2):
                depth = min(2, k - k0)
                result += [
                    command("CONFIG_LD", stride=k),
                    command(
                        "MVIN",
                        1000 + m0 * k + k0,
                        rows=rows,
                        cols=depth,
                        addr=0,
                        is_acc_addr=False,
                    ),
                    command("CONFIG_LD", stride=n),
                    command(
                        "MVIN",
                        2000 + k0 * n + n0,
                        rows=depth,
                        cols=cols,
                        addr=2,
                        is_acc_addr=False,
                    ),
                    command(
                        "PRELOAD",
                        shape(depth, cols, 2),
                        shape(rows, cols, 128 | (0 if overwrite and k0 == 0 else 64)),
                    ),
                    command("FENCE"),
                    command(
                        "COMPUTE_PRELOADED", shape(rows, depth, 0), garbage=True, bd=255
                    ),
                ]
            result += [
                command(
                    "CONFIG_ST",
                    out_stride_bytes=4 * n,
                    acc_scale=1.0,
                    **{
                        f: 0
                        for f in (
                            "acc_act",
                            "pool_stride",
                            "pool_size",
                            "upad",
                            "lpad",
                            "pool_out_dim",
                            "porows",
                            "pocols",
                            "orows",
                            "ocols",
                        )
                    },
                ),
                command(
                    "MVOUT",
                    3000 + 4 * (m0 * n + n0),
                    rows=rows,
                    cols=cols,
                    addr=128 | 32,
                    readout="i32",
                ),
            ]
    return result


def check(
    issued, commands, m, k, n, *, effect=True, memory_alias=False, on_readout=None
):
    _M, owner, *_ = issued
    P = importlib.import_module("gemmini_packet_contractions")

    def memory(address, width):
        if width == 4 and on_readout is not None:
            on_readout()
        for base, count, tensor, size in (
            (1000, m * k, "A", 1),
            (2000, k * n, "A" if memory_alias else "B", 1),
            (3000, m * n, "C", 4),
        ):
            if (
                base <= address < base + count * size
                and width == size
                and (address - base) % size == 0
            ):
                return tensor, (address - base) // size
        raise ValueError("unowned memory")

    return P.check_packets(
        commands,
        contractions=[P.Contraction(3, "A", "B", "C", m, k, n)],
        dim=2,
        spad_rows=32,
        acc_rows=32,
        address_bits=8,
        data_bits=5,
        accumulator_flag=128,
        accumulate_flag=64,
        full_width_flag=32,
        memory_cell=memory,
        zero_cell=lambda *a: False,
        execute_contract=lambda *a: None,
        weight_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        compute_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        preload_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        load_configuration=lambda *a: None,
        accumulator_overwrite=owner if effect else None,
    )


@pytest.mark.parametrize(
    "m,k,n", [(1, 1, 1), (2, 2, 2), (3, 3, 3), (3, 5, 1), (1, 7, 3)]
)
def test_first_write_without_zero_seed_complete_tails_and_deep_k(issued, m, k, n):
    row = check(issued, stream(m, k, n), m, k, n)
    assert row["products"] == m * k * n
    assert row["raw_i32_values"] == m * n
    assert row["accumulator_effects"]["protocol"] == issued[0].PROTOCOL
    with pytest.raises(ValueError, match="uninitialized"):
        check(issued, stream(m, k, n), m, k, n, effect=False)


@pytest.mark.parametrize(
    "fault",
    [
        "accumulate_uninitialized",
        "later_overwrite",
        "omit_first",
        "duplicate_first",
        "missing_tail",
        "wrong_readout",
        "alias",
        "unowned_operand",
        "biased_load",
    ],
)
def test_overwrite_never_waives_prefix_ownership_or_required_work(issued, fault):
    m = k = n = 3
    rows = stream(m, k, n)
    pre = [i for i, r in enumerate(rows) if r["class"] == "PRELOAD"]
    if fault == "accumulate_uninitialized":
        rows[pre[0]]["rs2"]["raw"] |= 64
    elif fault == "later_overwrite":
        rows[pre[1]]["rs2"]["raw"] &= ~64
    elif fault == "omit_first":
        del rows[1 : pre[0] + 3]
    elif fault == "duplicate_first":
        rows[pre[0] + 3 : pre[0] + 3] = copy.deepcopy(rows[1 : pre[0] + 3])
    elif fault == "missing_tail":
        rows = rows[:-1]
    elif fault == "wrong_readout":
        rows[-1]["rs1"]["raw"] += 4
    elif fault == "unowned_operand":
        rows[2]["rs1"]["raw"] = 999
    elif fault == "biased_load":
        rows[1:1] = [
            command("CONFIG_LD", stride=0),
            command("MVIN", 4000, rows=2, cols=2, addr=128, is_acc_addr=True),
        ]
    with pytest.raises(ValueError):
        check(issued, rows, m, k, n, memory_alias=fault == "alias")


def test_no_boolean_or_saved_effect_authority(issued):
    M, owner, *_ = issued
    P = importlib.import_module("gemmini_packet_contractions")
    for bad in (
        True,
        {},
        owner.record(),
        object.__new__(M.SelectedAccumulatorOverwrite),
    ):
        with pytest.raises((TypeError, ValueError)):
            P.check_packets(
                [],
                contractions=[],
                dim=2,
                spad_rows=32,
                acc_rows=32,
                address_bits=8,
                data_bits=5,
                accumulator_flag=128,
                accumulate_flag=64,
                full_width_flag=32,
                memory_cell=None,
                zero_cell=None,
                execute_contract=None,
                weight_shape=None,
                compute_shape=None,
                preload_shape=None,
                load_configuration=None,
                accumulator_overwrite=bad,
            )


@pytest.mark.parametrize("fault", ["source", "callback"])
def test_final_revalidation_refuses_changes_during_complete_induction(issued, fault):
    _, _, _, arithmetic, _, receiver_path, _ = issued

    def change():
        if fault == "source":
            receiver_path.write_text("changed after first product checks")
        else:
            arithmetic.revalidate = lambda: None

    with pytest.raises(ValueError):
        check(issued, stream(3, 3, 3), 3, 3, 3, on_readout=change)


@pytest.mark.parametrize("field", ["data[0][0]", "mask[0]"])
def test_shadowing_execute_write_source_refuses(adapter, field):
    m = modules(adapter)
    body = m["Scratchpad"]
    row = next(
        r
        for r in body.connects
        if r.lhs == adapter.F.parse_expr("bank_ios_1[0].write.bits." + field)
    )
    body.connects.append(replace(row, rhs=adapter.F.Lit("UInt", 1, 0), guards=()))
    with pytest.raises(ValueError):
        prove(adapter, m)
