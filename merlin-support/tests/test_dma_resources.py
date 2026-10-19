"""Local source cone checks are independent of saved status/timing verdicts."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_dma_resources as R


def parse(name, lines):
    return R.F.parse_module(
        name, list(enumerate((f"module {name} :\n" + lines).splitlines(), 1))
    )


def fixtures():
    channel = "{ flip ready : UInt<1>, valid : UInt<1>, bits : { source : UInt<2>, data : UInt<8>, size : UInt<3>, opcode : UInt<3>}}"
    common = f"  output auto : {{ out : {{ a : {channel}, flip d : {channel}}}}}\n  wire nodeOut : {{ a : {channel}, flip d : {channel}}}\n"
    reader = parse(
        "Reader",
        common
        + """  reg state : UInt<1>, clock
  output io : {reserve : {valid : UInt<1>, flip ready : UInt<1>, flip xactid : UInt<2>}, beatData : {valid : UInt<1>, flip ready : UInt<1>, bits : {last : UInt<1>}}}
  wire untranslated_a : {valid : UInt<1>, flip ready : UInt<1>, bits : {tl_a : {opcode : UInt<3>}}}
  connect untranslated_a.bits.tl_a.opcode, UInt<3>(4)
  connect untranslated_a.valid, and(eq(state, UInt<1>(1)), io.reserve.ready)
  connect io.reserve.valid, and(eq(state, UInt<1>(1)), untranslated_a.ready)
  connect nodeOut.d.ready, io.beatData.ready
  connect io.beatData.valid, nodeOut.d.valid
  regreset io_beatData_bits_last_counter : UInt<3>, clock, reset, UInt<3>(0)
  node beats1 = tail(sub(dshl(UInt<1>(1), nodeOut.d.bits.size), UInt<1>(1)), 1)
  node last = or(eq(io_beatData_bits_last_counter, UInt<1>(1)), eq(beats1, UInt<1>(0)))
  connect io.beatData.bits.last, last
  node fire = and(nodeOut.d.ready, nodeOut.d.valid)
  node next_counter = mux(eq(io_beatData_bits_last_counter, UInt<1>(0)), beats1, tail(sub(io_beatData_bits_last_counter, UInt<1>(1)), 1))
  when fire :
    connect io_beatData_bits_last_counter, next_counter
""",
    )
    writer = parse(
        "Writer",
        common
        + """  output io : {busy : UInt<1>}
  reg state : UInt<2>, clock
  reg xactBusy : UInt<4>, clock
  wire write_packet : {is_full : UInt<1>}
  wire untranslated_a : {valid : UInt<1>, flip ready : UInt<1>, bits : {tl_a : {opcode : UInt<3>}}}
  connect untranslated_a.bits.tl_a.opcode, mux(write_packet.is_full, UInt<3>(0), UInt<3>(1))
  node xactId = mux(eq(bits(xactBusy, 0, 0), UInt<1>(0)), UInt<2>(0), mux(eq(bits(xactBusy, 1, 1), UInt<1>(0)), UInt<2>(1), mux(eq(bits(xactBusy, 2, 2), UInt<1>(0)), UInt<2>(2), UInt<2>(3))))
  connect untranslated_a.valid, and(or(eq(state, UInt<2>(1)), eq(state, UInt<2>(2))), not(andr(xactBusy)))
  wire xactBusy_fire : UInt<1>
  connect xactBusy_fire, and(and(untranslated_a.valid, untranslated_a.ready), eq(state, UInt<2>(1)))
  connect nodeOut.d.ready, orr(xactBusy)
  connect io.busy, or(orr(xactBusy), neq(state, UInt<2>(0)))
  node next_busy = and(or(xactBusy, mux(xactBusy_fire, dshl(UInt<1>(1), xactId), UInt<1>(0))), not(mux(and(nodeOut.d.ready, nodeOut.d.valid), dshl(UInt<1>(1), nodeOut.d.bits.source), UInt<1>(0))))
  connect xactBusy, next_busy
""",
    )
    tracker_lines = """  output io : {alloc : {valid : UInt<1>, flip ready : UInt<1>, flip xactid : UInt<2>}, peek : {flip pop : UInt<1>, flip xactid : UInt<2>}, busy : UInt<1>}
  reg entries : {valid : UInt<1>}[4], clock
  input reset : UInt<1>
  node free_entry = mux(not(entries[0].valid), UInt<2>(0), mux(not(entries[1].valid), UInt<2>(1), mux(not(entries[2].valid), UInt<2>(2), UInt<2>(3))))
  connect io.alloc.xactid, free_entry
  connect io.alloc.ready, not(and(and(entries[0].valid, entries[1].valid), and(entries[2].valid, entries[3].valid)))
  connect io.busy, or(or(entries[0].valid, entries[1].valid), or(entries[2].valid, entries[3].valid))
  node allocate = and(io.alloc.valid, io.alloc.ready)
  when allocate :
    connect entries[free_entry].valid, UInt<1>(1)
  when io.peek.pop :
    connect entries[io.peek.xactid].valid, UInt<1>(0)
  when reset :
""" + "\n".join(f"    connect entries[{i}].valid, UInt<1>(0)" for i in range(4))
    tracker = parse("Tracker", tracker_lines)
    merger = parse("Merger", "  output io : {in : {ready : UInt<1>, valid : UInt<1>}}")
    wrapper = parse(
        "Wrapper",
        """  inst core of Reader
  inst beatPacker of Merger
  inst xactTracker of Tracker
  connect xactTracker.io.peek.pop, and(and(beatPacker.io.in.ready, beatPacker.io.in.valid), core.io.beatData.bits.last)
""",
    )
    modules = {"Reader": reader, "Writer": writer, "Tracker": tracker, "Merger": merger}
    return reader, writer, tracker, wrapper, modules


def cones(parts=None):
    return R.DmaOwnershipCones(*(parts or fixtures()), new_write_state=1)


def test_source_cones_check_nonselected_geometry_and_finite_ownership():
    owner = cones()
    assert owner.read_slots == owner.write_slots == 4
    assert owner.check()["local_cutpoint_cases"] > 100
    assert owner.check()["reachability_proven"] is False


def test_source_id_can_survive_idle_until_response_and_release_is_not_local_write():
    owner = cones()
    assert owner.writer_step(
        state=0, occupied=2, request_ready=1, response_valid=0, response_source=1
    ) == {
        "allocate": 0,
        "acknowledge": 0,
        "occupied_next": 2,
        "source": None,
        "busy": 1,
        "request_valid": 0,
        "response_ready": 1,
    }
    assert owner.reader_release(ready=0, valid=1, last=1) == 0
    assert owner.reader_release(ready=1, valid=1, last=1) == 1
    assert (
        owner.writer_step(
            state=0, occupied=2, request_ready=0, response_valid=1, response_source=1
        )["occupied_next"]
        == 0
    )


def test_writer_continuation_does_not_allocate_again_and_full_store_stalls():
    owner = cones()
    assert (
        owner.writer_step(
            state=2, occupied=1, request_ready=1, response_valid=0, response_source=0
        )["allocate"]
        == 0
    )
    result = owner.writer_step(
        state=1, occupied=15, request_ready=1, response_valid=1, response_source=0
    )
    assert result["allocate"] == result["request_valid"] == 0
    assert result["occupied_next"] == 14


def test_same_cycle_new_source_and_existing_ack_preserve_both_effects():
    result = cones().writer_step(
        state=1, occupied=2, request_ready=1, response_valid=1, response_source=1
    )
    assert result["allocate"] == result["acknowledge"] == 1
    assert result["source"] == 0 and result["occupied_next"] == 1


def test_packet_response_beats_do_not_advance_without_handshake():
    owner = cones()
    for counter in range(4):
        assert (
            owner.reader_response_step(
                packet_bytes=4, beat_bytes=1, counter=counter, ready=0, valid=1
            )["counter_next"]
            == counter
        )
    assert (
        owner.reader_response_step(
            packet_bytes=4, beat_bytes=1, counter=1, ready=1, valid=1
        )["accepted_final"]
        == 1
    )
    with pytest.raises(ValueError, match="complete-beat"):
        owner.reader_response_step(
            packet_bytes=4, beat_bytes=2, counter=1, ready=1, valid=1
        )


@pytest.mark.parametrize("mutation", ["allocation", "release", "busy"])
def test_mutated_source_effects_refuse_instead_of_reusing_known_equation(mutation):
    parts = fixtures()
    _reader, writer, tracker, _wrapper, _ = parts
    if mutation == "allocation":
        row = next(
            row
            for row in tracker.connects
            if row.lhs == R.F.parse_expr("io.alloc.ready")
        )
        tracker.connects[tracker.connects.index(row)] = replace(
            row, rhs=R.F.Lit("UInt", 1, 1)
        )
    elif mutation == "release":
        expression, line, locator = writer.nodes["next_busy"]
        writer.nodes["next_busy"] = (R.F.Prim("or", expression.args), line, locator)
    else:
        row = next(
            row for row in writer.connects if row.lhs == R.F.parse_expr("io.busy")
        )
        writer.connects[writer.connects.index(row)] = replace(
            row, rhs=R.F.Lit("UInt", 1, 0)
        )
    with pytest.raises(ValueError):
        cones(parts).check()


@pytest.mark.parametrize(
    "arguments",
    [
        {"state": 3, "occupied": 0, "response_source": 0},
        {"state": 1, "occupied": -1, "response_source": 0},
        {"state": 1, "occupied": 1, "response_source": 4},
    ],
)
def test_unowned_or_out_of_scope_responses_fail_closed(arguments):
    with pytest.raises(ValueError):
        cones().writer_step(**arguments, request_ready=1, response_valid=1)
    with pytest.raises(ValueError, match="no outstanding"):
        cones().writer_step(
            state=1, occupied=0, request_ready=1, response_valid=1, response_source=0
        )


def test_saved_receipt_cannot_issue_original_contract():
    with pytest.raises(ValueError, match="original selected source owner"):
        R.DmaTransferContract({"status": "complete", "timing_qualified": True})


def test_saved_source_and_dma_receipts_cannot_issue_live_join():
    from gemmini_source_dma_transfers import SourceDmaTransferEffects

    with pytest.raises(ValueError, match="original live source/resource owners"):
        SourceDmaTransferEffects(
            source_effects={"status": "pass"}, transfer_contract={"status": "pass"}
        )
