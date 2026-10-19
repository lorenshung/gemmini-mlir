"""Packet arithmetic checks source cones, not names or physical measurements."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_dma_service as S


def fixture(direction, sizes=(4, 8, 16), beat=4):
    prefix = "read_packets_" if direction == "read" else "write_packets_"
    moved = "bytes_read" if direction == "read" else "bytes_written"
    chosen = "read_packet" if direction == "read" else "best_write_packet"
    body = [
        "module Example :",
        "  input state : UInt<2>",
        "  reg req : { vaddr : UInt<32>}, clock",
        "  wire bytesLeft : UInt<8>",
        f"  output auto : {{ out : {{ a : {{valid : UInt<1>, bits : {{data : UInt<{beat * 8}>}}}}, d : {{valid : UInt<1>, bits : {{data : UInt<{beat * 8}>}}}}}}}}",
        "  wire untranslated_a : { bits : { tl_a : {size : UInt<5>}}}",
    ]
    for i, size in enumerate(sizes):
        log = size.bit_length() - 1
        body += [
            f"  wire {prefix}{i} : {{size : UInt<8>, lg_size : UInt<5>, vaddr : UInt<32>, {moved} : UInt<8>}}",
            f"  connect {prefix}{i}.size, UInt<8>({size})",
            f"  connect {prefix}{i}.lg_size, UInt<5>({log})",
            f"  connect {prefix}{i}.vaddr, cat(bits(req.vaddr, 31, {log}), UInt<{log}>(0))",
            f"  node left{i} = tail(sub(UInt<8>({size}), bits(req.vaddr, {log - 1}, 0)), 1)",
            f"  connect {prefix}{i}.{moved}, mux(lt(left{i}, bytesLeft), left{i}, bytesLeft)",
        ]
    previous = prefix + "0"
    for i in range(1, len(sizes)):
        name = chosen if i == len(sizes) - 1 else "choose" + str(i)
        body += [
            f"  node {name} = mux(gt({prefix}{i}.{moved}, {previous}.{moved}), {prefix}{i}, {previous})"
        ]
        previous = name
    if direction == "write":
        body += [
            f"  reg saved : {{size : UInt<8>, lg_size : UInt<5>, vaddr : UInt<32>, {moved} : UInt<8>}}, clock",
            "  node new_state = eq(state, UInt<2>(1))",
            "  node write_packet = mux(new_state, best_write_packet, saved)",
            f"  node write_beats = div(write_packet.size, UInt<8>({beat}))",
        ]
    body += [
        f"  connect untranslated_a.bits.tl_a.size, {chosen}.lg_size",
        f"  node next_vaddr = add(req.vaddr, {chosen}.{moved})",
    ]
    return S.F.parse_module("Example", list(enumerate(body, 1)))


def test_packet_rule_rederives_nonselected_sizes_and_tail_packet_envelopes():
    selector = S.PacketSelectors(fixture("read"), fixture("write"))
    assert selector.check_all_offsets_and_lengths()["cutpoint_cases"] == 512
    packet = selector.packet("read", 3, 2)
    assert packet["packet_bytes"] == 8 and packet["payload_bytes"] == 2
    assert (
        packet["virtual_packet_address"] == 0
        and packet["physical_packet_address"] is None
    )
    # Choosing the largest payload, with smaller tie, is not round_up(length, beat).
    assert packet["packet_bytes"] > 4
    assert selector.packet("read", 0, 2)["packet_bytes"] == 4


def test_source_selector_changes_refuse_rather_than_reuse_known_formula():
    writer = fixture("write")
    expression, line, locator = writer.nodes["best_write_packet"]
    predicate = expression.args[0]
    writer.nodes["best_write_packet"] = (
        S.F.Prim("mux", (S.F.Prim("lt", predicate.args), *expression.args[1:])),
        line,
        locator,
    )
    selector = S.PacketSelectors(fixture("read"), writer)
    with pytest.raises(ValueError, match="packet cone differs"):
        selector.packet("write", 0, 16)


def test_missing_or_out_of_scope_packet_inputs_stay_refused():
    selector = S.PacketSelectors(fixture("read"), fixture("write"))
    for address, length in [(-1, 1), (0, 0), (0, 17), (2**40, 1)]:
        with pytest.raises(ValueError):
            selector.packet("read", address, length)
    for direction, length in [("read", 0), ("write", -1), ("other", 1)]:
        with pytest.raises(ValueError):
            selector.row(direction, 0, length)


def test_packet_inventory_preserves_payload_separate_from_full_bus_beats():
    selector = S.PacketSelectors(fixture("read"), fixture("write"))
    packets = selector.row("write", 3, 16)
    assert sum(row["payload_bytes"] for row in packets) == 16
    assert sum(row["packet_bytes"] for row in packets) > 16
    assert all(
        row["data_beats"] * row["data_beat_bytes"] == row["packet_bytes"]
        for row in packets
    )
    assert all(row["physical_packet_address"] is None for row in packets)


def test_saved_receipt_cannot_construct_live_source_owner():
    with pytest.raises(ValueError, match="original selected source owner"):
        S.DmaServiceContract({"status": "verified", "timing_qualified": True})
