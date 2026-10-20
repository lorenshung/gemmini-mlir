"""Decoded structural controls, never live target qualification fixtures."""

import copy

import gemmini_v26_staged_actual_pair as live
import pytest
from gemmini_v26_staged_pair_effects import compare_decoded_pair
from merlin.compile.scheduling.staged_dense import StagedDenseContract

PORTS = {"MVIN": 1, "MVIN2": 0, "MVIN3": 2}


def packed(addr, rows, cols):
    return addr | rows << 48 | cols << 32


def row(kind, first=0, second=0, **decoded):
    funct = {
        "CONFIG_EX": 0,
        "CONFIG_LD": 0,
        "CONFIG_ST": 0,
        "MVIN": 1,
        "MVIN2": 2,
        "MVIN3": 14,
        "PRELOAD": 6,
        "COMPUTE_PRELOADED": 4,
        "MVOUT": 3,
    }.get(kind, 0)
    raw = (
        first
        if type(first) is int
        else {"lhs": 0x1000, "rhs": 0x2000, "out": 0x3000, "private_zero": 0x4000}[
            first[0]
        ]
        + first[1]
    )
    return (kind, first, second), {
        "class": kind,
        "funct": funct,
        "rs1": {"raw": raw},
        "rs2": {"raw": second},
        "decoded": decoded,
        "transport": (funct << 25) | (3 << 12) | 0x7B,
    }


def fence(word=0x0FF0000F):
    return ("FENCE", 0, 0), {"class": "FENCE", "decoded": {}, "transport": word}


def tile(m, n, *, msize=4, nsize=4):
    mr, nr = min(2, msize - m), min(2, nsize - n)
    return [
        fence(),
        row(
            "MVIN3",
            ("private_zero", 0),
            packed(0x80000000, mr, nr),
            addr=0x80000000,
            rows=mr,
            cols=nr,
            is_acc_addr=True,
        ),
        row(
            "MVIN2",
            ("lhs", m * 2),
            packed(0, mr, 2),
            addr=0,
            rows=mr,
            cols=2,
            is_acc_addr=False,
        ),
        row(
            "MVIN",
            ("rhs", n),
            packed(2, 2, nr),
            addr=2,
            rows=2,
            cols=nr,
            is_acc_addr=False,
        ),
        row("PRELOAD", packed(2, 2, nr), packed(0xC0000000, mr, nr)),
        row("COMPUTE_PRELOADED", packed(0, mr, 2), packed(0xFFFFFFFF, mr, nr)),
        fence(),
        row("CONFIG_ST", 2, (0x3F800000 << 32) | nsize * 4, out_stride_bytes=nsize * 4),
        row(
            "MVOUT",
            ("out", (m * nsize + n) * 4),
            packed(0xA0000000, mr, nr),
            addr=0xA0000000,
            rows=mr,
            cols=nr,
            readout="i32",
        ),
        fence(),
    ]


def program(order, *, m=4, n=4, ports=PORTS, omit=False):
    prefix = [fence(), row("CONFIG_EX", 9, 8)]
    for kind, stride in [("MVIN2", 2), ("MVIN", n), ("MVIN3", 0)]:
        state = ports[kind]
        prefix.append(
            row("CONFIG_LD", state * 8 + 1, stride, state_id=state, stride=stride)
        )
    seen, body = set(), []
    for mi, nj in order:
        block = tile(mi, nj, msize=m, nsize=n)
        if omit and nj in seen:
            del block[3]
        seen.add(nj)
        body.extend(block)
    return (*prefix, *body, fence(), fence())


def check(before, after, **kwargs):
    return compare_decoded_pair(
        before,
        after,
        ports=kwargs.pop("ports", PORTS),
        rhs="rhs",
        operand_rows=4,
        dim=2,
        **kwargs,
    )


@pytest.mark.parametrize("m,n", [(4, 4), (3, 3), (1, 4), (4, 1)])
def test_exact_fence_groups_per_state_configuration_and_live_cells(m, n):
    before = program(
        [(mi, nj) for mi in range(0, m, 2) for nj in range(0, n, 2)], m=m, n=n
    )
    after = program(
        [(mi, nj) for nj in range(0, n, 2) for mi in range(0, m, 2)],
        m=m,
        n=n,
        omit=True,
    )
    original = copy.deepcopy((before, after))
    report = check(before, after)
    assert report["actual_rhs_loads_removed"] == ((m + 1) // 2 - 1) * ((n + 1) // 2)
    assert report[
        "original_leading_reduction_readout_fences_and_configurations_preserved"
    ]
    assert (
        report["target_capability_issued"] is False
        and report["timing_authority"] == "NONE"
    )
    assert original == (before, after)
    assert check(before, before)["actual_rhs_loads_removed"] == 0


def test_selected_mapping_is_explicit_not_class_name_or_last_setter():
    ports = {"MVIN": 2, "MVIN2": 1, "MVIN3": 0}
    before = program([(0, 0), (0, 2), (2, 0), (2, 2)], ports=ports)
    after = program([(0, 0), (2, 0), (0, 2), (2, 2)], ports=ports, omit=True)
    assert check(before, after, ports=ports)["actual_rhs_loads_removed"] == 2
    with pytest.raises(ValueError, match="load-state configuration"):
        # A data-only helper cannot infer hardware mapping. The live wrapper
        # supplies SelectedLoadStreams; an unconfigured declared port refuses.
        check(before, after, ports={"MVIN": 9, "MVIN2": 1, "MVIN3": 0})


@pytest.mark.parametrize(
    "fault",
    [
        "drop_tile",
        "duplicate_tile",
        "readout",
        "leading_fence",
        "reduction_fence",
        "readout_fence",
        "extra_fence",
        "fence_bits",
        "config",
        "state",
        "compute",
        "transport",
        "first_rhs",
        "unknown",
        "missing_decode",
    ],
)
def test_changes_outside_exact_tile_permutation_and_reload_refuse(fault):
    before = program([(0, 0), (0, 2), (2, 0), (2, 2)])
    after = list(copy.deepcopy(program([(0, 0), (2, 0), (0, 2), (2, 2)], omit=True)))
    if fault == "drop_tile":
        del after[5:15]
    elif fault == "duplicate_tile":
        after[15:24] = copy.deepcopy(after[5:15])
    elif fault == "readout":
        after[13] = row(
            "MVOUT",
            ("out", 4),
            packed(0xA0000000, 2, 2),
            addr=0xA0000000,
            rows=2,
            cols=2,
            readout="i32",
        )
    elif fault in {"leading_fence", "reduction_fence", "readout_fence"}:
        del after[
            {"leading_fence": 5, "reduction_fence": 11, "readout_fence": 14}[fault]
        ]
    elif fault == "extra_fence":
        after.insert(11, fence())
    elif fault == "fence_bits":
        after[5] = fence(0x0330000F)
    elif fault == "config":
        after[12] = row("CONFIG_ST", 2, 17, out_stride_bytes=17)
    elif fault == "state":
        after[3] = row("CONFIG_LD", 9, 5, state_id=1, stride=5)
    elif fault == "compute":
        after[10] = row("COMPUTE_PRELOADED", 1, 2)
    elif fault == "transport":
        after[10][1]["transport"] ^= 1 << 7
    elif fault == "first_rhs":
        del after[8]
    elif fault == "unknown":
        after.insert(10, row("UNKNOWN", 1, 2))
    else:
        del after[8][1]["decoded"]
    with pytest.raises(ValueError):
        check(before, tuple(after))


def test_lhs_clobber_invalidates_retained_rhs_cells():
    before = list(program([(0, 0), (0, 2), (2, 0), (2, 2)]))
    after = list(program([(0, 0), (2, 0), (0, 2), (2, 2)], omit=True))
    for stream in (before, after):
        for i, (event, command) in enumerate(stream):
            if event[0] == "MVIN2" and event[1] == ("lhs", 4):
                stream[i] = row(
                    "MVIN2",
                    ("lhs", 4),
                    packed(2, 2, 2),
                    addr=2,
                    rows=2,
                    cols=2,
                    is_acc_addr=False,
                )
    with pytest.raises(ValueError, match="live exact"):
        check(tuple(before), tuple(after))


def test_same_actual_stream_projects_into_original_shared_source_verifier():
    binding = {"lhs": "lhs", "rhs": "rhs", "output": "out", "m": 4, "k": 2, "n": 4}
    contract = StagedDenseContract(
        "facts",
        "sw",
        "numeric",
        "execution",
        2,
        4,
        2,
        8,
        20,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        False,
    )

    class ShapeFixture:
        def identity_ws(self, *args):
            pass

        def weight_shape(self, raw):
            return raw >> 48, (raw >> 32) & 0xFFFF

    stream = program([(0, 0), (2, 0), (0, 2), (2, 2)], omit=True)
    report = live._logical(
        stream,
        binding=binding,
        contract=contract,
        shape=lambda bundle, raw: (raw >> 48, (raw >> 32) & 0xFFFF),
        data_mask=(1 << 29) - 1,
        continuation=1 << 30,
        ex=ShapeFixture(),
        ports=PORTS,
    )
    assert report["products"] == 32 and report["outputs"] == 16
    assert report["computes"] == 4 and report["timing_authority"] == "NONE"


@pytest.mark.parametrize("value", [None, {}, {"status": "PASS"}, object()])
def test_saved_verdict_or_foreign_owner_cannot_enter_live_checker(value):
    with pytest.raises(TypeError, match="original live"):
        live.check_actual_staged_pair(
            baseline_owner=value,
            candidate_owner=value,
            source=None,
            proof=None,
            capsule_dir=None,
            completion=value,
        )


def test_original_callback_and_current_helper_replacement_refuse(monkeypatch):
    owner = object.__new__(live.SelectedLoadStreams)
    owner.revalidate = lambda: None
    with pytest.raises(ValueError, match="bound callback"):
        live._original_methods(owner)
    monkeypatch.setattr(
        live, "compare_decoded_pair", lambda *a, **k: {"status": "PASS"}
    )
    with pytest.raises(ValueError, match="checker callback"):
        live._current_implementation()


def test_complete_transport_retains_fence_and_rocc_destination_bits():
    opcode = 0x7B
    custom = (6 << 25) | (7 << 15) | (8 << 20) | (3 << 12) | opcode
    log = f"core 0: 3 0x8000 (0x{custom:08x})\ncore 0: 3 0x8004 (0x0330000f)\n"
    result = live._transport(log, opcode)
    assert result == [custom & ~((31 << 15) | (31 << 20)), 0x0330000F]
    with pytest.raises(ValueError, match="single-hart"):
        live._transport(log.replace("core 0:", "core 1:"), opcode)


def test_imported_checker_source_and_original_owner_method_remain_live(monkeypatch):
    source = next(iter(live._IMPORTED_SOURCES))
    pin = dict(live._IMPORTED_SOURCES[source])
    pin["sha256"] = "0" * 64
    with monkeypatch.context() as changed:
        changed.setitem(live._IMPORTED_SOURCES, source, pin)
        with pytest.raises(ValueError):
            live._current_implementation()
    owner = object.__new__(live.SelectedLoadStreams)
    monkeypatch.setattr(live.SelectedLoadStreams, "revalidate", lambda self: None)
    with pytest.raises(ValueError, match="bound callback"):
        live._original_methods(owner)
