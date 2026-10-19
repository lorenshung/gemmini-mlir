"""Adversarial state/product checks on synthetic operands, not qualification."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def packet(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_packet_contractions")


def command(kind, first=0, second=0, **decoded):
    return {
        "class_": kind,
        "rs1": {"raw": first},
        "rs2": {"raw": second},
        "decoded": decoded,
    } | {"class": kind}


def shape(rows, cols, addr):
    return rows << 16 | cols << 8 | addr


def stream(k=3, *, reuse=False):
    result = [command("CONFIG_EX")]
    for m0, rows in ((0, 2), (2, 1)):
        result += [
            command("CONFIG_LD", stride=0),
            command("MVIN", 4000, rows=rows, cols=2, addr=128, is_acc_addr=True),
        ]
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
                command("CONFIG_LD", stride=2),
                command(
                    "MVIN", 2000 + k0 * 2, rows=depth, cols=2, addr=2, is_acc_addr=False
                ),
                command(
                    "PRELOAD",
                    shape(depth, 2, 255 if reuse and m0 else 2),
                    shape(rows, 2, 128 | 64),
                ),
                command("FENCE"),
                command(
                    "COMPUTE_ACCUMULATE" if reuse and m0 else "COMPUTE_PRELOADED",
                    shape(rows, depth, 0),
                    garbage=True,
                    bd=255,
                ),
            ]
        result += [
            command(
                "CONFIG_ST",
                out_stride_bytes=8,
                acc_scale=1.0,
                **{
                    key: 0
                    for key in (
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
                "MVOUT", 3000 + m0 * 8, rows=rows, cols=2, addr=128 | 32, readout="i32"
            ),
        ]
    return result


def check(packet, commands, *, k=3, zero=lambda a, w: 4000 <= a < 4008):
    def memory(address, width):
        for start, count, name, element in (
            (1000, 3 * k, "A", 1),
            (2000, k * 2, "B", 1),
            (3000, 6, "C", 4),
        ):
            if (
                start <= address < start + count * element
                and width == element
                and (address - start) % width == 0
            ):
                return name, (address - start) // width
        raise ValueError("actual address lacks original tensor ownership")

    return packet.check_packets(
        commands,
        contractions=[packet.Contraction(7, "A", "B", "C", 3, k, 2)],
        dim=2,
        spad_rows=32,
        acc_rows=32,
        address_bits=8,
        data_bits=5,
        accumulator_flag=128,
        accumulate_flag=64,
        full_width_flag=32,
        memory_cell=memory,
        zero_cell=zero,
        execute_contract=lambda a, b: None,
        weight_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        compute_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        preload_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
        load_configuration=lambda row: None,
    )


def test_independent_m_and_k_tails_complete_products_and_pending_fences(packet):
    report = check(packet, stream())
    assert report["products"] == 18
    assert report["raw_i32_values"] == 6
    assert report["computes"] == 4
    assert report["fences_preserving_pending_preload"] == 4
    assert report["hardware_qualification"] is False


def test_retained_weight_values_survive_scratchpad_clobber(packet):
    commands = stream(2, reuse=True)
    # The second M tile overwrites the former B scratchpad cells with A. The
    # retained PE weights must still denote the first independently loaded B.
    second = [i for i, c in enumerate(commands) if c["class"] == "PRELOAD"][1]
    commands[second - 1] = command(
        "MVIN", 1000, rows=2, cols=2, addr=2, is_acc_addr=False
    )
    report = check(packet, commands, k=2)
    assert report["products"] == 12


@pytest.mark.parametrize(
    "mutation",
    [
        "no_compute",
        "no_zero",
        "bad_d",
        "wrong_k",
        "clobber_a",
        "wrong_output",
        "duplicate_output",
        "pending_ex",
        "pending_end",
        "unknown",
        "reuse_without_weights",
    ],
)
def test_effect_and_required_work_mutations_refuse(packet, mutation):
    commands = copy.deepcopy(stream())
    compute = next(
        i for i, c in enumerate(commands) if c["class"] == "COMPUTE_PRELOADED"
    )
    preload = compute - 2
    readout = next(i for i, c in enumerate(commands) if c["class"] == "MVOUT")
    if mutation == "no_compute":
        del commands[compute]
    elif mutation == "no_zero":
        del commands[1:3]
    elif mutation == "bad_d":
        commands[compute]["decoded"]["garbage"] = False
    elif mutation == "wrong_k":
        commands[compute]["rs1"]["raw"] = shape(2, 1, 0)
    elif mutation == "clobber_a":
        commands.insert(
            compute, command("MVIN", 2000, rows=2, cols=2, addr=0, is_acc_addr=False)
        )
    elif mutation == "wrong_output":
        commands[readout]["rs1"]["raw"] += 4
    elif mutation == "duplicate_output":
        commands.insert(readout, copy.deepcopy(commands[readout]))
    elif mutation == "pending_ex":
        commands.insert(compute, command("CONFIG_EX"))
    elif mutation == "pending_end":
        commands = commands[: preload + 1] + [command("FENCE")]
    elif mutation == "unknown":
        commands.insert(compute, command("FOREIGN_EFFECT"))
    else:
        commands[compute]["class"] = "COMPUTE_ACCUMULATE"
    with pytest.raises(ValueError):
        check(packet, commands)


def test_nonzero_or_unowned_initialization_refuses(packet):
    with pytest.raises(ValueError, match="independently proven zero"):
        check(packet, stream(), zero=lambda address, width: False)
