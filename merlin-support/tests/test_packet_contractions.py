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


def check(
    packet,
    commands,
    *,
    k=3,
    zero=lambda a, w: 4000 <= a < 4008,
    load_streams=None,
    zero_fill=None,
):
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
        load_streams=load_streams,
        zero_fill=zero_fill,
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


def banked_stream():
    commands = stream()
    for row in commands:
        if row["class"] == "CONFIG_LD":
            row["decoded"]["state_id"] = {0: 2, 3: 0, 2: 1}[row["decoded"]["stride"]]
        elif row["class"] == "MVIN":
            row["class"] = (
                "MVIN3"
                if row["rs1"]["raw"] == 4000
                else "MVIN2"
                if row["rs1"]["raw"] >= 2000
                else "MVIN"
            )
    # Configure every state before any load: scalar 'last stride' would be 2,
    # which is wrong for A (stride3) and zero D (stride0).
    configurations = [row for row in commands if row["class"] == "CONFIG_LD"]
    return [
        commands[0],
        *configurations[:3],
        *[row for row in commands[1:] if row["class"] != "CONFIG_LD"],
    ]


def test_selected_load_streams_keep_independent_a_b_and_zero_strides(packet):
    report = check(
        packet, banked_stream(), load_streams={"MVIN": 0, "MVIN2": 1, "MVIN3": 2}
    )
    assert report["products"] == 18 and report["raw_i32_values"] == 6


@pytest.mark.parametrize(
    "fault",
    [
        "unconfigured",
        "foreign_state",
        "wrong_mapping",
        "overwrite_a",
        "duplicate",
        "bool",
    ],
)
def test_selected_stream_state_or_mapping_mutations_refuse(packet, fault):
    commands = banked_stream()
    mapping = {"MVIN": 0, "MVIN2": 1, "MVIN3": 2}
    if fault == "unconfigured":
        del commands[2]
    elif fault == "foreign_state":
        commands[1]["decoded"]["state_id"] = 3
    elif fault == "wrong_mapping":
        mapping["MVIN"], mapping["MVIN2"] = 1, 0
    elif fault == "overwrite_a":
        commands.insert(4, command("CONFIG_LD", stride=2, state_id=0))
    elif fault == "duplicate":
        mapping["MVIN3"] = 0
    else:
        mapping["MVIN3"] = True
    with pytest.raises(ValueError):
        check(packet, commands, load_streams=mapping)


def test_legacy_default_does_not_admit_additional_load_opcodes(packet):
    with pytest.raises(ValueError):
        check(packet, banked_stream())


def generated_zero_stream():
    commands = banked_stream()
    for row in commands:
        if row["class"] == "MVIN3":
            row["rs1"]["raw"] = 0
    return commands


def test_explicit_selected_zero_generator_is_separate_from_elf_zero_memory(packet):
    commands = generated_zero_stream()
    mapping = {"MVIN": 0, "MVIN2": 1, "MVIN3": 2}
    with pytest.raises(ValueError, match="independently proven zero"):
        check(packet, commands, load_streams=mapping, zero=lambda a, w: False)
    report = check(
        packet,
        commands,
        load_streams=mapping,
        zero=lambda a, w: False,
        zero_fill=lambda row: (
            row["rs1"]["raw"] == 0 and row["decoded"]["is_acc_addr"] is True
        ),
    )
    assert report["products"] == 18 and report["raw_i32_values"] == 6


@pytest.mark.parametrize(
    "fault", ["nonzero_address", "accumulate", "scratchpad", "untyped_grant"]
)
def test_zero_fill_cannot_initialize_foreign_or_accumulating_storage(packet, fault):
    commands = generated_zero_stream()
    zero = next(row for row in commands if row["class"] == "MVIN3")
    callback = lambda row: (
        row["rs1"]["raw"] == 0 and row["decoded"]["is_acc_addr"] is True
    )
    if fault == "nonzero_address":
        zero["rs1"]["raw"] = 1
    elif fault == "accumulate":
        zero["decoded"]["addr"] |= 64
    elif fault == "scratchpad":
        callback = lambda row: True
    else:
        callback = lambda row: 1
    with pytest.raises(ValueError):
        check(
            packet,
            commands,
            zero=lambda a, w: False,
            load_streams={"MVIN": 0, "MVIN2": 1, "MVIN3": 2},
            zero_fill=callback,
        )
