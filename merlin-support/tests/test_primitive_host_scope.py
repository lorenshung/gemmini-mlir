"""Actual call scope rejects CPU output stores while preserving setup writes."""

import pytest
from gemmini_primitive_host_scope import check_host_output_writes


def _line(pc, word, tail=""):
    return f"core 0: 3 0x{pc:016x} (0x{word:08x}) {tail}"


def _log(*, inside=(), before=(), after=(), call=0x000000EF, link=0x104):
    return "\n".join(
        [
            *before,
            _line(0x100, call, f"x1 0x{link:016x}"),
            _line(0x200, 0x00000013),
            *inside,
            _line(0x204, 0x00008067),
            _line(link, 0x00000013),
            *after,
        ]
    )


def _check(log):
    return check_host_output_writes(
        log, entry=0x200, main=(0x100, 0x80), outputs=[(0x1000, 64)], custom_opcode=0x7B
    )


def test_setup_poison_is_outside_current_kernel_and_never_subtracted_from_cycles():
    checked = _check(_log(before=[_line(0x90, 0x00002023, "mem 0x1000 0x00000001")]))
    assert checked["kernel_cpu_output_writes"] == 0
    assert checked["target_cycles"] is None


def test_device_writes_are_checked_by_independent_packet_owner_not_mislabeled_cpu():
    assert (
        _check(_log(inside=[_line(0x202, 0x7B, "mem 0x1000 0x01")]))[
            "kernel_cpu_output_writes"
        ]
        == 0
    )


@pytest.mark.parametrize("pc", [0x202, 0x900])
def test_direct_or_callee_cpu_output_overwrite_refuses(pc):
    with pytest.raises(ValueError, match="CPU overwrites"):
        _check(_log(inside=[_line(pc, 0x00002023, "mem 0x1000 0x00000001")]))


def test_unrelated_stack_stores_remain_allowed_accelerator_feeding_work():
    checked = _check(_log(inside=[_line(0x202, 0x00002023, "mem 0x9000 0x00000001")]))
    assert checked["kernel_retired_instructions_observed"] == 3


@pytest.mark.parametrize(
    "call,link", [(0x00000013, 0x104), (0x000000EF, 0x108), (0x0000006F, 0x104)]
)
def test_invented_or_inconsistent_link_cannot_select_scope(call, link):
    with pytest.raises(ValueError, match="link"):
        _check(_log(call=call, link=link))


def test_missing_actual_return_refuses():
    with pytest.raises(ValueError, match="did not close"):
        _check("\n".join(_log().splitlines()[:-1]))


def test_multiple_actual_entries_refuse():
    with pytest.raises(ValueError, match="ambiguous or repeated"):
        _check(_log(after=[_line(0x108, 0xEF, "x1 0x10c"), _line(0x200, 0x13)]))


def _initialized(*, damage=None):
    stores = [
        _line(0x202, 0xC218, f"mem 0x{address:x} 0x00000000")
        for address in range(0x1000, 0x1040, 4)
    ]
    commands = [{"class": "FLUSH", "funct": 7}, {"class": "MVIN", "funct": 2}]
    first, work = _line(0x202, 7 << 25 | 0x7B), _line(0x202, 2 << 25 | 0x7B)
    if damage == "missing":
        stores.pop()
    elif damage == "repeated":
        stores.append(stores[0])
    elif damage == "nonzero":
        stores[0] = stores[0].replace("0x00000000", "0x00000001")
    elif damage == "partial":
        stores[0] = _line(0x202, 0x00001023, "mem 0x1000 0x0000")
    elif damage == "atomic":
        stores[0] = _line(0x202, 0x0000202F, "mem 0x1000 0x00000000")
    elif damage == "unaligned":
        stores[0] = stores[0].replace("0x1000", "0x1002")
    elif damage == "framing":
        commands[0]["funct"] += 1
    elif damage == "extra_command":
        commands.append({"class": "MVOUT", "funct": 3})
    body = [first, *stores, work] if damage != "after_work" else [first, work, *stores]
    return _log(inside=body), commands


def test_explicit_complete_zero_init_before_device_work_is_execution_overhead():
    log, commands = _initialized()
    row = check_host_output_writes(
        log,
        entry=0x200,
        main=(0x100, 0x80),
        outputs=[(0x1000, 64)],
        custom_opcode=0x7B,
        allow_complete_zero_initialization=True,
        commands=commands,
    )
    assert row["kernel_cpu_output_writes"] == 16
    assert row["complete_pre_device_zero_initialization_bytes"] == 64
    assert row["kernel_cpu_result_overwrites"] == 0
    assert row["target_cycles"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "repeated",
        "nonzero",
        "partial",
        "atomic",
        "unaligned",
        "framing",
        "extra_command",
        "after_work",
    ],
)
def test_zero_init_allowance_requires_complete_actual_bytes_and_order(damage):
    log, commands = _initialized(damage=damage)
    with pytest.raises(ValueError, match="primitive"):
        check_host_output_writes(
            log,
            entry=0x200,
            main=(0x100, 0x80),
            outputs=[(0x1000, 64)],
            custom_opcode=0x7B,
            allow_complete_zero_initialization=True,
            commands=commands,
        )


@pytest.mark.parametrize("word", [0x00002023, 0x00003023, 0xC218, 0xE218, 0x0000202F])
def test_unresolved_retired_store_cannot_masquerade_as_no_output_write(word):
    with pytest.raises(ValueError, match="unresolved committed memory"):
        _check(_log(inside=[_line(0x202, word)]))
