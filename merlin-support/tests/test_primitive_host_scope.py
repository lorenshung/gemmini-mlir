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
