"""Independent store-observation and output-boundary controls for call proof."""

import pytest
from gemmini_primitive_host_scope import check_host_output_writes


def _line(pc, word, observation=""):
    return f"core 0: 3 0x{pc:016x} (0x{word:08x}) {observation}"


def _log(body):
    return "\n".join(
        [
            _line(0x100, 0xEF, "x1 0x104"),
            _line(0x200, 0x13),
            *body,
            _line(0x204, 0x8067),
            _line(0x104, 0x13),
        ]
    )


def _check(body, *, commands=None, outputs=((0x1000, 64),)):
    return check_host_output_writes(
        _log(body),
        entry=0x200,
        main=(0x100, 0x80),
        outputs=list(outputs),
        custom_opcode=0x7B,
        allow_complete_zero_initialization=commands is not None,
        commands=commands,
    )


@pytest.mark.parametrize(
    "word,width",
    [(0x2023, 4), (0x3023, 8), (0xC000, 4), (0xE000, 8), (0xC002, 4), (0xE002, 8)],
)
def test_all_integer_store_forms_keep_complete_zero_work_in_boundary(word, width):
    body = [
        _line(0x202, word, f"mem 0x{address:x} 0x{'00' * width}")
        for address in range(0x1000, 0x1040, width)
    ]
    body.append(_line(0x202, 2 << 25 | 0x7B))
    row = _check(body, commands=[{"class": "MVIN", "funct": 2}])
    assert row["kernel_cpu_output_writes"] == 64 // width
    assert row["complete_pre_device_zero_initialization_bytes"] == 64
    assert row["kernel_cpu_result_overwrites"] == 0
    assert row["target_cycles"] is None


@pytest.mark.parametrize(
    "device_class",
    ["MVIN", "PRELOAD", "COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE", "MVOUT"],
)
def test_every_device_memory_phase_closes_host_zero_permission(device_class):
    body = [_line(0x202, 2 << 25 | 0x7B)]
    body.extend(
        _line(0x202, 0x3023, f"mem 0x{address:x} 0x0000000000000000")
        for address in range(0x1000, 0x1040, 8)
    )
    with pytest.raises(ValueError, match="overwrites"):
        _check(body, commands=[{"class": device_class, "funct": 2}])


def test_each_disjoint_output_must_have_exact_full_initialization():
    outputs = ((0x1000, 16), (0x1020, 16))
    body = [
        _line(0x202, 0x3023, f"mem 0x{address:x} 0x0000000000000000")
        for address in (0x1000, 0x1008, 0x1020, 0x1028)
    ]
    body.append(_line(0x202, 2 << 25 | 0x7B))
    row = _check(body, commands=[{"class": "MVIN", "funct": 2}], outputs=outputs)
    assert row["complete_pre_device_zero_initialization_bytes"] == 32
    with pytest.raises(ValueError, match="complete|repeated"):
        _check(body[1:], commands=[{"class": "MVIN", "funct": 2}], outputs=outputs)


def test_zero_store_cannot_bridge_an_output_and_unowned_gap():
    with pytest.raises(ValueError, match="exceeds"):
        _check(
            [
                _line(0x202, 0x3023, "mem 0x100c 0x0000000000000000"),
                _line(0x202, 2 << 25 | 0x7B),
            ],
            commands=[{"class": "MVIN", "funct": 2}],
            outputs=((0x1000, 16), (0x1020, 16)),
        )


@pytest.mark.parametrize("word", [0x2027, 0x3027, 0xA000, 0xA002])
def test_floating_store_encoding_is_not_integer_zero_initialization(word):
    with pytest.raises(ValueError, match="integer|store"):
        _check(
            [
                _line(0x202, word, "mem 0x1000 0x00000000"),
                _line(0x202, 2 << 25 | 0x7B),
            ],
            commands=[{"class": "MVIN", "funct": 2}],
        )


@pytest.mark.parametrize("word", [0x2023, 0x3023, 0xC000, 0xE002, 0x202F])
@pytest.mark.parametrize("explicit_zero_mode", [False, True])
def test_missing_committed_store_observation_never_means_no_output_write(
    word, explicit_zero_mode
):
    # With no address, a retired store could touch any required output. This
    # must refuse even if no other zero stores were observed and cover is empty.
    body = [_line(0x202, 2 << 25 | 0x7B), _line(0x202, word)]
    with pytest.raises(ValueError, match="store|memory|observation"):
        _check(
            body,
            commands=[{"class": "MVIN", "funct": 2}] if explicit_zero_mode else None,
        )
