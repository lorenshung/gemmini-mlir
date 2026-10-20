"""Complete committed storage initialization, not inferred ELF BSS contents."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def M(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_private_zero_storage")


def store(address, value=0, word=0xE3023):
    return f"core 0: 3 0x80000000 (0x{word:08x}) mem 0x{address:016x} 0x{value:016x}"


DEVICE = "core 0: 3 0x80000004 (0x0000007b)"


def prove(M, lines, *, symbols=((0x1000, 16, "private"),), abi=()):
    return M.observed_private_zero_storage(
        "\n".join(lines), symbols=symbols, abi_symbols=abi, custom_opcode=0x7B
    )


def test_complete_observed_zero_coverage_then_device_use(M):
    assert prove(M, [store(0x1000), store(0x1008), DEVICE]) == (
        (0x1000, 16, "private"),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "nonzero",
        "late_zero",
        "late_nonzero",
        "outside",
        "abi",
        "implicit_bss",
    ],
)
def test_missing_coverage_or_any_late_clobber_never_proves_zero(M, fault):
    rows = [store(0x1000), store(0x1008), DEVICE]
    abi = ()
    if fault == "missing":
        del rows[1]
    elif fault == "nonzero":
        rows[0] = store(0x1000, 1)
    elif fault == "late_zero":
        rows.append(store(0x1000))
    elif fault == "late_nonzero":
        rows.append(store(0x1000, 1))
    elif fault == "outside":
        rows[0] = store(0xFFC)
    elif fault == "abi":
        abi = ("private",)
    else:
        rows = [DEVICE]
    assert prove(M, rows, abi=abi) == ()


def test_unknown_store_address_duplicate_memory_and_missing_device_refuse(M):
    for rows in (
        [store(0x1000), "core 0: 3 0x80000000 (0x000e3023)", DEVICE],
        [store(0x1000) + " mem 0x1008 0x00", DEVICE],
        [store(0x1000), store(0x1008)],
    ):
        with pytest.raises(ValueError):
            prove(M, rows)


def test_custom_host_write_after_first_device_command_clobbers_private_zero(M):
    assert (
        prove(
            M,
            [
                store(0x1000),
                store(0x1008),
                DEVICE,
                "core 0: 3 0x80000008 (0x0000007b) mem 0x1000 0x00",
            ],
        )
        == ()
    )
