"""Complete observed memory widths remain separate from packet/source admission."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_source_packet_artifacts")


def test_scalar_stores_keep_conservative_complete_extent(adapter):
    log = "core 0: 3 0x80000000 (0x000e3023) mem 0x90000000 0x0000000000000000"
    assert adapter._cpu_writes(log, 0x6B) == [(0x90000000, 8)]


def test_custom_memory_effects_keep_each_actual_byte(adapter):
    log = "core 0: 3 0x80000000 (0x0000006b) mem 0x90000000 0xff mem 0x90000001 0x00"
    assert adapter._cpu_writes(log, 0x6B) == [(0x90000000, 1), (0x90000001, 1)]


@pytest.mark.parametrize(
    "log",
    [
        "core 1: 3 0x80000000 (0x000e3023) mem 0x90000000 0x00",
        "core 0: 3 0x80000000 (0x00007027) mem 0x90000000 0x00",
        "core 0: 3 0x80000000 (0x0000006b) mem 0x90000000 0xffff",
    ],
)
def test_unknown_hart_width_and_incomplete_custom_effects_refuse(adapter, log):
    with pytest.raises(ValueError):
        adapter._cpu_writes(log, 0x6B)
