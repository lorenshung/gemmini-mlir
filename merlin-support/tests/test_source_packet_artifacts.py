"""Complete observed memory widths remain separate from packet/source admission."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_source_packet_artifacts")


def test_scalar_stores_keep_exact_complete_extent(adapter):
    log = "core 0: 3 0x80000000 (0x000e3023) mem 0x90000000 0x0000000000000000"
    assert adapter._cpu_writes(log, 0x6B) == [(0x90000000, 8)]


@pytest.mark.parametrize(
    "word,width",
    [
        (0x23, 1), (0x1023, 2), (0x2023, 4), (0x3023, 8),
        (0x2027, 4), (0x3027, 8), (0x202F, 4), (0x302F, 8),
        (0xA000, 8), (0xC000, 4), (0xE000, 8),
        (0xA002, 8), (0xC002, 4), (0xE002, 8),
    ],
)
def test_instruction_width_controls_scalar_float_atomic_and_rv64_compressed_stores(adapter, word, width):
    value = "0x" + "00" * width
    log = f"core 0: 3 0x80000000 (0x{word:08x}) mem 0x90000000 {value}"
    assert adapter._cpu_writes(log, 0x7B) == [(0x90000000, width)]
    for wrong in ("0x" + "00" * (width + 1), "0x" + "00" * max(0, width - 1)):
        with pytest.raises(ValueError, match="complete committed memory width"):
            adapter._cpu_writes(log.replace(value, wrong), 0x7B)


def test_rebound_or_modified_store_decoder_refuses_before_memory_effects(adapter, monkeypatch):
    original = adapter.rv64_scalar_store_width
    replacement = lambda _word: 1
    monkeypatch.setattr(adapter, "rv64_scalar_store_width", replacement)
    with pytest.raises(ValueError, match="effect decoder changed"):
        adapter._cpu_writes("", 0x7B)
    monkeypatch.setattr(adapter, "rv64_scalar_store_width", original)
    monkeypatch.setattr(original, "__code__", replacement.__code__)
    with pytest.raises(ValueError, match="effect decoder changed"):
        adapter._cpu_writes("", 0x7B)


def test_custom_memory_effects_keep_each_actual_byte(adapter):
    log = "core 0: 3 0x80000000 (0x0000006b) mem 0x90000000 0xff mem 0x90000001 0x00"
    assert adapter._cpu_writes(log, 0x6B) == [(0x90000000, 1), (0x90000001, 1)]


@pytest.mark.parametrize(
    "log",
    [
        "core 1: 3 0x80000000 (0x000e3023) mem 0x90000000 0x00",
        "core 0: 3 0x80000000 (0x00007027) mem 0x90000000 0x00",
        "core 0: 3 0x80000000 (0x0000006b) mem 0x90000000 0xffff",
        "core 0: 3 0x80000000 (0x1000202f) mem 0x90000000 0x00000000",
        "core 0: 3 0x80000000 (0x00000023) mem 0x90000000 0xgg",
    ],
)
def test_unknown_hart_width_and_incomplete_custom_effects_refuse(adapter, log):
    with pytest.raises(ValueError):
        adapter._cpu_writes(log, 0x6B)
