"""Streaming changes transport, never the original command/effect semantics."""

import importlib
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.common.replayable_text import ReplayableText

ISA = {"CUSTOM_OPCODE": 0x0B, "FUNCT_CLASS": {1: "WORK"}, "CONFIG_CLASS": None}
COMMAND = (1 << 25) | (11 << 20) | (10 << 15) | (3 << 12) | 0x0B
LOG = (
    "core 0: 3 0x80000000 (0x00500513) x10 0x05\r\n"
    "core 0: 3 0x80000004 (0x00700593) x11 0x07\n"
    f"core 0: 3 0x80000008 (0x{COMMAND:08x}) mem 0x90000000 0xff\n"
    "core 0: 3 0x8000000c (0x0ff0000f)\n"
    "core 0: 3 0x80000010 (0x000e3023) mem 0x90000008 0x0000000000000000\n"
    f"core 0: 3 0x80000014 (0x{COMMAND:08x})"
)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_streamed_packet_observation")


@pytest.mark.parametrize("chunk", [1, 7, 65536])
def test_original_complete_decoder_fences_registers_and_memory_writes_match(adapter, tmp_path, chunk):
    path = tmp_path / "stderr.bin"
    path.write_bytes(LOG.encode())
    log = ReplayableText(path, max_bytes=64 << 20, chunk_bytes=chunk)
    actual = adapter.decode_packet_stream(log, isa=ISA, symbols={})
    checker, _path = adapter._checker()
    trace, events = checker._events(LOG, ISA, {})
    assert actual["trace"] == trace
    assert actual["events"] == events
    assert actual["host_writes"] == adapter.original._cpu_writes(LOG, ISA["CUSTOM_OPCODE"])
    assert [row["class"] for row in actual["stream"]] == ["WORK", "FENCE", "WORK"]
    assert actual["trace"]["instructions"][0]["rs1"]["raw"] == 5
    assert actual["trace"]["instructions"][0]["rs2"]["raw"] == 7
    assert actual["source_correspondence_qualified"] is False
    assert actual["writer_admission"] is False


@pytest.mark.parametrize("replacement", ["core 1:", "core 0: 3 0x8000000c (0x00007027) mem 0x90000000 0x00\ncore 0:"])
def test_original_unknown_hart_or_store_refusal_is_not_filtered(adapter, tmp_path, replacement):
    path = tmp_path / "stderr.bin"
    path.write_text(LOG.replace("core 0:", replacement, 1))
    with pytest.raises(ValueError):
        adapter.decode_packet_stream(ReplayableText(path, max_bytes=64 << 20), isa=ISA, symbols={})


def _actual_process(tmp_path):
    argv = [sys.executable, "-c", "import sys;sys.stderr.write(" + repr(LOG) + ")"]
    invocation_record.run(argv, directory=tmp_path / "evidence", stage="neutral_complete_trace", capture_output=True)
    return argv, next((tmp_path / "evidence").rglob("invocation.json"))


def test_actual_original_process_custody_and_changed_stdout_refusal(adapter, tmp_path):
    argv, invocation = _actual_process(tmp_path)
    result = adapter.read_packet_observation(invocation=invocation, expected_trace_argv=argv, isa=ISA, symbols={})
    assert result["trace"]["retired_instructions"] == 6
    record = invocation_record.verify(invocation)
    Path(record["stdout"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        adapter.read_packet_observation(invocation=invocation, expected_trace_argv=argv, isa=ISA, symbols={})


def test_wrong_original_runtime_or_flags_refuses_before_decoding(adapter, tmp_path, monkeypatch):
    argv, invocation = _actual_process(tmp_path)
    monkeypatch.setattr(adapter, "decode_packet_stream", lambda *_a, **_k: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="different observation"):
        adapter.read_packet_observation(
            invocation=invocation, expected_trace_argv=argv + ["foreign"], isa=ISA, symbols={}
        )


def test_raw_log_over_original_64_mib_is_still_refused_before_decoding(adapter, tmp_path):
    path = tmp_path / "stderr.bin"
    with path.open("wb") as stream:
        stream.truncate((64 << 20) + 1)
    with pytest.raises(ValueError, match="byte budget"):
        ReplayableText(path, max_bytes=64 << 20)


def test_decoder_source_change_during_complete_observation_refuses(adapter, tmp_path, monkeypatch):
    path = tmp_path / "stderr.bin"
    path.write_text(LOG)
    checker, _original_path = adapter._checker()
    selected_decoder = tmp_path / "selected-decoder.py"
    selected_decoder.write_text("selected original decoder bytes")

    class CurrentDecoder:
        @staticmethod
        def _events(log, isa, symbols):
            result = checker._events(log, isa, symbols)
            selected_decoder.write_text("changed decoder bytes")
            return result

    monkeypatch.setattr(adapter, "_checker", lambda: (CurrentDecoder, selected_decoder))
    with pytest.raises(ValueError, match="decoder source changed"):
        adapter.decode_packet_stream(ReplayableText(path, max_bytes=64 << 20), isa=ISA, symbols={})


def test_original_retirement_ceiling_is_passed_without_override(adapter, tmp_path, monkeypatch):
    from merlin.perf import executed_trace

    path = tmp_path / "stderr.bin"
    path.write_text(LOG)
    original_decoder = executed_trace.executed_trace
    observed = []

    def retain_original_ceiling(lines, *, isa, max_retired):
        observed.append(max_retired)
        return original_decoder(lines, isa=isa, max_retired=max_retired)

    monkeypatch.setattr(executed_trace, "executed_trace", retain_original_ceiling)
    adapter.decode_packet_stream(ReplayableText(path, max_bytes=64 << 20), isa=ISA, symbols={})
    assert observed == [2_000_000]
