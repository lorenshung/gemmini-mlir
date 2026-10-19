"""Complete raw-stream observation using the original packet decoders.

This additive transport entrypoint supplies no source, arithmetic, timing or
writer grant. In particular the original independent 64 MiB *raw* ceiling and
two-million-retired-instruction decoder ceiling still apply. Larger historical
traces remain refused; this is not a compressed transport exemption.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import gemmini_source_packet_artifacts as original

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.common.replayable_text import ReplayableText


def _checker():
    path = Path(__file__).resolve().parents[2] / "experiments/shared_load_configuration/qualify.py"
    spec = importlib.util.spec_from_file_location("streamed_original_packet_framing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _pins(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def decode_packet_stream(log: ReplayableText, *, isa, symbols):
    """Replay all bytes through unchanged framing, command and write semantics.

    ``isa`` and ``symbols`` remain explicit caller-owned decoder inputs. This
    result alone cannot authenticate a target, ELF or source correspondence.
    No decoded command, fence or observed memory write is filtered out.
    """
    checker, checker_path = _checker()
    paths = (
        checker_path,
        Path(original.__file__),
        Path(__file__),
        module_source_path("merlin.common.replayable_text"),
        module_source_path("merlin.perf.executed_trace"),
    )
    pins = _pins(paths)
    trace, events = checker._events(log, isa, symbols)
    stream, commands = [], iter(trace["instructions"])
    for event in events:
        if event[0] == "FENCE":
            stream.append({"class": "FENCE"})
        else:
            command = next(commands)
            if command["class"] != event[0]:
                raise ValueError("actual packet/fence framing changed")
            stream.append(command)
    if next(commands, None) is not None:
        raise ValueError("actual packet framing is incomplete")
    writes = original._cpu_writes(log, isa["CUSTOM_OPCODE"])
    log.verify()
    if _pins(paths) != pins:
        raise ValueError("original packet decoder source changed")
    return {
        "schema": "gemmini.complete_streamed_packet_observation.v1",
        "trace": trace,
        "events": events,
        "stream": stream,
        "host_writes": writes,
        "raw_stream": log.record(),
        "decoder_sources": pins,
        "source_correspondence_qualified": False,
        "writer_admission": False,
        "timing_authority": "NONE",
    }


def read_packet_observation(*, invocation, expected_trace_argv, isa, symbols):
    """Reopen an original complete process before/after streamed observation.

    Caller still owns exact runtime/ELF/ISA/symbol/source applicability. This
    verifies byte provenance, never a saved numeric or source-success verdict.
    """
    declaration = invocation_record.verify(Path(invocation))
    if declaration.get("kind") != "subprocess" or declaration.get("argv") != expected_trace_argv:
        raise ValueError("actual command trace belongs to a different observation invocation")
    log = ReplayableText(Path(declaration["stderr"]["path"]), max_bytes=64 << 20)
    if log.record()["sha256"] != declaration["stderr"]["sha256"]:
        raise ValueError("raw stream differs from its original invocation")
    result = decode_packet_stream(log, isa=isa, symbols=symbols)
    if invocation_record.verify(Path(invocation)) != declaration:
        raise ValueError("original observation changed during packet verification")
    return result
