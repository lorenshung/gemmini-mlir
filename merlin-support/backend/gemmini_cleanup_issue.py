"""Shared bounded complete-output and actual-command checks for final artifacts.

This target checker supplies no grade, source ownership or timing authority.
The original caller must independently admit the exact source and executable.
"""

from __future__ import annotations

import importlib.util
import resource
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.runtime.out_b64 import OutB64Decoder


def check_pair_artifacts(owner, before, after):
    """Dispatch only the original live owner's additional artifact checker."""
    callback = getattr(owner, "check_pair_artifacts", None)
    if callback is None:
        result = owner.check_actual_pair(
            before["events"], after["events"], execution_domain="functional_spike"
        )
    else:
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise ValueError(
                "final artifact checker requires an original bound owner method"
            )
        result = callback(before, after, execution_domain="functional_spike")
    owner.revalidate()
    return result


def checked_qualifier(admission):
    record = admission.revalidate()
    pin = record["pinned_inputs"]["checker"]
    if _pin(pin["path"]) != pin:
        raise ValueError("original target event decoder changed")
    spec = importlib.util.spec_from_file_location(
        "original_cleanup_event_checker", pin["path"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values(path):
    values, decoder = {}, OutB64Decoder()
    for line in Path(path).read_text().splitlines():
        decoder.consume(line.split(), values)
    decoder.require_closed()
    if not values:
        raise ValueError("final realization has no complete native output values")
    return {name: np.asarray(value).reshape(-1) for name, value in values.items()}


def _limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))


def check_issued_program(owner, *, elf, command_buffer, reference_console, work_root):
    owner.revalidate()
    runtime = owner.cleanup.revalidate()["runtime_inputs"]
    qualify = checked_qualifier(owner.cleanup)
    cb = command_buffer
    # Use the original selected runtime, never an ambient target/backend lookup.
    argv = [
        runtime["spike"]["path"],
        "--extlib=" + runtime["functional_extension"]["path"],
        "--extension=gemmini",
        "-l",
        "--log-commits",
        str(elf),
    ]
    import os

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(runtime["functional_extension"]["path"]).parent)
    with (
        shared_compute_pool(timeout_s=120),
        invocation_record.observe(
            work_root,
            stage="original_final_cleanup_issue",
            argv=argv,
            inputs=(elf,),
            dependencies=(Path(__file__),),
            env=env,
        ) as observed,
    ):
        with (
            (observed.directory / "stdout.bin").open("w") as stdout,
            (observed.directory / "stderr.bin").open("w") as stderr,
        ):
            completed = subprocess.run(
                argv,
                check=False,
                stdout=stdout,
                stderr=stderr,
                env=env,
                timeout=60,
                preexec_fn=_limits,
            )
        observed.complete_streams(completed.returncode)
        invocation = observed.path
    if completed.returncode:
        raise ValueError("bounded final cleanup issue execution did not complete")
    trace = invocation_record.verify(invocation)
    if Path(trace["stderr"]["path"]).stat().st_size > 64 << 20:
        raise ValueError(
            "final cleanup trace exceeds unchanged independent byte budget"
        )
    actual, expected = (
        _values(trace["stdout"]["path"]),
        _values(reference_console),
    )
    if set(actual) != set(expected) or any(
        actual[name].dtype != value.dtype or not np.array_equal(actual[name], value)
        for name, value in expected.items()
    ):
        raise ValueError(
            "final cleanup trace changed complete original native output values"
        )
    outputs = {
        name: tensor
        for name, tensor in cb["tensors"].items()
        if tensor.get("role") == "output"
    }
    if set(actual) != set(outputs) or any(
        actual[name].size != int(np.prod(tensor["shape"]))
        for name, tensor in outputs.items()
    ):
        raise ValueError(
            "final cleanup trace does not cover every complete compiled output"
        )
    symbols = qualify._symbols(elf, owner.nm)
    decoded, events = qualify._events(
        Path(trace["stderr"]["path"]).read_text(),
        owner.cleanup.selected.operand_observer().isa_constants("gemmini"),
        symbols,
    )
    windows = qualify._check_transfer_windows(decoded, symbols, cb)
    owner.revalidate()
    return {
        "command_buffer": cb,
        "events": events,
        "commands": len(decoded["instructions"]),
        "outputs": sum(value.size for value in actual.values()),
        "logical_windows": windows,
        "issue_invocation": _pin(invocation),
    }
