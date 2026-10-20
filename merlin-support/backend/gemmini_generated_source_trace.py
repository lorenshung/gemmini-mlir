"""Fresh complete-commit observation for a source-only generated compiler.

Reuse the original artifact/runtime precheck, invocation custody and independent
SourceExecutionOwner. This grants no scheduling or compiler qualification.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from gemmini_source_execution import source_trace_flags
from gemmini_source_execution_issuer import _limits, _native_artifacts
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool


def observe_current_source(
    issuer, *, native_result, command_buffer, lowered, work_root, remaining_s
):
    deadline = time.monotonic() + remaining_s
    issuer.revalidate()
    state = issuer._original_state()
    paths, pins, elf = _native_artifacts(
        native_result, command_buffer, lowered, state["runtime"]
    )
    if state["binding_protocol"] != "whole_program" and (
        json.loads(paths[0].read_bytes()).get("schema")
        != "merlin_candidate_native_pointer_source_v1"
        or json.loads(paths[0].read_bytes())
        .get("pointer_execution", {})
        .get("protocol")
        != state["binding_protocol"]
    ):
        raise ValueError(
            "direct source observation requires its explicit original native execution record"
        )
    work = Path(work_root)
    if (
        not work.is_absolute()
        or work.resolve() != work
        or any(p.is_symlink() for p in (work, *work.parents))
        or work.exists()
        or any(
            work == p or work.is_relative_to(p) or p.is_relative_to(work)
            for p in (*paths, elf)
        )
    ):
        raise ValueError(
            "generated source observation needs a fresh disjoint original output directory"
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError(
            "generated source observation budget expired before original execution"
        )
    argv = [
        state["runtime"]["spike"]["path"],
        "--extlib=" + state["runtime"]["functional_extension"]["path"],
        "--extension=gemmini",
        *source_trace_flags(state["protocol"]),
        str(elf),
    ]
    env = dict(
        os.environ,
        LD_LIBRARY_PATH=str(
            Path(state["runtime"]["functional_extension"]["path"]).parent
        ),
    )
    with shared_compute_pool(timeout_s=min(120, remaining)):
        seconds = min(60, deadline - time.monotonic())
        if seconds <= 0:
            raise ValueError(
                "generated source observation budget expired in CPU admission"
            )
        work.mkdir(parents=True, exist_ok=False)
        with invocation_record.observe(
            work,
            stage="original_current_source_issue",
            argv=argv,
            inputs=(elf, *paths),
            dependencies=tuple(Path(p["path"]) for p in state["sources"].values()),
            env=env,
        ) as observed:
            with (
                (observed.directory / "stdout.bin").open("w") as stdout,
                (observed.directory / "stderr.bin").open("w") as stderr,
            ):
                completed = subprocess.run(
                    argv,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    env=env,
                    timeout=seconds,
                    preexec_fn=lambda: _limits(seconds),
                )
            observed.complete_streams(completed.returncode)
            invocation = observed.path
        if completed.returncode:
            raise ValueError("bounded generated-source complete trace execution failed")
        trace = invocation_record.verify(invocation)
        if any(
            Path(trace[role]["path"]).stat().st_size > 64 << 20
            for role in ("stdout", "stderr")
        ):
            raise ValueError(
                "generated source trace exceeds the unchanged 64MiB ceiling"
            )
        if tuple(_pin(path) for path in paths) != pins:
            raise ValueError(
                "current generated-source artifacts changed during observation"
            )
        issuer.revalidate()
        owner = state["owner_type"](
            arithmetic=state["arithmetic"],
            selected=state["selected"],
            result=paths[0],
            command_buffer=paths[1],
            lowered=paths[2],
            trace_invocation=invocation,
            nm=Path(state["nm"]["path"]),
            observation_protocol=state["protocol"],
            **(
                {"execution_contract": state["execution_contract"]}
                if state["binding_protocol"] != "whole_program"
                else {}
            ),
        )
        if time.monotonic() >= deadline:
            raise ValueError(
                "generated source observation budget expired during original owner construction"
            )
        return owner
