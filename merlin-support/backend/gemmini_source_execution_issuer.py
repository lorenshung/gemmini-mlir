"""Original host issuer for a newly emitted, bounded native source realization.

This does not deserialize a verdict or grant general target support. Each call
observes the current exact executable with the selected original functional
runtime, then returns the existing independent source/packet/output owner.
The ordinary source qualifier must still call that live owner's verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path

from gemmini_source_execution import SourceExecutionOwner
from gemmini_staged_dense_capability import _digest
from gemmini_staged_tile_capability import QualifiedStagedTileOwner
from gemmini_timing_resources import _revalidate_sources
from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import _pin
from merlin.targetgen.native_model_execution import _pinned_native_file


def _limits(seconds):
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    cpu = max(1, math.ceil(seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def _native_artifacts(result, command_buffer, lowered, runtime):
    """Check current artifacts before executing; this supplies no source proof."""
    paths = tuple(Path(path) for path in (result, command_buffer, lowered))
    pins = tuple(_pin(path) for path in paths)
    row = json.loads(paths[0].read_bytes())
    cb = json.loads(paths[1].read_bytes())
    if (
        row.get("failure")
        or row.get("numeric", {}).get("status") != "pass"
        or row.get("candidate", {}).get("command_buffer_sha256")
        != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
        or row.get("candidate", {}).get("lowered_mlir_sha256") != pins[2]["sha256"]
    ):
        raise ValueError(
            "source issuer needs the current complete native/compiler artifacts"
        )
    tier = row.get("tiers", {}).get("L2", {})
    citation = tier.get("engine_citation", {})
    if (
        tier.get("status") != "pass"
        or tier.get("engine") != "spike"
        or citation.get("engine") != "spike"
        or citation.get("target") != "gemmini"
        or citation.get("flags")
        != [
            "--extlib=" + runtime["functional_extension"]["path"],
            "--extension=gemmini",
        ]
        or any(
            {key: citation.get(role, {}).get(key) for key in ("path", "sha256")}
            != runtime[name]
            for role, name in (
                ("binary", "spike"),
                ("extension", "functional_extension"),
            )
        )
    ):
        raise ValueError(
            "source issuer cannot substitute a different functional runtime"
        )
    elf = _pinned_native_file(row, "elf")
    if tier.get("elf") != row["elf"]:
        raise ValueError("source issuer native tier ran a different executable")
    for group in (row["source"], row["build_artifacts"]):
        for role in group:
            _pinned_native_file(group, role)
    _pinned_native_file(tier, "console")
    for pin in row["host_build_sources"]:
        if _pin(Path(pin["path"])) != {key: pin[key] for key in ("path", "sha256")}:
            raise ValueError("source issuer native build source changed")
    return paths, pins, elf


class SourceExecutionIssuer:
    """Extra diagnostic capability, never a compiler preparation mechanism."""

    def __init__(self, qualification):
        if type(qualification) is not QualifiedStagedTileOwner:
            raise ValueError(
                "source issuer requires the original live qualified target owner"
            )
        qualification.revalidate()
        self._qualification = qualification
        self._qualification_identity = qualification.identity
        self._prototype = qualification.prototype.prototype
        self._prototype_identity = self._prototype.identity
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_source_execution_issuer",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
                "merlin.common.invocation_record",
                "merlin.common.cpu_pool",
                "merlin.targetgen.native_model_execution",
            )
        }
        self._identity = _digest(
            [self._qualification_identity, self._prototype_identity, self._sources]
        )
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.original_source_execution_issuer.v1:" + self._identity

    def revalidate(self):
        self._qualification.revalidate()
        _revalidate_sources(self._sources)
        if (
            self._qualification.identity != self._qualification_identity
            or self._qualification.prototype.prototype is not self._prototype
            or self._prototype.identity != self._prototype_identity
            or _digest(
                [self._qualification_identity, self._prototype_identity, self._sources]
            )
            != self._identity
        ):
            raise ValueError(
                "original source issuer target/library/owner selection changed"
            )

    def source_rule(self):
        self.revalidate()
        return self._prototype.arithmetic.source_rule()

    def create_source_execution_owner(
        self, *, native_result, command_buffer, lowered, work_root, remaining_s
    ):
        if (
            type(remaining_s) not in (int, float)
            or not math.isfinite(remaining_s)
            or remaining_s <= 0
        ):
            raise ValueError("source issuer requires a positive remaining wall budget")
        deadline = time.monotonic() + remaining_s
        self.revalidate()
        runtime = self._prototype.cleanup.revalidate()["runtime_inputs"]
        for pin in runtime.values():
            if _pin(Path(pin["path"])) != {key: pin[key] for key in ("path", "sha256")}:
                raise ValueError("original selected source runtime changed")
        paths, pins, elf = _native_artifacts(
            native_result, command_buffer, lowered, runtime
        )
        work = Path(work_root)
        if (
            not work.is_absolute()
            or work.is_symlink()
            or work.resolve() != work
            or work.exists()
        ):
            raise ValueError(
                "source issuer requires a fresh ordinary absolute work directory"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("source issuer budget expired before original execution")
        argv = [
            runtime["spike"]["path"],
            "--extlib=" + runtime["functional_extension"]["path"],
            "--extension=gemmini",
            "-l",
            "--log-commits",
            str(elf),
        ]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(
            Path(runtime["functional_extension"]["path"]).parent
        )
        with shared_compute_pool(timeout_s=min(120, remaining)):
            seconds = min(60, deadline - time.monotonic())
            if seconds <= 0:
                raise ValueError(
                    "source issuer budget expired while awaiting shared CPU"
                )
            work.mkdir(parents=True, exist_ok=False)
            with invocation_record.observe(
                work,
                stage="original_current_source_issue",
                argv=argv,
                inputs=(elf, *paths),
                dependencies=tuple(Path(pin["path"]) for pin in self._sources.values()),
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
                raise ValueError(
                    "bounded original current-source trace execution failed"
                )
            trace = invocation_record.verify(invocation)
            if any(
                Path(trace[role]["path"]).stat().st_size > 64 << 20
                for role in ("stdout", "stderr")
            ):
                raise ValueError(
                    "current source trace exceeds the unchanged independent byte ceiling"
                )
            if tuple(_pin(path) for path in paths) != pins:
                raise ValueError(
                    "current source artifacts changed during original trace execution"
                )
            self.revalidate()
            # This is the original verifier, not a receipt constructor. Its
            # subsequent verify_source_execution call must check source proof,
            # actual products/settings/state, all inputs and complete outputs.
            owner = SourceExecutionOwner(
                arithmetic=self._prototype.arithmetic,
                selected=self._prototype.selected,
                result=paths[0],
                command_buffer=paths[1],
                lowered=paths[2],
                trace_invocation=invocation,
                nm=self._prototype.nm,
            )
            if time.monotonic() >= deadline:
                raise ValueError(
                    "source issuer budget expired during original owner construction"
                )
            return owner
