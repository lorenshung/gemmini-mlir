"""Original final realization consumer for source-bound native model pairs.

An original complete functional grade remains mandatory. This adds actual issue
operand/state checks; it cannot upgrade a pending grade or certify timing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import resource
import subprocess
from pathlib import Path

import numpy as np
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import native_model_execution as native


def _logical(cb):
    result = copy.deepcopy(cb)
    plan = result.get("params", {}).get("global_program_plan")
    if plan is not None:
        for key in (
            "schedule_instruction_count",
            "prologue_instruction_range",
            "epilogue_instruction_range",
        ):
            plan.pop(key, None)
        for task in plan.get("tasks", ()):
            task.pop("instruction_start", None)
            task.pop("instruction_end", None)
    return result


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


def check_native_arm(owner, *, native_result, command_buffer, lowered, work_root):
    """Consume the exact original native artifacts; run a new bounded issue trace."""
    import qualify

    owner.revalidate()
    row = json.loads(Path(native_result).read_bytes())
    cb = json.loads(Path(command_buffer).read_bytes())
    text = Path(lowered).read_text()
    evidence = owner.cleanup.revalidate()
    if (
        row.get("schema") != "merlin_candidate_native_model_execution_v1"
        or row.get("failure")
        or row.get("numeric", {}).get("status") != "pass"
        or row.get("tiers", {}).get("L2", {}).get("status") != "pass"
        or row.get("candidate", {}).get("command_buffer_sha256")
        != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
        or row.get("candidate", {}).get("lowered_mlir_sha256")
        != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise ValueError(
            "final cleanup requires this native model's complete graded outputs and exact compiler artifacts"
        )
    for group in (row["source"], row["build_artifacts"]):
        for role in group:
            native._pinned_native_file(group, role)
    for pin in row["host_build_sources"]:
        if _pin(pin["path"]) != {key: pin[key] for key in ("path", "sha256")}:
            raise ValueError("original native build source changed")
    if (
        native._pinned_native_file(
            row["build_artifacts"], "kernel.llvm.mlir"
        ).read_text()
        != text
    ):
        raise ValueError("final native build used different compiler LLVM")
    citation = row["tiers"]["L2"].get("engine_citation", {})
    runtime = evidence["runtime_inputs"]
    if (
        citation.get("engine") != "spike"
        or citation.get("target") != "gemmini"
        or {key: citation.get("binary", {}).get(key) for key in ("path", "sha256")}
        != runtime["spike"]
        or {key: citation.get("extension", {}).get(key) for key in ("path", "sha256")}
        != runtime["functional_extension"]
        or citation.get("flags")
        != [
            "--extlib=" + runtime["functional_extension"]["path"],
            "--extension=gemmini",
        ]
    ):
        raise ValueError(
            "final cleanup has a different functional runtime or command semantics"
        )
    elf = native._pinned_native_file(row, "elf")
    if native._pinned_native_file(row["tiers"]["L2"], "elf") != elf:
        raise ValueError("complete outputs and final trace do not name the same ELF")
    # Use the original native citation, never an ambient target/backend lookup.
    argv = [
        runtime["spike"]["path"],
        *citation["flags"],
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
        _values(native._pinned_native_file(row["tiers"]["L2"], "console")),
    )
    if set(actual) != set(expected) or any(
        not np.array_equal(actual[name], value) for name, value in expected.items()
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
        "native_result": _pin(native_result),
        "source": row["source"],
        "command_buffer": cb,
        "events": events,
        "commands": len(decoded["instructions"]),
        "outputs": sum(value.size for value in actual.values()),
        "logical_windows": windows,
        "issue_invocation": _pin(invocation),
    }


def check_native_pair(owner, *, baseline, candidate, work_root):
    """Original host API, not a saved-success importer or qualification factory."""
    before = check_native_arm(owner, **baseline, work_root=Path(work_root) / "baseline")
    after = check_native_arm(
        owner, **candidate, work_root=Path(work_root) / "candidate"
    )
    if _logical(before["command_buffer"]) != _logical(after["command_buffer"]) or {
        key: (pin["sha256"], pin.get("size_bytes"))
        for key, pin in before["source"].items()
    } != {
        key: (pin["sha256"], pin.get("size_bytes"))
        for key, pin in after["source"].items()
    }:
        raise ValueError(
            "final cleanup pair changes original source, inputs, goldens or logical work"
        )
    checked = owner.check_actual_pair(
        before["events"], after["events"], execution_domain="functional_spike"
    )
    return {
        "status": "checked_actual_final_pair",
        "baseline": before,
        "candidate": after,
        "actual_pair": checked,
        "timing_authority": "NONE",
        "hardware_execution_qualified": False,
    }


def check_functional_guard(
    owner, baseline, candidate, baseline_result, candidate_result, *, grade_root
):
    """Bind complete original guard rosters to exact newly graded native programs."""
    if grade_root is None:
        raise ValueError(
            "final cleanup requires durable original functional grade artifacts"
        )
    arms = {}
    for label, package, result in (
        ("baseline", baseline, baseline_result),
        ("candidate", candidate, candidate_result),
    ):
        grade = result.get("full_output_regrade") or {}
        expected_root = (
            Path(grade_root)
            if label == "candidate"
            else Path(grade_root).with_name(Path(grade_root).name + "_library_baseline")
        )
        if (
            result.get("status") != "clean"
            or grade.get("status") != "pass"
            or grade.get("evidence_root") != str(expected_root)
        ):
            raise ValueError(
                "original complete functional/certification gate is pending or not bound to this final grade"
            )
        roster = grade.get("score", {}).get("per_capsule", ())
        names = [entry.get("capsule") for entry in roster]
        if (
            not names
            or grade.get("score", {}).get("integrity_status") != "clean"
            or result.get("capsules") != len(names)
            or any(
                type(name) is not str
                or not name
                or name in {".", ".."}
                or Path(name).name != name
                for name in names
            )
            or len(names) != len(set(names))
            or any(
                entry.get("status") != "pass"
                or entry.get("numeric") != "pass"
                or entry.get("cert_verdict")
                for entry in roster
            )
        ):
            raise ValueError(
                "original final functional roster is incomplete or unqualified"
            )
        from merlin.targetgen.compiler_pass_selection import _tree

        if _tree(Path(package)) != _tree(expected_root / "compiler"):
            raise ValueError(
                "final graded compiler differs from the selected retained compiler"
            )
        found = {}
        for path in (expected_root / "runs").rglob("capsule_result.json"):
            record = json.loads(path.read_bytes())
            name = record.get("capsule")
            if name not in names or name in found or record.get("status") != "pass":
                raise ValueError(
                    "final grade has foreign or duplicated native capsule evidence"
                )
            native_record, emission = (
                record.get("candidate_native_execution"),
                record.get("candidate_emission"),
            )
            if type(native_record) is not dict or type(emission) is not dict:
                raise ValueError(
                    "final cleanup native-model realization unavailable for this capsule family"
                )
            native_path = path.parent / "candidate_native/result.json"
            if json.loads(native_path.read_bytes()) != native_record:
                raise ValueError(
                    "original graded native result differs from its complete durable receipt"
                )
            found[name] = {
                "native_result": native_path,
                "command_buffer": native._pinned_native_file(
                    emission, "command_buffer"
                ),
                "lowered": native._pinned_native_file(emission, "lowered_mlir"),
            }
        if set(found) != set(names):
            raise ValueError("final cleanup omitted required original native evidence")
        arms[label] = found
    if set(arms["baseline"]) != set(arms["candidate"]):
        raise ValueError("final cleanup changed the frozen functional cohort")
    return {
        name: check_native_pair(
            owner,
            baseline=arms["baseline"][name],
            candidate=arms["candidate"][name],
            work_root=Path(grade_root) / "independent_cleanup_pairs" / name,
        )
        for name in sorted(arms["baseline"])
    }
