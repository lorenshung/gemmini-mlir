"""Original final realization consumer for source-bound native model pairs.

An original complete functional grade remains mandatory. This adds actual issue
operand/state checks; it cannot upgrade a pending grade or certify timing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from gemmini_cleanup_issue import check_issued_program, check_pair_artifacts
from gemmini_timing_resources import _pin
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


def check_native_arm(owner, *, native_result, command_buffer, lowered, work_root):
    """Consume the exact original native artifacts; run a new bounded issue trace."""
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
    checked = check_issued_program(
        owner,
        elf=elf,
        command_buffer=cb,
        reference_console=native._pinned_native_file(row["tiers"]["L2"], "console"),
        work_root=work_root,
    )
    return {
        **checked,
        "native_result": _pin(native_result),
        "source": row["source"],
        "command_buffer_artifact": _pin(command_buffer),
        "lowered": _pin(lowered),
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
    checked = check_pair_artifacts(owner, before, after)
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
                if type(record.get("compiler_source")) is not dict:
                    raise ValueError(
                        "final cleanup native-model/layer realization unavailable for this capsule family"
                    )
                from gemmini_cleanup_layer_artifacts import read_layer_arm

                arguments = {"result_path": path, "package": expected_root / "compiler"}
                read_layer_arm(owner, **arguments)
                found[name] = arguments
                continue
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
    if any(
        set(arms["baseline"][name]) != set(arms["candidate"][name])
        for name in arms["baseline"]
    ):
        raise ValueError("final cleanup changed the original capsule execution route")
    from gemmini_cleanup_layer_artifacts import check_layer_pair

    return {
        name: (
            check_layer_pair
            if "result_path" in arms["baseline"][name]
            else check_native_pair
        )(
            owner,
            baseline=arms["baseline"][name],
            candidate=arms["candidate"][name],
            work_root=Path(grade_root) / "independent_cleanup_pairs" / name,
        )
        for name in sorted(arms["baseline"])
    }
