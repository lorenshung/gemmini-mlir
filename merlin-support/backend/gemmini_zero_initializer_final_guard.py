"""Original full-grade gates with an independently checked source/ABI inverse.

Legacy cleanup consumers remain unchanged. This composite consumes only their
actual graded native artifacts and preserves their complete roster/certification
requirements. Saved successful reports cannot substitute any original gate.
"""

import json
from pathlib import Path

from gemmini_cleanup_final_guard import check_native_arm
from gemmini_cleanup_issue import check_pair_artifacts

from merlin.targetgen import native_model_execution as native


def check_native_pair(owner, *, baseline, candidate, work_root):
    before = check_native_arm(owner, **baseline, work_root=Path(work_root) / "baseline")
    after = check_native_arm(owner, **candidate, work_root=Path(work_root) / "candidate")
    # The new original bound checker first proves identical source/input/reference
    # identities, then checks the complete allowed source/ABI inverse and actual
    # products, zero state, outputs, commands, configurations and fences.
    checked = check_pair_artifacts(owner, before, after)
    return dict(
        status="checked_actual_final_source_initializer_pair",
        baseline=before,
        candidate=after,
        actual_pair=checked,
        timing_authority="NONE",
        hardware_execution_qualified=False,
    )


def check_functional_guard(owner, baseline, candidate, baseline_result, candidate_result, *, grade_root):
    """Bind complete original guard rosters to exact newly graded native programs."""
    if grade_root is None:
        raise ValueError("final cleanup requires durable original functional grade artifacts")
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
                type(name) is not str or not name or name in {".", ".."} or Path(name).name != name for name in names
            )
            or len(names) != len(set(names))
            or any(
                entry.get("status") != "pass" or entry.get("numeric") != "pass" or entry.get("cert_verdict")
                for entry in roster
            )
        ):
            raise ValueError("original final functional roster is incomplete or unqualified")
        from merlin.targetgen.compiler_pass_selection import _tree

        if _tree(Path(package)) != _tree(expected_root / "compiler"):
            raise ValueError("final graded compiler differs from the selected retained compiler")
        from gemmini_cleanup_layer_artifacts import check_compiler_binding

        check_compiler_binding(owner, expected_root / "compiler")
        found = {}
        for path in (expected_root / "runs").rglob("capsule_result.json"):
            record = json.loads(path.read_bytes())
            name = record.get("capsule")
            if name not in names or name in found or record.get("status") != "pass":
                raise ValueError("final grade has foreign or duplicated native capsule evidence")
            native_record, emission = (
                record.get("candidate_native_execution"),
                record.get("candidate_emission"),
            )
            if type(native_record) is not dict or type(emission) is not dict:
                if type(record.get("compiler_source")) is not dict:
                    raise ValueError("final cleanup native-model/layer realization unavailable for this capsule family")
                from gemmini_cleanup_layer_artifacts import read_layer_arm

                arguments = {"result_path": path, "package": expected_root / "compiler"}
                read_layer_arm(owner, **arguments)
                found[name] = arguments
                continue
            native_path = path.parent / "candidate_native/result.json"
            if json.loads(native_path.read_bytes()) != native_record:
                raise ValueError("original graded native result differs from its complete durable receipt")
            found[name] = {
                "native_result": native_path,
                "command_buffer": native._pinned_native_file(emission, "command_buffer"),
                "lowered": native._pinned_native_file(emission, "lowered_mlir"),
            }
        if set(found) != set(names):
            raise ValueError("final cleanup omitted required original native evidence")
        arms[label] = found
    if set(arms["baseline"]) != set(arms["candidate"]):
        raise ValueError("final cleanup changed the frozen functional cohort")
    if any(set(arms["baseline"][name]) != set(arms["candidate"][name]) for name in arms["baseline"]):
        raise ValueError("final cleanup changed the original capsule execution route")
    from gemmini_cleanup_layer_artifacts import check_layer_pair

    return {
        name: (check_layer_pair if "result_path" in arms["baseline"][name] else check_native_pair)(
            owner,
            baseline=arms["baseline"][name],
            candidate=arms["candidate"][name],
            work_root=Path(grade_root) / "independent_cleanup_pairs" / name,
        )
        for name in sorted(arms["baseline"])
    }
