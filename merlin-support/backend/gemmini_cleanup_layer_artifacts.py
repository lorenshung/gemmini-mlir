"""Consume exact ordinary layer artifacts under the original final grade.

No copied status creates a live owner. The original final functional/certification
roster must pass before calling this additional source/program/issue check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from gemmini_cleanup_issue import check_issued_program, check_pair_artifacts
from gemmini_timing_resources import _pin
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _ordinary(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("final layer evidence needs an exact ordinary original file")
    return path


def _member(root, pin):
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "bytes"}:
        raise ValueError("retained layer member lacks its complete original byte pin")
    name = PurePosixPath(pin["path"])
    if name.is_absolute() or ".." in name.parts or name.as_posix() != pin["path"]:
        raise ValueError("retained layer member escapes its original generated root")
    path = _ordinary(root / name)
    if (
        type(pin["bytes"]) is not int
        or not 0 <= pin["bytes"] <= 64 << 20
        or path.stat().st_size != pin["bytes"]
    ):
        raise ValueError("retained layer member exceeds its original bounded byte pin")
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise ValueError("retained layer member bytes changed")
    return path


def read_layer_arm(owner, *, result_path, package):
    """Read only a freshly graded complete-value layer on the selected runtime."""
    owner.revalidate()
    result_path = _ordinary(result_path)
    if (
        result_path.name != "capsule_result.json"
        or result_path.stat().st_size > 16 << 20
    ):
        raise ValueError("final layer has no bounded original capsule-result receipt")
    row = json.loads(result_path.read_bytes())
    tier = row.get("tiers", {}).get("L2") or {}
    source = row.get("compiler_source") or {}
    package = Path(package)
    generated = result_path.parent / "generated"
    runtime = owner.cleanup.revalidate()["runtime_inputs"]
    provenance = tier.get("sim_provenance") or {}
    if (
        row.get("status") != "pass"
        or row.get("failure")
        or row.get("numeric", {}).get("status") != "pass"
        or any(
            row.get("tiers", {}).get(name, {}).get("status") != "pass"
            for name in ("L0", "L1", "L2")
        )
        or tier.get("measured_now") is not True
        or tier.get("carried")
        or provenance.get("engine") != "spike"
        or provenance.get("sha256") != runtime["spike"]["sha256"]
        or provenance.get("binary") != runtime["spike"]["path"]
        or tier.get("toolchain") != str(package.resolve())
        or source.get("schema") != "merlin.capsule_compiler_source.v1"
        or source.get("status") != "stable"
        or source.get("package") != str(package.resolve())
    ):
        raise ValueError(
            "original complete layer grade, compiler or selected runtime is unavailable"
        )
    from merlin_experiments.phase2.retained_revisions import source_inventory

    source_inventory(
        package, max_bytes=32 << 20, max_files=8192, exclude_generated=True
    )
    expected = source.get("candidate_sha256")
    if source.get("snapshot") != "compiler-source/" + str(expected):
        raise ValueError("final layer compiler snapshot has ambiguous source ownership")
    snapshot = generated / source["snapshot"]
    source_inventory(
        snapshot, max_bytes=32 << 20, max_files=8192, exclude_generated=True
    )
    if (
        hash_tree(package)["sha256"] != expected
        or hash_tree(snapshot)["sha256"] != expected
    ):
        raise ValueError("original final layer compiler source or snapshot changed")
    if E.collect(generated) != source.get("executed_programs"):
        raise ValueError("original per-engine layer artifacts changed since grading")
    candidates = [
        entry
        for entry in source["executed_programs"]
        if entry.get("schema") == E.SCHEMA_V2
        and entry.get("status") == "retained"
        and entry.get("engine") == "spike"
        and entry.get("engine_sha256") == provenance["sha256"]
    ]
    if len(candidates) != 1:
        raise ValueError(
            "final layer requires one exact original functional execution, not an inferred representative"
        )
    execution = candidates[0]
    artifacts = execution.get("compiler_artifacts") or {}
    if artifacts.get("status") != "retained" or set(artifacts.get("members", {})) != {
        "command_buffer",
        "lowered_mlir",
        "console",
        "build",
        "invocation",
        "kernel_object",
    }:
        raise ValueError(
            "final layer complete compiler/execution artifacts were not retained"
        )
    members = {
        role: _member(generated, pin) for role, pin in artifacts["members"].items()
    }
    elf, harness = (_member(generated, execution[role]) for role in ("elf", "harness"))
    cb = json.loads(members["command_buffer"].read_bytes())
    build = json.loads(members["build"].read_bytes())
    invocation = json.loads(members["invocation"].read_bytes())
    original = artifacts.get("original_execution_invocation") or {}
    if (
        _pin(original["path"]) != original
        or Path(original["path"]).read_bytes() != members["invocation"].read_bytes()
        or invocation.get("kind") != "python_call"
        or invocation.get("stage") != "execution"
        or invocation.get("status") != "completed"
        or invocation.get("outcome") != "returned"
        or any(
            invocation.get(key) is not True
            for key in (
                "inputs_unchanged",
                "dependencies_unchanged",
                "executable_unchanged",
            )
        )
        or invocation.get("arguments", {}).get("simulator") != "spike"
        or invocation.get("stdout", {}).get("sha256")
        != artifacts["members"]["console"]["sha256"]
        or not any(
            pin.get("sha256") == execution["elf"]["sha256"]
            for pin in invocation.get("inputs", ())
        )
        or build.get("status") != "completed"
        or build.get("readback_policy", {}).get("transport") != "out_b64_v1"
        or build.get("command_buffer_sha256") != canonical_sha256(cb)
        or build.get("elf_sha256") != execution["elf"]["sha256"]
        or build.get("harness_sha256") != execution["harness"]["sha256"]
        or build.get("kernel_object_sha256")
        != artifacts["members"]["kernel_object"]["sha256"]
        or build.get("build_identity_sha256")
        != canonical_sha256(
            {k: v for k, v in build.items() if k != "build_identity_sha256"}
        )
    ):
        raise ValueError(
            "final layer original build, complete-value transport or execution binding changed"
        )
    source_file = generated / "input.interface.mlir"
    if not source_file.is_file() or hashlib.sha256(
        source_file.read_bytes()
    ).hexdigest() != source.get("artifacts", {}).get("input.interface.mlir"):
        raise ValueError("final layer lost its exact originally graded input source")
    owner.revalidate()
    return {
        "result": _pin(result_path),
        "source": _pin(source_file),
        "elf": elf,
        "command_buffer": cb,
        "command_buffer_artifact": _pin(members["command_buffer"]),
        "lowered": members["lowered_mlir"],
        "reference_console": members["console"],
        "retained_artifacts": artifacts,
        "harness": _pin(harness),
        "execution": execution,
        "compiler_sha256": expected,
    }


def check_layer_pair(owner, *, baseline, candidate, work_root):
    """Recheck both exact final ELFs and compare actual issued commands/outputs."""
    from gemmini_cleanup_final_guard import _logical

    before = read_layer_arm(owner, **baseline)
    after = read_layer_arm(owner, **candidate)
    if before["source"]["sha256"] != after["source"]["sha256"] or _logical(
        before["command_buffer"]
    ) != _logical(after["command_buffer"]):
        raise ValueError(
            "final layer pair changed original source, inputs, outputs or required logical work"
        )
    checked = {}
    for label, arm in (("baseline", before), ("candidate", after)):
        checked[label] = check_issued_program(
            owner,
            elf=arm["elf"],
            command_buffer=arm["command_buffer"],
            reference_console=arm["reference_console"],
            work_root=Path(work_root) / label,
        )
        checked[label].update(
            capsule_result=arm["result"],
            command_buffer_artifact=arm["command_buffer_artifact"],
            lowered=_pin(arm["lowered"]),
            source=arm["source"],
        )
    pair = check_pair_artifacts(owner, checked["baseline"], checked["candidate"])
    return {
        "status": "checked_actual_final_layer_pair",
        "baseline": checked["baseline"],
        "candidate": checked["candidate"],
        "original_capsule_results": {
            "baseline": before["result"],
            "candidate": after["result"],
        },
        "source_sha256": before["source"]["sha256"],
        "actual_pair": pair,
        "timing_authority": "NONE",
        "hardware_execution_qualified": False,
    }
