"""Original ordinary primitive numerical/build/caller custody, with L3 pending."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gemmini_cleanup_layer_artifacts import _member, _ordinary
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _diagnostic_grade(row, capsule):
    """Preserve original mandatory L3 and accept no other missing/failed role."""
    failure = row.get("failure")
    if row.get("status") == "incomplete":
        if (
            not isinstance(failure, dict)
            or failure.get("plane") != "oracle_unavailable"
            or failure.get("category") != "NOT_RUN_IS_NOT_PASS"
            or failure.get("tier") != "L3"
        ):
            raise ValueError(
                "primitive ordinary grade has an unresolved role other than mandatory pending L3"
            )
    else:
        raise ValueError(
            "primitive ordinary grade did not complete its functional roles"
        )
    tiers = row.get("tiers", {})
    if (
        row.get("capsule") != capsule["name"]
        or row.get("kind") != "layer"
        or row.get("label") != "public"
        or row.get("numeric", {}).get("status") != "pass"
        or row.get("trace_check", {}).get("status") != "pass"
        or any(
            tiers.get(name, {}).get("status") != "pass"
            or tiers[name].get("mandatory") is not True
            or tiers[name].get("measured_now") is not True
            or tiers[name].get("carried")
            for name in ("L0", "L1", "L2")
        )
    ):
        raise ValueError(
            "primitive original full-value L0/L1/L2 or actual device trace is missing/stale"
        )


def _read_arm(owner, *, result_path, package, capsule):
    result_path, package = _ordinary(result_path), Path(package)
    if (
        result_path.name != "capsule_result.json"
        or result_path.stat().st_size > 16 << 20
    ):
        raise ValueError("primitive needs its original bounded ordinary capsule result")
    row = json.loads(result_path.read_bytes())
    _diagnostic_grade(row, capsule)
    owner.check_package_binding(package)
    source, tier = row.get("compiler_source", {}), row["tiers"]["L2"]
    runtime = owner.cleanup.revalidate()["runtime_inputs"]
    provenance = tier.get("sim_provenance", {})
    expected = source.get("candidate_sha256")
    generated = result_path.parent / "generated"
    if (
        type(expected) is not str
        or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
        or source.get("schema") != "merlin.capsule_compiler_source.v1"
        or source.get("status") != "stable"
        or source.get("package") != str(package.resolve())
        or source.get("snapshot") != "compiler-source/" + str(expected)
        or tier.get("engine") != "spike"
        or tier.get("toolchain") != str(package.resolve())
        or provenance.get("engine") != "spike"
        or provenance.get("binary") != runtime["spike"]["path"]
        or provenance.get("sha256") != runtime["spike"]["sha256"]
        or hash_tree(package)["sha256"] != expected
        or hash_tree(generated / source["snapshot"])["sha256"] != expected
        or E.collect(generated) != source.get("executed_programs")
    ):
        raise ValueError(
            "primitive current compiler/source snapshot/runtime/execution changed"
        )
    executions = [
        entry
        for entry in source["executed_programs"]
        if entry.get("schema") == E.SCHEMA_V2
        and entry.get("status") == "retained"
        and entry.get("engine") == "spike"
        and entry.get("engine_sha256") == provenance["sha256"]
    ]
    if len(executions) != 1:
        raise ValueError(
            "primitive needs one exact original retained functional execution v2"
        )
    execution = executions[0]
    artifacts = execution.get("compiler_artifacts", {})
    if artifacts.get("status") != "retained" or set(artifacts.get("members", {})) != {
        "command_buffer",
        "lowered_mlir",
        "console",
        "build",
        "invocation",
        "kernel_object",
    }:
        raise ValueError(
            "primitive original compiler/native/caller artifact roles are incomplete"
        )
    members = {
        role: _member(generated, pin) for role, pin in artifacts["members"].items()
    }
    elf, harness = (_member(generated, execution[role]) for role in ("elf", "harness"))
    cb = json.loads(members["command_buffer"].read_bytes())
    build = json.loads(members["build"].read_bytes())
    invocation = json.loads(members["invocation"].read_bytes())
    original = artifacts.get("original_execution_invocation", {})
    if (
        set(original) != {"path", "sha256"}
        or _pin(original["path"]) != original
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
            "primitive original complete build, input/caller or execution custody changed"
        )
    invocation_record.verify(Path(original["path"]))
    source_path = _ordinary(generated / "input.interface.mlir")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source.get(
        "artifacts", {}
    ).get("input.interface.mlir"):
        raise ValueError("primitive original graded input source changed")
    original_cb_path = _ordinary(generated / "command_buffer.json")
    if hashlib.sha256(original_cb_path.read_bytes()).hexdigest() != source.get(
        "artifacts", {}
    ).get("command_buffer.json"):
        raise ValueError("primitive original emitted pointer-order artifact changed")
    original_cb = json.loads(original_cb_path.read_bytes())
    if (
        set(original_cb.get("tensors", {})) != set(cb["tensors"])
        or "kernel_abi" in cb
        or "kernel_abi" in original_cb
    ):
        raise ValueError(
            "primitive implicit source pointer roster changed or became an unreviewed explicit ABI"
        )
    return {
        "result": _pin(result_path),
        "source": _pin(source_path),
        "elf": elf,
        "harness": _pin(harness),
        "command_buffer": cb,
        "command_buffer_artifact": _pin(members["command_buffer"]),
        "lowered": _pin(members["lowered_mlir"]),
        "reference_console": members["console"],
        "original_execution_invocation": original,
        "ordinary_status": row["status"],
        "ordinary_failure": row.get("failure"),
        "complete_artifacts": artifacts,
        "build_record": build,
        "original_command_buffer": original_cb,
        "caller_roster_basis": _pin(original_cb_path),
    }
