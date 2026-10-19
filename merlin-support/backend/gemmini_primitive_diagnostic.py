"""Live original primitive-pair L2 diagnostic, never a final qualification.

Ordinary numerical grading stays untouched, including mandatory pending L3.
This separate owner reopens the original grade/build/caller and freshly checks
actual packets, immutable inputs, all products and outputs for each current ELF.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import yaml
from gemmini_cleanup_issue import check_issued_program
from gemmini_cleanup_layer_artifacts import _member, _ordinary
from gemmini_primitive_packets import check_primitive_packets
from gemmini_primitive_source import bind_primitive, check_integer_values
from gemmini_staged_tile_effects import check_output_tile_permutation
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_initializer_capability import QualifiedZeroInitializerOwner
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.common.tree_hash import hash_tree
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _original_methods(owner):
    callbacks = {}
    for name in ("revalidate", "check_compiler_binding"):
        callback = getattr(owner, name, None)
        if (
            type(callback) is not MethodType
            or callback.__self__ is not owner
            or callback.__func__ is not getattr(QualifiedZeroInitializerOwner, name)
        ):
            raise ValueError(
                "primitive diagnostic requires original bound compiler-owner callbacks"
            )
        callbacks[name] = callback.__func__.__code__
    return callbacks


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
    owner.qualification.check_compiler_binding(package)
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


class PrimitiveDiagnosticOwner:
    """Explicit additional original effect checker; no JSON qualification loader."""

    def __init__(self, qualification):
        if type(qualification) is not QualifiedZeroInitializerOwner:
            raise ValueError(
                "primitive diagnostic requires its original live qualified compiler owner"
            )
        self._callbacks = _original_methods(qualification)
        qualification.revalidate()
        self.qualification = qualification
        self._qualification_identity = qualification.identity
        self.selected = qualification.prototype.tile.selected
        self.arithmetic = qualification.prototype.tile.arithmetic
        self.cleanup, self.nm = qualification.cleanup, qualification.nm
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                __name__,
                "gemmini_primitive_source",
                "gemmini_primitive_packets",
            "gemmini_primitive_host_scope",
                "gemmini_packet_contractions",
                "gemmini_source_packet_artifacts",
                "gemmini_cleanup_issue",
                "gemmini_cleanup_layer_artifacts",
                "gemmini_staged_tile_effects",
                "gemmini_staged_rhs_effects",
                "gemmini_execute_state",
                "gemmini_zero_initializer_capability",
                "merlin.runtime.harness_render",
                "merlin.runtime.direct_kernel_harness",
                "merlin.runtime.backends.chipyard_rocc_bound",
                "merlin.targetgen.contract.interface_emit",
                "merlin.targetgen.capsule_golden",
                "merlin.targetgen.capsule_inputs",
                "merlin.targetgen.executed_program",
            )
        }
        self._identity = canonical_sha256([self._qualification_identity, self._sources])

    @property
    def identity(self):
        return "gemmini.primitive_diagnostic.v1:" + self._identity

    def revalidate(self):
        if type(self.qualification) is not QualifiedZeroInitializerOwner:
            raise ValueError("primitive original live qualification was replaced")
        if _original_methods(self.qualification) != self._callbacks:
            raise ValueError("primitive original compiler-owner callbacks changed")
        self.qualification.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.qualification.identity != self._qualification_identity
            or canonical_sha256([self._qualification_identity, self._sources])
            != self._identity
            or self.selected is not self.qualification.prototype.tile.selected
            or self.arithmetic is not self.qualification.prototype.tile.arithmetic
            or self.cleanup is not self.qualification.cleanup
            or self.nm != self.qualification.nm
        ):
            raise ValueError(
                "primitive original selection/semantics/checker identity changed"
            )

    def check_pair(
        self,
        *,
        capsule_dir,
        baseline_result,
        baseline_package,
        candidate_result,
        candidate_package,
        work_root,
    ):
        from merlin.targetgen import capsule_golden as CG

        self.revalidate()
        started = time.monotonic()
        capsule_dir, work_root = Path(capsule_dir), Path(work_root)
        if (
            not work_root.is_absolute()
            or work_root.exists()
            or work_root.resolve() != work_root
            or any(p.is_symlink() for p in work_root.parents)
        ):
            raise ValueError(
                "primitive proof requires a new ordinary independent work root"
            )
        pins = {
            name: _pin(_ordinary(capsule_dir / name))
            for name in (
                "capsule.yaml",
                "capsule.interface.mlir",
                "golden.yaml",
                "golden.npz",
            )
        }
        if any(Path(pin["path"]).stat().st_size > 64 << 20 for pin in pins.values()):
            raise ValueError(
                "primitive original source/reference leaves exceed the bounded byte domain"
            )
        capsule = yaml.safe_load(Path(pins["capsule.yaml"]["path"]).read_bytes())
        capsule["__dir__"] = str(capsule_dir)
        source = Path(pins["capsule.interface.mlir"]["path"]).read_text()
        bind_primitive(capsule, source)
        inputs = {
            name: np.asarray(tensor.data).reshape(tensor.shape)
            for name, tensor in CG.materialize_capsule_leaves(capsule).items()
        }
        golden = CG.golden(capsule, capsule_dir)
        arms, streams, rhs_names, costs = {}, [], [], {}
        for label, result, package in (
            ("baseline", baseline_result, baseline_package),
            ("candidate", candidate_result, candidate_package),
        ):
            tick = time.monotonic()
            arm = _read_arm(self, result_path=result, package=package, capsule=capsule)
            if Path(arm["source"]["path"]).read_text() != source:
                raise ValueError(
                    "primitive ordinary grade used another original source"
                )
            contraction = bind_primitive(capsule, source, arm["command_buffer"])
            bind_primitive(capsule, source, arm["original_command_buffer"])
            caller_order = list(arm["original_command_buffer"]["tensors"])
            if caller_order != [contraction.rhs, contraction.lhs, contraction.output]:
                raise ValueError(
                    "primitive original source pointer order differs from resident weight/lhs/output ABI"
                )
            numeric = check_integer_values(
                contraction, inputs, golden, self.arithmetic.source_rule()
            )
            actual, decoder = {}, OutB64Decoder()
            for line in arm["reference_console"].read_text().splitlines():
                decoder.consume(line.split(), actual)
            decoder.require_closed()
            if set(actual) != set(golden) or any(
                not np.array_equal(
                    np.asarray(actual[name]).reshape(-1), np.asarray(value).reshape(-1)
                )
                for name, value in golden.items()
            ):
                raise ValueError(
                    "primitive original actual complete outputs differ from independent reference"
                )
            current = check_issued_program(
                self,
                elf=arm["elf"],
                command_buffer=arm["command_buffer"],
                reference_console=arm["reference_console"],
                work_root=work_root / label,
            )
            current.update(
                elf=arm["elf"],
                build_record=arm["build_record"],
                caller_order=caller_order,
            )
            packets, stream, rhs = check_primitive_packets(
                self, current, contraction, inputs
            )
            arms[label] = {
                "original_result": arm["result"],
                "ordinary_status": arm["ordinary_status"],
                "ordinary_failure": arm["ordinary_failure"],
                "elf": _pin(arm["elf"]),
                "harness": arm["harness"],
                "command_buffer": arm["command_buffer_artifact"],
                "lowered": arm["lowered"],
                "original_execution": arm["original_execution_invocation"],
                "caller_roster_basis": arm["caller_roster_basis"],
                "fresh_actual_issue": current["issue_invocation"],
                "source_arithmetic": numeric,
                "actual_packets": packets,
            }
            streams.append(stream)
            rhs_names.append(rhs)
            costs[label + "_original_custody_and_fresh_packet_proof_wall_seconds"] = (
                time.monotonic() - tick
            )
        if len(set(rhs_names)) != 1:
            raise ValueError(
                "primitive pair changed exact immutable linked RHS ownership"
            )
        geometry = self.selected.report()["geometry"]
        stores = {row["name"]: row for row in geometry["stores"]}
        effects = check_output_tile_permutation(
            *streams,
            rhs=rhs_names[0],
            operand_rows=stores["scratchpad"]["banks"]
            * stores["scratchpad"]["rows_per_bank"],
            dim=geometry["array_rows"],
        )
        self.revalidate()
        _revalidate_sources(pins)
        return {
            "schema": "gemmini.primitive_pair_diagnostic.v1",
            "status": "checked_actual_primitive_L2_pair",
            "owner": self.identity,
            "source": pins,
            "arms": arms,
            "actual_issue_effects": effects,
            "costs": costs,
            "wall_seconds": time.monotonic() - started,
            "original_required_tiers": capsule["required_oracle_tiers"],
            "L3": "pending; no waiver or copied verdict",
            "scope": "original ordinary primitive arithmetic and actual complete current ELF issue",
            "final_functional_guard_qualified": False,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "performance_promoted": False,
        }
