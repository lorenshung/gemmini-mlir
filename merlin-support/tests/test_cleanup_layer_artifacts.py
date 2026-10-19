"""Synthetic final-grade artifact controls; no simulator or device qualification."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module(
        "gemmini_cleanup_layer_artifacts"
    ), importlib.import_module("gemmini_cleanup_issue")


def _oracle():
    return b"complete original fixture outputs\n"


@pytest.fixture
def arm(tmp_path, modules):
    from merlin.common import invocation_record as I
    from merlin.targetgen import executed_program as E
    from merlin.targetgen.capsule_source_identity import (
        _capture_compiler_source,
        _finish_compiler_source,
    )
    from merlin.targetgen.contract.readback_policy import canonical_sha256

    package = tmp_path / "compiler"
    package.mkdir()
    (package / "compiler.py").write_text("# neutral synthetic source\n")
    run = tmp_path / "run"
    work = run / "generated"
    work.mkdir(parents=True)
    captured = _capture_compiler_source(package, work)
    files = {
        "package_kernel.elf": b"ELF fixture",
        "harness.c": b"harness fixture",
        "kernel.o": b"object fixture",
        "input.interface.mlir": b"module {}",
    }
    for name, data in files.items():
        (work / name).write_bytes(data)
    pin = lambda name: hashlib.sha256(files[name]).hexdigest()
    cb = {"tensors": {}, "params": {"source": "original"}}
    build = {
        "status": "completed",
        "command_buffer_sha256": canonical_sha256(cb),
        "readback_policy": {"transport": "out_b64_v1"},
        "elf_sha256": pin("package_kernel.elf"),
        "harness_sha256": pin("harness.c"),
        "kernel_object_name": "kernel.o",
        "kernel_object_sha256": pin("kernel.o"),
    }
    build["build_identity_sha256"] = canonical_sha256(build)
    with I.observe_call(
        work,
        stage="execution",
        function=_oracle,
        arguments={"simulator": "spike"},
        inputs=(work / "package_kernel.elf",),
    ) as observed:
        observed.returned(stdout=_oracle())
    binary = tmp_path / "selected-functional-runtime"
    binary.write_text("synthetic runtime; never executed")
    runtime = {
        "spike": {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        }
    }
    provenance = {
        "engine": "spike",
        "binary": str(binary),
        "sha256": runtime["spike"]["sha256"],
    }
    res = {
        "elf": str(work / "package_kernel.elf"),
        "oracle": {"provenance": provenance},
        "readback_build": build,
        "console": _oracle(),
        "execution_invocation": {
            "path": str(observed.path),
            "sha256": hashlib.sha256(observed.path.read_bytes()).hexdigest(),
        },
    }
    E.retain(
        res,
        sim="spike",
        workdir=work,
        compiler_artifacts={"command_buffer": cb, "lowered_mlir": "module {}"},
    )
    row = {
        "status": "pass",
        "numeric": {"status": "pass"},
        "failure": None,
        "tiers": {k: {"status": "pass"} for k in ("L0", "L1", "L2")},
        "compiler_source": _finish_compiler_source(captured, work),
    }
    row["tiers"]["L2"].update(
        measured_now=True, sim_provenance=provenance, toolchain=str(package)
    )
    result = run / "capsule_result.json"
    result.write_text(json.dumps(row))
    owner = SimpleNamespace(
        revalidate=lambda: None,
        cleanup=SimpleNamespace(revalidate=lambda: {"runtime_inputs": runtime}),
    )
    return owner, result, package, row


def test_exact_original_layer_program_can_be_read_without_issuing_a_verdict(
    arm, modules
):
    owner, path, package, _row = arm
    result = modules[0].read_layer_arm(owner, result_path=path, package=package)
    assert result["elf"].read_bytes() == b"ELF fixture"
    assert result["command_buffer"]["params"]["source"] == "original"
    assert result["retained_artifacts"]["status"] == "retained"
    assert "qualified" not in result and "timing_cycles" not in result


@pytest.mark.parametrize(
    "change",
    [
        "pending",
        "numeric",
        "carried",
        "runtime",
        "snapshot",
        "source",
        "duplicate",
        "console",
        "invocation",
        "build",
    ],
)
def test_partial_stale_or_foreign_layer_artifacts_refuse(arm, modules, change):
    owner, path, package, row = arm
    work = path.parent / "generated"
    if change == "pending":
        row["status"] = "incomplete"
    elif change == "numeric":
        row["numeric"]["status"] = "pending"
    elif change == "carried":
        row["tiers"]["L2"]["measured_now"] = False
    elif change == "runtime":
        row["tiers"]["L2"]["sim_provenance"]["sha256"] = "f" * 64
    elif change == "snapshot":
        (work / row["compiler_source"]["snapshot"] / "compiler.py").write_text(
            "changed"
        )
    elif change == "source":
        (work / "input.interface.mlir").write_text("changed")
    elif change == "duplicate":
        row["compiler_source"]["executed_programs"].append(
            copy.deepcopy(row["compiler_source"]["executed_programs"][0])
        )
    else:
        members = row["compiler_source"]["executed_programs"][0]["compiler_artifacts"][
            "members"
        ]
        (work / members[change]["path"]).write_text("changed")
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        modules[0].read_layer_arm(owner, result_path=path, package=package)


def test_original_artifact_method_dispatch_cannot_be_replaced_by_saved_callback(
    modules,
):
    class Owner:
        def revalidate(self):
            pass

        def check_pair_artifacts(self, before, after, *, execution_domain):
            return (before["original"], after["original"], execution_domain)

    result = modules[1].check_pair_artifacts(Owner(), {"original": 1}, {"original": 2})
    assert result == (1, 2, "functional_spike")
    owner = Owner()
    owner.check_pair_artifacts = lambda *a, **kw: {"success": True}
    with pytest.raises(ValueError, match="bound owner"):
        modules[1].check_pair_artifacts(owner, {}, {})
