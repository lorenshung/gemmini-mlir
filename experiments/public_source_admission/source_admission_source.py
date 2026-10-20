"""Reopen original Phase0 source, complete references and role inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from merlin.targetgen.native_pointer_source_binding import _bounded_arrays
from merlin_experiments.phase0 import development_capture_intake as I
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0 import public_source_roles as R
from merlin_experiments.phase0 import sealed_native_program as N
from source_admission_inputs import (
    compiler_python,
    copy_compiler,
    reopen_plan,
    require_fields,
)


def _native_inputs(binding, member, intake):
    require_fields(binding, {"native_result"}, "original supplemental input producer")
    native = intake.document(binding["native_result"])
    source = native["source"]
    for key in ("interface", "inputs", "capsule_declaration"):
        row = source[key]
        intake.read(
            {"path": row["path"], "sha256": row["sha256"], "bytes": row["size_bytes"]}
        )
    loader = Path(source["capsule_declaration"]["path"]).parent / "capsule.pytorch.py"
    intake.read(dict(path=str(loader), **member["loader_identity"]))
    return Path(source["inputs"]["path"]), [
        f"in{i}" for i in range(len(member["description"]["inputs"]))
    ]


def check(declaration, intake):
    """Verify inputs only. Original target/compiler/source effects remain separate."""
    planned = reopen_plan(declaration, intake)
    compiler_python(declaration, intake)
    compiler_sha256 = copy_compiler(declaration, intake, None)
    original_plan = intake.document(declaration["job_plan"])
    if compiler_sha256 != original_plan["compiler_sha256"]:
        raise ValueError("compiler differs from original complete development plan")
    request = declaration["source"]
    supplements = []
    if type(request["supplements"]) is not list or len(request["supplements"]) > 8:
        raise ValueError("reference supplements need a finite selected roster")
    for pin in request["supplements"]:
        intake.document(pin)
        supplements.append((Path(pin["path"]), pin["sha256"]))
    manifest_pin = request["manifest"]
    intake.document(manifest_pin)
    manifest, refs = I.references(
        Path(manifest_pin["path"]).parent,
        manifest_sha256=manifest_pin["sha256"],
        supplements=supplements,
    )
    n = request["ordinal"]
    if n >= len(manifest["members"]) or n not in refs:
        raise ValueError("complete independent source/reference is unavailable")
    member = manifest["members"][n]
    if (
        member["request_sha256"] != planned["request_sha256"]
        or member.get("numeric_policy") != planned["numerical_policy"]
        or member.get("description", {}).get("declaration_sha256")
        != planned["declaration_sha256"]
    ):
        raise ValueError(
            "source, arithmetic or role request differs from original complete plan"
        )
    screen = I.plan(manifest, refs, **request["cost_limits"])["members"][n]
    if screen["status"] != "capture_cost_screen_passed_not_admission":
        raise ValueError(
            "original source/reference cost or numeric screen refused: "
            + "; ".join(screen["reasons"])
        )
    root = Path(manifest_pin["path"]).parent
    loader = root / member["loader"]
    raw_loader = intake.read(dict(path=str(loader), **member["loader_identity"]))
    ref = refs[n]
    if ref["kind"] == "original_frozen_npz":
        intake.read({"path": ref["path"], **ref["identity"]}, maximum=16 << 20)
        _bounded_arrays(Path(ref["path"]))
    else:
        for item in ref["outputs"]:
            intake.read(
                {key: item[key] for key in ("path", "bytes", "sha256")},
                maximum=16 << 20,
            )
    expected = I.load_outputs(member, ref)
    if ref["kind"] == "original_frozen_npz":
        if request["input_binding"] is not None:
            raise ValueError(
                "original full reference cannot select a foreign input producer"
            )
        path = Path(ref["path"])
        intake.read(dict(path=str(path), **ref["identity"]), maximum=16 << 20)
        keys = [f"input_{i}" for i in range(len(member["description"]["inputs"]))]
    else:
        if member["description"]["declaration"]["pattern"] != "matmul":
            raise ValueError(
                "supplementary input producer currently supports original raw rank2 sources only"
            )
        path, keys = _native_inputs(request["input_binding"], member, intake)
    _bounded_arrays(path)
    with np.load(path, allow_pickle=False) as archive:
        inputs = [archive[key].copy() for key in keys]
    for value, abi in zip(inputs, member["description"]["inputs"], strict=True):
        if value.dtype != np.dtype(abi["dtype"]) or list(value.shape) != abi["shape"]:
            raise ValueError("complete original input ABI differs")
    if ref["kind"] != "original_frozen_npz":
        from merlin_experiments.phase0.development_blocked_reference import seed_matrix

        for i, value in enumerate(inputs):
            wanted = seed_matrix(
                value.shape,
                i,
                np.arange(value.shape[0], dtype=np.int64),
                np.arange(value.shape[1], dtype=np.int64),
            )
            np.testing.assert_array_equal(value, wanted)
    attestation = intake.document(request["attestation"])
    artifact = N.read(
        attestation,
        program=request["program"],
        source_member=request["source_member"],
        max_payload_bytes=16 << 20,
    )
    if artifact.pytorch_src.encode() != raw_loader:
        raise ValueError(
            "original sealed loader differs from the selected public source"
        )
    if len(artifact.inputs) != len(inputs):
        raise ValueError("sealed source changed complete input arity")
    for got, want in zip(artifact.inputs, inputs, strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)
    values = [artifact.golden] if len(expected) == 1 else artifact.golden
    if len(values) != len(expected):
        raise ValueError("sealed source changed complete output arity")
    for got, want in zip(values, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)
    private = E.select(
        Path(declaration["roles"]["private_exclusion"]["path"]),
        declaration["roles"]["private_exclusion"]["sha256"],
    )
    roles = R.select(
        Path(declaration["roles"]["profile"]["path"]),
        declaration["roles"]["profile"]["sha256"],
        operator_exclusion=private,
    )
    factory_inputs = intake.document(declaration["factory_inputs"])
    intake.read(declaration["execution_contract"])
    # Factory inputs are closed by the ORIGINAL constructor after emitted ABI selection.
    for key in ("facts", "observed_contract", "software_spec", "library_selection"):
        path = Path(factory_inputs[key])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("factory selected inputs must retain canonical paths")
    intake.finish()
    return {
        "member": member,
        "artifact": artifact,
        "attestation": attestation,
        "roles": roles,
        "private": private,
        "factory_inputs": factory_inputs,
        "expected": expected,
        "planned": planned,
        "loader": loader,
        "compiler_sha256": compiler_sha256,
    }
