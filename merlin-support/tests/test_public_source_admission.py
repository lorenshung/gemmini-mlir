"""Neutral operator-input/ABI controls; no compiler or simulator jobs."""

import hashlib
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/public_source_admission"))
I = importlib.import_module("source_admission_inputs")
RUN = importlib.import_module("admit_sources")


def pin(path):
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def file(tmp_path, name="input.json", content=b"{}"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_pin_reopens_changed_data(tmp_path):
    path = file(tmp_path)
    intake = I.Intake()
    assert intake.read(pin(path)) == b"{}"
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="bytes changed"):
        intake.finish()


def test_pin_rejects_symlink_parent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = file(source)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    row = dict(pin(path), path=str(alias / path.name))
    with pytest.raises(ValueError, match="canonical"):
        I.Intake().read(row)


@pytest.mark.parametrize(
    "change", ["wrong_sha", "wrong_size", "bool_size", "extra_field", "oversize"]
)
def test_pin_bad_inputs_refuse(tmp_path, change):
    path = file(tmp_path)
    row = pin(path)
    kwargs = {}
    if change == "wrong_sha":
        row["sha256"] = "0" * 64
    if change == "wrong_size":
        row["bytes"] += 1
    if change == "bool_size":
        row["bytes"] = True
    if change == "extra_field":
        row["pass"] = True
    if change == "oversize":
        kwargs["maximum"] = 1
    with pytest.raises(ValueError):
        I.Intake().read(row, **kwargs)


def test_abi_uses_actual_emission_without_rewriting():
    cb = {"kernel_abi": {"kind": "whole_program"}, "params": {"a": 1}}
    old = json.loads(json.dumps(cb))
    inputs = {
        "source_binding_protocol": "merlin.native_pointer_source.v1",
        "execution_contract": "/old",
    }
    route, chosen = I.select_abi(cb, inputs, "/current")
    assert route == "whole_program"
    assert chosen == {"source_binding_protocol": "whole_program"}
    assert cb == old and inputs["execution_contract"] == "/old"


def test_standalone_abi_selects_original_direct_protocol():
    route, chosen = I.select_abi(
        {"compiler_pointer_abi": {"version": 1, "arguments": ["a", "b", "y"]}},
        {},
        "/contract",
    )
    assert route == "merlin.native_pointer_source.v1"
    assert chosen["execution_contract"] == "/contract"


@pytest.mark.parametrize(
    "cb",
    [
        {},
        {"kernel_abi": {"kind": "foreign"}},
        {"compiler_pointer_abi": {"version": True}},
        {"kernel_abi": {}, "compiler_pointer_abi": {"version": 1}},
    ],
)
def test_unknown_abi_is_not_made_whole_program(cb):
    with pytest.raises(ValueError, match="no original"):
        I.select_abi(cb, {}, "/contract")


def compiler(tmp_path):
    root = tmp_path / "compiler"
    root.mkdir()
    files = []
    digest = hashlib.sha256()
    for name, raw in [
        ("compiler.py", b"public original\n"),
        ("manifest.yaml", b"entry: compiler.py\n"),
    ]:
        path = file(root, name, raw)
        files.append({"path": name, "bytes": len(raw), "sha256": pin(path)["sha256"]})
        digest.update(name.encode() + b"\0" + raw + b"\0")
    inventory = file(tmp_path, "inventory.json", json.dumps({"files": files}).encode())
    return root, {
        "compiler": {
            "root": str(root),
            "inventory": pin(inventory),
            "sha256": digest.hexdigest(),
        }
    }


def test_exact_compiler_copy_leaves_original_unchanged(tmp_path):
    root, declaration = compiler(tmp_path)
    original = {p.name: p.read_bytes() for p in root.iterdir()}
    out = tmp_path / "copied"
    assert (
        I.copy_compiler(declaration, I.Intake(), out)
        == declaration["compiler"]["sha256"]
    )
    assert {p.name: p.read_bytes() for p in out.iterdir()} == original
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original


@pytest.mark.parametrize("kind", ["extra", "mutated", "symlink", "special"])
def test_compiler_complete_namespace_and_bytes_required(tmp_path, kind):
    root, declaration = compiler(tmp_path)
    if kind == "extra":
        file(root, "extra.py")
    if kind == "mutated":
        (root / "compiler.py").write_bytes(b"changed")
    if kind == "symlink":
        (root / "alias").symlink_to(root / "compiler.py")
    if kind == "special":
        import os

        os.mkfifo(root / "pipe")
    with pytest.raises(ValueError):
        I.copy_compiler(declaration, I.Intake(), None)


def purpose(tmp_path, **limits):
    now = datetime.now(timezone.utc)
    value = {
        "start_utc": (now - timedelta(seconds=1)).isoformat(),
        "deadline_utc": (now + timedelta(seconds=90)).isoformat(),
        "identity": "public:1",
        "limits": {
            "aggregate_cpu_seconds": 80,
            "wall_seconds": 180,
            "rss_bytes": 1 << 30,
            "file_bytes": 64 << 20,
            "disk_bytes": 128 << 20,
        },
    }
    value["limits"].update(limits)
    path = file(tmp_path, "purpose.json", json.dumps(value).encode())
    return {"purpose": pin(path), "source": {"identity": "public:1"}}


@pytest.mark.parametrize(
    "limits",
    [
        {"file_bytes": (64 << 20) + 1},
        {"aggregate_cpu_seconds": 121},
        {"rss_bytes": float(1 << 30)},
        {"wall_seconds": True},
    ],
)
def test_purpose_cannot_enlarge_original_job_ceilings(tmp_path, limits):
    with pytest.raises(ValueError, match="ceilings"):
        RUN.require_purpose(purpose(tmp_path, **limits), I.Intake())


def test_expired_clock_is_not_restarted(tmp_path):
    declaration = purpose(tmp_path)
    path = Path(declaration["purpose"]["path"])
    value = json.loads(path.read_bytes())
    value["deadline_utc"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()
    path.write_text(json.dumps(value))
    declaration["purpose"] = pin(path)
    with pytest.raises(ValueError, match="fixed clock"):
        RUN.require_purpose(declaration, I.Intake())


@pytest.mark.parametrize("under", ["capture", "source", "compiler"])
def test_output_cannot_mutate_original_input_namespace(tmp_path, under):
    roots = {key: tmp_path / key for key in ("capture", "source", "compiler")}
    for path in roots.values():
        path.mkdir()
    att = file(
        tmp_path,
        "att.json",
        json.dumps(
            {"capture": {"kind": "session", "capture_path": str(roots["capture"])}}
        ).encode(),
    )
    d = {
        "source": {
            "attestation": pin(att),
            "manifest": {"path": str(roots["source"] / "manifest.json")},
        },
        "compiler": {"root": str(roots["compiler"])},
    }
    with pytest.raises(ValueError, match="immutable"):
        RUN.require_output(d, I.Intake(), roots[under] / "new-output")
    assert not (roots[under] / "new-output").exists()


def test_check_inputs_never_enters_execution(tmp_path, monkeypatch):
    att = file(
        tmp_path,
        "check-att.json",
        json.dumps(
            {"capture": {"kind": "session", "capture_path": str(tmp_path / "capture")}}
        ).encode(),
    )
    declaration = {
        "source": {
            "attestation": pin(att),
            "identity": "member:1",
            "manifest": {"path": str(tmp_path / "sources/manifest.json")},
        },
        "compiler": {"root": str(tmp_path / "compiler")},
    }
    monkeypatch.setattr(RUN, "read_declaration", lambda *a: (declaration, I.Intake()))
    monkeypatch.setattr(
        RUN,
        "check",
        lambda *a: {
            "compiler_sha256": "0" * 64,
            "expected": [np.asarray([1])],
            "member": {"description": {"scalar_products": 1}},
            "planned": {"prospective_role": "guard"},
        },
    )
    monkeypatch.setattr(
        RUN, "execute", lambda *a: pytest.fail("metadata-only mode started execution")
    )
    out = tmp_path / "check-output"
    assert (
        RUN.main(
            [
                "--declaration",
                str(tmp_path / "request.json"),
                "--sha256",
                "0" * 64,
                "--output",
                str(out),
                "--check-inputs",
            ]
        )
        == 0
    )
    report = json.loads((out / "report.json").read_bytes())
    assert report["status"] == "original_inputs_checked_no_target_execution"
    assert report["internal_resource_supervision"] is False
    assert report["formal_cohort_frozen"] is False


def test_reopen_plan_refuses_changed_original_role(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import development_admission_plan as P

    value = {
        "schema": "merlin.phase0.development_admission_job_plan.v1",
        "jobs": [{"limits": {}}],
        "members": [{"identity": "member:0", "prospective_role": "guard"}],
    }
    p = file(tmp_path, "plan.json", json.dumps(value).encode())
    actual = json.loads(json.dumps(value))
    actual["members"][0]["prospective_role"] = "objective"
    monkeypatch.setattr(P, "plan", lambda *a, **k: actual)
    with pytest.raises(ValueError, match="complete original"):
        I.reopen_plan(
            {
                "job_plan": pin(p),
                "readiness": {"path": str(p), "sha256": pin(p)["sha256"]},
                "source": {"identity": "member:0"},
            },
            I.Intake(),
        )
