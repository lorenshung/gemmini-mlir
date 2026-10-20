"""Early availability uses the original loader selector; no runtime execution."""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/public_source_admission"))
I = importlib.import_module("source_admission_inputs")
RUN = importlib.import_module("admit_sources")


@pytest.fixture
def selected_environment(tmp_path, monkeypatch):
    from merlin.integrations import model2mlir
    from merlin.targetgen import capsule_source

    checkout = tmp_path / "capture-checkout"
    # Isolate .env lookups while preserving the actual original selector.
    monkeypatch.setattr(model2mlir, "env", os.environ.get)
    monkeypatch.setattr(capsule_source, "_m2m_dir", lambda: checkout)
    monkeypatch.delenv("MERLIN_M2M_PYTHON", raising=False)
    monkeypatch.delenv("MERLIN_M2M_VENV", raising=False)
    return checkout


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 99\n")
    path.chmod(0o700)
    return path


def test_explicit_interpreter_uses_original_precedence(
    selected_environment, tmp_path, monkeypatch
):
    default = executable(selected_environment / ".venv/bin/python")
    chosen = executable(tmp_path / "explicit-python")
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(chosen))
    report = I.native_loader_python_preflight()
    assert report["python"] == str(chosen) != str(default)
    assert report["runtime_qualification"] == "NONE"
    assert report["framework_or_loader_execution"] is False


def test_valid_default_needs_no_explicit_environment(selected_environment):
    chosen = executable(selected_environment / ".venv/bin/python")
    assert I.native_loader_python_preflight()["python"] == str(chosen)


def test_venv_selection_preserves_symlink_spelling(
    selected_environment, tmp_path, monkeypatch
):
    venv = tmp_path / "selected-venv"
    chosen = venv / "bin/python"
    chosen.parent.mkdir(parents=True)
    chosen.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setenv("MERLIN_M2M_VENV", str(venv))
    assert I.native_loader_python_preflight()["python"] == str(chosen)
    assert chosen.resolve() != chosen


@pytest.mark.parametrize("kind", ["missing", "not_executable", "directory"])
def test_explicit_bad_selection_never_falls_back(
    selected_environment, tmp_path, monkeypatch, kind
):
    executable(selected_environment / ".venv/bin/python")
    chosen = tmp_path / "bad-python"
    if kind == "not_executable":
        chosen.write_text("not executable")
    elif kind == "directory":
        chosen.mkdir()
    monkeypatch.setenv("MERLIN_M2M_PYTHON", str(chosen))
    with pytest.raises(ValueError, match="missing or not executable") as refused:
        I.native_loader_python_preflight()
    assert str(chosen) in str(refused.value)


def test_missing_default_refuses_the_original_loader_path(selected_environment):
    with pytest.raises(ValueError, match="missing or not executable") as refused:
        I.native_loader_python_preflight()
    assert str(selected_environment / ".venv/bin/python") in str(refused.value)


@pytest.mark.parametrize("mode", ["--check-inputs", "--execute"])
def test_runtime_refusal_precedes_source_intake_and_paid_work(
    selected_environment, tmp_path, monkeypatch, mode
):
    declaration = {"source": {"identity": "public:0"}}
    monkeypatch.setattr(
        RUN, "read_declaration", lambda *args: (declaration, I.Intake())
    )
    monkeypatch.setattr(
        RUN, "require_output", lambda declaration, intake, output: output
    )
    monkeypatch.setattr(
        RUN,
        "check",
        lambda *args: pytest.fail("source intake preceded runtime availability"),
    )
    monkeypatch.setattr(
        RUN,
        "execute",
        lambda *args: pytest.fail("unavailable runtime entered execution"),
    )
    output = tmp_path / "refused"
    assert (
        RUN.main(
            [
                "--declaration",
                str(tmp_path / "unused.json"),
                "--sha256",
                "0" * 64,
                "--output",
                str(output),
                mode,
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_bytes())
    assert report["status"] == "precise_original_refusal"
    assert report["stage"] == "input_verification"
    assert "selected frozen-loader Python" in report["reason"]


def test_direct_execute_caller_also_checks_before_materialization(
    selected_environment, monkeypatch
):
    monkeypatch.setattr(RUN, "require_purpose", lambda *args: {})
    monkeypatch.setattr(
        RUN,
        "prepare",
        lambda *args: pytest.fail("unavailable runtime materialized source"),
    )
    with pytest.raises(ValueError, match="selected frozen-loader Python"):
        RUN.execute({}, I.Intake(), {}, None, {}, lambda: None)
