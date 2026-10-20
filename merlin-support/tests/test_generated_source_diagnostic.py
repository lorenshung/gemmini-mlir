"""Neutral constructor/transport controls; no target or scheduling qualification."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def issued(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    M = importlib.import_module("gemmini_generated_source_diagnostic")
    calls = []
    for name in ("facts", "observed_contract", "software_spec", "nm"):
        (tmp_path / name).write_text("original " + name)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib").mkdir()
    (runtime / "bin/spike").write_text("neutral binary")
    (runtime / "lib/libgemmini.so").write_text("neutral extension")
    selection = {
        "contract": {"review_id": "neutral", "public_modules": [], "members": []},
        "root": str(tmp_path),
    }
    (tmp_path / "library_selection").write_text(json.dumps({"selection": selection}))
    pins = {
        name: {
            "path": str(tmp_path / name),
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in ("facts", "observed_contract")
    }

    class Target:
        @classmethod
        def read(cls, *_):
            return cls()

        def revalidate(self):
            calls.append("target")

        def report(self):
            return {"selected_inputs": pins}

    class Arithmetic:
        identity = "original arithmetic"

        def __init__(self, selected, **kwargs):
            self.selected = selected

        def revalidate(self):
            calls.append("arithmetic")

        def source_rule(self):
            return {"primitive": 20}

        def record(self):
            return {"source": "neutral"}

    class Completion:
        identity = "original completion"

        def __init__(self, selected, **kwargs):
            self.selected = selected

        def revalidate(self):
            calls.append("completion")

        def record(self):
            return {"proof": {"original": True}, "selected_inputs": pins}

    class LibraryContract:
        sha256 = "neutral library"

        def __init__(self, *args):
            pass

    class Library:
        def __init__(self, contract, root):
            self.contract = contract

        def record(self):
            calls.append("library")
            return selection

    def trace(owner, **kwargs):
        owner.revalidate()
        calls.append(("trace", kwargs))
        return returned

    returned = object()
    for name, value in (
        ("SelectedSchedulingTarget", Target),
        ("IntegerArithmeticOwner", Arithmetic),
        ("StagedCompletionOwner", Completion),
        ("CompilerLibraryContract", LibraryContract),
        ("CompilerLibrarySelection", Library),
        ("observe_current_source", trace),
    ):
        monkeypatch.setattr(M, name, value)
    inputs = {
        name: str(tmp_path / name) for name in M._FIELDS - {"observation_protocol"}
    }
    inputs.update(
        source_root=str(tmp_path),
        receiver_source=str(tmp_path),
        observation_protocol=M.COMMIT_ONLY_PROTOCOL,
    )
    owner = M.GeneratedSourceDiagnosticIssuer(inputs)
    return M, owner, inputs, calls, returned


def test_only_source_diagnostic_is_issued_without_cleanup_reports(issued, tmp_path):
    M, owner, inputs, calls, returned = issued
    registry = M.restore_registry(inputs, work_root=tmp_path / "fresh")
    row = registry.record()["capabilities"]
    assert len(row) == 1 and row[0]["mechanism"] == "source_execution_diagnostic"
    assert not hasattr(owner, "prepare_compiler") and not hasattr(
        owner, "check_functional_guard"
    )
    result = owner.record()
    assert result["scheduling_qualification"] == result["timing_authority"] == "NONE"
    assert result["hardware_execution_qualified"] is False
    assert (
        owner.create_source_execution_owner(
            native_result=tmp_path / "r",
            command_buffer=tmp_path / "cb",
            lowered=tmp_path / "llvm",
            work_root=tmp_path / "trace",
            remaining_s=10,
        )
        is returned
    )
    assert any(type(row) is tuple and row[0] == "trace" for row in calls)


@pytest.mark.parametrize(
    "fault", ["missing", "report", "protocol", "metadata", "symlink"]
)
def test_unselected_or_legacy_inputs_refuse(issued, tmp_path, fault):
    M, _, inputs, *_ = issued
    inputs = dict(inputs)
    if fault == "missing":
        del inputs["software_spec"]
    elif fault == "report":
        inputs["reports"] = ["old saved success"]
    elif fault == "protocol":
        inputs["observation_protocol"] = "gemmini.source_observation.v1"
    elif fault == "metadata":
        inputs["metadata_verification_protocol"] = "foreign"
    else:
        link = tmp_path / "linked"
        link.symlink_to(tmp_path / "software_spec")
        inputs["software_spec"] = str(link)
    with pytest.raises((ValueError, TypeError)):
        M.GeneratedSourceDiagnosticIssuer(inputs)


@pytest.mark.parametrize(
    "fault",
    [
        "input",
        "runtime",
        "binding",
        "source",
        "callback",
        "semantic_callback",
        "arithmetic_identity",
        "completion_identity",
    ],
)
def test_current_sources_owners_callbacks_and_binding_stay_fresh(
    issued, tmp_path, monkeypatch, fault
):
    M, owner, _, calls, _ = issued
    state = M._ISSUED[owner]
    if fault == "input":
        (tmp_path / "software_spec").write_text("changed")
    elif fault == "runtime":
        (tmp_path / "runtime/bin/spike").write_text("changed")
    elif fault == "binding":
        object.__setattr__(state["binding"], "software", "foreign")
    elif fault == "source":
        state["sources"]["gemmini_generated_source_diagnostic"]["sha256"] = "0" * 64
    elif fault == "callback":
        monkeypatch.setattr(owner, "_original_state", lambda: state)
    elif fault == "semantic_callback":
        monkeypatch.setattr(state["arithmetic"], "revalidate", lambda: None)
    elif fault == "arithmetic_identity":
        state["arithmetic"].identity = "foreign"
    else:
        state["completion"].identity = "foreign"
    with pytest.raises(ValueError):
        owner.revalidate()
    assert not any(type(row) is tuple and row[0] == "trace" for row in calls)


@pytest.mark.parametrize(
    "remaining", [True, False, 0, -1, float("nan"), float("inf"), "10"]
)
def test_unbounded_trace_never_admitted(issued, tmp_path, remaining):
    _, owner, _, calls, _ = issued
    with pytest.raises(ValueError, match="positive finite"):
        owner.create_source_execution_owner(
            native_result=tmp_path / "r",
            command_buffer=tmp_path / "cb",
            lowered=tmp_path / "llvm",
            work_root=tmp_path / "trace",
            remaining_s=remaining,
        )
    assert not any(type(row) is tuple and row[0] == "trace" for row in calls)


def test_saved_success_unissued_and_subclassed_owners_refuse(issued):
    M, _, *_ = issued
    with pytest.raises(ValueError):
        M.GeneratedSourceDiagnosticIssuer({"status": "pass"})
    forged = object.__new__(M.GeneratedSourceDiagnosticIssuer)
    with pytest.raises(ValueError, match="not originally issued"):
        forged.revalidate()

    class Foreign(M.GeneratedSourceDiagnosticIssuer):
        pass

    with pytest.raises(TypeError):
        Foreign({})
