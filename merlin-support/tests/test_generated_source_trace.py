"""Real harmless host-process trace custody; no accelerator execution."""

import importlib
from types import SimpleNamespace

import pytest
from test_source_execution_issuer import (  # noqa: F401 -- shared neutral inputs
    fixture,
    module,
)


@pytest.fixture
def generated(fixture, monkeypatch):  # noqa: F811
    T = importlib.import_module("gemmini_generated_source_trace")
    state = {
        "runtime": fixture.runtime,
        "arithmetic": object(),
        "selected": object(),
        "protocol": "gemmini.source_observation.v2",
        "nm": {"path": str(fixture.source)},
        "sources": fixture.issuer._sources,
        "binding_protocol": "whole_program",
    }
    owner = SimpleNamespace(revalidate=lambda: None, _original_state=lambda: state)
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(original_constructor_arguments=kwargs)

    state["owner_type"] = constructor
    return T, fixture, owner, state, calls


def test_original_current_native_runtime_and_complete_invocation_survive(generated):
    T, f, owner, _, calls = generated
    actual = T.observe_current_source(
        owner,
        native_result=f.result,
        command_buffer=f.cb,
        lowered=f.lowered,
        work_root=f.root / "trace",
        remaining_s=10,
    )
    from merlin.common import invocation_record

    record = invocation_record.verify(calls[0]["trace_invocation"])
    assert record["argv"][-2:] == ["--log-commits", str(f.elf)]
    assert "-l" not in record["argv"]
    assert actual.original_constructor_arguments == calls[0]
    assert record["returncode"] == 0 and record["stdout"]["sha256"]


def test_foreign_current_runtime_refuses_before_process(generated):
    T, f, owner, state, calls = generated
    state["runtime"]["spike"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different functional runtime"):
        T.observe_current_source(
            owner,
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "trace",
            remaining_s=10,
        )
    assert not calls and not (f.root / "trace").exists()


def test_existing_output_refuses_before_process(generated):
    T, f, owner, _, calls = generated
    work = f.root / "trace"
    work.mkdir()
    with pytest.raises(ValueError, match="fresh disjoint"):
        T.observe_current_source(
            owner,
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=work,
            remaining_s=10,
        )
    assert not calls


def test_changed_current_artifacts_after_complete_process_refuse(
    generated, monkeypatch
):
    T, f, owner, _, calls = generated
    original = T.subprocess.run

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        f.lowered.write_text("changed after actual process")
        return result

    monkeypatch.setattr(T.subprocess, "run", run)
    with pytest.raises(ValueError, match="unchanged observed inputs"):
        T.observe_current_source(
            owner,
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "trace",
            remaining_s=10,
        )
    assert not calls


def test_direct_protocol_refuses_old_native_schema_before_process(generated):
    T, f, owner, state, calls = generated
    state["binding_protocol"] = "merlin.native_pointer_source.v1"
    with pytest.raises(ValueError, match="explicit original native execution record"):
        T.observe_current_source(
            owner,
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "trace",
            remaining_s=10,
        )
    assert not calls and not (f.root / "trace").exists()


def test_direct_protocol_preserves_current_native_record_and_selected_contract(
    generated,
):
    import json

    T, f, owner, state, calls = generated
    state["binding_protocol"] = "merlin.native_pointer_source.v1"
    state["execution_contract"] = {
        "path": str(f.source),
        "sha256": "neutral original pin",
    }
    row = json.loads(f.result.read_bytes())
    row.update(
        schema="merlin_candidate_native_pointer_source_v1",
        pointer_execution={"protocol": state["binding_protocol"]},
    )
    f.result.write_text(json.dumps(row))
    T.observe_current_source(
        owner,
        native_result=f.result,
        command_buffer=f.cb,
        lowered=f.lowered,
        work_root=f.root / "trace",
        remaining_s=10,
    )
    assert calls[0]["execution_contract"] == state["execution_contract"]
