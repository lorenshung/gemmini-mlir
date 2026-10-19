"""Native process/custody plumbing fixtures; these grant no target support."""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin.common import invocation_record
from merlin.targetgen.compiler_pass_selection import _pin


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_source_execution_issuer")


def native_pin(path):
    return {**_pin(path), "size_bytes": path.stat().st_size}


@pytest.fixture
def fixture(module, tmp_path, monkeypatch):
    runner = tmp_path / "fixture-runtime"
    runner.write_text(f"#!{sys.executable}\nprint('neutral native plumbing output')\n")
    runner.chmod(0o700)
    extension = tmp_path / "fixture-extension"
    extension.write_text("fixture only")
    elf = tmp_path / "fixture.elf"
    elf.write_text("neutral fixture input")
    source = tmp_path / "source.mlir"
    source.write_text("original independent source fixture")
    console = tmp_path / "console"
    console.write_text("neutral native plumbing output\n")
    cb = tmp_path / "command-buffer.json"
    cb.write_text('{"complete": "fixture"}')
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text("actual fixture lowered bytes")
    runtime = {"spike": _pin(runner), "functional_extension": _pin(extension)}
    row = {
        "failure": None,
        "numeric": {"status": "pass"},
        "candidate": {
            "command_buffer_sha256": hashlib.sha256(
                json.dumps(json.loads(cb.read_bytes()), sort_keys=True).encode()
            ).hexdigest(),
            "lowered_mlir_sha256": _pin(lowered)["sha256"],
        },
        "elf": native_pin(elf),
        "source": {"interface": native_pin(source)},
        "build_artifacts": {},
        "host_build_sources": [native_pin(source)],
        "tiers": {
            "L2": {
                "status": "pass",
                "engine": "spike",
                "elf": native_pin(elf),
                "console": native_pin(console),
                "engine_citation": {
                    "engine": "spike",
                    "target": "gemmini",
                    "binary": native_pin(runner),
                    "extension": native_pin(extension),
                    "flags": ["--extlib=" + str(extension), "--extension=gemmini"],
                },
            }
        },
    }
    result = tmp_path / "native-result.json"
    result.write_text(json.dumps(row))
    # This bypass is solely a plumbing test double. The real constructor below
    # rejects it; no source/numerical/effect verification is asserted here.
    issuer = object.__new__(module.SourceExecutionIssuer)
    issuer.revalidate = lambda: None
    issuer._prototype = SimpleNamespace(
        cleanup=SimpleNamespace(
            revalidate=lambda: {"runtime_inputs": copy.deepcopy(runtime)}
        ),
        arithmetic=object(),
        selected=object(),
        nm=source,
    )
    issuer._sources = {"fixture": _pin(Path(__file__).resolve())}
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(original_constructor_arguments=kwargs)

    monkeypatch.setattr(module, "SourceExecutionOwner", constructor)
    return SimpleNamespace(
        issuer=issuer,
        runtime=runtime,
        row=row,
        result=result,
        cb=cb,
        lowered=lowered,
        elf=elf,
        source=source,
        calls=calls,
        root=tmp_path,
    )


def test_saved_success_and_foreign_owner_cannot_construct_original_issuer(module):
    for value in ({"status": "qualified"}, None, object()):
        with pytest.raises(ValueError, match="original live qualified"):
            module.SourceExecutionIssuer(value)


def test_current_native_artifacts_run_fresh_original_observed_process_per_call(fixture):
    f = fixture
    for i in range(2):
        owner = f.issuer.create_source_execution_owner(
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / f"fresh-{i}",
            remaining_s=20,
        )
        args = owner.original_constructor_arguments
        trace = invocation_record.verify(args["trace_invocation"])
        assert trace["returncode"] == 0
        assert trace["argv"] == [
            f.runtime["spike"]["path"],
            "--extlib=" + f.runtime["functional_extension"]["path"],
            "--extension=gemmini",
            "-l",
            "--log-commits",
            str(f.elf),
        ]
        assert args["result"] == f.result
        assert args["command_buffer"] == f.cb
        assert args["lowered"] == f.lowered
    assert len(f.calls) == 2
    assert f.calls[0]["trace_invocation"] != f.calls[1]["trace_invocation"]


@pytest.mark.parametrize(
    "mutation",
    ["numeric", "cb", "lowered", "elf", "flags", "runtime", "tier_elf", "host_source"],
)
def test_stale_forged_wrong_runtime_or_incomplete_native_artifacts_refuse_before_trace(
    fixture, mutation
):
    f = fixture
    if mutation == "numeric":
        f.row["numeric"]["status"] = "fail"
    elif mutation == "cb":
        f.cb.write_text('{"different": "current candidate"}')
    elif mutation == "lowered":
        f.lowered.write_text("different executable IR")
    elif mutation == "elf":
        f.elf.write_text("different ELF")
    elif mutation == "flags":
        f.row["tiers"]["L2"]["engine_citation"]["flags"].append("--different")
    elif mutation == "runtime":
        f.row["tiers"]["L2"]["engine_citation"]["binary"]["sha256"] = "0" * 64
    elif mutation == "tier_elf":
        f.row["tiers"]["L2"]["elf"] = native_pin(f.source)
    else:
        f.source.write_text("changed original native build dependency")
    f.result.write_text(json.dumps(f.row))
    with pytest.raises(ValueError):
        f.issuer.create_source_execution_owner(
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "must-not-execute",
            remaining_s=20,
        )
    assert not (f.root / "must-not-execute").exists()
    assert not f.calls


@pytest.mark.parametrize("budget", [0, -1, True, float("nan"), float("inf"), "60"])
def test_invalid_or_expired_budget_never_executes(fixture, budget):
    f = fixture
    with pytest.raises(ValueError, match="positive remaining"):
        f.issuer.create_source_execution_owner(
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "must-not-execute",
            remaining_s=budget,
        )
    assert not f.calls


def test_existing_output_directory_and_linked_current_artifacts_refuse(fixture):
    f = fixture
    with pytest.raises(ValueError, match="fresh ordinary"):
        f.issuer.create_source_execution_owner(
            native_result=f.result,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root,
            remaining_s=20,
        )
    link = f.root / "alias.json"
    link.symlink_to(f.result)
    with pytest.raises(ValueError, match="regular absolute leaf"):
        f.issuer.create_source_execution_owner(
            native_result=link,
            command_buffer=f.cb,
            lowered=f.lowered,
            work_root=f.root / "must-not-execute",
            remaining_s=20,
        )
    assert not f.calls
