"""Actual harmless host invocation custody; no target qualification fixture."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from merlin.common import invocation_record
from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.compiler_emission_binding import revalidate_compiler_emission
from merlin.targetgen.compiler_pass_selection import _pin, _tree


@pytest.fixture
def observed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    helper = importlib.import_module("gemmini_staged_compile_custody")
    package = tmp_path / "package"
    package.mkdir()
    script = package / "compiler.py"
    script.write_text(
        "from pathlib import Path\nimport sys\np=next(a.split('=',1)[1] for a in sys.argv if a.startswith('--emit-command-buffer='))\nPath(p).write_text('{}')\nprint('fixture lowered bytes')\n"
    )
    for name in (
        "direct_codegen.py",
        "shared-library-sources.json",
        "_selected_shared_sources.py",
        "gemmini_staged_tile_hook.py",
        "gemmini_staged_tile_source.py",
        "staged_dense_policy.py",
    ):
        (package / name).write_text("fixture")
    (package / "selected-shared-library.json").write_text('"frozen-release"')
    source = tmp_path / "source.mlir"
    source.write_text("independent source fixture")
    cb = tmp_path / "command_buffer.json"
    lowered = tmp_path / "lowered.mlir"
    argv = [
        sys.executable,
        str(script),
        "--convert-iface-to-gemmini",
        "--emit-command-buffer=" + str(cb),
        "--emit-target-artifact",
        str(source),
    ]
    with shared_compute_pool(timeout_s=120):
        result = invocation_record.run(
            argv,
            directory=tmp_path / "invocation",
            stage="harmless_compiler_custody_fixture",
            inputs=(source,),
            outputs=(cb,),
            dependencies=tuple(package.iterdir()),
            cwd=package,
            capture_output=True,
            text=True,
            timeout=10,
        )
    lowered.write_text(result.stdout)
    native = tmp_path / "native.json"
    native.write_text(json.dumps({"source": {"interface": _pin(source)}}))
    arm = {
        "compiler_package": {"path": str(package), "sha256": _tree(package)},
        "ordinary_invocation": _pin(
            next((tmp_path / "invocation").rglob("invocation.json"))
        ),
        "command_buffer_artifact": _pin(cb),
        "lowered": _pin(lowered),
        "native_result": _pin(native),
    }

    class Owner:
        def check_compiler_binding(self, path):
            if (
                json.loads((path / "selected-shared-library.json").read_bytes())
                != "frozen-release"
            ):
                raise ValueError("foreign release")

        def revalidate(self):
            pass

    return helper, Owner(), arm


def test_initial_source_binding_reopens_actual_compiler_invocation_and_exact_output_bytes(
    observed,
):
    helper, owner, arm = observed
    assert (
        helper.check_original_emission(owner, arm)["status"]
        == "checked_original_selected_compiler_emission"
    )


def test_issued_emission_revalidation_reopens_bytes_without_claiming_new_compilation(
    observed, monkeypatch
):
    helper, owner, arm = observed
    record = helper.check_original_emission(owner, arm)

    def no_new_execution(*args, **kwargs):
        raise AssertionError(
            "byte custody must not be reported as another compiler execution"
        )

    monkeypatch.setattr(invocation_record, "run", no_new_execution)
    revalidate_compiler_emission(record)
    package = Path(record["compiler_package"]["path"])
    (package / "new-unobserved-source.py").write_text("changed source roster")
    with pytest.raises(ValueError, match="source tree changed"):
        revalidate_compiler_emission(record)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_invocation",
        "missing_tree",
        "stale_tree",
        "foreign_release",
        "lowered",
        "source",
        "cb",
        "forged_package",
    ],
)
def test_missing_stale_forged_or_wrong_release_emissions_refuse(observed, damage):
    helper, owner, arm = observed
    if damage == "missing_invocation":
        arm.pop("ordinary_invocation")
    elif damage == "missing_tree":
        arm.pop("compiler_package")
    elif damage == "stale_tree":
        (Path(arm["compiler_package"]["path"]) / "extra.py").write_text("late addition")
    elif damage == "foreign_release":
        p = Path(arm["compiler_package"]["path"])
        (p / "selected-shared-library.json").write_text('"different-release"')
        arm["compiler_package"]["sha256"] = _tree(p)
    elif damage == "lowered":
        p = Path(arm["lowered"]["path"])
        p.write_text("self-consistent forged lowered bytes")
        arm["lowered"] = _pin(p)
    elif damage == "source":
        p = Path(arm["native_result"]["path"])
        x = json.loads(p.read_bytes())
        x["source"]["interface"]["path"] = str(p)
        p.write_text(json.dumps(x))
        arm["native_result"] = _pin(p)
    elif damage == "cb":
        p = Path(arm["command_buffer_artifact"]["path"])
        p.write_text("changed")
        arm["command_buffer_artifact"] = _pin(p)
    else:
        arm["compiler_package"]["path"] = str(
            Path(arm["compiler_package"]["path"]).parent
        )
    with pytest.raises(ValueError):
        helper.check_original_emission(owner, arm)
