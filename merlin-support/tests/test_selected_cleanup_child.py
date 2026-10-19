"""Opt-in actual owner replay through the original sealed child transport.

The parent profile is a neutral fixture: this checks transport, not formal
accelerator admission. The selected cleanup factory still reruns its original
real linked/source/output checks in the child. No author or simulator is run.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from merlin.common import cpu_pool, invocation_record
from merlin.common.paths import repo_root
from merlin.targetgen.compiler_library import scoped_compiler_library
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    scoped_compiler_passes,
)
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD

sys.path.insert(0, str(repo_root() / "merlin/tests/targetgen"))
from test_compiler_profile_child import _CONTROL, _scope
from test_compiler_profile_child import original as original  # noqa: PLC0414


@pytest.mark.skipif(
    not os.environ.get("MERLIN_SELECTED_CLEANUP_TEST_INPUTS"),
    reason="original host cleanup inputs not explicitly selected",
)
def test_actual_cleanup_factory_replayed_under_original_sealed_parent(
    original, tmp_path, monkeypatch
):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    from gemmini_compiler_passes import capability_binding, restore_registry

    inputs = json.loads(
        Path(os.environ["MERLIN_SELECTED_CLEANUP_TEST_INPUTS"]).read_bytes()
    )
    factory = OriginalPassFactory(restore_registry, inputs)
    # No parent compute lease: the original independent checker child owns it.
    registry = factory.create(work_root=tmp_path / "parent-admission")
    owner = registry._capabilities[0].owner
    selection = CompilerPassSelection(
        registry,
        capability_binding(owner),
        mechanisms=("configuration_cleanup",),
        factory=factory,
    )
    script = _CONTROL.replace(
        "    job=json.loads(raw)",
        """    job=json.loads(raw)
    from merlin.targetgen.compiler_pass_selection import active_compiler_passes
    selected=active_compiler_passes()
    capability=selected.resolve('configuration_cleanup')
    assert capability is not None
    assert callable(capability.owner.check_functional_guard)
    assert receipt['schema']==CHILD.PASSES_SCHEMA
    assert receipt['compiler_passes_sha256']==CHILD._digest(selected.record())
    assert (spec.parent/'compiler/pass-admission/independently-rechecked-receipt.json').is_file()
    assert capability.owner.record()['hardware_execution_qualified'] is False
    assert capability.owner.record()['timing_authority']=='NONE'""",
    )
    with (
        scoped_compiler_library(owner.library),
        scoped_compiler_passes(selection),
        _scope(original),
    ):
        environment = cpu_pool.child_environment(os.environ)
        with CHILD.prepare_model_child(
            spec_path=original.spec,
            environment=environment,
            package_dir=original.candidate,
        ) as transport:
            child = invocation_record.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(transport.fd),
                    str(original.spec),
                ],
                directory=tmp_path / "child-observation",
                stage="actual_cleanup_sealed_child_transport",
                pass_fds=transport.pass_fds,
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
            assert child.returncode == 0, child.stderr
            output = json.loads(child.stdout)
            transport.verify_result(output)
            assert output["status"] == "source-scope-control-only"
