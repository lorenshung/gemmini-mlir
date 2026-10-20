"""Isolated host emission from one explicit SDK; no target execution."""

import json
import runpy
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_bytes())
sdk = Path(config["sdk"])
assert sys.dont_write_bytecode
assert not any(name == "merlin" or name.startswith("merlin.") for name in sys.modules)
sys.path[:0] = [str(sdk), *config["dependencies"]]
for name in (
    "merlin.llvmlower.direct_source_contraction",
    "merlin.llvmlower.late_quant_rne",
    "merlin_experiments.phase0.connected_integer_bounds",
):
    try:
        __import__(name)
    except ImportError:
        pass
    else:
        raise AssertionError("private import became available: " + name)

try:
    emit = runpy.run_path(config["adapter"])["emit_v26_staged_dense"]
except ImportError:
    if config["selected"]:
        raise
    result = {"unselected_import": "refused"}
else:
    assert config["selected"]
    from merlin.compile.scheduling.staged_dense import StagedDenseContract

    options = config["options"]
    options["contract"] = StagedDenseContract(**options["contract"])
    baseline, before = emit(**options, rewrite_policy="verify_only")
    candidate, after = emit(**options, rewrite_policy="n_before_m_rhs")
    assert baseline != candidate
    assert before["physical_rhs_calls_removed"] == 0
    assert after["physical_rhs_calls_removed"] == 2
    assert before["physical_fence_count"] == after["physical_fence_count"] == 15
    assert all(
        row["products"] == 16384 and row["outputs"] == 1024
        for report in (before, after)
        for row in report["proposal"]["checks"]
    )
    for name, module in tuple(sys.modules.items()):
        if name == "merlin" or name.startswith("merlin."):
            assert Path(module.__file__).resolve().is_relative_to(sdk)
    result = {
        "host_emission": "PASS",
        "rhs_calls_removed": 2,
        "products": 16384,
        "outputs": 1024,
        "target_executions": 0,
        "qualification": False,
        "loaded_merlin_modules": sorted(
            name for name in sys.modules if name == "merlin" or name.startswith("merlin.")
        ),
    }
Path(config["result"]).write_text(json.dumps(result, sort_keys=True) + "\n")
