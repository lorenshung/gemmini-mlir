"""Replay the retained bounded diagnostic, with a fresh explicit output root.

This reuses its original operator and limits. It does not start an author,
change the selected SDK, issue a capability from saved JSON or launch gSIM.
The original managed task and exact retained inputs must remain available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

TASK = Path("/work/merlin/tasks/phase2-pass-integration-20261018")
ORIGINAL = (
    TASK
    / "out/artifacts/target-adapters/gemmini/combined-live-rhs-retained-primitive-v4"
)
PINS = {
    "run_combined_rhs_trial_v4.py": "515eb275e2016c339666b9126878ca5d1ee8ac4a9c89a9120d6701d0271c1ffa",
    "prepare_retained_rhs_policy_v2.py": "9720ac98fd2cfde866cd8751d3ccf6f4dd9943ac344983b05e709bb86a138e4a",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_root
    if (
        not output.is_absolute()
        or output.resolve() != output
        or output.exists()
        or output.is_symlink()
        or not output.is_relative_to(TASK / "out")
        or not output.parent.is_dir()
    ):
        parser.error(
            "output must be a fresh canonical path below the original task's out/, with an existing parent"
        )
    payloads = {}
    for name, expected in PINS.items():
        path = ORIGINAL / "operator-sources" / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 << 10:
            parser.error("original operator snapshot is absent, linked or oversized")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            parser.error("original operator snapshot changed: " + name)
        payloads[name] = payload
    runner = TASK / ".tmp" / ("rhs-reproduction-" + uuid.uuid4().hex)
    runner.mkdir(mode=0o700)
    helper = runner / "prepare_retained_rhs_policy_v2.py"
    helper.write_bytes(payloads[helper.name])
    source = payloads["run_combined_rhs_trial_v4.py"].decode()
    replacements = {
        "OUT=A/'combined-live-rhs-retained-primitive-v4'": "OUT=Path("
        + repr(str(output))
        + ")",
        "B/'.tmp/prepare_retained_rhs_policy_v2.py'": "Path(" + repr(str(helper)) + ")",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(
                "original operator does not have its exact mechanical output/helper seam"
            )
        source = source.replace(old, new)
    script = runner / "run_combined_rhs_trial_v4.py"
    script.write_text(source)
    # The original operator owns its process-tree budgets and all leaf loans.
    result = subprocess.run(
        ["bash", str(TASK / ".tmp/run_hybrid_python.sh"), str(script)],
        cwd=TASK,
        check=False,
    )
    if output.is_dir():
        record = {
            "schema": "phase2.original_retained_operator_reproduction.v1",
            "original": str(ORIGINAL),
            "original_operator_sources": PINS,
            "mechanical_changes": [
                "fresh output path",
                "byte-identical adjacent retained helper",
            ],
            "generated_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "returncode": result.returncode,
            "authority": "replay evidence only; inspect original status/refusals; no performance promotion",
        }
        (output / "reproduction.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
