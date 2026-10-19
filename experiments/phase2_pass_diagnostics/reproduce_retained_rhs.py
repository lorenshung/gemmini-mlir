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
ARTIFACTS = TASK / "out/artifacts/target-adapters/gemmini"
OPERATORS = {
    "combined": (
        "combined-live-rhs-retained-primitive-v4",
        "run_combined_rhs_trial_v4.py",
        "515eb275e2016c339666b9126878ca5d1ee8ac4a9c89a9120d6701d0271c1ffa",
    ),
    "boxed": (
        "boxed-retained-rhs-source-v5",
        "run_boxed_retained_rhs_source_v5.py",
        "fd51745e23c4b221b29aa358855c6065bf3674b7b0277e361fe7d7afdcacfdb5",
    ),
}
HELPER = "prepare_retained_rhs_policy_v2.py"
HELPER_SHA256 = "9720ac98fd2cfde866cd8751d3ccf6f4dd9943ac344983b05e709bb86a138e4a"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(OPERATORS), default="combined")
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
    directory, script_name, script_sha256 = OPERATORS[args.kind]
    original = ARTIFACTS / directory
    pins = {script_name: script_sha256, HELPER: HELPER_SHA256}
    for name, expected in pins.items():
        path = original / "operator-sources" / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 << 10:
            parser.error("original operator snapshot is absent, linked or oversized")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            parser.error("original operator snapshot changed: " + name)
        payloads[name] = payload
    if (
        args.kind == "boxed"
        and hashlib.sha256((TASK / ".tmp" / HELPER).read_bytes()).hexdigest()
        != HELPER_SHA256
    ):
        parser.error("original boxed supervisor's retained helper changed")
    runner = TASK / ".tmp" / ("rhs-reproduction-" + uuid.uuid4().hex)
    runner.mkdir(mode=0o700)
    helper = runner / HELPER
    helper.write_bytes(payloads[helper.name])
    source = payloads[script_name].decode()
    replacements = {
        "OUT=A/'" + directory + "'": "OUT=Path(" + repr(str(output)) + ")",
    }
    if args.kind == "combined":
        replacements["B/'.tmp/prepare_retained_rhs_policy_v2.py'"] = (
            "Path(" + repr(str(helper)) + ")"
        )
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(
                "original operator does not have its exact mechanical output/helper seam"
            )
        source = source.replace(old, new)
    script = runner / script_name
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
            "original": str(original),
            "original_operator_sources": pins,
            "mechanical_changes": [
                "fresh output path",
                "byte-identical retained helper",
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
