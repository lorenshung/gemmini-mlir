"""Prepare one static candidate from the immutable complete V26 compiler package.

This experiment changes target store-configuration placement only. It issues no
compiler, execution, qualification, timing or edit authority.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BASELINE_SHA256 = "29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24"
FILES = (
    "REPORT.md", "backend.py", "builder.py", "compiler.py", "contraction.py",
    "conv.py", "docs/iteration_notes.md", "elementwise.py", "frontend.py",
    "linalg_frontend.py", "manifest.yaml", "mixed_codegen.py", "mixed_frontend.py",
    "mixed_scalar.py", "pooling.py", "task_tags.py",
)
SETUP = "    _mvin_config(b, 2, 0)\n"
HOISTED = (
    SETUP
    + "    # Raw readout state is invariant across every tile in this contraction.\n"
    + "    # Other config families do not modify store state; retain all fences.\n"
    + "    if dtype == 'i32':\n"
    + "        _mvout_config(b, n * 4, 1.0, False)\n"
)
PER_TILE = "            if dtype == 'i32':\n                _mvout_config(b, n * 4, 1.0, False)\n"
WITHOUT_REPEAT = "            if dtype == 'i32':\n"


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("baseline must be a real complete compiler directory")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("compiler inventory contains a link or special entry")
        if path.is_file():
            files.append(path)
    if tuple(path.relative_to(root).as_posix() for path in files) != FILES:
        raise ValueError("compiler inventory must retain exactly the original 16 files")
    digest = hashlib.sha256()
    rows = []
    for path in files:
        data = path.read_bytes()
        name = path.relative_to(root).as_posix()
        digest.update(name.encode() + b"\0" + data + b"\0")
        rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return digest.hexdigest(), rows


def rewrite(source):
    if source.count(SETUP) != 1 or source.count(PER_TILE) != 1:
        raise ValueError("the original contraction configuration sites changed")
    updated = source.replace(SETUP, HOISTED).replace(PER_TILE, WITHOUT_REPEAT)
    ast.parse(updated)
    return updated


def prepare(baseline, destination):
    baseline, destination = Path(baseline), Path(destination)
    original_sha, original_rows = inventory(baseline)
    if original_sha != BASELINE_SHA256:
        raise ValueError("complete original V26 compiler bytes changed")
    if destination.exists() or destination.is_symlink():
        raise ValueError("candidate destination must be absent")
    destination.mkdir(parents=True)
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((baseline / name).read_bytes())
    source = (destination / "contraction.py").read_text()
    (destination / "contraction.py").write_text(rewrite(source))
    for name in FILES:
        if name.endswith(".py"):
            ast.parse((destination / name).read_text(), filename=name)
    candidate_sha, candidate_rows = inventory(destination)
    if inventory(baseline) != (original_sha, original_rows):
        raise ValueError("original compiler changed during candidate preparation")
    changed = [a["path"] for a, b in zip(original_rows, candidate_rows, strict=True) if a != b]
    if changed != ["contraction.py"]:
        raise ValueError("candidate changed files outside the selected configuration placement")
    return {
        "schema": "gemmini.store_configuration_hoist_candidate.v1",
        "baseline_sha256": original_sha,
        "candidate_sha256": candidate_sha,
        "baseline_files": original_rows,
        "candidate_files": candidate_rows,
        "changed_files": changed,
        "compiler_execution": "NOT_RUN",
        "source_effect_qualification": "NOT_ISSUED",
        "timing": "UNKNOWN",
        "retained_rhs_85_dependency": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.baseline, args.destination)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
