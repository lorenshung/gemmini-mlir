"""Portable copied-compiler cleanup hook; projected public leaves supply no grant."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from gemmini_quiescent_load_config import (
    LoadConfigurationContract,
    _sha,
    optimize_statements,
)


def rewrite(statements, *, shape):
    from cleanup_policy import choose_cleanup
    from merlin.compile.scheduling.library_sources import shared_library_sources

    root = Path(__file__).resolve().parent
    if (
        json.loads((root / "shared-library-sources.json").read_bytes())
        != shared_library_sources()
    ):
        raise ValueError("copied compiler loaded a different reviewed shared release")
    projected = json.loads((root / "cleanup-public-inputs.json").read_bytes())
    if (
        set(projected) != {"schema", "origin_contract_sha256", "contract"}
        or projected["schema"] != "gemmini.cleanup_public_inputs.v1"
    ):
        raise ValueError("unknown public cleanup input projection")
    row = projected["contract"]
    size = 0

    def original(pin):
        nonlocal size
        if set(pin) != {"path", "sha256"}:
            raise ValueError("incomplete public cleanup leaf")
        relative = Path(pin["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or path.resolve() != path
        ):
            raise ValueError("public cleanup leaf escaped its copied package")
        payload = path.read_bytes()
        size += len(payload)
        if size > 1 << 20 or _sha(payload) != pin["sha256"]:
            raise ValueError("public cleanup leaf changed or exceeds its bound")
        return {"path": str(path), "sha256": pin["sha256"]}

    if not 4 <= len(row["sources"]) + len(row["selected_inputs"]) + 1 <= 64:
        raise ValueError("public cleanup leaf roster exceeds its bound")
    actual = {
        **row,
        "runtime": original(row["runtime"]),
        "selected_inputs": {
            name: original(pin) for name, pin in row["selected_inputs"].items()
        },
        "sources": [original(pin) for pin in row["sources"]],
    }
    enabled = choose_cleanup(tuple(shape))
    if type(enabled) is not bool:
        raise ValueError("cleanup policy must return a Boolean decision")
    # The unchanged original adapter requires an on-disk canonical contract.
    # Rebase only reviewed exact leaves into this current package invocation;
    # no original host path or source directory is exposed to the compiler.
    with tempfile.TemporaryDirectory(
        dir=os.environ["TMPDIR"], prefix="cleanup-inputs-"
    ) as scratch:
        path = Path(scratch) / "contract.json"
        path.write_text(json.dumps(actual, sort_keys=True))
        result, report = optimize_statements(
            statements, LoadConfigurationContract.read(path), enabled=enabled
        )
    print(
        "LOAD_CONFIGURATION " + json.dumps(report, sort_keys=True),
        file=__import__("sys").stderr,
    )
    return result
