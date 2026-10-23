import hashlib
import json
import os
from pathlib import Path
import subprocess
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_model2mlir_schema_drift():
    vendor = ROOT / "merlin-support/quant_schema.json"
    lock = ROOT / "merlin-support/quant_schema.lock.json"
    source = Path(os.environ.get("MODEL2MLIR_DIR", ROOT.parent / "model2mlir"))
    if not lock.exists():
        warnings.warn(
            "QUANT SCHEMA DRIFT NOT CHECKED: model2MLIR schema export is not committed/pinned yet",
            RuntimeWarning,
        )
        pytest.skip("QUANT SCHEMA DRIFT NOT CHECKED: waiting for schema owner commit")
    receipt = json.loads(lock.read_text())
    try:
        data = subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "show",
                receipt["revision"] + ":" + receipt["path"],
            ],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        warnings.warn(
            "QUANT SCHEMA DRIFT NOT CHECKED: pinned model2MLIR schema is unavailable",
            RuntimeWarning,
        )
        pytest.skip("QUANT SCHEMA DRIFT NOT CHECKED: pinned revision unavailable")
    assert data == vendor.read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
    exported = source / receipt["path"]
    if exported.exists():
        assert exported.read_bytes() == data, (
            "model2MLIR checkout schema changed; refresh the pin"
        )


def test_unknown_qdq_keys_are_rejected_without_rational_metadata():
    from fixture_source import read
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.frontend.parse import parse_module
    from xdsl.dialects.builtin import StringAttr

    source = read(ROOT / "tests/fixtures/core_aten/mean_dim/model.mlir")
    module = parse_module(source)
    for op in module.walk():
        nm = getattr(getattr(op, "op_name", None), "data", op.name)
        if nm.startswith("quant_ext."):
            op.attributes["prov.quant.typo"] = StringAttr("bad")
    _, manifest, matches = build(str(module))
    assert not matches
    assert any("unknown prov.quant" in row["reason"] for row in manifest["declines"])
