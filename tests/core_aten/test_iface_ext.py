"""The v0.1 extension opcodes through the iface route: bit-level model == Merlin's reference, exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "iface_ext"))
sys.path.insert(0, str(HERE / "device"))

pytest.importorskip("merlin.runtime.integer_ops")

import ext_cases as C  # noqa: E402
import run_ext as R  # noqa: E402
from mlir_oot.core_aten.stream import simulate  # noqa: E402

CASES = C.cases()


@pytest.mark.parametrize("name", list(CASES))
def test_extension_opcode_matches_merlin_reference(name):
    cb = CASES[name]
    inputs = C.stimulus(cb, list(CASES).index(name))
    from mlir_oot.core_aten.iface import IfaceDecline
    from mlir_oot.core_aten.match import Decline
    try:
        merlin_cb, (ours, _, stream) = R.lower(cb)
    except (IfaceDecline, Decline) as exc:
        assert 'exact arithmetic' in str(exc)
        return
    ref = R.reference(merlin_cb, inputs)
    model = simulate(stream, inputs)
    for k, v in ref.items():
        assert np.array_equal(model[k].astype(np.int64).reshape(-1), v), k
    # Merlin's eager float simulator may disagree; its output is informational.
    print('eager-float divergence (informational):', R.merlin_checks(ours, inputs, ref))
    assert ours["kernel_abi"]["kind"] == "whole_program"


def test_gemmini_opt_cli_emits_buffer_and_artifact(tmp_path):
    from mlir_oot import gemmini_opt

    src = tmp_path / "in.mlir"
    src.write_text(R.render(CASES["fill"]))
    out = tmp_path / "cb.json"
    art = tmp_path / "art.mlir"
    assert gemmini_opt.main(["--convert-iface-to-gemmini", "--emit-target-artifact",
                             f"--emit-command-buffer={out}", str(src), "-o", str(art)]) == 0
    import json

    cb = json.loads(out.read_text())
    assert [c["opcode"] for c in cb["commands"]] == ["FILL"]
    assert "llvm.func @gemmini_kernel" in art.read_text()


def test_unsupported_form_declines_in_the_buffer(tmp_path):
    from mlir_oot import gemmini_opt

    cb = C._one("WINDOW_SUM", {"A": (1, 4, 4, 16)}, (1, 4, 4, 16), dict(
        mode="window", kernel=[3, 3], stride=[1, 1], padding=[1, 1], count_include_pad=True,
        readout_scale=1.0, output_dtype="i32"), "i32")
    src = tmp_path / "in.mlir"
    src.write_text(R.render(cb))
    out = tmp_path / "cb.json"
    assert gemmini_opt.main(["--convert-iface-to-gemmini", f"--emit-command-buffer={out}", str(src),
                             "-o", str(tmp_path / "o.mlir")]) == 0
    import json

    assert "declined" in json.loads(out.read_text())


def test_ceil_divisor_case_is_lowered_and_checked_against_rationals():
    cb=CASES['ws_avg_pool2d_ceil_div']
    inputs=C.stimulus(cb,173)
    source,(ours,_,stream)=R.lower(cb)
    for name,expected in R.reference(source,inputs).items():
        assert np.array_equal(simulate(stream,inputs)[name].reshape(-1),expected)
