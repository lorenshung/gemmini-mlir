"""The 39 Core ATen ops: recipe numerics, structural matching of model2MLIR captures and the
chained bit-level models against eager PyTorch (device runs: tests/core_aten/device and e2e)."""

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402
from mlir_oot.core_aten.catalog import build  # noqa: E402
from mlir_oot.core_aten.lint import lint  # noqa: E402
from mlir_oot.core_aten.stream import simulate  # noqa: E402

FIXTURES = HERE.parent / "fixtures" / "core_aten"
SUPPORT_39 = {
    "aten._adaptive_avg_pool2d.default", "aten._adaptive_avg_pool2d_backward.default",
    "aten._adaptive_avg_pool3d.default", "aten._to_copy.default", "aten.adaptive_avg_pool1d.default",
    "aten.add.Scalar", "aten.add.Tensor", "aten.addmm.default", "aten.avg_pool1d.default", "aten.avg_pool2d.default",
    "aten.avg_pool2d_backward.default", "aten.avg_pool3d.default", "aten.bmm.default", "aten.cat.default",
    "aten.clone.default", "aten.col2im.default", "aten.constant_pad_nd.default", "aten.convolution.default",
    "aten.convolution_backward.default", "aten.div.Scalar", "aten.fill.Scalar", "aten.full.default",
    "aten.full_like.default", "aten.mean.dim", "aten.mm.default", "aten.mul.Scalar", "aten.neg.default",
    "aten.permute.default", "aten.reflection_pad1d.default", "aten.reflection_pad2d.default",
    "aten.reflection_pad3d.default", "aten.relu.default", "aten.repeat.default", "aten.replication_pad2d.default",
    "aten.replication_pad3d.default", "aten.select_scatter.default", "aten.slice_scatter.default", "aten.sub.Scalar",
    "aten.sub.Tensor",
}


def chained_model(case_dir):
    """Bind the capture, then run the bound kernels' bit-level models in binding order."""
    text = (case_dir / "model.mlir").read_text()
    device, manifest, matches = build(text)
    from xdsl.ir import OpResult
    from mlir_oot.frontend.parse import parse_module
    module = parse_module(text)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    fn = [op for op in module.walk() if op.name == "func.func"][0]
    args = list(fn.body.blocks[0].args)
    ret = [op for op in fn.walk() if op.name == "func.return"][0]
    vals = {("arg", int(k[3:])): v for k, v in np.load(case_dir / "inputs.npz").items()}
    key = (lambda src: ("arg", src["func_arg"]) if "func_arg" in src else ("op", src["op_ordinal"], src["result"]))
    for b, m in zip(manifest["bindings"], matches):
        lint(m.kernel.stream)
        feed = {name: vals[key(src)] for name, src in b["operands"].items()}
        out = simulate(m.kernel.stream, feed)
        (oname, osrc), = b["result"].items()
        vals[key(osrc)] = out[oname]
    outs = []
    for v in ret.operands:
        k = ("arg", args.index(v)) if not isinstance(v, OpResult) else ("op", ordinals[v.owner], 0)
        outs.append(vals[k])
    if (case_dir / "expected.npy").exists():
        exp = [np.load(case_dir / "expected.npy")]
    else:
        exp = [np.load(p) for p in sorted(case_dir.glob("expected[0-9]*.npy"))]
    return manifest, outs, exp


def gate3(got, ref, tol=1, min_exact=0.99):
    e = np.abs(np.asarray(got, np.int64).reshape(-1) - np.asarray(ref, np.int64).reshape(-1))
    return int(e.max()) <= tol and float((e == 0).mean()) >= min_exact, int(e.max()), float((e == 0).mean())


class TestRecipes(unittest.TestCase):
    def test_small_cases_meet_gate3(self):
        for c in cases.paper_cases(representative=False):
            with self.subTest(c.name):
                lint(c.kernel.stream)
                out = simulate(c.kernel.stream, c.inputs)
                for name, ref in c.reference.items():
                    ok, mx, ex = gate3(out[name], ref, c.tol, c.min_exact)
                    self.assertTrue(ok, f"{c.name}.{name}: max {mx} exact {ex}")


class TestCatalogOnCaptures(unittest.TestCase):
    """Every model2MLIR capture binds completely and its chained models meet gate 3."""

    def test_every_fixture_binds_and_matches_eager(self):
        dirs = sorted(p for p in FIXTURES.iterdir() if (p / "model.mlir").exists())
        covered = set()
        for d in dirs:
            with self.subTest(d.name):
                manifest, outs, exp = chained_model(d)
                self.assertFalse(manifest["declines"], manifest["declines"])
                self.assertTrue(manifest["bindings"])
                covered |= {b["aten"] for b in manifest["bindings"]}
                for got, ref in zip(outs, exp):
                    ok, mx, ex = gate3(got, ref)
                    self.assertTrue(ok, f"{d.name}: max {mx} exact {ex}")
        self.assertTrue(SUPPORT_39 <= covered, sorted(SUPPORT_39 - covered))


if __name__ == "__main__":
    unittest.main()
