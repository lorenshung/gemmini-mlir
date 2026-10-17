"""The 17-op paper subset: recipe numerics, structural matching of model2MLIR graphs, and the
end-to-end chain (model level; device runs are in tests/core_aten/device and e2e)."""

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
EXPECTED_OPS = {
    "mm": "aten.mm.default", "addmm": "aten.addmm.default", "bmm": "aten.bmm.default",
    "convolution": "aten.convolution.default", "convolution_patch16": "aten.convolution.default",
    "sub_Tensor": "aten.sub.Tensor", "neg": "aten.neg.default", "add_Tensor": "aten.add.Tensor",
    "mul_Scalar": "aten.mul.Scalar", "relu": "aten.relu.default", "mean_dim": "aten.mean.dim",
    "clone": "aten.clone.default", "permute": "aten.permute.default", "slice_scatter": "aten.slice_scatter.default",
    "constant_pad_nd": "aten.clone.default", "constant_pad_nd_p1": "aten.constant_pad_nd.default",
    "full_like": "aten.full.default", "cat": "aten.cat.default", "_to_copy": "aten._to_copy.default",
}


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
    def test_every_fixture_binds_and_matches_eager(self):
        dirs = sorted(p for p in FIXTURES.iterdir() if (p / "model.mlir").exists())
        self.assertEqual({d.name for d in dirs}, set(EXPECTED_OPS))
        for d in dirs:
            with self.subTest(d.name):
                device, manifest, matches = build((d / "model.mlir").read_text())
                self.assertEqual([b["op"] for b in manifest["bindings"]], [EXPECTED_OPS[d.name]])
                self.assertFalse(manifest["declines"])
                inputs = dict(np.load(d / "inputs.npz"))
                m, b = matches[0], manifest["bindings"][0]
                feed = {name: inputs[f"arg{src['func_arg']}"] for name, src in b["operands"].items()}
                lint(m.kernel.stream)
                out = simulate(m.kernel.stream, feed)[m.result[0]]
                ok, mx, ex = gate3(out, np.load(d / "expected.npy"))
                self.assertTrue(ok, f"{d.name}: max {mx} exact {ex}")


if __name__ == "__main__":
    unittest.main()
