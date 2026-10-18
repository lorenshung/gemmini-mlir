"""Exact arithmetic and explicit device refusals on real model2MLIR captures."""

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

FIXTURES = HERE.parent / "fixtures" / "core_aten"            # model2MLIR 9c8b69f structural forms
LEGACY = HERE.parent / "fixtures" / "core_aten_legacy"      # earlier gather-loop and LHS-weight forms
class TestRecipes(unittest.TestCase):
    def test_admitted_recipes_equal_exact_arithmetic(self):
        for case in cases.paper_cases(representative=False):
            with self.subTest(case.name):
                model = simulate(case.kernel.stream, case.inputs)
                for name, expected in case.reference.items():
                    self.assertTrue(np.array_equal(model[name], expected), case.name)


class TestCatalogOnCaptures(unittest.TestCase):
    """Admitted regions match rationals; refused regions survive the rewrite."""

    def test_every_fixture_is_exact_or_declined(self):
        from exact_oracle import evaluate
        from mlir_oot.core_aten.offload import rewrite, shim_source
        for root in (FIXTURES, LEGACY):
            covered, refused = set(), set()
            for d in sorted(p for p in root.iterdir() if (p / "model.mlir").exists()):
                with self.subTest(f"{root.name}/{d.name}"):
                    from fixture_source import read
                    text = read(d / "model.mlir")
                    _, manifest, matches = build(text)
                    values = {("arg", int(k[3:])): v for k, v in np.load(d / "inputs.npz").items()}
                    key = lambda src: ("arg", src["func_arg"]) if "func_arg" in src else ("op", src["op_ordinal"], src["result"])
                    checked = 0
                    for b, m in zip(manifest["bindings"], matches):
                        # Partial catalogs may consume host results. Their instruction
                        # domains are separately qualified by exhaustive property tests.
                        if not all(key(src) in values for src in b['operands'].values()):
                            continue
                        feed = {name: values[key(src)] for name, src in b['operands'].items()}
                        lint(m.kernel.stream)
                        model = simulate(m.kernel.stream, feed)
                        exact = evaluate(m, feed)
                        (oname, osrc), = b['result'].items()
                        self.assertTrue(np.array_equal(model[oname], exact[oname]), d.name)
                        values[key(osrc)] = exact[oname]
                        checked += 1
                    out = rewrite(text, manifest)
                    self.assertEqual(out.count("func.call @ca_gemmini_ca_"), len(matches))
                    if not matches:
                        self.assertTrue(manifest['declines'], d.name)
                    if 'quant_ext.' in out:
                        refused.add(d.name)
                    else:
                        self.assertGreater(checked, 0)
                        covered.add(d.name)
                    shim = shim_source(manifest)
                    for kernel in manifest['kernels']:
                        self.assertIn('_mlir_ciface_ca_' + kernel['symbol'], shim)
            self.assertTrue(covered)
            if root == FIXTURES:
                self.assertFalse(refused, "all current supported captures must remain on the device")
            print(root.name, 'exactly covered:', sorted(covered), 'device refusals:', sorted(refused))


if __name__ == "__main__":
    unittest.main()
