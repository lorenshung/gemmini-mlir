"""Stream model, reroll identity and hazard lint (no device tools required)."""

import unittest

import numpy as np

from mlir_oot.core_aten.emit import build_module, encode
from mlir_oot.core_aten.lint import HazardError, lint
from mlir_oot.core_aten.recipes.gemm import blocked_matmul
from mlir_oot.core_aten.reroll import iter_expand, node_count, reroll
from mlir_oot.core_aten.stream import ACC, Arg, SP, Stream, simulate
from mlir_oot.core_aten.trace import stream_from_module
from mlir_oot.golden_conv import ConvShape, GoldenConv


def conv_ref(cs, X, W):
    Xp = np.zeros((cs.h + 2, cs.w + 2, cs.cin), np.int64)
    Xp[1:-1, 1:-1] = X
    out = np.zeros((cs.oh, cs.ow, cs.cout), np.int64)
    for kh in range(3):
        for kw in range(3):
            patch = Xp[kh:kh + cs.stride * (cs.oh - 1) + 1:cs.stride, kw:kw + cs.stride * (cs.ow - 1) + 1:cs.stride]
            out += patch @ W[(kh * 3 + kw) * cs.cin:(kh * 3 + kw + 1) * cs.cin].astype(np.int64)
    return out


class TestLint(unittest.TestCase):
    def _stream(self):
        return Stream([Arg("x", "i8", (16, 16), "input"), Arg("y", "i8", (16, 16), "output")])

    def test_overwrite_then_overlapping_mvin_needs_fence(self):
        s = self._stream()
        s.config_ld(0, 16)
        s.mvin(None, SP(0), 16, 16, 0)
        s.mvin(s.at("x"), SP(4), 16, 4, 0)
        with self.assertRaisesRegex(HazardError, "HW3"):
            lint(s)
        s = self._stream()
        s.config_ld(0, 16)
        s.mvin(None, SP(0), 16, 16, 0)
        s.fence()
        s.mvin(s.at("x"), SP(4), 16, 4, 0)
        lint(s)

    def test_zero_fill_then_accumulate_requires_fence(self):
        s = self._stream()
        s.config_ld(0, 16, shrink=False)
        s.mvin(None, ACC(0), 16, 16, 0)
        s.config_ld(1, 16, shrink=True)
        s.mvin(s.at("x"), ACC(0, accumulate=True), 16, 16, 1)
        with self.assertRaisesRegex(HazardError, "HW3"):
            lint(s)
        s.ins.insert(2, type(s.ins[0])('fence', (), None, 0))
        lint(s)

    def test_hw2_shrunk_accumulator_zero_fill(self):
        s = self._stream()
        s.config_ld(0, 16, shrink=True)
        s.mvin(None, ACC(0), 16, 16, 0)
        with self.assertRaisesRegex(HazardError, "HW2"):
            lint(s)

    def test_dram_raw_and_waw(self):
        s = self._stream()
        s.config_ld(0, 16)
        s.config_st(16)
        s.mvin(s.at("x"), SP(0), 16, 16, 0)
        s.mvout(s.at("y"), SP(0), 16, 16)
        s.mvout(s.at("y", 16), SP(0), 16, 8)
        with self.assertRaisesRegex(HazardError, "WAW"):
            lint(s)

    def test_hw1_execute_from_accumulator(self):
        s = self._stream()
        s.preload(ACC(0), ACC(16), 16, 16, 16, 16)
        with self.assertRaisesRegex(HazardError, "HW1"):
            lint(s)


class TestReroll(unittest.TestCase):
    def test_blocked_matmul_rerolls_exactly_and_matches_numpy(self):
        M, K, N = 70, 200, 90
        s = Stream([Arg("A", "i8", (M, K), "input"), Arg("B", "i8", (K, N), "input"),
                    Arg("bias", "i32", (N,), "input"), Arg("C", "i8", (M, N), "output")])
        blocked_matmul(s, ("A", 0), K, ("B", 0), N, ("C", 0), N, M, K, N, bias=("bias", 0),
                       scale=1 / 1024, relu=True)
        lint(s)
        recs = [encode(i) for i in s.ins]
        tree = reroll(recs)
        self.assertEqual(list(iter_expand(tree)), recs)
        self.assertLess(node_count(tree), len(recs) // 4)
        rng = np.random.default_rng(3)
        A = rng.integers(-128, 128, (M, K)).astype(np.int8)
        B = rng.integers(-127, 128, (K, N)).astype(np.int8)
        b = rng.integers(-30000, 30000, N).astype(np.int32)
        acc = A.astype(np.int64) @ B.astype(np.int64) + b
        ref = np.clip(np.rint(np.maximum(acc, 0).astype(np.float32) * np.float32(1 / 1024)), -128, 127)
        self.assertTrue((simulate(s, {"A": A, "B": B, "bias": b})["C"] == ref).all())
        build_module(s, "k")


class TestGoldenConvPadding(unittest.TestCase):
    """The padded 3x3 kernel zero-fills only out-of-bounds lanes (no HW3 pair) and stays exact."""

    def test_lint_and_model(self):
        for cs in (ConvShape(5, 19, 20, 19, 1), ConvShape(6, 20, 32, 16, 2), ConvShape(1, 1, 20, 19)):
            args = [Arg("X", "i8", (cs.h, cs.w, cs.cin), "input"), Arg("W", "i8", (9 * cs.cin, cs.cout), "input"),
                    Arg("Y", "i32", (cs.oh, cs.ow, cs.cout), "output")]
            s = stream_from_module(GoldenConv(cs).build(), args)
            lint(s)
            rng = np.random.default_rng(1)
            X = rng.integers(-128, 128, (cs.h, cs.w, cs.cin)).astype(np.int8)
            W = rng.integers(-127, 128, (9 * cs.cin, cs.cout)).astype(np.int8)
            self.assertTrue((simulate(s, {"X": X, "W": W})["Y"] == conv_ref(cs, X, W)).all())


if __name__ == "__main__":
    unittest.main()


def test_repeated_accumulates_preserve_overwrite_hazard():
    import pytest
    s=Stream([Arg('x','i8',(16,16),'input')])
    s.config_ld(2,0,shrink=False)
    s.mvin(None,ACC(0),16,16,2)
    s.fence()
    s.config_ld(0,16,shrink=True)
    for _ in range(1000):
        s.mvin(s.at('x'),ACC(0,accumulate=True),16,16)
    lint(s)
    s.mvin(s.at('x'),ACC(0),16,16)
    with pytest.raises(HazardError,match='HW3'):
        lint(s)
    s.ins.pop();s.fence()
    s.mvin(s.at('x'),ACC(0),16,16)
    lint(s)
