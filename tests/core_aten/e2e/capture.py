"""Capture quantized Core ATen op graphs with model2MLIR (run with model2MLIR's interpreter).

Each case is an eager module  int8 inputs -> dequantize -> aten op -> quantize -> int8 output
(per-tensor symmetric, zero point 0; a bias is int32 at scale s_x s_w as in PT2E).  For every
case this writes  <out>/<name>/model.mlir, inputs.npz (kernel-order int8/int32 inputs, named
by forward() argument index) and expected.npy (the eager module's own int8 output).

    MODEL2MLIR=/path/to/model2mlir  python capture.py OUT_DIR [--only name ...] [--set paper|all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("MODEL2MLIR", ""))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.ao.quantization.fx._decomposed  # noqa: E402,F401  (registers quantized_decomposed)
import m2m  # noqa: E402

A = torch.ops.aten
QD = torch.ops.quantized_decomposed


def sc(x):
    return max(float(x.abs().max()), 1e-8) / 127.5


def qz(x, s, dtype=torch.int8):
    lo, hi = (-128, 127) if dtype == torch.int8 else (-(2 ** 31), 2 ** 31 - 1)
    return torch.clamp(torch.round(x / s), lo, hi).to(dtype)


class Q(torch.nn.Module):
    """forward(*int inputs) = Q(fn(*DQ(inputs)), s_out)."""

    def __init__(self, fn, scales, dtypes, s_out):
        super().__init__()
        self.fn, self.scales, self.dtypes, self.s_out = fn, scales, dtypes, s_out

    def forward(self, *xs):
        ds = []
        for x, s, dt in zip(xs, self.scales, self.dtypes):
            if dt == "i32":
                ds.append(QD.dequantize_per_tensor(x, s, 0, -(2 ** 31), 2 ** 31 - 1, torch.int32))
            else:
                ds.append(QD.dequantize_per_tensor(x, s, 0, -128, 127, torch.int8))
        y = self.fn(*ds)
        return QD.quantize_per_tensor(y, self.s_out, 0, -128, 127, torch.int8)


def g(seed):
    return torch.Generator().manual_seed(seed)


def rnd(*shape, seed=0, scale=1.0, relu=False):
    x = torch.randn(*shape, generator=g(seed)) * scale
    return x.clamp(min=0) if relu else x


def case(name, fn, floats, kinds=None, s_out=None, bias_of=None):
    """floats: float example tensors; kinds: 'i8' or ('i32', scale) per input."""
    kinds = kinds or ["i8"] * len(floats)
    scales = [sc(x) if k == "i8" else None for x, k in zip(floats, kinds)]
    for i, k in enumerate(kinds):
        if k != "i8":
            scales[i] = k[1](scales)
    ints = [qz(x, s, torch.int8 if k == "i8" else torch.int32) for x, s, k in zip(floats, scales, kinds)]
    dts = ["i8" if k == "i8" else "i32" for k in kinds]
    deq = [i.to(torch.float32) * s for i, s in zip(ints, scales)]
    y = fn(*deq)
    so = s_out if s_out is not None else sc(y)
    return name, Q(fn, scales, dts, so), ints


class Plain(torch.nn.Module):
    """An integer-domain graph (no quantize/dequantize): forward(*ints) = fn(*ints)."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.scales, self.dtypes, self.s_out = fn, [], [], None

    def forward(self, *xs):
        return self.fn(*xs)


def plain(name, fn, ints):
    return name, Plain(fn), ints


def paper_set(small=False):
    Z = (lambda big, sm: sm if small else big)
    out = []
    # contractions
    out.append(case("mm", lambda a, b: A.mm.default(a, b), [rnd(*Z((128, 2048), (37, 200)), seed=1), rnd(*Z((2048, 2048), (200, 45)), seed=2, scale=0.05)]))
    out.append(case("addmm", lambda bias, a, b: A.addmm.default(bias, a, b),
                    [rnd(Z(768, 70), seed=3, scale=2.0), rnd(*Z((1024, 768), (50, 96)), seed=4), rnd(*Z((768, 768), (96, 70)), seed=5, scale=0.04)],
                    kinds=[("i32", lambda s: s[1] * s[2]), "i8", "i8"]))
    out.append(case("bmm", lambda a, b: A.bmm.default(a, b), [rnd(*Z((32, 128, 64), (3, 20, 40)), seed=6), rnd(*Z((32, 64, 128), (3, 40, 18)), seed=7)]))
    out.append(case("convolution", lambda x, w, b: A.convolution.default(x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1),
                    [rnd(*Z((1, 256, 14, 14), (1, 40, 9, 11)), seed=8, relu=True), rnd(*Z((256, 256, 3, 3), (24, 40, 3, 3)), seed=9, scale=0.05),
                     rnd(Z(256, 24), seed=10, scale=0.5)], kinds=["i8", "i8", ("i32", lambda s: s[0] * s[1])]))
    out.append(case("convolution_patch16", lambda x, w, b: A.convolution.default(x, w, b, [16, 16], [0, 0], [1, 1], False, [0, 0], 1),
                    [rnd(*Z((1, 3, 64, 64), (1, 3, 32, 48)), seed=11), rnd(*Z((96, 3, 16, 16), (20, 3, 16, 16)), seed=12, scale=0.05), rnd(Z(96, 20), seed=13, scale=0.5)],
                    kinds=["i8", "i8", ("i32", lambda s: s[0] * s[1])]))
    # elementwise
    out.append(case("sub_Tensor", lambda a, b: A.sub.Tensor(a, b), [rnd(*Z((750, 32), (37, 21)), seed=14), rnd(*Z((750, 32), (37, 21)), seed=15, scale=0.8)]))
    out.append(case("neg", lambda x: A.neg.default(x), [rnd(*Z((4096, 32), (37, 21)), seed=16)]))
    out.append(case("add_Tensor", lambda a, b: A.add.Tensor(a, b), [rnd(*Z((128, 2048), (37, 21)), seed=17), rnd(*Z((128, 2048), (37, 21)), seed=18, scale=0.6)]))
    out.append(case("mul_Scalar", lambda x: A.mul.Scalar(x, 0.125), [rnd(*Z((4096, 64), (37, 21)), seed=19)]))
    out.append(case("relu", lambda x: A.relu.default(x), [rnd(*Z((256, 196), (37, 21)), seed=20)]))
    out.append(case("mean_dim", lambda x: A.mean.dim(x, [-1]), [rnd(*Z((128, 2048), (37, 50)), seed=21, relu=True)]))
    # movement (shared qparams: the output scale is the input's)
    def shared(name, fn, xs):
        n, mod, ints = case(name, fn, xs)
        mod.s_out = mod.scales[0]
        return n, mod, ints
    out.append(shared("clone", lambda x: A.clone.default(x), [rnd(Z(262144, 777), seed=22)]))
    out.append(shared("permute", lambda x: A.permute.default(x, [1, 0, 2]), [rnd(*Z((128, 32, 64), (20, 3, 21)), seed=23)]))
    out.append(case("slice_scatter", lambda a, b: A.slice_scatter.default(a, b, 1, 0, 32, 1),
                    [rnd(*Z((750, 64), (20, 64)), seed=24), rnd(*Z((750, 32), (20, 32)), seed=25)]))
    out.append(shared("constant_pad_nd", lambda x: A.constant_pad_nd.default(x, [0, 0, 0, 0], 0.0), [rnd(*Z((3, 512, 512), (3, 20, 21)), seed=26)]))
    out.append(case("constant_pad_nd_p1", lambda x: A.constant_pad_nd.default(x, [1, 1, 1, 1], 0.5),
                    [rnd(*Z((1, 64, 56, 56), (1, 5, 7, 9)), seed=30, relu=True)]))
    out.append(case("full_like", lambda x: A.full_like.default(x, -1.25), [rnd(*Z((12, 1024, 1024), (37, 21)), seed=27)], s_out=0.02))
    out.append(case("cat", lambda a, b: A.cat.default([a, b], 1), [rnd(*Z((4, 128, 64), (2, 5, 24)), seed=28), rnd(*Z((4, 1, 64), (2, 3, 24)), seed=29)]))
    out.append(plain("_to_copy", lambda x: A._to_copy.default(x, dtype=torch.int32),
                     [torch.randint(-128, 128, Z((50,), (5, 10)), generator=g(31), dtype=torch.int8)]))
    return out


def run(cases, root: Path, only=None):
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, mod, ints in cases:
        if only and name not in only:
            continue
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            expected = mod(*ints)
        r = m2m.convert(mod.eval(), tuple(ints), backend="fx_importer")
        summary[name] = dict(ok=bool(r.ok), diagnostics=[str(x)[:300] for x in (r.diagnostics or [])][:4])
        if not r.ok:
            continue
        (d / "model.mlir").write_text(r.mlir_text)
        np.savez(d / "inputs.npz", **{f"arg{i}": t.numpy() for i, t in enumerate(ints)})
        np.save(d / "expected.npy", expected.numpy())
        (d / "meta.json").write_text(json.dumps(dict(scales=mod.scales, s_out=mod.s_out, dtypes=mod.dtypes)))
    (root / "capture_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--set", default="paper")
    a = ap.parse_args()
    sets = {"paper": paper_set, "paper_small": lambda: paper_set(small=True)}
    try:
        from capture_rest import rest_set  # noqa: E402
        sets["rest"] = rest_set
        sets["all"] = lambda: paper_set() + rest_set()
    except ImportError:
        pass
    print(json.dumps(run(sets[a.set](), a.out, a.only), indent=1))
