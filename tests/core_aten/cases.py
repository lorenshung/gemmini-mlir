"""Calibrated Core ATen cases with rational references for admitted arithmetic.

PyTorch supplies calibration data and integer input codes. Device admission
requires exactness on the full input domain; unsupported cases are reported.
Every admitted output is compared with exact arithmetic and has zero tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from exact_oracle import quantize as exact_quantize

import numpy as np
import torch

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.recipes.conv import ConvGeom

A = torch.ops.aten


def scale_of(x) -> float:
    return max(float(torch.as_tensor(x).abs().max()), 1e-8) / 127.5


def q(x, s) -> torch.Tensor:
    return torch.clamp(torch.round(torch.as_tensor(x, dtype=torch.float64) / s), -128, 127).to(torch.int8)


def dq(xq, s) -> torch.Tensor:
    return xq.to(torch.float64) * s


@dataclass
class Case:
    name: str
    op: str
    kernel: ops.Kernel
    inputs: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]           # kernel-layout expected outputs
    tol: int = 0
    min_exact: float = 1.0
    info: dict = field(default_factory=dict)


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _act(shape, seed, relu_like=False):
    x = torch.randn(*shape, generator=_g(seed), dtype=torch.float64)
    return x.clamp(min=0) * 1.7 if relu_like else x


def _np(t):
    return t.contiguous().numpy()


# ---------------------------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------------------------

def mm_case(M, K, N, seed=0, name=None):
    a, b = _act((M, K), seed), _act((K, N), seed + 1) * 0.05
    sa, sb = scale_of(a), scale_of(b)
    aq, bq = q(a, sa), q(b, sb)
    y = A.mm.default(dq(aq, sa), dq(bq, sb))
    so = scale_of(y)
    k = ops.matmul("aten.mm.default", M, K, N, s_a=sa, s_b=sb, s_out=so)
    return Case(name or f"mm_{M}x{K}x{N}", "aten.mm.default", k, {"a": _np(aq), "b": _np(bq)}, {"out": exact_quantize(_np(aq).astype(object) @ _np(bq).astype(object) * F(sa) * F(sb), so)})


def addmm_case(M, K, N, seed=0, name=None, beta=1.0, alpha=1.0):
    a, w, bias = _act((M, K), seed), _act((K, N), seed + 1) * 0.04, _act((N,), seed + 2) * 2.0
    sa, sw = scale_of(a), scale_of(w)
    aq, wq = q(a, sa), q(w, sw)
    # PT2E bias: int32 at scale s_a s_w (beta and alpha folded at compile time)
    bq = torch.round(beta * bias / (alpha * sa * sw)).to(torch.int32)
    y = A.addmm.default(dq(bq, sa * sw) * alpha, dq(aq, sa), dq(wq, sw), beta=1.0, alpha=alpha)
    so = scale_of(y)
    k = ops.matmul("aten.addmm.default", M, K, N, s_a=sa, s_b=sw, s_out=so, alpha=alpha, bias=True,
                   bias_range=(int(bq.min()), int(bq.max())))
    return Case(name or f"addmm_{M}x{K}x{N}", "aten.addmm.default", k,
                {"a": _np(aq), "b": _np(wq), "bias": _np(bq)}, {"out": _np(q(y, so))})


def bmm_case(B, M, K, N, seed=0, name=None):
    a, b = _act((B, M, K), seed), _act((B, K, N), seed + 1)
    sa, sb = scale_of(a), scale_of(b)
    aq, bq = q(a, sa), q(b, sb)
    y = A.bmm.default(dq(aq, sa), dq(bq, sb))
    so = scale_of(y)
    k = ops.matmul("aten.bmm.default", M, K, N, batch=B, s_a=sa, s_b=sb, s_out=so)
    return Case(name or f"bmm_{B}x{M}x{K}x{N}", "aten.bmm.default", k, {"a": _np(aq), "b": _np(bq)},
                {"out": exact_quantize(_np(aq).astype(object) @ _np(bq).astype(object) * F(sa) * F(sb), so)})


def conv_case(N, C, H, W, cout, k, s=1, p=0, d=1, groups=1, seed=0, name=None):
    """Without a seed: production refuses a runtime convolution seed (no nonoverflow range proof)."""
    x = _act((N, C, H, W), seed, relu_like=True)
    w = _act((cout, C // groups, k, k), seed + 1) * 0.1
    sx, sw = scale_of(x), scale_of(w)
    xq, wq = q(x, sx), q(w, sw)
    conv = lambda a, b: A.convolution.default(a, b, None, [s, s], [p, p], [d, d], False, [0, 0], groups)
    so = scale_of(conv(dq(xq, sx), dq(wq, sw)))
    acc = conv(xq.to(torch.float64), wq.to(torch.float64)).to(torch.int64)   # integer codes: exact in f64
    g = ConvGeom(N, H, W, C, cout, k, k, s, s, p, p, d, d, groups)
    kern = ops.convolution(g, s_x=sx, s_w=sw, s_out=so)
    inputs = {"x": _np(xq.permute(0, 2, 3, 1)), "w": _np(wq.permute(2, 3, 1, 0))}
    return Case(name or f"conv_{N}x{C}x{H}x{W}_k{k}s{s}p{p}_{cout}", "aten.convolution.default", kern, inputs,
                {"y": exact_quantize(_np(acc.permute(0, 2, 3, 1)).astype(object) * F(sx) * F(sw), so)})


# ---------------------------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------------------------

def scale_case(op, shape, c, seed=0, name=None, narrow=False):
    x = _act(shape, seed)
    sx = scale_of(x) if not narrow else float(x.abs().max()) / 127.0
    xq = q(x, sx)
    if narrow:
        xq = xq.clamp(min=-127)
    fn = {"aten.neg.default": lambda t: A.neg.default(t), "aten.mul.Scalar": lambda t: A.mul.Scalar(t, c),
          "aten.div.Scalar": lambda t: A.div.Scalar(t, c)}[op]
    y = fn(dq(xq, sx))
    so = scale_of(y)
    cc = {"aten.neg.default": -1.0, "aten.mul.Scalar": c, "aten.div.Scalar": F(1) / F(c) if c else F(0)}[op]
    k = ops.scale_map(op, shape, cc, s_x=sx, s_out=so)
    return Case(name or f"{op.split('.')[1]}_{'x'.join(map(str, shape))}", op, k, {"x": _np(xq)}, {"y": exact_quantize(_np(xq).astype(object) * F(sx) * F(cc), so)})


def relu_case(shape, seed=0, name=None):
    x = _act(shape, seed)
    sx = scale_of(x)
    xq = q(x, sx)
    y = A.relu.default(dq(xq, sx))
    so = scale_of(y)
    k = ops.relu(shape, s_x=sx, s_out=so)
    return Case(name or f"relu_{'x'.join(map(str, shape))}", "aten.relu.default", k, {"x": _np(xq)},
                {"y": exact_quantize(np.maximum(_np(xq).astype(object), 0) * F(sx), so)})


def add_case(op, shape, b_shape=None, alpha=1.0, seed=0, name=None):
    b_shape = b_shape or shape
    a, b = _act(shape, seed), _act(b_shape, seed + 1) * 0.6
    sa, sb = scale_of(a), scale_of(b)
    aq, bq = q(a, sa), q(b, sb)
    fn = A.add.Tensor if op == "aten.add.Tensor" else A.sub.Tensor
    y = fn(dq(aq, sa), dq(bq, sb), alpha=alpha)
    so = scale_of(y)
    k = ops.add_tensor(op, shape, b_shape, alpha=alpha if op == "aten.add.Tensor" else -alpha, s_a=sa, s_b=sb,
                       s_out=so)
    return Case(name or f"{op.split('.')[1]}_{'x'.join(map(str, shape))}", op, k, {"a": _np(aq), "b": _np(bq)},
                {"y": exact_quantize(_np(aq).astype(object) * F(sa) + _np(bq).astype(object) * F(sb) * F(alpha) * (1 if op == "aten.add.Tensor" else -1), so)})


def add_scalar_case(op, shape, c, seed=0, name=None):
    x = _act(shape, seed)
    sx = scale_of(x)
    xq = q(x, sx)
    fn = A.add.Scalar if op == "aten.add.Scalar" else A.sub.Scalar
    y = fn(dq(xq, sx), c)
    so = scale_of(y)
    k = ops.add_scalar(op, shape, c if op == "aten.add.Scalar" else -c, s_x=sx, s_out=so)
    return Case(name or f"{op.split('.')[1]}_{'x'.join(map(str, shape))}", op, k, {"x": _np(xq)},
                {"y": exact_quantize(_np(xq).astype(object) * F(sx) + F(c) * (1 if op == "aten.add.Scalar" else -1), so)})


def mean_case(shape, dims, keepdim=False, seed=0, name=None):
    x = _act(shape, seed, relu_like=True)
    sx = scale_of(x)
    xq = q(x, sx)
    y = A.mean.dim(dq(xq, sx), dims, keepdim)
    so = scale_of(y)
    k = ops.mean_dim(shape, dims, keepdim, s_x=sx, s_out=so)
    return Case(name or f"mean_{'x'.join(map(str, shape))}", "aten.mean.dim", k, {"x": _np(xq)},
                {"y": exact_quantize(_np(xq).astype(object).sum(axis=tuple(dims), keepdims=keepdim) * F(sx) / int(np.prod([shape[d] for d in dims])), so)})


# ---------------------------------------------------------------------------------------------
# Movement (shared qparams: the output observer sees the input; values are byte moves)
# ---------------------------------------------------------------------------------------------

def _codes(shape, seed):
    return torch.randint(-128, 128, shape, generator=_g(seed), dtype=torch.int8)


def clone_case(shape, seed=0, name=None):
    xq = _codes(shape, seed)
    return Case(name or f"clone_{'x'.join(map(str, shape))}", "aten.clone.default", ops.clone(shape),
                {"x": _np(xq)}, {"y": _np(A.clone.default(xq))}, tol=0, min_exact=1.0)


def permute_case(shape, perm, seed=0, name=None):
    xq = _codes(shape, seed)
    return Case(name or f"permute_{'x'.join(map(str, shape))}_{''.join(map(str, perm))}", "aten.permute.default",
                ops.permute(shape, perm), {"x": _np(xq)}, {"y": _np(A.permute.default(xq, perm))}, tol=0,
                min_exact=1.0)


def cat_case(shapes, dim, seed=0, name=None, independent=False):
    xs = [_act(sh, seed + i) * (1 + i) for i, sh in enumerate(shapes)]
    if independent:
        ss = [scale_of(x) for x in xs]
        so = scale_of(torch.cat([x.flatten() for x in xs]))
    else:
        ss = [scale_of(torch.cat([x.flatten() for x in xs]))] * len(xs)
        so = ss[0]
    xq = [q(x, s) for x, s in zip(xs, ss)]
    y = A.cat.default([dq(t, s) for t, s in zip(xq, ss)], dim)
    k = ops.cat(shapes, dim, s_in=ss, s_out=so)
    return Case(name or f"cat_{len(shapes)}x{'x'.join(map(str, shapes[0]))}_d{dim}", "aten.cat.default", k,
                {f"x{i}": _np(t) for i, t in enumerate(xq)}, {"y": exact_quantize(np.concatenate([_np(t).astype(object) * F(si) for t,si in zip(xq,ss)],axis=dim), so)})


def pad_case(shape, pad, value, seed=0, name=None):
    x = _act(shape, seed)
    sx = scale_of(x)
    xq = q(x, sx)
    y = A.constant_pad_nd.default(dq(xq, sx), pad, value)
    so = scale_of(y) if abs(value) > float(x.abs().max()) else sx
    k = ops.constant_pad_nd(shape, pad, value, s_x=sx, s_out=so)
    return Case(name or f"pad_{'x'.join(map(str, shape))}", "aten.constant_pad_nd.default", k, {"x": _np(xq)},
                {"y": _np(q(y, so))})


def slice_scatter_case(shape, src_shape, dim, start, end, step, seed=0, name=None):
    a, b = _act(shape, seed), _act(src_shape, seed + 1)
    s = scale_of(torch.cat([a.flatten(), b.flatten()]))
    aq, bq = q(a, s), q(b, s)
    y = A.slice_scatter.default(dq(aq, s), dq(bq, s), dim, start, end, step)
    k = ops.slice_scatter(shape, src_shape, dim, start, end, step, s_self=s, s_src=s, s_out=s)
    return Case(name or f"slice_scatter_{'x'.join(map(str, shape))}", "aten.slice_scatter.default", k,
                {"self": _np(aq), "src": _np(bq)}, {"y": _np(q(y, s))}, tol=0, min_exact=1.0)


def full_like_case(shape, value, seed=0, name=None, s_out=None):
    x = _act(shape, seed)
    so = s_out if s_out is not None else max(abs(value), 1e-8) / 127.5
    y = A.full_like.default(x, value)
    k = ops.full("aten.full_like.default", shape, value, s_out=so, like=True)
    return Case(name or f"full_like_{'x'.join(map(str, shape))}", "aten.full_like.default", k,
                {"x": _np(q(x, scale_of(x)))}, {"y": _np(q(y, so))}, tol=0, min_exact=1.0)


def to_copy_case(shape, seed=0, name=None):
    xq = _codes(shape, seed)
    y = A._to_copy.default(xq, dtype=torch.int32)
    return Case(name or f"to_copy_{'x'.join(map(str, shape))}", "aten._to_copy.default", ops.to_copy_widen(shape),
                {"x": _np(xq)}, {"y": _np(y)}, tol=0, min_exact=1.0)


# ---------------------------------------------------------------------------------------------
# The paper subset: small edge shapes
# ---------------------------------------------------------------------------------------------

def paper_cases() -> list[Case]:
    """Every case must be admitted: a device refusal raises Decline instead of dropping the case."""
    out = [
        lambda: conv_case(1, 64, 9, 11, 48, 3, 1, 1, name="conv_k3s1p1_tails"),
        lambda: conv_case(1, 32, 12, 12, 40, 3, 2, 1, name="conv_k3s2p1"),
        lambda: conv_case(1, 3, 32, 32, 64, 16, 16, 0, name="conv_patch16"),
        lambda: conv_case(1, 48, 8, 8, 32, 1, 1, 0, name="conv_1x1"),
        lambda: conv_case(1, 32, 9, 9, 24, 7, 2, 3, name="conv_k7s2p3"),
        lambda: mm_case(37, 200, 45, name="mm_tails"),
        lambda: addmm_case(50, 96, 70, name="addmm_tails"),
        lambda: bmm_case(3, 20, 40, 18, name="bmm_tails"),
        lambda: scale_case("aten.neg.default", (37, 21), 0, name="neg_tails"),
        lambda: scale_case("aten.mul.Scalar", (37, 21), 0.25, name="mul_Scalar_tails"),
        lambda: add_case("aten.sub.Tensor", (37, 21), name="sub_Tensor_tails"),
        lambda: add_case("aten.add.Tensor", (37, 21), name="add_Tensor_tails"),
        lambda: add_case("aten.add.Tensor", (40, 48), (48,), alpha=0.5, name="add_Tensor_rowbcast_alpha"),
        lambda: relu_case((37, 21), name="relu_tails"),
        lambda: mean_case((37, 50), [-1], name="mean_last_tails"),
        lambda: mean_case((3, 20, 21), [1], keepdim=True, name="mean_middle"),
        lambda: clone_case((37, 21), name="clone_tails"),
        lambda: permute_case((3, 20, 21), (1, 0, 2), name="permute_102"),
        lambda: permute_case((2, 20, 37), (0, 2, 1), name="permute_021_transposer"),
        lambda: cat_case([(2, 5, 24), (2, 3, 24)], 1, name="cat_d1"),
        lambda: cat_case([(2, 5, 24), (2, 5, 9)], -1, independent=True, name="cat_last_independent"),
        lambda: pad_case((2, 7, 9), [1, 2, 0, 1], 0.0, name="pad_zero"),
        lambda: pad_case((2, 7, 9), [2, 1, 1, 0], 9.0, name="pad_value_requant"),
        lambda: slice_scatter_case((20, 64), (20, 32), 1, 0, 32, 1, name="slice_scatter_c0"),
        lambda: slice_scatter_case((20, 33), (20, 11), 1, 1, None, 3, name="slice_scatter_step3"),
        lambda: full_like_case((37, 21), -1.7, s_out=0.05, name="full_like_tails"),
        lambda: to_copy_case((5, 10), name="to_copy_50"),
    ]
    return [factory() for factory in out]
