"""Core ATen op lowerings: semantic parameters + quantization -> one device kernel stream.

Every kernel takes dense row-major int8 tensors (int32 for bias seeds and widened outputs)
in the argument order of ``Kernel.args``; activations of convolutions are channels-last
(NHWC) and weights HWIO.  Quantization is per-tensor symmetric int8 with zero point 0:
a tensor q with scale s represents s q.  Each lowering reads its inputs' and output's
scales and folds them into the recipe's single rounding step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .numerics import best_int_multiplier, code, parts127, rational_weights
from .recipes import elementwise as ew
from .recipes import movement as mv
from .recipes import reduce as rd
from .recipes.conv import ConvGeom, conv2d
from .recipes.gemm import blocked_matmul
from .stream import Arg, Stream


@dataclass
class Kernel:
    op: str
    stream: Stream
    meta: dict = field(default_factory=dict)

    @property
    def args(self) -> list[Arg]:
        return [a for a in self.stream.args if a.role != "const"]


def _numel(shape) -> int:
    return int(np.prod(shape)) if len(shape) else 1


def _begin(args) -> Stream:
    s = Stream(list(args))
    s.fence()
    s.flush()
    return s


def _end(op, s, **meta) -> Kernel:
    s.fence()
    return Kernel(op, s, meta)


# ---------------------------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------------------------

def matmul(op: str, M: int, K: int, N: int, *, batch: int = 0, s_a: float = 1.0, s_b: float = 1.0,
           s_out: float | None = None, alpha: float = 1.0, bias: bool = False, relu: bool = False,
           bias_const: np.ndarray | None = None) -> Kernel:
    """mm / bmm (batch > 0) / addmm (bias: int32 [N] in accumulator units, i.e. at scale alpha s_a s_b).

    ``bias_const`` makes the (compile-time quantized) bias a private constant instead of an argument.
    s_out None selects the raw int32 readout of the exact integer contraction.
    """
    B = max(batch, 1)
    lead = (batch,) if batch else ()
    out_i32 = s_out is None
    args = [Arg("a", "i8", lead + (M, K), "input"), Arg("b", "i8", lead + (K, N), "input")]
    if bias and bias_const is None:
        args.append(Arg("bias", "i32", (N,), "input"))
    args.append(Arg("out", "i32" if out_i32 else "i8", lead + (M, N), "output"))
    s = _begin(args)
    bname = None
    if bias:
        bname = "bias"
        if bias_const is not None:
            bname = "bias_c"
            s.const(bname, np.asarray(bias_const, np.int32).reshape(N))
    scale = 1.0 if out_i32 else alpha * s_a * s_b / s_out
    for bi in range(B):
        blocked_matmul(s, ("a", bi * M * K), K, ("b", bi * K * N), N, ("out", bi * M * N * (4 if out_i32 else 1)), N,
                       M, K, N, bias=(bname, 0) if bias else None, scale=scale, relu=relu, out_i32=out_i32,
                       configure=bi == 0)
    return _end(op, s, readout_scale=scale)


def _conv_body(s: Stream, g: ConvGeom, x, w, y, bname, scale, relu, out_i32):
    if (g.kh, g.kw, g.sh, g.sw, g.ph, g.pw, g.groups) == (1, 1, 1, 1, 0, 0, 1):
        M = g.n * g.h * g.w
        blocked_matmul(s, (x, 0), g.cin, (w, 0), g.cout, (y, 0), g.cout, M, g.cin, g.cout,
                       bias=(bname, 0) if bname else None, scale=scale, relu=relu, out_i32=out_i32)
    else:
        conv2d(s, g, x, w, y, bias=bname, scale=scale, relu=relu, out_i32=out_i32)


def convolution(g: ConvGeom, *, s_x: float = 1.0, s_w: float = 1.0, s_out: float | None = None,
                bias: bool = False, relu: bool = False, alpha: float = 1.0) -> Kernel:
    """Channels-last: NHWC x HWIO -> NHWC; bias int32 [Cout] at scale s_x s_w (PT2E)."""
    g.validate()
    out_i32 = s_out is None
    args = [Arg("x", "i8", (g.n, g.h, g.w, g.cin), "input"),
            Arg("w", "i8", (g.kh, g.kw, g.cin // g.groups, g.cout), "input")]
    if bias:
        args.append(Arg("bias", "i32", (g.cout,), "input"))
    args.append(Arg("y", "i32" if out_i32 else "i8", (g.n, g.oh, g.ow, g.cout), "output"))
    s = _begin(args)
    scale = 1.0 if out_i32 else alpha * s_x * s_w / s_out
    _conv_body(s, g, "x", "w", "y", "bias" if bias else None, scale, relu, out_i32)
    return _end("aten.convolution.default", s, readout_scale=scale, layout="NHWC,HWIO->NHWC")


def convolution_nchw(g: ConvGeom, *, s_x: float, s_w: float, s_out: float, alpha: float = 1.0,
                     bias: bool = False, relu: bool = False, bias_const: np.ndarray | None = None) -> Kernel:
    """The source layout (NCHW x OIHW -> NCHW) with on-device transposes around the channels-last
    convolution: x and w are transposed into caller-provided scratch, fenced (DRAM RAW), convolved,
    fenced, and the NHWC result transposed into y.  Merlin's layout propagation can bind the
    channels-last kernel (``convolution``) instead and drop the transposes."""
    g.validate()
    if g.groups != 1:
        raise ValueError("grouped convolution needs the channels-last kernel")
    args = [Arg("x", "i8", (g.n, g.cin, g.h, g.w), "input"), Arg("w", "i8", (g.cout, g.cin, g.kh, g.kw), "input")]
    if bias and bias_const is None:
        args.append(Arg("bias", "i32", (g.cout,), "input"))
    args += [Arg("y", "i8", (g.n, g.cout, g.oh, g.ow), "output"),
             Arg("ws_x", "i8", (g.n, g.h, g.w, g.cin), "scratch"),
             Arg("ws_w", "i8", (g.kh, g.kw, g.cin, g.cout), "scratch"),
             Arg("ws_y", "i8", (g.n, g.oh, g.ow, g.cout), "scratch")]
    s = _begin(args)
    bname = None
    if bias:
        bname = "bias"
        if bias_const is not None:
            bname = "bias_c"
            s.const(bname, np.asarray(bias_const, np.int32).reshape(g.cout))
    mv.emit_transpose(s, "x", [g.n, g.cin, g.h, g.w], "ws_x", [0, 2, 3, 1])
    mv.emit_transpose(s, "w", [g.cout, g.cin, g.kh, g.kw], "ws_w", [2, 3, 1, 0])
    s.fence()
    scale = alpha * s_x * s_w / s_out
    _conv_body(s, g, "ws_x", "ws_w", "ws_y", bname, scale, relu, False)
    s.fence()
    mv.emit_transpose(s, "ws_y", [g.n, g.oh, g.ow, g.cout], "y", [0, 3, 1, 2])
    return _end("aten.convolution.default", s, readout_scale=scale, layout="NCHW,OIHW->NCHW via NHWC")


def conv_backward_input(g: ConvGeom, *, s_g: float, s_w: float, s_out: float) -> Kernel:
    """grad_input (NCHW) of a forward convolution ``g`` from grad_output (NCHW) and OIHW weights."""
    from .recipes.backward import conv_transpose
    args = [Arg("g", "i8", (g.n, g.cout, g.oh, g.ow), "input"), Arg("w", "i8", (g.cout, g.cin, g.kh, g.kw), "input"),
            Arg("y", "i8", (g.n, g.cin, g.h, g.w), "output"),
            Arg("ws_g", "i8", (g.n, g.oh, g.ow, g.cout), "scratch"),
            Arg("ws_w", "i8", (g.kh, g.kw, g.cout, g.cin), "scratch"),
            Arg("ws_y", "i8", (g.n, g.h, g.w, g.cin), "scratch")]
    s = _begin(args)
    mv.emit_transpose(s, "g", [g.n, g.cout, g.oh, g.ow], "ws_g", [0, 2, 3, 1])
    mv.emit_transpose(s, "w", [g.cout, g.cin, g.kh, g.kw], "ws_w", [2, 3, 0, 1])
    s.fence()
    scale = s_g * s_w / s_out
    conv_transpose(s, g, "ws_g", "ws_w", "ws_y", scale)
    s.fence()
    mv.emit_transpose(s, "ws_y", [g.n, g.h, g.w, g.cin], "y", [0, 3, 1, 2])
    return _end("aten.convolution_backward.default", s, output="grad_input", readout_scale=scale)


def conv_backward_weight(g: ConvGeom, *, s_g: float, s_x: float, s_out: float) -> Kernel:
    """grad_weight (OIHW) from grad_output (NCHW) and the forward input (NCHW); Q11 readout."""
    from .recipes.backward import conv_wgrad
    args = [Arg("g", "i8", (g.n, g.cout, g.oh, g.ow), "input"), Arg("x", "i8", (g.n, g.cin, g.h, g.w), "input"),
            Arg("y", "i8", (g.cout, g.cin, g.kh, g.kw), "output"),
            Arg("ws_x", "i8", (g.n, g.h, g.w, g.cin), "scratch"),
            Arg("ws_gw", "i8", (g.kh * g.kw, g.cout, g.cin), "scratch")]
    s = _begin(args)
    mv.emit_transpose(s, "x", [g.n, g.cin, g.h, g.w], "ws_x", [0, 2, 3, 1])
    s.fence()
    scale = s_g * s_x / s_out
    conv_wgrad(s, g, "g", "ws_x", "ws_gw", scale)
    s.fence()
    mv.emit_transpose(s, "ws_gw", [g.kh * g.kw, g.cout, g.cin], "y", [1, 2, 0])
    return _end("aten.convolution_backward.default", s, output="grad_weight", readout_scale=scale)


# ---------------------------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------------------------

def scale_map(op: str, shape, c: float, *, s_x: float, s_out: float) -> Kernel:
    """y = c x (neg: c = -1, mul.Scalar: c, div.Scalar: 1 / c), one MVIN-scale rounding."""
    n = _numel(shape)
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", tuple(shape), "output")])
    ew.unary_scale(s, "x", "y", n, c * s_x / s_out)
    return _end(op, s, mvin_scale=c * s_x / s_out)


def relu(shape, *, s_x: float, s_out: float) -> Kernel:
    n = _numel(shape)
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", tuple(shape), "output")])
    ew.unary_readout(s, "x", "y", n, s_x / s_out, relu=True)
    return _end("aten.relu.default", s, readout_scale=s_x / s_out)


def add_tensor(op: str, shape, b_shape, *, alpha: float = 1.0, s_a: float, s_b: float, s_out: float,
               wa: float = 1.0, c: float = 0.0, relu: bool = False, b_lead: int | None = None) -> Kernel:
    """y = act(wa a + alpha b + c) (add.Tensor; sub.Tensor has alpha < 0), one rounding (Q2/Q10).

    ``b`` is same-shape or broadcast over the leading ``b_lead`` output dims (a stride-0 row).
    """
    shape, b_shape = tuple(shape), tuple(b_shape)
    n = _numel(shape)
    if b_lead is None:
        b_lead = 0 if b_shape == shape else len(shape) - len(b_shape)
        while b_lead < len(shape) and _numel(shape[b_lead:]) != _numel(b_shape):
            b_lead += 1
    inner = _numel(shape[b_lead:])
    if _numel(b_shape) != inner or (b_lead and tuple(shape[b_lead:]) != tuple(b_shape[-(len(shape) - b_lead):])):
        raise ValueError(f"unsupported broadcast {b_shape} -> {shape}")
    if b_lead == 0:
        rows, cols = (n // 64, 64) if n % 64 == 0 else (n // (shape[-1] if shape else 1), shape[-1] if shape else 1)
        a_stride = b_stride = cols
    else:
        rows, cols, a_stride, b_stride = n // inner, inner, inner, 0
    if wa == 0 or alpha == 0:
        raise ValueError("degenerate linear combination")
    al, be = abs(wa) * s_a / s_out, abs(alpha) * s_b / s_out
    Wa, Wb = rational_weights(al, be)
    Wa_s = Wa if wa > 0 else -Wa
    Wb_s = Wb if alpha > 0 else -Wb
    bias = int(round(c * Wa / (abs(wa) * s_a))) if c else None
    terms = [("a", a_stride, parts127(Wa_s)), ("b", b_stride, parts127(Wb_s))]
    s = _begin([Arg("a", "i8", shape, "input"), Arg("b", "i8", b_shape, "input"), Arg("y", "i8", shape, "output")])
    ew.lincomb(s, terms, "y", rows, cols, al / Wa, bias=bias, relu=relu)
    return _end(op, s, weights=(Wa_s, Wb_s), bias=bias, readout_scale=al / Wa)


def add_scalar(op: str, shape, c: float, *, s_x: float, s_out: float, w: float = 1.0, relu: bool = False) -> Kernel:
    """y = act(w x + c): ACC = G x + B exactly with B = round(G c / (w s_x)) (Q9), one rounding."""
    n = _numel(shape)
    k = w * s_x
    G = best_int_multiplier(c / k)
    B = int(round(G * c / k))
    sc = k / (G * s_out)
    if sc < 0:                         # keep the readout scale positive (ReLU before scale, C7)
        G, B, sc = -G, -B, -sc
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", tuple(shape), "output")])
    rows, cols = (n // 64, 64) if n % 64 == 0 else (n // shape[-1], shape[-1])
    ew.lincomb(s, [("x", cols, parts127(G))], "y", rows, cols, sc, bias=B, relu=relu)
    return _end(op, s, G=G, B=B, readout_scale=sc)


# ---------------------------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------------------------

def mean_dim(shape, dims, keepdim: bool, *, s_x: float, s_out: float, w: float | None = None,
             op: str = "aten.mean.dim") -> Kernel:
    """sum (w = 1) or mean (w = 1 / L, the default) over dims, one readout scale w s_x / s_out."""
    r = len(shape)
    dims = sorted(d % r for d in dims)
    # size-1 dims never break contiguity of the reduced block
    eff = [d for d in dims if shape[d] != 1] or dims[:1]
    if any(d not in dims and shape[d] != 1 for d in range(eff[0], eff[-1] + 1)):
        raise ValueError("mean over non-contiguous dims")
    O = _numel([shape[d] for d in range(eff[0]) if d not in dims])
    L = _numel([shape[d] for d in dims])
    I = _numel([shape[d] for d in range(eff[-1] + 1, r) if d not in dims])
    oshape = [1 if d in dims else shape[d] for d in range(r)] if keepdim else \
        [shape[d] for d in range(r) if d not in dims]
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", tuple(oshape), "output")])
    sc = (1.0 / L if w is None else w) * s_x / s_out
    if I == 1:
        rd.mean_last(s, "x", "y", O, L, sc)
    else:
        rd.mean_middle(s, "x", "y", O, L, I, sc)
    return _end(op, s, readout_scale=sc, reduced=L)


# ---------------------------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------------------------

def _plan_kernel(op, args, plan: mv.Plan, **meta) -> Kernel:
    s = _begin(args)
    mv.emit_plan(s, plan)
    return _end(op, s, **meta)


def clone(shape, *, s_x: float = 1.0, s_out: float = 1.0) -> Kernel:
    n = _numel(shape)
    plan = mv.Plan()
    mv.flat_copy(plan.stage(), ("x", 0), ("y", 0), n, s_x / s_out)
    return _plan_kernel("aten.clone.default", [Arg("x", "i8", tuple(shape), "input"),
                                                Arg("y", "i8", tuple(shape), "output")], plan)


def permute(shape, perm, *, s_x: float = 1.0, s_out: float = 1.0) -> Kernel:
    perm = [p % len(shape) for p in perm]
    oshape = tuple(shape[p] for p in perm)
    args = [Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", oshape, "output")]
    if perm[-1] == len(shape) - 1:
        plan, _ = mv.permute_plan("x", shape, "y", perm, s_x / s_out)
        return _plan_kernel("aten.permute.default", args, plan, form="strided_dma")
    s = _begin(args)
    mv.emit_transpose(s, "x", list(shape), "y", perm, s_x / s_out)
    return _end("aten.permute.default", s, form="transposer")


def _permute_into(s: Stream, x: str, shape, out: str, perm) -> None:
    """Emit a permutation of x into out (strided DMA if the innermost dim stays, else the transposer)."""
    perm = list(perm)
    if perm[-1] == len(shape) - 1:
        plan, _ = mv.permute_plan(x, list(shape), out, perm)
        mv.emit_plan(s, plan)
    else:
        mv.emit_transpose(s, x, list(shape), out, perm)


def cat(shapes, dim: int, *, s_in=None, s_out: float = 1.0) -> Kernel:
    names = [f"x{i}" for i in range(len(shapes))]
    scales = None if s_in is None else [si / s_out for si in s_in]
    plan = mv.Plan()
    oshape = mv.cat(plan, list(zip(names, [tuple(sh) for sh in shapes])), "y", dim, scales)
    args = [Arg(nm, "i8", tuple(sh), "input") for nm, sh in zip(names, shapes)] + \
        [Arg("y", "i8", tuple(oshape), "output")]
    return _plan_kernel("aten.cat.default", args, plan)


def constant_pad_nd(shape, pad, value: float, *, s_x: float = 1.0, s_out: float = 1.0) -> Kernel:
    """``pad`` in ATen order: (last lo, last hi, second-to-last lo, ...)."""
    r = len(shape)
    lohi = [(0, 0)] * r
    for i in range(len(pad) // 2):
        lohi[r - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    plan = mv.Plan()
    oshape = mv.constant_pad(plan, "x", list(shape), "y", lohi, code(value, s_out), s_x / s_out)
    args = [Arg("x", "i8", tuple(shape), "input"), Arg("y", "i8", tuple(oshape), "output")]
    return _plan_kernel("aten.constant_pad_nd.default", args, plan, fill_code=code(value, s_out))


def slice_scatter(shape, src_shape, dim: int, start, end, step: int, *, s_self: float = 1.0, s_src: float = 1.0,
                  s_out: float = 1.0) -> Kernel:
    D = shape[dim % len(shape)]
    start = 0 if start is None else start
    end = D if end is None else end
    plan = mv.Plan()
    mv.slice_scatter(plan, "self", list(shape), "src", "y", dim, start, end, step, s_self / s_out, s_src / s_out)
    args = [Arg("self", "i8", tuple(shape), "input"), Arg("src", "i8", tuple(src_shape), "input"),
            Arg("y", "i8", tuple(shape), "output")]
    return _plan_kernel("aten.slice_scatter.default", args, plan)


def full(op: str, shape, value: float, *, s_out: float, like: bool = False) -> Kernel:
    n = _numel(shape)
    plan = mv.Plan()
    mv.full(plan, "y", n, code(value, s_out))
    args = ([Arg("x", "i8", tuple(shape), "input")] if like else []) + [Arg("y", "i8", tuple(shape), "output")]
    return _plan_kernel(op, args, plan, fill_code=code(value, s_out))


def to_copy_widen(shape) -> Kernel:
    """int8 -> int32 (the quantized codes widen exactly through the accumulator readout)."""
    n = _numel(shape)
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i32", tuple(shape), "output")])
    mv.emit_widen(s, "x", "y", n)
    return _end("aten._to_copy.default", s)
