"""The other 22 Core ATen ops of the support set, in the same quantized capture form as
``capture.py`` (imported by it; run ``capture.py OUT --set rest|rest_small|all``)."""

from __future__ import annotations

import torch

from capture import A, case, rnd


def _shared(name, fn, xs, **kw):
    n, mod, ints = case(name, fn, xs, **kw)
    mod.s_out = mod.scales[0]
    return n, mod, ints


class Multi(torch.nn.Module):
    """forward(*ints) = tuple of Q(out_k) for a multi-output op (convolution_backward)."""

    def __init__(self, fn, scales, s_outs):
        super().__init__()
        self.fn, self.scales, self.s_outs = fn, scales, s_outs
        self.dtypes, self.s_out = ["i8"] * len(scales), s_outs

    def forward(self, *xs):
        QD = torch.ops.quantized_decomposed
        ds = [QD.dequantize_per_tensor(x, s, 0, -128, 127, torch.int8) for x, s in zip(xs, self.scales)]
        outs = self.fn(*ds)
        return tuple(QD.quantize_per_tensor(y, s, 0, -128, 127, torch.int8) for y, s in zip(outs, self.s_outs))


def conv_backward_case(name, N, C, H, W, F, k, s, p, small):
    from capture import qz, sc
    g = rnd(N, F, (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1, seed=40, scale=0.3)
    x = rnd(N, C, H, W, seed=41, relu=True)
    w = rnd(F, C, k, k, seed=42, scale=0.05)
    scales = [sc(g), sc(x), sc(w)]
    ints = [qz(g, scales[0]), qz(x, scales[1]), qz(w, scales[2])]
    deq = [i.to(torch.float32) * s_ for i, s_ in zip(ints, scales)]

    def fn(g_, x_, w_):
        gi, gw, gb = A.convolution_backward.default(g_, x_, w_, [F], [s, s], [p, p], [1, 1], False, [0, 0], 1,
                                                    [True, True, True])
        return gi, gw, gb

    outs = fn(*deq)
    return name, Multi(fn, scales, [sc(o) for o in outs]), ints


def rest_set(small: bool = False):
    Z = (lambda big, sm: sm if small else big)
    out = []
    out.append(case("div_Scalar", lambda x: A.div.Scalar(x, 8.0), [rnd(*Z((256, 256), (37, 21)), seed=50)]))
    out.append(case("add_Scalar", lambda x: A.add.Scalar(x, 1.0), [rnd(*Z((256, 256), (37, 21)), seed=51).tanh()]))
    out.append(case("sub_Scalar", lambda x: A.sub.Scalar(x, 0.5),
                    [rnd(*Z((256, 256), (37, 21)), seed=52).sigmoid()]))
    out.append(case("fill_Scalar", lambda x: A.fill.Scalar(x, 0.5), [rnd(*Z((256, 256), (37, 21)), seed=53)],
                    s_out=0.0125))
    out.append(case("full", lambda x: A.full.default(list(x.shape), -1.7, dtype=torch.float32),
                    [rnd(*Z((256, 256), (37, 21)), seed=54)], s_out=0.05))
    # pools (NCHW / NCL / NCDHW), count_include_pad defaults
    out.append(case("avg_pool1d", lambda x: A.avg_pool1d.default(x, [3], [2], [1]),
                    [rnd(*Z((1, 64, 3136), (1, 20, 37)), seed=55, relu=True)]))
    out.append(case("avg_pool2d", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]),
                    [rnd(*Z((1, 64, 56, 56), (1, 20, 9, 11)), seed=56, relu=True)]))
    out.append(case("avg_pool3d", lambda x: A.avg_pool3d.default(x, [3, 3, 3], [2, 2, 2], [1, 1, 1]),
                    [rnd(*Z((1, 64, 16, 14, 14), (1, 16, 5, 6, 7)), seed=57, relu=True)]))
    out.append(case("adaptive_avg_pool1d", lambda x: A.adaptive_avg_pool1d.default(x, [Z(49, 5)]),
                    [rnd(*Z((1, 64, 3136), (1, 20, 37)), seed=58, relu=True)]))
    out.append(case("_adaptive_avg_pool2d", lambda x: A._adaptive_avg_pool2d.default(x, [Z(7, 3), Z(7, 4)]),
                    [rnd(*Z((1, 64, 56, 56), (1, 20, 9, 11)), seed=59, relu=True)]))
    out.append(case("_adaptive_avg_pool3d", lambda x: A._adaptive_avg_pool3d.default(x, [2, Z(7, 3), Z(7, 4)]),
                    [rnd(*Z((1, 64, 16, 14, 14), (1, 16, 5, 6, 7)), seed=60, relu=True)]))
    gshape = Z((1, 64, 28, 28), (1, 20, 5, 6))
    xshape = Z((1, 64, 56, 56), (1, 20, 9, 11))
    out.append(case("avg_pool2d_backward",
                    lambda g, x: A.avg_pool2d_backward.default(g, x, [3, 3], [2, 2], [1, 1], False, True, None),
                    [rnd(*gshape, seed=61, scale=0.2), rnd(*xshape, seed=62)]))
    out.append(case("_adaptive_avg_pool2d_backward", lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x),
                    [rnd(*Z((1, 64, 7, 7), (1, 20, 3, 4)), seed=63, scale=0.2), rnd(*xshape, seed=64)]))
    C2, H2, W2 = Z((64, 56, 56), (5, 7, 9))
    out.append(case("col2im", lambda c: A.col2im.default(c, [H2, W2], [3, 3], [1, 1], [1, 1], [1, 1]),
                    [rnd(1, C2 * 9, H2 * W2, seed=65)]))
    # movement
    out.append(_shared("repeat", lambda x: A.repeat.default(x, [2, 2]), [rnd(*Z((256, 64), (13, 21)), seed=66)]))
    out.append(_shared("reflection_pad1d", lambda x: A.reflection_pad1d.default(x, [2, 1]),
                       [rnd(*Z((64, 56, 56), (5, 7, 9)), seed=67)]))
    out.append(_shared("reflection_pad2d", lambda x: A.reflection_pad2d.default(x, [1, 1, 1, 1]),
                       [rnd(*Z((1, 64, 56, 56), (1, 5, 7, 9)), seed=68)]))
    out.append(_shared("reflection_pad3d", lambda x: A.reflection_pad3d.default(x, [1, 1, 1, 1, 1, 1]),
                       [rnd(*Z((1, 64, 4, 28, 28), (1, 3, 4, 5, 6)), seed=69)]))
    out.append(_shared("replication_pad2d", lambda x: A.replication_pad2d.default(x, [1, 1, 1, 1]),
                       [rnd(*Z((1, 64, 56, 56), (1, 5, 7, 9)), seed=70)]))
    out.append(_shared("replication_pad3d", lambda x: A.replication_pad3d.default(x, [1, 1, 1, 1, 1, 1]),
                       [rnd(*Z((1, 64, 4, 28, 28), (1, 3, 4, 5, 6)), seed=71)]))
    N = Z(512, 20)
    out.append(case("select_scatter", lambda a, b: A.select_scatter.default(a, b, 1, 2),
                    [rnd(N, 4, 64, seed=72), rnd(N, 64, seed=73)]))
    out.append(conv_backward_case("convolution_backward", 1, *Z((64, 56, 56, 64), (20, 9, 11, 24)), 3, 1, 1, small))
    return out
