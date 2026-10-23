"""Small real captures covering every support operation and spatial/channel tails.

This is a simulator qualification driver. Sizes select test inputs only; no
production schedule uses test names, provenance IDs or expected outputs.
"""

import torch
from capture import A, _shared, case, plain, rnd
from capture_rest import conv_backward_case


def rtl_set():
    r = lambda *s: rnd(*s, seed=731)
    cases = [
        case("mm", lambda a, b: A.mm.default(a, b), [r(3, 17), r(17, 5)]),
        case(
            "addmm",
            lambda b, a, w: A.addmm.default(b, a, w),
            [r(5), r(3, 17), r(17, 5)],
            kinds=[("i32", lambda s: s[1] * s[2]), "i8", "i8"],
        ),
        case("bmm", lambda a, b: A.bmm.default(a, b), [r(2, 3, 17), r(2, 17, 5)]),
        case(
            "convolution",
            lambda x, w, b: A.convolution.default(
                x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            ),
            [r(1, 3, 4, 5), r(5, 3, 3, 3), r(5)],
            kinds=["i8", "i8", ("i32", lambda s: s[0] * s[1])],
        ),
        case(
            "convolution_patch16",
            lambda x, w, b: A.convolution.default(
                x, w, b, [2, 2], [0, 0], [1, 1], False, [0, 0], 1
            ),
            [r(1, 3, 5, 7), r(5, 3, 2, 2), r(5)],
            kinds=["i8", "i8", ("i32", lambda s: s[0] * s[1])],
        ),
        case("add_Tensor", lambda a, b: A.add.Tensor(a, b), [r(3, 5), r(3, 5)]),
        case("sub_Tensor", lambda a, b: A.sub.Tensor(a, b), [r(3, 5), r(3, 5) * 0.7]),
        case("add_Scalar", lambda x: A.add.Scalar(x, 0.75), [r(3, 5)]),
        case("sub_Scalar", lambda x: A.sub.Scalar(x, 0.5), [r(3, 5)]),
        case("mul_Scalar", lambda x: A.mul.Scalar(x, 0.125), [r(3, 5)]),
        case("div_Scalar", lambda x: A.div.Scalar(x, 3), [r(3, 5)]),
        case("neg", lambda x: A.neg.default(x), [r(3, 5)]),
        case("relu", lambda x: A.relu.default(x), [r(3, 5)]),
        case("mean_dim", lambda x: A.mean.dim(x, [-1]), [r(3, 17)]),
        _shared("clone", lambda x: A.clone.default(x), [r(17)]),
        _shared("permute", lambda x: A.permute.default(x, [1, 0, 2]), [r(3, 2, 5)]),
        case("cat", lambda a, b: A.cat.default([a, b], 1), [r(2, 3, 5), r(2, 2, 5)]),
        case(
            "slice_scatter",
            lambda a, b: A.slice_scatter.default(a, b, 1, 0, 3, 1),
            [r(3, 5), r(3, 3)],
        ),
        case(
            "select_scatter",
            lambda a, b: A.select_scatter.default(a, b, 1, 2),
            [r(2, 4, 5), r(2, 5)],
        ),
        _shared(
            "constant_pad_nd",
            lambda x: A.constant_pad_nd.default(x, [0, 0], 0.0),
            [r(3, 5)],
        ),
        case(
            "constant_pad_nd_p1",
            lambda x: A.constant_pad_nd.default(x, [1, 1, 1, 1], 0.5),
            [r(1, 2, 3, 5)],
        ),
        case(
            "full_like", lambda x: A.full_like.default(x, -1.25), [r(3, 5)], s_out=0.02
        ),
        case(
            "full",
            lambda x: A.full.default(list(x.shape), -1.7, dtype=torch.float32),
            [r(3, 5)],
            s_out=0.05,
        ),
        case("fill_Scalar", lambda x: A.fill.Scalar(x, 0.5), [r(3, 5)], s_out=0.0125),
        plain(
            "_to_copy",
            lambda x: A._to_copy.default(x, dtype=torch.int32),
            [torch.arange(-9, 8, dtype=torch.int8)],
        ),
        case(
            "avg_pool1d", lambda x: A.avg_pool1d.default(x, [3], [2], [1]), [r(1, 3, 7)]
        ),
        case(
            "avg_pool2d",
            lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]),
            [r(1, 3, 5, 7)],
        ),
        case(
            "avg_pool3d",
            lambda x: A.avg_pool3d.default(x, [3, 3, 3], [2, 2, 2], [1, 1, 1]),
            [r(1, 2, 3, 4, 5)],
        ),
        case(
            "adaptive_avg_pool1d",
            lambda x: A.adaptive_avg_pool1d.default(x, [3]),
            [r(1, 3, 7)],
        ),
        case(
            "_adaptive_avg_pool2d",
            lambda x: A._adaptive_avg_pool2d.default(x, [2, 3]),
            [r(1, 3, 5, 7)],
        ),
        case(
            "_adaptive_avg_pool3d",
            lambda x: A._adaptive_avg_pool3d.default(x, [2, 2, 3]),
            [r(1, 2, 3, 4, 5)],
        ),
        case(
            "avg_pool2d_backward",
            lambda g, x: A.avg_pool2d_backward.default(
                g, x, [3, 3], [2, 2], [1, 1], False, True, None
            ),
            [r(1, 3, 3, 4), r(1, 3, 5, 7)],
        ),
        case(
            "_adaptive_avg_pool2d_backward",
            lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x),
            [r(1, 3, 2, 3), r(1, 3, 5, 7)],
        ),
        case(
            "col2im",
            lambda x: A.col2im.default(x, [3, 5], [3, 3], [1, 1], [1, 1], [1, 1]),
            [r(1, 18, 15)],
        ),
        _shared("repeat", lambda x: A.repeat.default(x, [2, 2]), [r(3, 5)]),
        _shared(
            "reflection_pad1d",
            lambda x: A.reflection_pad1d.default(x, [2, 1]),
            [r(2, 3, 5)],
        ),
        _shared(
            "reflection_pad2d",
            lambda x: A.reflection_pad2d.default(x, [1, 1, 1, 1]),
            [r(1, 2, 3, 5)],
        ),
        _shared(
            "reflection_pad3d",
            lambda x: A.reflection_pad3d.default(x, [1, 1, 1, 1, 1, 1]),
            [r(1, 2, 3, 4, 5)],
        ),
        _shared(
            "replication_pad2d",
            lambda x: A.replication_pad2d.default(x, [1, 1, 1, 1]),
            [r(1, 2, 3, 5)],
        ),
        _shared(
            "replication_pad3d",
            lambda x: A.replication_pad3d.default(x, [1, 1, 1, 1, 1, 1]),
            [r(1, 2, 3, 4, 5)],
        ),
        conv_backward_case("convolution_backward", 1, 3, 4, 5, 5, 3, 1, 1, True),
    ]
    return cases
