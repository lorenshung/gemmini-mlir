"""Target schedule for rational sparse linear maps: i32 sums, exact host readout.

Coefficients are never approximated into int8 weights. Independent coefficient
classes receive independent i32 tiles; the CPU combines their rational values
before the sole quantization. Sparse maps derive solely from the source IR.
"""

from fractions import Fraction as Q

import numpy as np

from . import ops
from .exactness import accumulator_bound, require
from .requant import Readout
from .stream import Arg
from .recipes.asm import accumulate_sum


def kernel(op, lm, srcs, s_out):
    n = int(np.prod(lm.out_shape)) if lm.out_shape else 1
    require(
        n > 0 and all(all(d > 0 for d in src.shape) for src in srcs.values()),
        "empty or dynamic linear map",
    )
    args = [
        Arg(srcs[k].name, "i8", tuple(srcs[k].shape), "input") for k in sorted(srcs)
    ]
    args.append(Arg("y", "i8", tuple(lm.out_shape), "output"))
    s = ops._begin(args)
    components = []
    bounds = []
    input_names = {sid: src.name for sid, src in srcs.items()}
    for sid, src in srcs.items():
        if src.unsigned:
            packed = src.name + "__signed"
            s._index[packed] = len(s.args)
            s.args.append(Arg(packed, "i8", tuple(src.shape), "scratch"))
            s.packs.append((src.name, packed))
            input_names[sid] = packed
    for sid in sorted(srcs):
        # Within each output, combine equal exact coefficients. Their ordinal
        # class may have a different factor at another output (pool divisors).
        groups = {}
        for term in np.flatnonzero(lm.src == sid):
            p, coefficient = int(lm.out[term]), Q(lm.coef[term]) / Q(s_out)
            if coefficient:
                groups.setdefault(p, {}).setdefault(coefficient, []).append(
                    int(lm.idx[term])
                )
        count = max((len(g) for g in groups.values()), default=0)
        for rank in range(count):
            out, idx = [], []
            factors = np.full(n, Q(0), dtype=object)
            for p, classes in groups.items():
                if rank >= len(classes):
                    continue
                coefficient, indices = sorted(classes.items())[rank]
                factors[p] = coefficient
                out.extend([p] * len(indices))
                idx.extend(indices)
            name = f"acc_{sid}_{rank}"
            s.args.append(Arg(name, "i32", tuple(lm.out_shape), "scratch"))
            s._index[name] = len(s.args) - 1
            o, i = np.asarray(out, np.int64), np.asarray(idx, np.int64)
            weights = np.ones(len(out), np.int64)
            bound = int(np.bincount(o, minlength=n).max(initial=0)) * 128
            accumulator_bound(bound)
            bounds.append(
                dict(tile=name, accumulator_min=-bound, accumulator_max=bound)
            )
            accumulate_sum(
                s, input_names[sid], name, 1, n, o, i, weights, np.ones(n), out_i32=True
            )
            components.append((name, factors.reshape(lm.out_shape)))
    s.readouts.append(
        Readout(
            "y",
            components,
            np.asarray(lm.const, dtype=object).reshape(lm.out_shape) / Q(s_out),
        )
    )
    return ops._end(
        op,
        s,
        form="exact_linear_i32",
        readout="full_i32",
        rounding="host_exact_rational_RNE",
        components=len(components),
        accumulator_bounds=bounds,
        terms=int(lm.out.size),
    )
