"""Target schedule for rational sparse linear maps: i32 sums, exact host readout.

Integral weights in proved common units combine on the device. When their
bounds or instruction budget fail, separate i32 coefficient-class sums retain
an exact rational host combination before the sole quantization. Coefficients
are never approximated. Sparse maps derive solely from the source IR.
"""

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import reduce
from math import gcd, lcm

import numpy as np

from . import ops
from .exactness import accumulator_bound, require
from .requant import Readout
from .stream import Arg
from .recipes.asm import accumulate_sum


@dataclass
class Src:
    """A sparse-map input: kernel argument name, shape, code unit (scale) and uint8 storage."""
    name: str
    shape: tuple
    scale: object
    unsigned: bool = False


def _integer_plan(lm, srcs, s_out):
    """Find exact common units, never approximate rational coefficients."""
    n = lm.const.size
    factors = np.ones(n, object)
    weights = np.zeros(lm.coef.size, object)
    seed = np.zeros(n, object)
    residual = np.array([Q(v)/Q(s_out) for v in lm.const], object)
    by_output = {}
    for t, p in enumerate(lm.out):
        by_output.setdefault(int(p), []).append(t)
    work = 0
    for p, terms in by_output.items():
        coefficients = [Q(lm.coef[t])/Q(s_out) for t in terms]
        unit = abs(Q(srcs[int(lm.src[terms[0]])].scale)/Q(s_out))
        if not unit:
            return None
        # Prefer the input code unit: integer scalar multiplication/addition
        # then remains device arithmetic. A small rational common divisor can
        # also combine noninteger-ratio terms, subject to the same hard bounds.
        if not all((c/unit).denominator == 1 for c in coefficients):
            values = [unit, *coefficients]
            denominator = lcm(*(v.denominator for v in values))
            numerator = reduce(gcd, (abs(v.numerator)*(denominator//v.denominator) for v in values))
            unit = Q(numerator, denominator)
        ws = [int(c/unit) for c in coefficients]
        candidate = residual[p]/unit
        b = int(candidate) if candidate.denominator == 1 else 0
        bound = abs(b)+128*sum(abs(w) for w in ws)
        work += sum((abs(w)+126)//127 for w in ws)
        if bound >= 1 << 31 or work > 1 << 18:
            return None
        factors[p], seed[p] = unit, b
        residual[p] -= b*unit
        weights[terms] = ws
    return factors, weights, seed, residual


def _affine_tiles(s, lm, weights, seed, input_names, out):
    """Use vector c*I tiles for exact dense/broadcast affine maps when legal."""
    from .recipes.elementwise import lincomb
    from .numerics import parts127
    n = lm.const.size
    terms = []
    period = n
    for sid, name in input_names.items():
        t = np.flatnonzero(lm.src == sid)
        if not t.size:
            continue
        if t.size != n or not np.array_equal(lm.out[t], np.arange(n)) or not np.all(weights[t] == weights[t[0]]):
            return False
        indices = lm.idx[t]
        size = int(indices.max())+1
        if size <= 0 or n % size or not np.array_equal(indices, np.arange(n)%size):
            return False
        if size != n:
            if period != n and period != size:
                return False
            period = size
        w = int(weights[t[0]])
        if abs(w) > 12*127:
            return False
        terms.append((name, size, parts127(w)))
    if not terms or len(terms) > 2 or sum(len(parts) for _, _, parts in terms) > 12 or not np.all(seed == seed[0]):
        return False
    lincomb(s, [(name, 0 if size != n else period, parts) for name, size, parts in terms],
            out, n//period, period, 1.0, bias=int(seed[0]) if seed[0] else None, out_i32=True)
    return True


def kernel(op, lm, srcs, s_out, *, relu=False):
    from dataclasses import replace
    nonzero = np.array([Q(c) != 0 for c in lm.coef], bool)
    lm = replace(lm, out=lm.out[nonzero], src=lm.src[nonzero],
                 idx=lm.idx[nonzero], coef=lm.coef[nonzero])
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
            input_names[sid] = s.signed_scratch(src.name, tuple(src.shape))
    plan = _integer_plan(lm, srcs, s_out)
    if plan is not None and lm.coef.size:
        factors, weights, seed, residual = plan
        name = 'acc_combined'
        s._index[name] = len(s.args)
        s.args.append(Arg(name, 'i32', tuple(lm.out_shape), 'scratch'))
        integer_map = replace(lm, coef=weights, const=seed)
        mode = 'weighted_integer_compute'
        device_relu = (relu and len(input_names) == 1 and np.all(weights == 1) and
                       np.all(seed == 0) and np.all(residual == 0) and
                       np.array_equal(lm.out, np.arange(n)) and np.array_equal(lm.idx, np.arange(n)))
        if device_relu:
            from .recipes.elementwise import exact_byte_relu
            exact_byte_relu(s, next(iter(input_names.values())), name, n)
            mode = 'signed_byte_execute_relu'
        elif np.all(weights == 1):
            accumulate_sum(s, next(iter(input_names.values())), name, 1, n, lm.out, lm.idx,
                           np.ones(lm.out.size, np.int64), np.ones(n), out_i32=True,
                           source_id=lm.src, sources=input_names, seed=seed)
            repeated = np.bincount(lm.out, minlength=n).max(initial=0) > 1
            mode = 'integer_accumulating_mvin' if repeated or np.any(seed) else 'integer_widening'
        elif not _affine_tiles(s, lm, weights, seed, input_names, name):
            from .recipes.limbs import linear
            # The common sparse integer primitive has a configurable output.
            linear(s, integer_map, input_names, out=name)
        bound = np.abs(seed)
        for p, w in zip(lm.out, weights):
            bound[int(p)] += 128*abs(int(w))
        s.readouts.append(Readout('y', [(name, factors.reshape(lm.out_shape))],
                                  residual.reshape(lm.out_shape), relu=relu and not device_relu))
        return ops._end(op, s, form='exact_linear_i32', readout='full_i32',
                        rounding='host_exact_rational_RNE', components=1, terms=int(lm.out.size),
                        arithmetic=mode, accumulator_bounds=[dict(tile=name, accumulator_min=-int(max(bound)), accumulator_max=int(max(bound)))],
                        integer_weights=[str(w) for w in sorted(set(weights))],
                        device_constant=bool(np.any(seed)), device_relu=bool(device_relu),
                        execute_activation_proof=(dict(operation='identity_signed_byte_then_ReLU',
                            input_range=[-128,127], output_range=[0,127], shift=0,
                            clipping_is_identity=True) if device_relu else None))
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
            relu=relu,
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
        arithmetic='separate_rational_terms_host_combination',
        combination_refusal='common integer units exceed exact i32 or instruction-part budget',
    )
