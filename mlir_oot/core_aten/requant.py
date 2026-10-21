"""Exact int32 readout ABI and portable host references.

Device operations return full-width signed i32. Factors are reduced rationals,
serialized as decimal numerator/denominator strings; zero point is applied AFTER
RNE. No floating intermediate participates in a code decision.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class Readout:
    out: str
    terms: list[tuple[str, object]]
    constant: object = 0
    relu: bool = False
    zero_point: int = 0
    quant_min: int = -128
    quant_max: int = 127
    permutations: dict = None
    broadcasts: dict = None
    after: int | None = None
    placement: tuple | dict | None = None  # scalar-parameter destination region, in elements
    axis: int = -1


def expanded(value, shape, axis=-1):
    a = np.asarray(value, dtype=object)
    size = int(np.prod(shape))
    if a.size == size:
        return a.reshape(-1)
    if a.ndim == 1 and shape and a.size == shape[axis]:
        dims = [1]*len(shape)
        dims[axis] = a.size
        a = a.reshape(dims)
    return np.broadcast_to(a, shape).reshape(-1)


def parameters(readout, value, shape):
    return expanded(value, shape, readout.axis)


def evaluate(
    values,
    numerator,
    denominator,
    *,
    zero_point=0,
    quant_min=-128,
    quant_max=127,
    relu=False,
):
    """Pure-Python reference, including negative ties and asymmetric parity."""
    factor = Fraction(int(numerator), int(denominator))

    def scalar(x):
        real = int(x) * factor
        if relu:
            real = max(real, 0)
        return min(quant_max, max(quant_min, round(real) + zero_point))

    dt = np.uint8 if quant_min >= 0 else np.int8
    a = np.asarray(values)
    return np.array([scalar(x) for x in a.flat], dtype=dt).reshape(a.shape)


def apply(stream, buffers, readouts=None):
    """Instruction-model host epilogue, separate from the Gemmini datapath."""
    by_name = {a.name: buffers[i] for i, a in enumerate(stream.args)}
    for readout in stream.readouts if readouts is None else readouts:
        out = by_name[readout.out]
        if readout.placement is not None:
            if isinstance(readout.placement,dict):
                from math import prod
                p = readout.placement
                count = prod(p['shape'])
                logical = np.arange(count,dtype=np.int64)
                positions = np.full(count,p['offset'],dtype=np.int64)
                for axis,(extent,stride) in enumerate(zip(p['shape'],p['strides'])):
                    positions += (logical//prod(p['shape'][axis+1:])%extent)*stride
            else:
                count, block, offset, stride = readout.placement
                positions = np.arange(count,dtype=np.int64)
                positions = positions//block*stride + positions%block + offset
            values = np.full(count,readout.constant,dtype=object)
            for name,factor in readout.terms:
                values += by_name[name].reshape(-1)[positions].astype(object)*factor
            if readout.relu:
                values = np.maximum(values, 0)
            out.reshape(-1)[positions] = [min(readout.quant_max,max(readout.quant_min,round(Fraction(v))+readout.zero_point)) for v in values]
            continue
        values = parameters(readout, readout.constant, out.shape).copy()
        for name, factor in readout.terms:
            raw = by_name[name]
            if readout.permutations and name in readout.permutations:
                shp, perm = readout.permutations[name]
                raw = raw.reshape(shp).transpose(perm)
            if readout.broadcasts and name in readout.broadcasts:
                raw = raw.reshape(readout.broadcasts[name])
            raw_values = expanded(raw, out.shape)
            values += raw_values.astype(object)*parameters(readout, factor, out.shape)
        if readout.relu:
            values = np.maximum(values, 0)
        zps, lows, highs = [parameters(readout, v, out.shape) for v in
                           (readout.zero_point, readout.quant_min, readout.quant_max)]
        out.flat[:] = np.array([min(int(hi), max(int(lo), round(Fraction(v))+int(zp)))
                                     for v,zp,lo,hi in zip(values,zps,lows,highs)], dtype=np.int64).astype(out.dtype)
