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


def evaluate(values, numerator, denominator, *, zero_point=0, quant_min=-128, quant_max=127, relu=False):
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


def thresholds(numerator, denominator, *, zero_point=0, quant_min=-128, quant_max=127, relu=False):
    """Exact monotone transition table for any positive rational, full i32 domain.

    Thresholds are i64: INT32_MAX+1 denotes an unreachable code transition.
    The C implementation needs no wide multiply or floating point.
    """
    scale = Fraction(int(numerator), int(denominator))
    if scale <= 0:
        raise ValueError('positive readout factor required')
    result = []
    for code in range(quant_min + 1, quant_max + 1):
        q = code - zero_point
        boundary = Fraction(2*q-1, 2) / scale
        # At the half tie, the upper code wins iff its unshifted code is even.
        t = -(-boundary.numerator // boundary.denominator)
        if boundary.denominator == 1 and q % 2:
            t += 1
        if relu and q <= 0:
            t = -(1 << 31)
        result.append(min(1 << 31, max(-(1 << 31), t)))
    return result


def c_reference(symbol='gemmini_exact_requant_i32'):
    """C reference ABI: table from thresholds(), output is byte storage bits."""
    return f'''#include <stdint.h>
#include <stddef.h>
void {symbol}(const int32_t *acc, uint8_t *out, size_t count,
             const int64_t *thresholds, int quant_min, int quant_max) {{
  for (size_t i=0; i<count; ++i) {{
    unsigned lo=0, hi=(unsigned)(quant_max-quant_min);
    while (lo<hi) {{
      unsigned mid=lo+(hi-lo)/2;
      if ((int64_t)acc[i]>=thresholds[mid]) lo=mid+1; else hi=mid;
    }}
    out[i]=(uint8_t)(quant_min+(int)lo);
  }}
}}
/* Signed 64-bit rational factors: i32 * i64 fits exactly in i128.
   Arbitrary-size positive factors use the generated transition table above. */
void {symbol}_ratio(const int32_t *acc, uint8_t *out, size_t count,
                    int64_t numerator, uint64_t denominator, int zero_point,
                    int quant_min, int quant_max) {{
  if (!denominator) __builtin_trap();
  for (size_t i=0; i<count; ++i) {{
    __int128 product=(__int128)acc[i]*numerator;
    __int128 magnitude=product<0 ? -product : product;
    __int128 quotient=magnitude/denominator;
    uint64_t remainder=(uint64_t)(magnitude%denominator);
    if (remainder>denominator-remainder ||
        (remainder==denominator-remainder && (quotient&1))) ++quotient;
    __int128 code=(product<0 ? -quotient : quotient)+zero_point;
    if (code<quant_min) code=quant_min;
    if (code>quant_max) code=quant_max;
    out[i]=(uint8_t)code;
  }}
}}
'''


def apply(stream, buffers, readouts=None):
    """Instruction-model host epilogue, separate from the Gemmini datapath."""
    by_name = {a.name: buffers[i] for i, a in enumerate(stream.args)}
    for readout in stream.readouts if readouts is None else readouts:
        out = by_name[readout.out]
        constant=np.asarray(readout.constant,dtype=object)
        values=(constant.reshape(out.shape) if constant.size==out.size else np.broadcast_to(constant,out.shape)).copy()
        for name, factor in readout.terms:
            raw = by_name[name]
            if readout.permutations and name in readout.permutations:
                shp,perm = readout.permutations[name]
                raw = raw.reshape(shp).transpose(perm)
            if readout.broadcasts and name in readout.broadcasts:
                raw = raw.reshape(readout.broadcasts[name])
            if raw.size==out.size: raw=raw.reshape(out.shape)
            factor=np.asarray(factor,dtype=object)
            if factor.size==out.size: factor=factor.reshape(out.shape)
            values += np.broadcast_to(raw,out.shape).astype(object) * factor
        if readout.relu:
            values = np.maximum(values, 0)
        out.flat[:] = np.array([min(readout.quant_max, max(readout.quant_min, round(Fraction(v))+readout.zero_point)) for v in values.flat],dtype=np.int64).astype(out.dtype)
