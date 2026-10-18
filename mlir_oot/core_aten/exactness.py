"""Admission proofs for Gemmini's binary32 scale units and bounded i32 SRAM.

These prove target instructions against rational semantics; they do not implement a
host requantizer. Negative evidence is a device refusal, never an error tolerance.
"""
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np


def rational(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def require(condition, reason):
    if not condition:
        from .match import Decline
        raise Decline('exact arithmetic: ' + reason)


def quantized(value, lo=-128, hi=127):
    return min(hi, max(lo, round(value)))


def accumulator_bound(bound):
    require(0 <= bound <= (1 << 31) - 1, 'unproved i32 accumulator overflow bound')


@lru_cache(maxsize=4096)
def prove_readout(scale, lo, hi, relu=False):
    """Prove exact conversion AND multiplication, before the quantize rounding.

    A binary32 has 24 significant bits. The reduced integer mantissa times
    every integer in the interval must fit that precision. Merely agreeing
    after saturation or RNE does not establish this invariant.
    """
    scale = rational(scale)
    require(-(1 << 31) <= lo <= hi < (1 << 31), 'invalid accumulator interval')
    if relu:
        lo, hi = max(0, lo), max(0, hi)
    maximum = max(abs(lo), abs(hi))
    require(maximum <= (1 << 24), 'i32 to FP32 conversion is not exact over the interval')
    device = np.float32(float(scale))
    require(math.isfinite(float(device)) and rational(float(device)) == scale,
            'scale is not exactly representable in FP32')
    if scale == 0 or maximum == 0:
        return True
    numerator = abs(scale.numerator)
    while numerator % 2 == 0:
        numerator //= 2
    require(maximum * numerator <= (1 << 24), 'FP32 product is not exact over the interval')
    # Multiplication must not underflow or overflow either.
    require(abs(scale) >= rational(float(np.nextafter(np.float32(0), np.float32(1))))
            and maximum * abs(scale) <= rational(float(np.finfo(np.float32).max)),
            'FP32 product exponent is outside the exact finite range')
    return True


def prove_affine(coefficients, constant, weights, bias, scale, relu=False):
    """Require the exact integer form, then prove its FP32 product exact."""
    coefficients = tuple(map(rational, coefficients))
    constant, scale = rational(constant), rational(scale)
    require(1 <= len(weights) <= 2 and len(weights) == len(coefficients), 'unsupported affine arity')
    bound = abs(bias) + sum(128 * abs(w) for w in weights)
    accumulator_bound(bound)
    require(constant == bias * scale and all(c == w * scale for c, w in zip(coefficients, weights)),
            'affine integer form changes the exact source coefficients')
    return prove_readout(scale, -bound, bound, relu)
