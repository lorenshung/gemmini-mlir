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
    """Compare every monotone transition, including ties, over the entire domain.

    Float conversion, multiplication, RNE and clamp are monotone. Equality of
    all 255 transition locations proves equality at every integer, even above
    2**24 where conversion to binary32 is inexact.
    """
    scale = rational(scale)
    if scale == 0:
        return True
    device_scale = np.float32(float(scale))
    require(math.isfinite(float(device_scale)) and device_scale != 0, 'nonfinite or zero hardware scale')
    require(-(1 << 31) <= lo <= hi < (1 << 31), 'invalid accumulator interval')
    direction = 1 if scale > 0 else -1
    def exact(x):
        return direction * quantized((max(x, 0) if relu else x) * scale)
    def hardware(x):
        x = max(x, 0) if relu else x
        return direction * int(min(127, max(-128, np.rint(np.float32(x) * device_scale))))
    def transition(fn, q):
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if fn(mid) >= q:
                right = mid
            else:
                left = mid + 1
        return left
    # For decreasing original functions direction reverses the values, not x.
    for q in range(-128, 129):
        a, b = transition(exact, q), transition(hardware, q)
        require(a == b, f'FP32 readout differs at accumulator transition {q}: exact={a}, device={b}')
    return True


def prove_affine(coefficients, constant, weights, bias, scale, relu=False):
    """Exhaustive proof over one/two signed byte inputs, including all rounding ties."""
    coefficients = tuple(map(rational, coefficients))
    constant, scale = rational(constant), rational(scale)
    require(1 <= len(weights) <= 2 and len(weights) == len(coefficients), 'unsupported affine arity')
    bound = abs(bias) + sum(128 * abs(w) for w in weights)
    accumulator_bound(bound)
    # Fast sufficient proof, followed by a complete finite-domain check when
    # integer weight synthesis has approximated the source coefficients.
    if constant == bias * scale and all(c == w * scale for c, w in zip(coefficients, weights)):
        return prove_readout(scale, -bound, bound, relu)
    import itertools
    for values in itertools.product(range(-128, 128), repeat=len(weights)):
        source = sum(c * x for c, x in zip(coefficients, values)) + constant
        acc = sum(w * x for w, x in zip(weights, values)) + bias
        expected = quantized(max(source, 0) if relu else source)
        got = min(127, max(-128, int(np.rint(np.float32(max(acc, 0) if relu else acc) * np.float32(float(scale))))))
        require(got == expected, f'affine mismatch at codes {values}: exact={expected}, device={got}')
    return True
