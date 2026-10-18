"""Quantized numerics for the Core ATen recipes (per-tensor symmetric int8, zero point 0).

Candidate synthesis never grants device admission; exactness.py must prove it.
The device rounds once per scale unit: ``sat8(RNE(fp32(v) * fp32(scale)))``.  The
compile-time constants here (rational integer weights, integer constant multipliers,
fill codes) are the gate-3 fixes Q1/Q2/Q9/Q10/Q13 of ``analysis/recipe_corrections.md``.
"""

from __future__ import annotations

import math

import numpy as np

WMAX = 381          # c*I parts of at most 3 x 127


def code(value: float, scale: float, lo: int = -128, hi: int = 127) -> int:
    """Compile-time quantized code of a constant (Q13), with the reference quantizer's own
    arithmetic: RNE(fp32(v) * fp32(1 / s)), saturated (``quantize_per_tensor``)."""
    from .exactness import rational, quantized, require
    require(math.isfinite(value) and math.isfinite(scale) and scale > 0, "invalid constant quantization")
    return quantized(rational(value) / rational(scale), lo, hi)



def parts127(w: int) -> list[int]:
    """Split |w| <= WMAX into int8 c*I weights; the sign follows w."""
    from .exactness import require
    require(abs(w) <= 12 * 127, "integer weight exceeds the instruction-part budget")
    sign = -1 if w < 0 else 1
    w = abs(w)
    out = []
    while w > 0:
        out.append(sign * min(127, w))
        w -= min(127, w)
    return out


def rational_weights(alpha: float, beta: float, wmax: int = WMAX) -> tuple[int, int]:
    """Integer (Wa, Wb) with Wa / Wb the best approximation of alpha / beta (both > 0)."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("rational weights take positive magnitudes")
    best = None
    r = alpha / beta
    for wb in range(1, wmax + 1):
        wa = round(wb * r)
        if wa < 1 or wa > wmax:
            continue
        err = abs(wa / wb - r) / r
        key = (err, -max(wa, wb))
        if best is None or key < best[0]:
            best = (key, wa, wb)
    if best is None:
        raise ValueError(f"no integer weights within {wmax} for ratio {r}")
    return best[1], best[2]


def best_int_multiplier(v: float, gmax: int = WMAX) -> int:
    """G <= gmax with G * v closest to an integer (relative to G), preferring large G (Q9)."""
    best = None
    for g in range(1, gmax + 1):
        e = abs(g * v - round(g * v)) / g
        key = (round(e, 12), -g)
        if best is None or key < best[0]:
            best = (key, g)
    return best[1]
