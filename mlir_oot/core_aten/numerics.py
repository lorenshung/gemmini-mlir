"""Exact compile-time constant quantization and integer instruction weights."""

from __future__ import annotations

import math


def code(value: float, scale: float, lo: int = -128, hi: int = 127) -> int:
    """One exact rational RNE and clamp, evaluated at compile time."""
    from .exactness import rational, quantized, require
    require(math.isfinite(value) and math.isfinite(scale) and scale > 0, "invalid constant quantization")
    return quantized(rational(value) / rational(scale), lo, hi)



def parts127(w: int) -> list[int]:
    """Split |w| <= 12*127 into int8 c*I weights; the sign follows w."""
    from .exactness import require
    require(abs(w) <= 12 * 127, "integer weight exceeds the instruction-part budget")
    sign = -1 if w < 0 else 1
    w = abs(w)
    out = []
    while w > 0:
        out.append(sign * min(127, w))
        w -= min(127, w)
    return out
