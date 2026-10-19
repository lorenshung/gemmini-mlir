"""Target accumulator intervals for exact host integer normal forms."""

from operator import index

def interval(readout, coeffs, meta):
    """Conservative combined numerator interval from recorded tile bounds."""
    tiles = {b['tile']: b for key in ('accumulator_bounds', 'zero_point_corrections')
             for b in meta.get(key, [])}
    lo = hi = coeffs[-1]
    sources = []
    safe = -(1 << 63) < lo < 1 << 63
    for (name, _), weight in zip(readout.terms, coeffs):
        b = tiles.get(name)
        # Legacy single accumulator kernels record a top-level interval. Extra
        # zero-point tiles must never inherit that primary accumulator's bound.
        if b is None and name == readout.terms[0][0] and 'accumulator_min' in meta:
            b = meta
        low, high = ((index(b['accumulator_min']), index(b['accumulator_max'])) if b is not None
                     else (-(1 << 31), (1 << 31) - 1))
        if not -(1 << 31) <= low <= high < 1 << 31:
            raise ValueError('invalid recorded accumulator interval')
        pl, ph = sorted((weight * low, weight * high))
        lo += pl
        hi += ph
        safe &= (-(1 << 63) < min(pl, lo) <= max(ph, hi) < 1 << 63
                 and -(1 << 63) <= weight < 1 << 63)
        sources.append(dict(tile=name, accumulator_min=low, accumulator_max=high,
                            bound_source='kernel_meta' if b is not None else 'full_int32_storage'))
    return lo, hi, sources, safe

