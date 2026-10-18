"""Compile-time certificate for the integer host readout boundary.

This prototype is isolated for promotion to Merlin's shared host code generator.
The target supplies accumulator intervals; no calibration values participate.
"""
from functools import lru_cache
from fractions import Fraction
from itertools import product
from math import prod


def rne(n, d):
    q, r = divmod(abs(n), d)
    q += r > d - r or (r == d - r and q & 1)
    return -q if n < 0 else q


@lru_cache(maxsize=4096)
def prove(lo, hi, denominator, quant_min, quant_max, zero_point, relu):
    """Prove floor(|n|/d), then exact RNE, for EVERY n in [lo, hi].

    M=ceil(2**s/d), e=M*d-2**s. If A*e < 2**s, the error
    in n*M/2**s is <1/d. Since n/d has fractional part <=(d-1)/d,
    floor(n*M/2**s)=floor(n/d) for all 0<=n<=A. Products fit u128.
    r=n-q*d is then exact. Comparing r with d-r avoids doubling overflow.
    For every tie n=(2*k+1)*d/2, the correction is exactly k&1;
    RNE(-x)=-RNE(x), so this covers both signs and all tie parities.
    """
    d = denominator
    a = max(abs(lo), abs(hi))
    if not (0 < d < 1 << 63 and -(1 << 63) < lo <= hi < 1 << 63):
        return dict(method="comparison_search", reason="integer numerator or divisor exceeds signed i64")
    if a // d >= (1 << 63) - 1:
        return dict(method="comparison_search", reason="rounded quotient exceeds signed i64")
    for shift in range(128):
        m = ((1 << shift) + d - 1) // d
        error = m * d - (1 << shift)
        if m < 1 << 64 and a * error < 1 << shift:
            break
    else:
        return dict(method="comparison_search", reason="no certified u64 reciprocal")

    def fast(n):
        mag = abs(n)
        q = mag * m >> shift
        r = mag - q * d
        q += r > d - r or (r == d - r and q & 1)
        return -q if n < 0 else q

    def code(n, fn):
        if relu:
            n = max(n, 0)
        return min(quant_max, max(quant_min, fn(n) + zero_point))

    # Every visible rounding/clamp edge, plus extremes and both signs. Outside
    # the clamp interval, the quotient theorem still proves the unbounded RNE.
    points = {lo, hi, 0, -1, 1, -d, d, -(d // 2), d // 2}
    for k in range(quant_min - zero_point - 1, quant_max - zero_point + 2):
        edge = (2 * k + 1) * d // 2
        for x in (edge, -edge):
            points.update(range(x - 2, x + 3))
    points = {n for n in points if lo <= n <= hi}
    exhaustive = hi - lo <= 65536
    samples = range(lo, hi + 1) if exhaustive else sorted(points)
    count = 0
    for n in samples:
        if (fast(n) != round(Fraction(n, d)) or
                code(n, fast) != code(n, lambda v: rne(v, d))):
            return dict(method="comparison_search", reason="reciprocal verification failed")
        count += 1
    return dict(method="exact_reciprocal_multiply_shift", numerator_min=str(lo),
                numerator_max=str(hi), denominator=str(d), multiplier=str(m), shift=shift,
                max_magnitude=str(a), reciprocal_error=str(error),
                error_product=str(a * error), error_limit=str(1 << shift),
                quotient_proof="A*(M*d-2^s)<2^s; floor(n*M/2^s)=floor(n/d)",
                rounding_proof="exact remainder; r>d-r or (r=d-r and quotient odd); signed symmetry",
                ties="all n=+/-((2*k+1)*d/2), when integral, proved symbolically for every k",
                verification="exhaustive" if exhaustive else "formal_bound_and_boundary_tests",
                tested_values=count, boundary_values=len(points))


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
        low, high = ((int(b['accumulator_min']), int(b['accumulator_max'])) if b is not None
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


@lru_cache(maxsize=4096)
def _verify_tuples(coeffs, bounds, denominator, m, shift):
    count = prod(hi-lo+1 for lo, hi in bounds)
    if count > 65536:
        return None
    ranges = [range(lo, hi+1) for lo, hi in bounds]
    for values in product(*ranges):
        n = coeffs[-1] + sum(a*w for a, w in zip(values, coeffs))
        q = abs(n)*m >> shift
        r = abs(n)-q*denominator
        q += r > denominator-r or (r == denominator-r and q & 1)
        if (-q if n < 0 else q) != round(Fraction(n, denominator)):
            return -1
    return count


def verify_reachable(proof, coeffs, sources, denominator):
    """Exhaust every tuple when the recorded accumulator domain is feasible."""
    count = _verify_tuples(tuple(coeffs), tuple((b['accumulator_min'], b['accumulator_max'])
                          for b in sources), denominator,
                          int(proof['multiplier']), proof['shift'])
    if count == -1:
        proof.clear()
        proof.update(method="comparison_search", reason="reachable-domain verification failed")
    elif count is not None:
        proof['reachable_verification'] = 'exhaustive_accumulator_tuples'
        proof['reachable_values_tested'] = count
