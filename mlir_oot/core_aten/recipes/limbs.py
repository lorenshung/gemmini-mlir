"""Signed sparse integer maps: K=1 WS products, exact accumulator SRAM addition."""
import numpy as np

from ..stream import ACC, SP, DIM


def linear(s, lm, names):
    n = lm.const.size
    s.const('seed', np.asarray(lm.const, np.int32))
    coefficients = sorted({int(c) for c in lm.coef})
    parts = {c: ([127 if c > 0 else -127] * (abs(c) // 127) +
                  ([((1 if c > 0 else -1) * (abs(c) % 127))] if abs(c) % 127 else []))
             for c in coefficients}
    weights = sorted({v for vs in parts.values() for v in vs})
    s.const('weights', np.asarray(weights or [1], np.int8))
    s.config_ex()
    s.config_ld(1, 0)
    s.config_ld(2, 4)
    s.config_st(4)
    # A fence at each reused tile boundary covers DMA/EX/STORE ordering and
    # makes arbitrary gathers independent of load queue issue order.
    for base in range(0, n, DIM):
        rows = min(DIM, n-base)
        s.fence()
        s.mvin(s.at('seed', base*4), ACC(0), 1, rows, 2)
        selected = np.flatnonzero((lm.out >= base) & (lm.out < base+rows))
        for term in selected:
            p, sid, idx, c = (int(lm.out[term]), int(lm.src[term]),
                              int(lm.idx[term]), int(lm.coef[term]))
            for weight in parts[c]:
                s.fence()
                s.config_ld(0, 0)
                s.mvin(s.at(names[sid], idx), SP(32), 1, 1, 0)
                s.mvin(s.at('weights', weights.index(weight)), SP(0), 1, 1, 1)
                s.matmul_tile(SP(32), SP(0), ACC(p-base, accumulate=True), 1, 1, 1)
        s.fence()
        s.mvout(s.at('out', base*4), ACC(0, full=True), 1, rows)
