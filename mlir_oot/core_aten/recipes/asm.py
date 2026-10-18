"""Accumulate-on-MVIN weighted sums (``qx_avgpool_nhwc``, pool backwards, ``qx_col2im_dma``).

y[r, :] = sat8(RNE(rho_r * sum_t W_t x[in_t, :])) over rows of a shared contiguous channel
block of C elements (C = 1 views every element as a row).  The int32 sums are exact: every
term is an int8 MVIN into the accumulator (shrunk state, unit scale: the MVIN scale unit
saturates to int8, so an integer weight W is W accumulating MVINs of the row), accumulated
onto a zero-filled tile (zero-fill through a shrink=0 state, HW2;
zero-fill followed by accumulates is the ordering the RTL guarantees, HW3).  Consecutive
output rows whose t-th terms read input rows in arithmetic progression (any non-negative
DRAM stride, including 0 for a broadcast) share one MVIN.  Output rows are processed in
chunks of one accumulator half; readouts are grouped by equal scale.
"""

from __future__ import annotations

import numpy as np

from ..stream import ACC, DIM, NO_ACT, Stream


def accumulate_sum(s: Stream, x: str, y: str, C: int, rows_out: int, out_row: np.ndarray, in_row: np.ndarray,
                   weight: np.ndarray, rho: np.ndarray, *, x_pitch: int | None = None,
                   y_pitch: int | None = None, out_i32: bool = False) -> None:
    """``out_row``/``in_row``/``weight``: one entry per term; ``rho``: readout scale per output row."""
    xp = C if x_pitch is None else x_pitch
    yp = C if y_pitch is None else y_pitch
    if out_i32:
        yp *= 4
    nct = (C + DIM - 1) // DIM
    R = max(1, 512 // nct)
    order = np.lexsort((in_row, out_row))
    out_row, in_row, weight = out_row[order], in_row[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(out_row)) + 1] if out_row.size else np.zeros(0, np.int64)
    rank = np.arange(out_row.size) - np.repeat(starts, np.diff(np.r_[starts, out_row.size])) \
        if out_row.size else np.zeros(0, np.int64)
    s.config_ld(2, 0, 1.0, shrink=False)
    ld = None
    st = None
    for chunk, r0 in enumerate(range(0, rows_out, R)):
        rr = min(R, rows_out - r0)
        half = (chunk & 1) * 512
        for ct in range(nct):
            cw = min(DIM, C - ct * DIM)
            for b in range(0, rr, DIM):
                s.mvin(None, ACC(half + ct * R + b), cw, min(DIM, rr - b), 2)
        sel = (out_row >= r0) & (out_row < r0 + rr)
        o, i, w, k = out_row[sel], in_row[sel], weight[sel], rank[sel]
        for t in range(int(k.max()) + 1 if k.size else 0):
            m = k == t
            ot, it, wt = o[m], i[m], w[m]
            j = 0
            while j < ot.size:
                e = j
                d = None
                while e + 1 < ot.size and e + 1 - j < DIM and ot[e + 1] == ot[e] + 1 and wt[e + 1] == wt[j]:
                    dd = it[e + 1] - it[e]
                    if dd < 0 or (d is not None and dd != d):
                        break
                    d = dd
                    e += 1
                d = 0 if d is None else int(d)
                cfg = (d * xp, 1.0)
                if ld != cfg:
                    s.config_ld(0, d * xp, 1.0, shrink=True)
                    ld = cfg
                for _ in range(int(wt[j])):
                    for ct in range(nct):
                        cw = min(DIM, C - ct * DIM)
                        s.mvin(s.at(x, int(it[j]) * xp + ct * DIM),
                               ACC(half + ct * R + int(ot[j]) - r0, accumulate=True), cw, e - j + 1, 0)
                j = e + 1
        for ct in range(nct):
            cw = min(DIM, C - ct * DIM)
            r = 0
            while r < rr:
                e = r
                while e + 1 < rr and e + 1 - r < DIM and rho[r0 + e + 1] == rho[r0 + r]:
                    e += 1
                cfg = (yp, 1.0 if out_i32 else float(rho[r0 + r]))
                if st != cfg:
                    s.config_st(yp, NO_ACT, cfg[1])
                    st = cfg
                s.mvout(s.at(y, (r0 + r) * yp + ct * DIM * (4 if out_i32 else 1)), ACC(half + ct * R + r, full=out_i32), cw, e - r + 1)
                r = e + 1
