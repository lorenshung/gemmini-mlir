"""Convolution backward contractions (``em_conv_tr``, ``qx_conv_wgrad``), channels-last.

* ``conv_transpose``: grad_input = transposed convolution of grad_output in scatter form.
  Per output tile (n, ih, 16 input columns, 16 input channels) the accumulator is zeroed;
  for every tap whose grad row exists, the run of grad pixels landing in the tile is one
  MVIN and the mesh result rows are scattered into the tile with CONFIG_EX c_stride = s_w
  (no zero-insertion).  Weights are [tap][Cout][Cin] (the forward Cout is the K dim).
* ``conv_wgrad``: grad_weight[tap][co][ci] = sum over output pixels of grad[co, p] x[p(tap), ci]
  with grad in NCHW (rows co, pixels contiguous: the A operand) and x channels-last (strided
  rows: the B operand, out-of-bounds lanes zero-filled as disjoint runs); K = output pixels
  of one row, 16 deep per COMPUTE; one readout scale s_g s_x / s_gw (gate-3 Q11).
"""

from __future__ import annotations

from ..stream import ACC, DIM, NO_ACT, SP, Stream
from .conv import ConvGeom


def _cdiv(a, b):
    return -(-a // b)


def _fdiv(a, b):
    return a // b


def conv_transpose(s: Stream, g: ConvGeom, gy: str, wb: str, gx: str, scale: float) -> None:
    """gy: NHWC [N, OH, OW, F]; wb: [KH, KW, F, C]; gx: NHWC [N, H, W, C] int8 at ``scale``."""
    F, C = g.cout, g.cin
    s.config_ex(c_stride=g.sw)
    s.config_ld(0, F)
    s.config_ld(1, C)
    s.config_ld(2, 0, 1.0, shrink=False)
    s.config_st(C, NO_ACT, scale)
    t = u = 0
    for n in range(g.n):
        for ih in range(g.h):
            for iw0 in range(0, g.w, DIM):
                mr = min(DIM, g.w - iw0)
                for c0 in range(0, C, DIM):
                    nc = min(DIM, C - c0)
                    acc = (t & 1) * DIM
                    t += 1
                    s.mvin(None, ACC(acc), nc, mr, 2)
                    for kh in range(g.kh):
                        nh = ih + g.ph - kh * g.dh
                        oh = _fdiv(nh, g.sh)
                        if oh * g.sh != nh or not 0 <= oh < g.oh:
                            continue
                        for kw in range(g.kw):
                            off = kw * g.dw - g.pw            # iw = ow * sw + off
                            lo = max(0, _cdiv(iw0 - off, g.sw))
                            hi = min(g.ow - 1, _fdiv(iw0 + mr - 1 - off, g.sw))
                            if lo > hi:
                                continue
                            v = hi - lo + 1
                            orow = lo * g.sw + off - iw0
                            tap = kh * g.kw + kw
                            for f0 in range(0, F, DIM):
                                kr = min(DIM, F - f0)
                                a = 32 + (u & 1) * 2 * DIM
                                b = a + DIM
                                u += 1
                                s.mvin(s.at(gy, ((n * g.oh + oh) * g.ow + lo) * F + f0), SP(a), kr, v, 0)
                                s.mvin(s.at(wb, (tap * F + f0) * C + c0), SP(b), nc, kr, 1)
                                s.preload(SP(b), ACC(acc + orow, accumulate=True), nc, kr, nc, v)
                                s.compute(SP(a), kr, v)
                    s.mvout(s.at(gx, ((n * g.h + ih) * g.w + iw0) * C + c0), ACC(acc), nc, mr)
    s.config_ex()


def conv_wgrad(s: Stream, g: ConvGeom, gy: str, x: str, gw: str, scale: float) -> None:
    """gy: NCHW [N, F, OH, OW]; x: NHWC [N, H, W, C]; gw: [KH*KW, F, C] int8 at ``scale``."""
    F, C = g.cout, g.cin
    OP = g.oh * g.ow
    s.config_ex()
    s.config_ld(0, OP)
    s.config_ld(1, g.sw * C)
    s.config_ld(2, 0, 1.0, shrink=False)
    s.config_st(C, NO_ACT, scale)
    t = u = 0
    for tap in range(g.kh * g.kw):
        kh, kw = divmod(tap, g.kw)
        for co0 in range(0, F, DIM):
            nco = min(DIM, F - co0)
            for ci0 in range(0, C, DIM):
                nci = min(DIM, C - ci0)
                acc = (t & 1) * DIM
                t += 1
                s.mvin(None, ACC(acc), nci, nco, 2)
                for n in range(g.n):
                    for oh in range(g.oh):
                        ih = oh * g.sh - g.ph + kh * g.dh
                        if not 0 <= ih < g.h:
                            continue
                        for ow0 in range(0, g.ow, DIM):
                            mk = min(DIM, g.ow - ow0)
                            off = kw * g.dw - g.pw
                            lo = max(ow0, _cdiv(-off, g.sw))
                            hi = min(ow0 + mk - 1, _fdiv(g.w - 1 - off, g.sw))
                            if lo > hi:
                                continue
                            a = 32 + (u & 1) * 2 * DIM
                            b = a + DIM
                            u += 1
                            s.mvin(s.at(gy, (n * F + co0) * OP + oh * g.ow + ow0), SP(a), mk, nco, 0)
                            if lo > ow0:
                                s.mvin(None, SP(b), nci, lo - ow0, 1)
                            if hi < ow0 + mk - 1:
                                s.mvin(None, SP(b + hi + 1 - ow0), nci, ow0 + mk - 1 - hi, 1)
                            iw = lo * g.sw + off
                            s.mvin(s.at(x, ((n * g.h + ih) * g.w + iw) * C + ci0), SP(b + lo - ow0), nci,
                                   hi - lo + 1, 1)
                            s.preload(SP(b), ACC(acc, accumulate=True), nci, mk, nci, nco)
                            s.compute(SP(a), mk, nco)
                s.mvout(s.at(gw, (tap * F + co0) * C + ci0), ACC(acc), nci, nco)
