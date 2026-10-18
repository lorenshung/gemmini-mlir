"""Channels-last convolution as an implicit GEMM with accumulator blocking (no im2col copy).

Activations are NHWC int8, weights HWIO int8 ([KH][KW][Cin/groups][Cout]), output NHWC.
M = output pixel tiles (16 consecutive output columns of one output row), N = output
channel tiles, K = (tap, input-channel chunk) steps.  The A tile of a step is DMA'd
straight from the activation with DRAM row stride ``stride_w * Cin`` (the strided DMA
im2col of P3); out-of-bounds lanes of a padded tap are zero-filled as disjoint lane runs
and fully out-of-bounds taps are skipped.  When the width has no padding and unit
dilation, a kernel row's (kw, ci) run is contiguous, so K steps cover KW*Cin at once and
A loads use 4-tile block MVINs (the patch-embedding case, stride == kernel).

An accumulator block of up to 32 tiles is seeded with the int32 bias (stride-0 full-width
MVIN3, Q1) or zero (shrink = 0 state, HW2) so skipped taps contribute nothing; every
weight tile is preloaded once per K step and reused across the block's pixel tiles with
COMPUTE_ACCUMULATED.  K enters the mesh 16 deep per COMPUTE (WS).  Readout: ReLU, fp32
scale, RNE and int8 saturation, or raw int32.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..stream import ACC, DIM, GARBAGE, NO_ACT, RELU, SP, Stream


@dataclass(frozen=True)
class ConvGeom:
    n: int
    h: int
    w: int
    cin: int
    cout: int
    kh: int
    kw: int
    sh: int = 1
    sw: int = 1
    ph: int = 0
    pw: int = 0
    dh: int = 1
    dw: int = 1
    groups: int = 1

    @property
    def oh(self):
        return (self.h + 2 * self.ph - self.dh * (self.kh - 1) - 1) // self.sh + 1

    @property
    def ow(self):
        return (self.w + 2 * self.pw - self.dw * (self.kw - 1) - 1) // self.sw + 1

    def validate(self):
        if min(self.n, self.h, self.w, self.cin, self.cout, self.kh, self.kw, self.sh, self.sw, self.dh, self.dw,
               self.groups) <= 0 or self.ph < 0 or self.pw < 0:
            raise ValueError("invalid convolution geometry")
        if self.cin % self.groups or self.cout % self.groups:
            raise ValueError("channels must divide by groups")
        if self.oh <= 0 or self.ow <= 0:
            raise ValueError("empty convolution output")


def _ksteps(g: ConvGeom):
    """(kh, kw or None for a merged row, channel offset within the run, kr, weight row)."""
    cig = g.cin // g.groups
    steps = []
    if g.pw == 0 and g.dw == 1 and g.groups == 1:
        run = g.kw * g.cin
        for kh in range(g.kh):
            for c0 in range(0, run, DIM):
                steps.append((kh, None, c0, min(DIM, run - c0), kh * run + c0))
    else:
        for kh in range(g.kh):
            for kw in range(g.kw):
                for c0 in range(0, cig, DIM):
                    steps.append((kh, kw, c0, min(DIM, cig - c0), (kh * g.kw + kw) * cig + c0))
    return steps


def conv2d(s: Stream, g: ConvGeom, x: str, w: str, y: str, *, bias: str | None = None, scale: float = 1.0,
           relu: bool = False, out_i32: bool = False, bias_repeat: int = 1) -> None:
    g.validate()
    if out_i32 and (scale != 1.0 or relu):
        raise ValueError("raw int32 readout applies neither scale nor ReLU")
    if relu and scale <= 0:
        raise ValueError("ReLU readout requires a positive scale (C7)")
    oh, ow = g.oh, g.ow
    cig, cog = g.cin // g.groups, g.cout // g.groups
    steps = _ksteps(g)
    tiles = [(n, r, c0) for n in range(g.n) for r in range(oh) for c0 in range(0, ow, DIM)]
    nt = (cog + DIM - 1) // DIM
    TJ = min(nt, 8)
    TI = min(len(tiles), 32 // TJ)
    TK = max(1, min(len(steps), 512 // (TI + TJ)))
    esz = 4 if out_i32 else 1
    s.config_ex()
    s.config_ld(0, g.sw * g.cin)
    s.config_ld(1, g.cout)
    s.config_ld(2, 0, 1.0, shrink=False)
    s.config_st(g.cout * esz, RELU if relu else NO_ACT, scale)
    blk = kblk = 0
    for grp in range(g.groups):
        for jb in range(0, nt, TJ):
            tj = min(TJ, nt - jb)
            for ib in range(0, len(tiles), TI):
                block = tiles[ib:ib + TI]
                acc0 = (blk & 1) * 512
                blk += 1
                for i, (n, r, c0) in enumerate(block):
                    mr = min(DIM, ow - c0)
                    for j in range(tj):
                        nc = min(DIM, cog - (jb + j) * DIM)
                        dst = ACC(acc0 + (i * tj + j) * DIM)
                        if bias is not None and bias_repeat == 1:
                            s.mvin(s.at(bias, (grp * cog + (jb + j) * DIM) * 4), dst, nc, mr, 2)
                        else:
                            s.mvin(None, dst, nc, mr, 2)
                            if bias is not None:
                                s.fence()
                            for _ in range(bias_repeat if bias is not None else 0):
                                s.mvin(s.at(bias, (grp * cog + (jb + j) * DIM) * 4),
                                       ACC(acc0 + (i * tj + j) * DIM, accumulate=True), nc, mr, 2)
                for kb in range(0, len(steps), TK):
                    ks = steps[kb:kb + TK]
                    tk = len(ks)
                    aB = (kblk & 1) * 8192
                    bB = aB + TI * TK * DIM
                    kblk += 1
                    valid = {}
                    for i, (n, r, c0) in enumerate(block):
                        mr = min(DIM, ow - c0)
                        k = 0
                        while k < tk:
                            kh, kw, cc, kr, _ = ks[k]
                            ih = r * g.sh - g.ph + kh * g.dh
                            if not 0 <= ih < g.h:
                                k += 1
                                continue
                            row = (n * g.h + ih) * g.w
                            local = aB + (i * TK + k) * DIM
                            if kw is None:
                                # merged (kw, ci) run: block-MVIN consecutive steps of the same kernel row
                                e = k
                                while (e + 1 < tk and e + 1 - k < 4 and ks[e + 1][0] == kh and ks[e + 1][1] is None
                                       and ks[e + 1][2] == ks[e][2] + DIM):
                                    e += 1
                                cols = sum(ks[q][3] for q in range(k, e + 1))
                                s.mvin(s.at(x, (row + c0 * g.sw) * g.cin + cc), SP(local), cols, mr, 0)
                                for q in range(k, e + 1):
                                    valid[(i, q)] = True
                                k = e + 1
                                continue
                            off = kw * g.dw - g.pw
                            lanes = [p for p in range(mr) if 0 <= (c0 + p) * g.sw + off < g.w]
                            if lanes:
                                lo, hi = lanes[0], lanes[-1]
                                if lo > 0:
                                    s.mvin(None, SP(local), kr, lo, 0)
                                if hi + 1 < mr:
                                    s.mvin(None, SP(local + hi + 1), kr, mr - hi - 1, 0)
                                iw = (c0 + lo) * g.sw + off
                                s.mvin(s.at(x, (row + iw) * g.cin + grp * cig + cc), SP(local + lo), kr, hi - lo + 1, 0)
                                valid[(i, k)] = True
                            k += 1
                    for k, (kh, kw, cc, kr, wrow) in enumerate(ks):
                        if not any((i, k) in valid for i in range(len(block))):
                            continue
                        for j in range(0, tj, 4):
                            cols = min(min(4, tj - j) * DIM, cog - (jb + j) * DIM)
                            s.mvin(s.at(w, wrow * g.cout + grp * cog + (jb + j) * DIM),
                                   SP(bB + (k * tj + j) * DIM), cols, kr, 1)
                    for j in range(tj):
                        nc = min(DIM, cog - (jb + j) * DIM)
                        for k, (kh, kw, cc, kr, wrow) in enumerate(ks):
                            first = True
                            for i, (n, r, c0) in enumerate(block):
                                if (i, k) not in valid:
                                    continue
                                mr = min(DIM, ow - c0)
                                c = ACC(acc0 + (i * tj + j) * DIM, accumulate=True)
                                s.preload(SP(bB + (k * tj + j) * DIM) if first else GARBAGE, c, nc, kr, nc, mr)
                                s.compute(SP(aB + (i * TK + k) * DIM), kr, mr, accumulate=not first)
                                first = False
                for i, (n, r, c0) in enumerate(block):
                    mr = min(DIM, ow - c0)
                    j = 0
                    while j < tj:
                        width = min(4, tj - j) if not out_i32 else 1
                        cols = min(width * DIM, cog - (jb + j) * DIM)
                        s.mvout(s.at(y, (((n * oh + r) * ow + c0) * g.cout + grp * cog + (jb + j) * DIM) * esz),
                                ACC(acc0 + (i * tj + j) * DIM, full=out_i32), cols, mr)
                        j += width
