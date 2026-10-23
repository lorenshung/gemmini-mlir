"""Elementwise affine maps: scale (``em_ew_spad``) and the
exact integer linear combination ``qx_lincomb2`` / ``qx_affine1`` in its blocked form
(``qf_lincomb_blocked``).

* ``unary_scale``: y = sat8(RNE(c x)) on the int8 MVIN path (neg, mul.Scalar, div.Scalar:
  c = k s_x / s_out; one rounding).
* ``lincomb``: ACC = sum_t W_t x_t (+ B) exactly in int32, W_t as int8 c*I matmul parts
  (|c| <= 127, negative weights allowed), B an int32 stride-0 MVIN3; one RNE readout.
  Groups of up to 32 output tiles; each c*I tile is preloaded once per group and reused
  with COMPUTE_ACCUMULATED.  An operand with row stride 0 is a broadcast row.
"""

from __future__ import annotations

import numpy as np

from ..stream import ACC, DIM, GARBAGE, NO_ACT, RELU, SP, Stream

W64 = 4 * DIM


def exact_byte_relu(s, x, out, n):
    """ReLU of a signed-byte value, then identity widening to full i32.

    Full ACC readout bypasses activation. Execute-to-SP clips before ReLU,
    which is exact here because the identity product is already in [-128,127].
    The second identity product widens [0,127] into ACC without activation.
    """
    s.const('relu_identity', np.eye(DIM, dtype=np.int8))
    s.config_ld(1, DIM)
    s.mvin(s.at('relu_identity'), SP(0), DIM, DIM, 1)
    off = 0
    while off < n:
        cols = min(DIM, n-off)
        rows = min(DIM, (n-off)//cols)
        s.fence()
        s.config_ld(0, cols)
        s.config_st(cols*4)
        s.mvin(s.at(x, off), SP(32), cols, rows)
        s.config_ex(act=RELU)
        s.matmul_tile(SP(32), SP(0), SP(64), rows, cols, cols)
        s.fence()
        s.config_ex()
        s.matmul_tile(SP(64), SP(0), ACC(0), rows, cols, cols)
        s.mvout(s.at(out, off*4), ACC(0, full=True), cols, rows)
        off += rows*cols


def _flat_blocks(n: int):
    """[n // 64, 64] row blocks of up to 16 rows, then one tail row of n % 64."""
    rows = n // W64
    for r0 in range(0, rows, DIM):
        yield r0 * W64, min(DIM, rows - r0), W64
    if n % W64:
        yield rows * W64, 1, n % W64


def unary_scale(s: Stream, x: str, out: str, n: int, scale: float) -> None:
    s.config_ld(0, W64, scale)
    s.config_st(W64)
    for t, (off, rows, cols) in enumerate(_flat_blocks(n)):
        slot = (t & 1) * W64
        s.mvin(s.at(x, off), SP(slot), cols, rows, 0)
        for c0 in range(0, cols, DIM):
            s.mvout(s.at(out, off + c0), SP(slot + c0), min(DIM, cols - c0), rows)


def lincomb(s: Stream, terms: list[tuple[str, int, list[int]]], out: str, M: int, N: int, scale: float, *,
            bias: int | None = None, relu: bool = False, out_stride: int | None = None,
            out_i32: bool = False) -> None:
    """``terms``: (arg, row stride in bytes (0 = broadcast row), int8 weight parts).

    Computes out[M, N] = sat8(RNE(scale act(sum_t (sum parts_t) x_t + bias))).
    """
    if relu and scale <= 0:
        raise ValueError("ReLU readout requires a positive scale (C7)")
    parts = [(ti, c) for ti, (_, _, cs) in enumerate(terms) for c in cs]
    if not parts or len(parts) > 12 or any(not -127 <= c <= 127 or c == 0 for _, c in parts):
        raise ValueError("lincomb needs 1..12 nonzero int8 weight parts")
    ld_out = N if out_stride is None else out_stride
    s.config_ex()
    for p, (_, c) in enumerate(parts):
        name = f"weight_eye_{c}"
        if name not in s._index:
            s.const(name, np.eye(DIM, dtype=np.int8) * c)
        s.config_ld(1, DIM, 1.0)
        s.mvin(s.at(name), SP(16 * p), DIM, DIM, 1)
    if bias is not None:
        name = f"brow{bias}".replace("-", "m")
        if name not in s._index:
            s.const(name, np.full(DIM, bias, np.int32))
    # one load state per operand (same row stride per term)
    states = []
    for ti, (_, stride, _) in enumerate(terms):
        states.append(ti)
        s.config_ld(ti, stride, 1.0)
    if bias is not None:
        s.config_ld(2, 0, 1.0)
    if out_i32 and (relu or scale != 1.0):
        raise ValueError("an int32 result has no readout scale or activation")
    s.config_st(ld_out * (4 if out_i32 else 1), RELU if relu else NO_ACT, scale)
    tiles = [(i0, j0) for i0 in range(0, M, DIM) for j0 in range(0, N, DIM)]
    base = 1024
    for grp, g0 in enumerate(range(0, len(tiles), 32)):
        group = tiles[g0:g0 + 32]
        half = grp & 1
        opbase = [base + half * 1024 + ti * 512 for ti in range(len(terms))]
        acc0 = half * 512
        # operand loads: consecutive tiles of one row block, up to 4 per block MVIN
        runs = []
        g = 0
        while g < len(group):
            i0, j0 = group[g]
            k = g
            while k + 1 < len(group) and k + 1 - g < 4 and group[k + 1][0] == i0:
                k += 1
            runs.append((g, k + 1))
            g = k + 1
        for a, b in runs:
            i0, j0 = group[a]
            mr = min(DIM, M - i0)
            cols = sum(min(DIM, N - group[g][1]) for g in range(a, b))
            for ti, (arg, stride, _) in enumerate(terms):
                s.mvin(s.at(arg, i0 * stride + j0), SP(opbase[ti] + a * DIM), cols, mr, states[ti])
            if bias is not None:
                for g in range(a, b):
                    s.mvin(s.at(name), ACC(acc0 + g * DIM), min(DIM, N - group[g][1]), mr, 2)
        for p, (ti, _) in enumerate(parts):
            accumulate = p > 0 or bias is not None
            for g, (i0, j0) in enumerate(group):
                mr, nc = min(DIM, M - i0), min(DIM, N - j0)
                s.preload(SP(16 * p) if g == 0 else GARBAGE, ACC(acc0 + g * DIM, accumulate=accumulate),
                          DIM, DIM, nc, mr)
                s.compute(SP(opbase[ti] + g * DIM), nc, mr, accumulate=g > 0)
        for a, b in runs:
            i0, j0 = group[a]
            mr = min(DIM, M - i0)
            if out_i32:
                for g in range(a, b):
                    gi, gj = group[g]
                    s.mvout(s.at(out, (gi * ld_out + gj) * 4), ACC(acc0 + g * DIM, full=True), min(DIM, N - gj), mr)
                continue
            cols = sum(min(DIM, N - group[g][1]) for g in range(a, b))
            s.mvout(s.at(out, i0 * ld_out + j0), ACC(acc0 + a * DIM), cols, mr)
