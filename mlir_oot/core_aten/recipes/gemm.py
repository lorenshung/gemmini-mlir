"""Blocked direct weight-stationary matmul (``ep_gm_mm_blocked``), the contraction primitive.

C[M,N] = rq(A[M,K] . B[K,N] (+ int32 bias row)) with an accumulator block of up to 32
output tiles (halves alternate per block), A/B tile blocks in half the scratchpad
(halves alternate per K block) loaded with 4-tile block MVINs, each B tile preloaded
once per (j, k) and reused across the block's row tiles with COMPUTE_ACCUMULATED.
The int32 bias is a stride-0 full-width MVIN3 (the gate-3 Q1 form) seeding the block
before its K loop.  K enters the mesh 16 deep per COMPUTE (WS; no 20-bit risk) and
accumulates in the int32 accumulator; the readout applies ReLU then the fp32 scale,
RNE and int8 saturation, or stores raw int32.
"""

from __future__ import annotations

from ..stream import ACC, DIM, GARBAGE, NO_ACT, RELU, SP, Stream


def _cdiv(a, b):
    return (a + b - 1) // b


def blocks(M: int, N: int, K: int) -> tuple[int, int, int]:
    Mt, Nt, Kt = _cdiv(M, DIM), _cdiv(N, DIM), _cdiv(K, DIM)
    TI, TJ = min(Mt, 4), min(Nt, 8)
    if TI < 4:
        TJ = min(Nt, 32 // TI)
    if TJ < 8:
        TI = min(Mt, 32 // TJ)
    TK = min(Kt, 512 // (TI + TJ))
    return TI, TJ, TK


def blocked_matmul(s: Stream, A: tuple[str, int], lda: int, B: tuple[str, int], ldb: int,
                   C: tuple[str, int], ldc: int, M: int, K: int, N: int, *,
                   bias: tuple[str, int] | None = None, scale: float = 1.0, relu: bool = False,
                   out_i32: bool = False, configure: bool = True, bias_rows: bool = False) -> None:
    """``A``/``B``/``C``/``bias`` are (argument name, byte offset); strides are in elements.

    ``bias_rows`` reads a full [M, N] int32 bias instead of one broadcast row.
    """
    if out_i32 and (scale != 1.0 or relu):
        raise ValueError("raw int32 readout applies neither scale nor ReLU")
    if scale <= 0 and relu:
        raise ValueError("ReLU readout requires a positive scale (C7)")
    Mt, Nt, Kt = _cdiv(M, DIM), _cdiv(N, DIM), _cdiv(K, DIM)
    TI, TJ, TK = blocks(M, N, K)
    if configure:
        s.config_ex()
        s.config_ld(0, lda)
        s.config_ld(1, ldb)
        s.config_ld(2, ldc * 4 if bias_rows else 0)
        s.config_st(ldc * (4 if out_i32 else 1), RELU if relu else NO_ACT, scale)
    an, ao = A
    bn, bo = B
    cn, co = C
    blk = kblk = 0
    for ib in range(0, Mt, TI):
        ti = min(TI, Mt - ib)
        for jb in range(0, Nt, TJ):
            tj = min(TJ, Nt - jb)
            acc0 = (blk & 1) * 32 * DIM
            blk += 1
            if bias is not None:
                for i in range(ti):
                    mr = min(DIM, M - (ib + i) * DIM)
                    for j in range(tj):
                        nc = min(DIM, N - (jb + j) * DIM)
                        boff = bias[1] + (((ib + i) * DIM * N if bias_rows else 0) + (jb + j) * DIM) * 4
                        tile = acc0 + (i * tj + j) * DIM
                        s.mvin(s.at(bias[0], boff), ACC(tile), nc, mr, 2)
            for kb in range(0, Kt, TK):
                tk = min(TK, Kt - kb)
                aB = (kblk & 1) * 8192
                bB = aB + ti * tk * DIM
                kblk += 1
                for i in range(ti):
                    mr = min(DIM, M - (ib + i) * DIM)
                    for k in range(0, tk, 4):
                        cols = min(min(4, tk - k) * DIM, K - (kb + k) * DIM)
                        s.mvin(s.at(an, ao + (ib + i) * DIM * lda + (kb + k) * DIM),
                               SP(aB + (i * tk + k) * DIM), cols, mr, 0)
                for k in range(tk):
                    kr = min(DIM, K - (kb + k) * DIM)
                    for j in range(0, tj, 4):
                        cols = min(min(4, tj - j) * DIM, N - (jb + j) * DIM)
                        s.mvin(s.at(bn, bo + (kb + k) * DIM * ldb + (jb + j) * DIM),
                               SP(bB + (k * tj + j) * DIM), cols, kr, 1)
                for j in range(tj):
                    nc = min(DIM, N - (jb + j) * DIM)
                    for k in range(tk):
                        kr = min(DIM, K - (kb + k) * DIM)
                        accumulate = bias is not None or (kb + k) > 0
                        for i in range(ti):
                            mr = min(DIM, M - (ib + i) * DIM)
                            c = ACC(acc0 + (i * tj + j) * DIM, accumulate=accumulate)
                            s.preload(SP(bB + (k * tj + j) * DIM) if i == 0 else GARBAGE, c, nc, kr, nc, mr)
                            s.compute(SP(aB + (i * tk + k) * DIM), kr, mr, accumulate=i > 0)
            for i in range(ti):
                mr = min(DIM, M - (ib + i) * DIM)
                for j in range(tj):
                    nc = min(DIM, N - (jb + j) * DIM)
                    s.mvout(s.at(cn, co + ((ib + i) * DIM * ldc + (jb + j) * DIM) * (4 if out_i32 else 1)),
                            ACC(acc0 + (i * tj + j) * DIM, full=out_i32), nc, mr)
