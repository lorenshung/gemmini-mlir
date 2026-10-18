"""Data movement through the scratchpad: strided 2-D DMA copies, fills, transposes.

These are the ``_lib_movement.py`` recipes (``mv_copy2d[_multi]``, ``mv_vec``, ``mv_fill2d``,
``mv_transpose2d``, ``mv_run_jobs``) as compile-time job lists:

* a copy job moves an int8 [rows x cols] block, source row stride ``sld`` bytes, to one or
  more destinations with row stride ``dld``; every <=16x16 tile is one MVIN and one MVOUT
  per destination, through four rotating scratchpad tiles (rows < 1024 so Spike can store
  them). Production copies retain integer codes; requantization uses the exact host boundary.
* a fill job stores a 16x16 constant tile: the zero-fill primitive for code 0, otherwise a
  stride-0 MVIN of literal integer codes with unit scale.
* jobs are grouped in stages; a fence separates stages whose DMA ranges overlap (a fill
  then the interior copy, a copy of ``self`` then the scattered ``src``, or a stage that
  reads output bytes a previous stage wrote): the reservation station does not order DRAM.
* the column (inner) DMA stride is always one element (C1). A permutation that moves the
  innermost dimension gathers one-element columns with strided DMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..stream import ACC, DIM, SP, Stream

SLOTS = (0, 16, 32, 48)          # rotating copy tiles
FILL_SLOT = 64


@dataclass
class Copy:
    src: tuple[str, int] | None      # None: fill with ``code``
    sld: int
    dsts: list[tuple[str, int]]
    dld: int
    rows: int
    cols: int
    scale: float = 1.0
    code: int = 0


@dataclass
class Plan:
    stages: list[list[Copy]] = field(default_factory=list)

    def stage(self) -> list[Copy]:
        self.stages.append([])
        return self.stages[-1]


def strides(shape) -> list[int]:
    out, acc = [], 1
    for d in reversed(shape):
        out.append(acc)
        acc *= d
    return list(reversed(out))


def _iter(ranges):
    return product(*[range(a, b) for a, b in ranges]) if ranges else [()]


def normalize(job: Copy) -> list[Copy]:
    """View a block that is contiguous on both sides as 16-byte rows (whole tiles, one tail)."""
    n = job.rows * job.cols
    contiguous_src = job.src is None or job.sld == job.cols or job.rows == 1
    contiguous_dst = job.dld == job.cols or job.rows == 1
    if contiguous_src and contiguous_dst and n > DIM and (job.cols % DIM or job.rows == 1):
        full = n // DIM
        out = [Copy(job.src, 0 if job.src is None else DIM, job.dsts, DIM, full, DIM, job.scale, job.code)]
        if n % DIM:
            src = None if job.src is None else (job.src[0], job.src[1] + full * DIM)
            out.append(Copy(src, DIM, [(a, o + full * DIM) for a, o in job.dsts], DIM, 1, n % DIM, job.scale,
                            job.code))
        return out
    return [job]


def emit_plan(s: Stream, plan: Plan) -> None:
    ld = st = None
    fill_code = None
    t = 0
    for si, stage in enumerate(plan.stages):
        if si and stage:
            s.fence()
            fill_code = None
        for job in [x for j in stage for x in normalize(j)]:
            if job.src is None:
                if fill_code != job.code:
                    if job.code == 0:
                        cfg = (0, 1.0)
                        if ld != cfg:
                            s.config_ld(0, 0, 1.0)
                            ld = cfg
                        s.mvin(None, SP(FILL_SLOT), DIM, DIM, 0)
                    else:
                        name = f"fill_code_{job.code}"
                        if name not in s._index:
                            s.const(name, np.full(DIM, job.code, np.int8))
                        cfg = (0, 1.0)
                        if ld != cfg:
                            s.config_ld(0, 0, 1.0)
                            ld = cfg
                        s.mvin(s.at(name), SP(FILL_SLOT), DIM, DIM, 0)
                    fill_code = job.code
            else:
                cfg = (job.sld, job.scale)
                if ld != cfg:
                    s.config_ld(0, job.sld, job.scale)
                    ld = cfg
            if st != job.dld:
                s.config_st(job.dld)
                st = job.dld
            for r0 in range(0, job.rows, DIM):
                rh = min(DIM, job.rows - r0)
                for c0 in range(0, job.cols, DIM):
                    cw = min(DIM, job.cols - c0)
                    if job.src is None:
                        slot = FILL_SLOT
                    else:
                        slot = SLOTS[t & 3]
                        t += 1
                        s.mvin(s.at(job.src[0], job.src[1] + r0 * job.sld + c0), SP(slot), cw, rh, 0)
                    for arg, off in job.dsts:
                        s.mvout(s.at(arg, off + r0 * job.dld + c0), SP(slot), cw, rh)


# ---------------------------------------------------------------------------------------------
# Op builders.  Shapes are row-major; offsets/strides are bytes of int8 elements.
# ---------------------------------------------------------------------------------------------

def flat_copy(plan_stage, src, dst, n, scale=1.0):
    plan_stage.append(Copy(src, n, [dst], n, 1, n, scale))


def gather_elements(s, sources, indices, itemsize, out, names, literal):
    """Static element selectors lifted to consecutive little-endian byte lanes.

    Runs share a nonnegative source stride. Reversals split into independent
    elements; an element's bytes always retain their original order.
    """
    plan = Plan()
    stage = plan.stage()
    start = 0
    while start < len(sources):
        sid = int(sources[start])
        index = start if sid < 0 else int(indices[start])
        end, step = start + 1, None
        while end < len(sources) and sources[end] == sid:
            following = end if sid < 0 else int(indices[end])
            delta = following - (end - 1 if sid < 0 else int(indices[end - 1]))
            if delta < 0 or (step is not None and delta != step):
                break
            step = delta
            end += 1
        name = literal if sid < 0 else names[sid]
        count = end - start
        if step == 1 or count == 1:
            size = count * itemsize
            stage.append(Copy((name, index * itemsize), size, [(out, start * itemsize)], size, 1, size))
        else:
            stage.append(Copy((name, index * itemsize), (step or 0) * itemsize,
                              [(out, start * itemsize)], itemsize, count, itemsize))
        start = end
    emit_plan(s, plan)


def block_copy(stage, src_name, src_shape, src_index, dst_name, dst_shape, dst_index, extents, scale=1.0):
    """Copy the block ``extents`` (per dim) from src[src_index:] to dst[dst_index:] (same rank)."""
    r = len(extents)
    ss, ds = strides(src_shape), strides(dst_shape)
    if r == 1:
        stage.append(Copy((src_name, src_index[0]), extents[0], [(dst_name, dst_index[0])], extents[0], 1,
                          extents[0], scale))
        return
    for outer in _iter([(0, e) for e in extents[:-2]]):
        so = sum((src_index[d] + i) * ss[d] for d, i in enumerate(outer)) + src_index[-2] * ss[-2] + src_index[-1]
        do = sum((dst_index[d] + i) * ds[d] for d, i in enumerate(outer)) + dst_index[-2] * ds[-2] + dst_index[-1]
        stage.append(Copy((src_name, so), ss[-2], [(dst_name, do)], ds[-2], extents[-2], extents[-1], scale))


def fill(stage, dst_name, n, code):
    stage.append(Copy(None, 0, [(dst_name, 0)], n, 1, n, code=code))


def constant_pad(plan: Plan, x: str, shape, out: str, pad_lohi, code, scale=1.0):
    """pad_lohi: per dim (lo, hi); negative pads crop.  Fill (if any pad > 0), fence, interior."""
    oshape = [d + lo + hi for d, (lo, hi) in zip(shape, pad_lohi)]
    n = int(np.prod(oshape))
    if any(lo > 0 or hi > 0 for lo, hi in pad_lohi):
        fill(plan.stage(), out, n, code)
    rng = [(max(0, -lo), min(d, d + hi)) for d, (lo, hi) in zip(shape, pad_lohi)]
    if any(a >= b for a, b in rng):
        return oshape
    stage = plan.stage()
    block_copy(stage, x, shape, [a for a, _ in rng], out, oshape, [a + lo for (a, _), (lo, _) in zip(rng, pad_lohi)],
               [b - a for a, b in rng], scale)
    return oshape


def cat(plan: Plan, inputs: list[tuple[str, tuple]], out: str, dim: int, scales=None):
    shape0 = inputs[0][1]
    r = len(shape0)
    dim %= r
    total = sum(sh[dim] for _, sh in inputs)
    outer = int(np.prod(shape0[:dim])) if dim else 1
    inner = int(np.prod(shape0[dim + 1:])) if dim + 1 < r else 1
    stage = plan.stage()
    off = 0
    for k, (name, sh) in enumerate(inputs):
        cols = sh[dim] * inner
        scale = 1.0 if scales is None else scales[k]
        if cols:
            stage.append(Copy((name, 0), cols, [(out, off * inner)], total * inner, outer, cols, scale))
        off += sh[dim]
    return [*shape0[:dim], total, *shape0[dim + 1:]]


def slice_scatter(plan: Plan, self_: str, shape, src: str, out: str, dim: int, start: int, end: int, step: int,
                  self_scale=1.0, src_scale=1.0):
    r = len(shape)
    dim %= r
    n = int(np.prod(shape))
    flat_copy(plan.stage(), (self_, 0), (out, 0), n, self_scale)
    D = shape[dim]
    idx = list(range(*slice(start, end, step).indices(D)))
    if not idx:
        return
    outer = int(np.prod(shape[:dim])) if dim else 1
    inner = int(np.prod(shape[dim + 1:])) if dim + 1 < r else 1
    k = len(idx)
    stage = plan.stage()
    st = idx[1] - idx[0] if k > 1 else 1
    if inner > 1 or st == 1:
        # rows = slice positions (stride st*inner), cols = inner block; one job per outer index
        if inner == 1:
            stage.append(Copy((src, 0), k, [(out, idx[0])], D, outer, k, src_scale))
        else:
            for o in range(outer):
                stage.append(Copy((src, o * k * inner), inner, [(out, (o * D + idx[0]) * inner)], st * inner, k,
                                  inner, src_scale))
    else:
        # strided last-dim slice: element-granular (cols = 1) copies, DRAM row stride = step
        for o in range(outer):
            stage.append(Copy((src, o * k), 1, [(out, o * D + idx[0])], st, k, 1, src_scale))


def full(plan: Plan, out: str, n: int, code: int):
    fill(plan.stage(), out, n, code)


def repeat(plan: Plan, x: str, shape, out: str, reps, scale=1.0):
    """out = x tiled ``reps`` times (len(reps) >= rank; leading reps add dims)."""
    shape = [1] * (len(reps) - len(shape)) + list(shape)
    oshape = [d * r for d, r in zip(shape, reps)]
    os_ = strides(oshape)
    stage = plan.stage()
    # copy the whole input block once per tile position; rows = dim -2, cols = dim -1
    tiles = list(product(*[range(r) for r in reps]))
    ss = strides(shape)
    for outer in _iter([(0, d) for d in shape[:-2]]):
        so = sum(i * ss[d] for d, i in enumerate(outer))
        dsts = []
        for t in tiles:
            idx = [t[d] * shape[d] + (outer[d] if d < len(outer) else 0) for d in range(len(shape))]
            do = sum(idx[d] * os_[d] for d in range(len(shape) - 2)) + t[-2] * shape[-2] * os_[-2] + t[-1] * shape[-1] \
                if len(shape) >= 2 else t[-1] * shape[-1]
            dsts.append((out, do))
        rows = shape[-2] if len(shape) >= 2 else 1
        stage.append(Copy((x, so), shape[-1], dsts, oshape[-1], rows, shape[-1], scale))
    return oshape


def permute_plan(x: str, shape, out: str, perm, scale=1.0):
    """Permutation keeping the innermost input dim innermost: strided row copies."""
    r = len(shape)
    oshape = [shape[p] for p in perm]
    ist, ost = strides(shape), strides(oshape)
    assert perm[-1] == r - 1
    plan = Plan()
    stage = plan.stage()
    if r == 1:
        flat_copy(stage, (x, 0), (out, 0), shape[0], scale)
        return plan, oshape
    rows_in = perm[-2]
    for outer in _iter([(0, oshape[d]) for d in range(r - 2)]):
        so = sum(i * ist[perm[d]] for d, i in enumerate(outer))
        do = sum(i * ost[d] for d, i in enumerate(outer))
        stage.append(Copy((x, so), ist[rows_in], [(out, do)], ost[-2], oshape[-2], oshape[-1], scale))
    return plan, oshape


def emit_transpose(s: Stream, x: str, shape, out: str, perm, scale=1.0):
    """Move a permutation's innermost output axis with strided byte DMA.

    Each DMA gathers a column of up to DIM source elements into scratchpad
    rows, then writes those bytes contiguously. This avoids accumulator
    narrowing even for layout conversions inside a contraction schedule.
    """
    from ..exactness import require
    require(scale == 1, "permutation scaling requires exact host requantization")
    r = len(shape)
    oshape = [shape[q] for q in perm]
    ist, ost = strides(shape), strides(oshape)
    p = perm[-1]
    others = [d for d in range(r) if d != p]
    s.config_ld(0, ist[p], 1.0)
    s.config_st(1)
    t = 0
    for idx in _iter([(0, shape[d]) for d in others]):
        so = sum(i * ist[d] for d, i in zip(others, idx))
        do = sum(i * ost[perm.index(d)] for d, i in zip(others, idx))
        for start in range(0, shape[p], DIM):
            rows = min(DIM, shape[p] - start)
            slot = SLOTS[t & 3]
            t += 1
            s.mvin(s.at(x, so + start * ist[p]), SP(slot), 1, rows, 0)
            s.mvout(s.at(out, do + start), SP(slot), 1, rows)
    return oshape


def emit_widen(s: Stream, x: str, out: str, n: int):
    """int8 -> int32 widening (``_to_copy``): shrunk MVIN into the accumulator, full-width readout."""
    full, tail = divmod(n, DIM)
    s.config_ld(0, DIM, 1.0, shrink=True)
    s.config_st(DIM * 4)
    t = 0
    for r0 in range(0, full, DIM):
        rh = min(DIM, full - r0)
        acc = (t & 3) * DIM
        t += 1
        s.mvin(s.at(x, r0 * DIM), ACC(acc), DIM, rh, 0)
        s.mvout(s.at(out, r0 * DIM * 4), ACC(acc, full=True), DIM, rh)
    if tail:
        acc = (t & 3) * DIM
        s.mvin(s.at(x, full * DIM), ACC(acc), tail, 1, 0)
        s.mvout(s.at(out, full * DIM * 4), ACC(acc, full=True), tail, 1)
