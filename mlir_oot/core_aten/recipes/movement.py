"""Data movement through the scratchpad: strided 2-D DMA copies, fills and widening.

These are the ``_lib_movement.py`` recipes (``mv_copy2d[_multi]``, ``mv_vec``, ``mv_fill2d``,
``mv_transpose2d``, ``mv_run_jobs``) as compile-time job lists:

* a copy job moves an int8 [rows x cols] block, source row stride ``sld`` bytes, to one or
  more destinations with row stride ``dld``; every <=16x16 tile is one MVIN and one MVOUT
  per destination, through a reused scratchpad tile (rows < 1024 so Spike can store
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
    from .. import layout
    ld = st = None
    fill_code = None
    for si, stage in enumerate(plan.stages):
        if si and stage:
            s.fence()
            fill_code = None
        for job in [x for j in stage for x in normalize(j)]:
            if job.src is not None:
                from ..exactness import require
                require(job.scale==1,'movement DMA requires unit scale; retain exact host requantization')
                for dst in job.dsts:
                    layout.rectangle(s,job.src,dst,job.rows,job.cols,job.sld,job.dld)
                ld=st=None
                continue
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
            if st != job.dld:
                s.config_st(job.dld)
                st = job.dld
            def tile(r,c,rh,cw):
                for arg,off in job.dsts:
                    s.mvout(s.at(arg,off+r*job.dld+c),SP(FILL_SLOT),cw,rh)
            for count,r,rh in ((job.rows//DIM,0,DIM),(bool(job.rows%DIM),job.rows//DIM*DIM,job.rows%DIM)):
                if not count:
                    continue
                def row(r=r,rh=rh):
                    layout.repeat(s,job.cols//DIM,{arg:DIM for arg,_ in job.dsts},lambda:tile(r,0,rh,DIM))
                    if job.cols%DIM:
                        tile(r,job.cols//DIM*DIM,rh,job.cols%DIM)
                layout.repeat(s,count,{arg:DIM*job.dld for arg,_ in job.dsts},row)


# ---------------------------------------------------------------------------------------------
# Op builders.  Shapes are row-major; offsets/strides are bytes of int8 elements.
# ---------------------------------------------------------------------------------------------

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


def fill(stage, dst_name, n, code):
    stage.append(Copy(None, 0, [(dst_name, 0)], n, 1, n, code=code))


def full(plan: Plan, out: str, n: int, code: int):
    fill(plan.stage(), out, n, code)


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
