"""Static hazard lint of a primitive stream against stock-RTL ordering rules.

Rules (analysis/recipe_corrections.md, HW1-HW4 and the DRAM rule):

* HW1  no EX operand (COMPUTE A/D, PRELOAD B/D) may address the accumulator: the bank
       read is never answered and the reservation station stalls.
* HW2  a zero-fill MVIN into the accumulator must use a load state configured with
       shrink = 0 (the shrunk ZeroWriter byte count trips DMACommandTracker).
* HW3  a MVIN must not follow an overlapping MVIN of the same local rows/columns unless
       both accumulate, the first is a zero-fill and the second accumulates, an EX or
       MVOUT touching those rows sits between them, or a fence does.
* HW4  pooled spad-source MVOUTs are never emitted (pooling stays disabled).
* DRAM a MVIN or MVOUT overlapping bytes an earlier MVOUT of this stream wrote needs a
       fence in between (the reservation station does not track DRAM addresses).
* C7   ReLU is never combined with a negative readout scale (Spike and RTL disagree).
* Spike a scratchpad MVOUT reads rows below 1024 only.
"""

from __future__ import annotations

import numpy as np

from .stream import ACC_BIT, ACCUM_BIT, DIM, GARBAGE, Stream


class HazardError(ValueError):
    pass


def _rows(local: int, rows: int, cols: int, block_stride: int):
    """(space, first row, row count, first col, col count) per DIM-wide block of a DMA."""
    space = "acc" if local & ACC_BIT else "sp"
    base = local & 0x3FFF if space == "acc" else local
    out = []
    for b0 in range(0, cols, DIM):
        out.append((space, base + (b0 // DIM) * block_stride, rows, 0, min(DIM, cols - b0)))
    return out


class _Dram:
    """Bytes written by MVOUTs since the last fence, as an epoch-stamped map per argument."""

    def __init__(self, stream: Stream):
        self.maps = {i: np.zeros(max(1, a.nbytes), np.int32) for i, a in enumerate(stream.args)}
        self.epoch = 1
        self.dirty: set[int] = set()

    def clear(self):
        self.epoch += 1
        self.dirty.clear()

    @staticmethod
    def spans(ins, stride, cols, width):
        rows = ins.a("rows")
        return [(ins.off + r * stride, ins.off + r * stride + cols * width) for r in range(rows)]

    def overlaps(self, arg, spans) -> bool:
        if arg not in self.dirty:
            return False
        m = self.maps[arg]
        return any(bool((m[lo:hi] == self.epoch).any()) for lo, hi in spans)

    def add(self, arg, spans):
        m = self.maps[arg]
        for lo, hi in spans:
            if hi > m.size:
                raise HazardError(f"DMA beyond argument {arg}: bytes [{lo}, {hi}) of {m.size}")
            m[lo:hi] = self.epoch
        self.dirty.add(arg)


def lint(stream: Stream) -> dict:
    ld = {s: dict(stride=0, shrunk=0, block_stride=DIM, scale=1.0) for s in range(3)}
    st = dict(stride=0, acc_act=0, acc_scale=1.0)
    pending: dict[tuple[str, int], list] = {}       # (space,row) -> [(c0, c1, accumulate, zero, index)]
    dram = _Dram(stream)
    fences = 0
    errors: list[str] = []

    def err(i, msg):
        errors.append(f"#{i} {msg}")

    def touch(space, row, rows):
        for r in range(row, row + rows):
            pending.pop((space, r), None)

    for i, ins in enumerate(stream.ins):
        k = ins.kind
        if k == "fence":
            pending.clear()
            dram.clear()
            fences += 1
        elif k == "config_ld":
            ld[ins.a("load_id")] = dict(stride=ins.a("stride"), shrunk=ins.a("shrunk"),
                                        block_stride=ins.a("block_stride"), scale=ins.a("scale"))
        elif k == "config_st":
            st = dict(stride=ins.a("stride"), acc_act=ins.a("acc_act"), acc_scale=ins.a("acc_scale"))
            if st["acc_act"] and st["acc_scale"] < 0:
                err(i, "C7: ReLU readout with a negative scale")
        elif k == "config_ex":
            if ins.a("act"):
                err(i, "EX activation is not used by these recipes")
        elif k == "mvin":
            cfg = ld[ins.a("load_id")]
            local = ins.a("local")
            is_acc = bool(local & ACC_BIT)
            accumulate = bool(local & ACCUM_BIT)
            zero = bool(ins.a("zero"))
            if is_acc and zero and cfg["shrunk"]:
                err(i, "HW2: accumulator zero-fill through a shrunk load state")
            if not zero:
                width = 4 if is_acc and not cfg["shrunk"] else 1
                spans = dram.spans(ins, cfg["stride"], ins.a("cols"), width)
                if dram.overlaps(ins.arg, spans):
                    err(i, "DRAM RAW: MVIN reads bytes an unfenced MVOUT wrote")
            for space, row, rows, c0, nc in _rows(local, ins.a("rows"), ins.a("cols"), cfg["block_stride"]):
                for r in range(row, row + rows):
                    prior = pending.get((space, r))
                    if prior:
                        for (p0, p1, pacc, pzero, pidx) in prior:
                            if p0 < c0 + nc and c0 < p1:
                                if (pacc and accumulate) or (pzero and accumulate):
                                    continue
                                err(i, f"HW3: MVIN after overlapping MVIN #{pidx} on {space} row {r} with no fence")
                                break
                    if accumulate:
                        history = pending.setdefault((space, r), [])
                        current = (c0, c0 + nc, accumulate, zero, i)
                        # Repeated accumulating writes grant exactly the same
                        # overlap permissions. Keep one fact for an identical
                        # interval instead of scanning its entire history.
                        if history and history[-1][:3] == current[:3]:
                            history[-1] = current
                        else:
                            history.append(current)
                    else:
                        pending[(space, r)] = [(c0, c0 + nc, accumulate, zero, i)]
        elif k == "mvout":
            local = ins.a("local")
            if local & ACC_BIT:
                if not local & (1 << 29):
                    pass
                width = 4 if local & (1 << 29) else 1
                space, row = "acc", local & 0x3FFF
                blocks = (ins.a("cols") + DIM - 1) // DIM
                touch(space, row, blocks * DIM if blocks > 1 else ins.a("rows"))
            else:
                width = 1
                if local + ins.a("rows") > 1024:
                    err(i, "Spike: scratchpad MVOUT from rows >= 1024")
                touch("sp", local, ins.a("rows"))
            spans = dram.spans(ins, st["stride"], ins.a("cols"), width)
            if dram.overlaps(ins.arg, spans):
                err(i, "DRAM WAW: MVOUT overlaps an unfenced earlier MVOUT")
            dram.add(ins.arg, spans)
        elif k == "preload":
            bd, c = ins.a("bd"), ins.a("c")
            if bd != GARBAGE and bd & ACC_BIT:
                err(i, "HW1: PRELOAD B from the accumulator")
            if bd != GARBAGE:
                touch("sp", bd, max(ins.a("bd_rows"), ins.a("bd_cols")))
            if c != GARBAGE:
                touch("acc" if c & ACC_BIT else "sp", c & 0x3FFF if c & ACC_BIT else c, ins.a("c_rows"))
        elif k == "compute":
            a, bd = ins.a("a"), ins.a("bd")
            if a & ACC_BIT and a != GARBAGE:
                err(i, "HW1: COMPUTE A from the accumulator")
            if bd != GARBAGE:
                err(i, "WS D operand is not used by these recipes")
            touch("sp", a, max(ins.a("a_rows"), ins.a("a_cols")))
    if errors:
        raise HazardError(f"{len(errors)} hazard(s): " + "; ".join(errors[:8]))
    return {"status": "pass", "instructions": len(stream.ins), "fences": fences}
