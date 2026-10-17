"""Affine loop rerolling of a static primitive command stream into ordinary CPU loops.

Each command is a record ``(key, values)``: ``key`` is the RoCC funct plus the DRAM
argument it addresses (or none), ``values`` are its encoded source words with the
argument's base pointer removed.  A block of ``L`` consecutive nodes repeats when the
keys are identical and every value advances by a constant per repetition; such a run
becomes ``Loop(count, body, step)``.  Passes repeat so inner loops become single nodes
of outer runs.  Because the comparison is on the packed words, local scratchpad and
accumulator addresses, DRAM offsets and even extents may vary affinely; ``expand``
reproduces the original words exactly and is checked before emission.  The loops are
plain CPU control flow; no Gemmini loop opcode is involved.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Rec:
    key: tuple
    values: tuple[int, ...]


@dataclass(frozen=True)
class Loop:
    count: int
    body: tuple
    step: tuple[int, ...]       # per flat value of one body iteration


def flat_values(node) -> tuple[int, ...]:
    if isinstance(node, Rec):
        return node.values
    out: list[int] = []
    for b in node.body:
        out.extend(flat_values(b))
    return tuple(out)


class _Interner:
    def __init__(self):
        self.ids: dict = {}

    def sig(self, node) -> int:
        if isinstance(node, Rec):
            key = ("r", node.key, len(node.values))
        else:
            key = ("l", node.count, tuple(self.sig(b) for b in node.body), node.step)
        hit = self.ids.get(key)
        if hit is None:
            hit = self.ids[key] = len(self.ids)
        return hit


def _roll_pass(nodes: list, interner: _Interner, max_period: int, min_cover: int) -> tuple[list, bool]:
    n = len(nodes)
    sid = [interner.sig(x) for x in nodes]
    vals = [flat_values(x) for x in nodes]
    where: dict[int, list[int]] = {}
    for i, s in enumerate(sid):
        where.setdefault(s, []).append(i)
    out: list = []
    changed = False
    i = 0

    def block(start, period):
        return [v for x in vals[start:start + period] for v in x]

    while i < n:
        best = None
        cands = where[sid[i]]
        k = bisect_right(cands, i)
        tried = 0
        while k < len(cands) and cands[k] - i <= max_period and tried < 16:
            j = cands[k]
            k += 1
            tried += 1
            period = j - i
            if j + period > n or sid[i:j] != sid[j:j + period]:
                continue
            base, nxt = block(i, period), block(j, period)
            step = tuple(b - a for a, b in zip(base, nxt))
            count, prev = 2, nxt
            while True:
                s0 = i + count * period
                if s0 + period > n or sid[s0:s0 + period] != sid[i:j]:
                    break
                cur = block(s0, period)
                if any(c - p != d for c, p, d in zip(cur, prev, step)):
                    break
                prev = cur
                count += 1
            if period * count >= min_cover:
                best = (period, count, step)
                break
        if best is not None:
            period, count, step = best
            out.append(Loop(count, tuple(nodes[i:i + period]), step))
            i += period * count
            changed = True
        else:
            out.append(nodes[i])
            i += 1
    return out, changed


def reroll(records: list[Rec], *, max_period: int = 512, min_cover: int = 4) -> list:
    nodes: list = list(records)
    interner = _Interner()
    for _ in range(16):
        nodes, changed = _roll_pass(nodes, interner, max_period, min_cover)
        if not changed:
            break
    return nodes


def _shift(nodes, deltas, pos: int = 0):
    out = []
    for node in nodes:
        if isinstance(node, Rec):
            w = len(node.values)
            node = Rec(node.key, tuple(v + d for v, d in zip(node.values, deltas[pos:pos + w])))
            pos += w
            out.append(node)
        else:
            body, pos = _shift(node.body, deltas, pos)
            out.append(Loop(node.count, tuple(body), node.step))
    return out, pos


def expand(nodes) -> list[Rec]:
    """Re-expand a loop tree; ``expand(reroll(r)) == r`` is checked before emission."""
    out: list[Rec] = []
    for node in nodes:
        if isinstance(node, Rec):
            out.append(node)
            continue
        for it in range(node.count):
            body, _ = _shift(node.body, [it * d for d in node.step])
            out.extend(expand(body))
    return out


def node_count(nodes) -> int:
    return sum(1 if isinstance(n, Rec) else 1 + node_count(n.body) for n in nodes)
