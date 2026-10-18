"""Classify an extracted linear map (``linmap``) into a device kernel.

* copy maps (every output element is at most one input element times a per-source
  requantization ratio, or a constant): a gather compiler turns output rows into maximal
  contiguous runs and stacks runs of consecutive rows with a constant source stride into
  2-D DMA jobs (reflection/replication pads, repeat, select/slice scatter, padding, cat).
  Every output byte is written once and only inputs are read: no fence is needed.
* sum maps (every output a weighted sum of one input's elements with exact integer weights
  up to a per-output readout scale): accumulate-on-MVIN (``recipes.asm``).  When the map is
  the identity on some dims (batch, channels) and the same elsewhere, those dims become one
  contiguous channel block: the operand and result are transposed channels-last on the
  device (avg/adaptive pools and their backwards); otherwise every element is a row (col2im).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ops
from .exactness import prove_readout, rational as Q, require, accumulator_bound
from .linmap import LinearMap
from .match import Decline
from .numerics import code as qcode
from .recipes import movement as mv
from .recipes.asm import accumulate_sum
from .stream import Arg


def _strides(shape):
    return mv.strides(list(shape))


@dataclass
class Src:
    name: str
    shape: tuple
    scale: float          # real value = scale * code


def gather_jobs(lm: LinearMap, srcs: dict[int, Src], s_out: float, out: str) -> list[mv.Copy]:
    if any(d <= 0 for d in lm.out_shape) or any(any(d <= 0 for d in src.shape) for src in srcs.values()):
        raise Decline("empty or dynamic gather shape")
    for coefficient in np.unique(lm.coef):
        prove_readout(Q(float(coefficient)) / Q(s_out), -128, 127)
    shp = lm.out_shape
    n = int(np.prod(shp)) if shp else 1
    cnt = np.bincount(lm.out, minlength=n)
    if np.any(cnt > 1):
        raise Decline("not a copy map")
    has = np.zeros(n, bool)
    src = np.full(n, -1, np.int64)
    idx = np.zeros(n, np.int64)
    ratio = np.zeros(n, np.float64)
    has[lm.out] = True
    src[lm.out] = lm.src
    idx[lm.out] = lm.idx
    ratio[lm.out] = lm.coef / s_out
    const = np.broadcast_to(lm.const, (n,))
    for sid in np.unique(lm.src):
        rs = np.unique(ratio[has & (src == sid)])
        if rs.size != 1:
            raise Decline("per-element requantization ratio")
    if np.any(const[has] != 0):
        raise Decline("copy with an offset")
    codes = np.array([qcode(float(c), s_out) for c in const]) if np.any(~has) else np.zeros(n, np.int64)
    Cl = shp[-1] if shp else 1
    R = n // Cl
    col = np.arange(n) % Cl
    # continuation of a run from element p-1 within one output row
    cont = np.zeros(n, bool)
    cont[1:] = (col[1:] != 0) & (has[1:] == has[:-1])
    cont[1:] &= np.where(has[1:], (src[1:] == src[:-1]) & (idx[1:] == idx[:-1] + 1) & (ratio[1:] == ratio[:-1]),
                         codes[1:] == codes[:-1])
    starts = np.flatnonzero(~cont)
    lens = np.diff(np.r_[starts, n])
    # stack runs of consecutive rows: same column, length and source, constant source delta
    open_jobs: dict[tuple, list] = {}
    jobs = []
    for p0, ln in zip(starts.tolist(), lens.tolist()):
        r, c0 = divmod(p0, Cl)
        if has[p0]:
            key = ("copy", c0, ln, int(src[p0]), float(ratio[p0]))
            base = int(idx[p0])
        else:
            key = ("fill", c0, ln, int(codes[p0]))
            base = 0
        j = open_jobs.get(key)
        if j is not None and j["row"] + j["rows"] == r and (j["rows"] == 1 or base - j["last"] == j["delta"]) \
                and base - j["last"] >= 0:
            if j["rows"] == 1:
                j["delta"] = base - j["last"]
            j["rows"] += 1
            j["last"] = base
            continue
        if j is not None:
            jobs.append(j)
        open_jobs[key] = dict(key=key, row=r, rows=1, first=base, last=base, delta=0)
    jobs.extend(open_jobs.values())
    out_jobs = []
    for j in sorted(jobs, key=lambda j: (j["row"], j["key"][1])):
        kind, c0, ln = j["key"][:3]
        dst = (out, j["row"] * Cl + c0)
        if kind == "copy":
            sid, r = j["key"][3], j["key"][4]
            out_jobs.append(mv.Copy((srcs[sid].name, j["first"]), j["delta"], [dst], Cl, j["rows"], ln, r))
        else:
            out_jobs.append(mv.Copy(None, 0, [dst], Cl, j["rows"], ln, code=j["key"][3]))
    return out_jobs


def copy_kernel(op: str, lm: LinearMap, srcs: dict[int, Src], s_out: float) -> ops.Kernel:
    jobs = gather_jobs(lm, srcs, s_out, "y")
    plan = mv.Plan()
    plan.stage().extend(jobs)
    args = [Arg(srcs[k].name, "i8", tuple(srcs[k].shape), "input") for k in sorted(srcs)] + \
        [Arg("y", "i8", tuple(lm.out_shape), "output")]
    return ops._plan_kernel(op, args, plan, form="gather_jobs", jobs=len(jobs))


# ---------------------------------------------------------------------------------------------
# Sum maps
# ---------------------------------------------------------------------------------------------

def _identity_dims(lm: LinearMap, in_shape, out_shape):
    """Pairs (out dim, in dim) on which every term maps an index to itself."""
    oc = np.unravel_index(lm.out, out_shape)
    ic = np.unravel_index(lm.idx, in_shape)
    pairs = []
    used = set()
    for i, od in enumerate(out_shape):
        for j, idm in enumerate(in_shape):
            if j in used or od != idm:
                continue
            if np.array_equal(oc[i], ic[j]):
                pairs.append((i, j))
                used.add(j)
                break
    return pairs, oc, ic


def _refine(lm: LinearMap, in_shape, out_shape):
    """Split input dims so that an output dim is the leading factor of one (col2im's C*KK rows)."""
    shape = list(in_shape)
    changed = True
    while changed:
        changed = False
        ic = np.unravel_index(lm.idx, shape)
        oc = np.unravel_index(lm.out, out_shape)
        for j, size in enumerate(shape):
            for i, osz in enumerate(out_shape):
                if osz <= 1 or size <= osz or size % osz:
                    continue
                inner = size // osz
                if np.array_equal(oc[i], ic[j] // inner):
                    shape[j:j + 1] = [osz, inner]
                    changed = True
                    break
            if changed:
                break
    return tuple(shape)


def sum_kernel(op: str, lm: LinearMap, src: Src, s_out: float, *, weights: np.ndarray | None = None,
               rho: np.ndarray | None = None) -> ops.Kernel:
    """``weights`` (per term) and ``rho`` (per output element) state the integer form explicitly;
    both are mandatory: floating coefficient approximation is not an exact source proof."""
    require(weights is not None and rho is not None, "linear sum weights lack a rational source proof")
    require(all(d > 0 for d in (*lm.out_shape, *src.shape)), "empty or dynamic sum shape")
    require(np.all(np.asarray(weights) == np.trunc(weights)) and np.all((np.asarray(weights) > 0) & (np.asarray(weights) <= 16)), "sum requires integer weights in 1..16")
    bounds = np.bincount(lm.out, weights=np.abs(weights), minlength=int(np.prod(lm.out_shape))) * 128
    for scale, bound in set(zip(np.asarray(rho).reshape(-1), bounds)):
        accumulator_bound(int(bound))
        prove_readout(Q(scale), -int(bound), int(bound))
    if np.any(lm.const != 0):
        raise Decline("sum with an offset")
    out_shape = tuple(lm.out_shape)
    arg_shape = tuple(src.shape)
    in_shape = _refine(lm, arg_shape, out_shape)
    pairs, oc, ic = _identity_dims(lm, in_shape, out_shape)
    C = int(np.prod([out_shape[i] for i, _ in pairs])) if pairs else 1
    if C >= 16:
        oid = [i for i, _ in pairs]
        iid = [j for _, j in pairs]
        osp = [d for d in range(len(out_shape)) if d not in oid]
        isp = [d for d in range(len(in_shape)) if d not in iid]
        o_sp_shape = [out_shape[d] for d in osp]
        i_sp_shape = [in_shape[d] for d in isp]
        id_flat = np.ravel_multi_index([oc[i] for i in oid], [out_shape[i] for i in oid])
        o_row = np.ravel_multi_index([oc[d] for d in osp], o_sp_shape) if osp else np.zeros_like(lm.out)
        i_row = np.ravel_multi_index([ic[d] for d in isp], i_sp_shape) if isp else np.zeros_like(lm.idx)
        # the spatial map must be identical for every channel index (checked for all of them)
        order = np.lexsort((lm.coef, i_row, o_row, id_flat))
        T = lm.out.size // C if lm.out.size % C == 0 else -1
        if T < 0 or np.bincount(id_flat, minlength=C).min() != T:
            raise Decline("map differs across channels")
        grid = [a[order].reshape(C, T) for a in (o_row, i_row, lm.coef)]
        if any(not np.all(g == g[0]) for g in grid):
            raise Decline("map differs across channels")
        ref = id_flat == 0
        rows_out = int(np.prod(o_sp_shape)) if osp else 1
        sub = LinearMap(tuple(o_sp_shape), o_row[ref], np.zeros(int(ref.sum()), np.int64), i_row[ref],
                        lm.coef[ref], np.zeros(rows_out), {})
        W = np.asarray(weights)[ref]
        rho_rows = np.zeros(rows_out)
        rho_rows[o_row[ref]] = np.asarray(rho).reshape(-1)[lm.out[ref]]
        in_perm = isp + iid
        out_perm = osp + oid
        args = [Arg(src.name, "i8", arg_shape, "input"), Arg("y", "i8", out_shape, "output"),
                Arg("ws_x", "i8", tuple(in_shape[d] for d in in_perm), "scratch"),
                Arg("ws_y", "i8", tuple(out_shape[d] for d in out_perm), "scratch")]
        s = ops._begin(args)
        x_in, y_out = src.name, "y"
        if in_perm != list(range(len(in_shape))):
            ops._permute_into(s, src.name, list(in_shape), "ws_x", in_perm)
            s.fence()
            x_in = "ws_x"
        direct_out = out_perm == list(range(len(out_shape)))
        accumulate_sum(s, x_in, "y" if direct_out else "ws_y", C, rows_out, sub.out, sub.idx, W, rho_rows)
        if not direct_out:
            s.fence()
            inv = [out_perm.index(d) for d in range(len(out_shape))]
            ops._permute_into(s, "ws_y", [out_shape[d] for d in out_perm], "y", inv)
        return ops._end(op, s, form="accumulate_sum", channels=C, rows=rows_out, terms=int(sub.out.size))
    n_out = int(np.prod(out_shape))
    W, rho = np.asarray(weights), np.asarray(rho).reshape(-1)
    args = [Arg(src.name, "i8", arg_shape, "input"), Arg("y", "i8", out_shape, "output")]
    s = ops._begin(args)
    accumulate_sum(s, src.name, "y", 1, n_out, lm.out, lm.idx, W, rho)
    return ops._end(op, s, form="accumulate_sum", channels=1, rows=n_out, terms=int(lm.out.size))


def kernel_for(op: str, lm: LinearMap, srcs: dict[int, Src], s_out: float) -> ops.Kernel:
    n = int(np.prod(lm.out_shape)) if lm.out_shape else 1
    cnt = np.bincount(lm.out, minlength=n) if lm.out.size else np.zeros(n, np.int64)
    if cnt.max(initial=0) <= 1:
        return copy_kernel(op, lm, srcs, s_out)
    if len(srcs) != 1 or len(np.unique(lm.src)) != 1:
        raise Decline("sum over several inputs")
    return sum_kernel(op, lm, next(iter(srcs.values())), s_out)
