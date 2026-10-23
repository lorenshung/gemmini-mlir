"""The merlin_iface v0.1 extension opcodes (Merlin 74bd4290, cc50cc01), lowered to one kernel.

AFFINE_MAP, the MOVEMENT forms, CONCAT, INSERT_SLICE, FILL, FENCE and WINDOW_SUM are implemented
with the Core ATen recipes, following each opcode's exact definition in
``command_buffer_abi.yaml``:

* movement forms, CONCAT, INSERT_SLICE and FILL are gather maps: the stated numpy semantics are
  applied to tagged index arrays of the sources, and the gather compiler turns the result into
  strided DMA jobs; an optional ``scale`` is the one MVIN-scale rounding (pad codes included).
* AFFINE_MAP is the exact int32 linear combination (signed integer weights as c*I parts, the int32
  constant as a stride-0 seed, ReLU on the accumulator) with one readout, or for a single unit
  weight the identical single MVIN-scale rounding; an i32 result is read out at full width.
* WINDOW_SUM is accumulate-on-MVIN with the definition's own integer form: unit weights and
  readout_scale / count per output, or the transposed modes' L / count weights and readout_scale / L.
* FENCE is a fence.

The program becomes one ``gemmini_kernel`` whose pointer arguments are declared by an explicit
``kernel_abi`` (every declared tensor once: leaves read, results written, kernel scratch as
``intermediate``).  Every command's kernel ends with a fence, so a later command never reads DRAM an
earlier one is still writing.
"""

from __future__ import annotations

import numpy as np
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, IntegerType, StringAttr, TensorType

from . import exact_linear, ops
from .exactness import rational as Q, accumulator_bound, quantized
from .emit import build_module
from .linmap import LinearMap
from .match import Decline
from .recipes import elementwise as ew
from .recipes import movement as mv
from .stream import Arg, Ins, Stream, f32

EXTENSION = {"affine_map", "concat", "insert_slice", "fill", "fence", "window_sum"}
_ROLE = {"input": "input", "weight": "input", "bias": "input", "output": "output"}
_RANGE = {"i8": (-128, 127), "i32": (-(2 ** 31), 2 ** 31 - 1)}


class IfaceDecline(Exception):
    def __init__(self, reason, op="", shape=()):
        super().__init__(reason)
        self.reason, self.op, self.shape = reason, op, list(shape)


def _py(attr):
    if isinstance(attr, IntegerAttr):
        t = attr.type
        if isinstance(t, IntegerType) and t.width.data == 1:
            return bool(attr.value.data)
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        return float(attr.value.data)
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, ArrayAttr):
        return [_py(a) for a in attr.data]
    return attr


def _mnemonic(op) -> str:
    return op.name.split(".", 1)[1] if op.name.startswith("merlin_iface.") else ""


def uses_extension(module) -> bool:
    for op in module.walk():
        m = _mnemonic(op)
        if m in EXTENSION or (m == "movement" and ("form" in op.attributes or "scale" in op.attributes)):
            return True
    return False


def _shape_dtype(v):
    t = v.type
    if not isinstance(t, TensorType):
        raise IfaceDecline("operand is not a tensor")
    shape = tuple(int(d) for d in t.get_shape())
    if any(d <= 0 for d in shape):
        raise IfaceDecline("empty or dynamic tensor shape")
    return shape, str(t.get_element_type())


# ---------------------------------------------------------------------------------------------
# Semantics as linear maps
# ---------------------------------------------------------------------------------------------

def _tags(sid, shape):
    n = int(np.prod(shape)) if shape else 1
    return (np.int64(sid) << 40) + np.arange(n, dtype=np.int64).reshape(shape)


def _gather_map(out_tags: np.ndarray, ratios: dict, fill_codes: np.ndarray | None) -> LinearMap:
    flat = out_tags.reshape(-1)
    has = flat >= 0
    p = np.arange(flat.size, dtype=np.int64)
    src = flat[has] >> 40
    idx = flat[has] & ((1 << 40) - 1)
    coef = np.array([ratios[int(s)] for s in src], np.float64) if src.size else np.zeros(0)
    const = np.zeros(flat.size) if fill_codes is None else np.where(has, 0, fill_codes.reshape(-1)).astype(np.float64)
    return LinearMap(tuple(out_tags.shape), p[has], src, idx, coef, const, {})


def _requant_code(code: int, scale) -> int:
    if scale is None:
        return code
    return int(quantized(Q(code) * Q(scale)))


def _check_range(attrs, dtype, op):
    lo, hi = _RANGE.get(dtype, (None, None))
    if lo is None:
        raise IfaceDecline(f"{op}: output dtype {dtype} is not implemented", op)
    if attrs.get("quant_min", lo) != lo or attrs.get("quant_max", hi) != hi:
        raise IfaceDecline(f"{op}: a narrowed saturation range is not implemented", op)


def movement_kernel(attrs, src_shape, src_dtype, dst_shape, dst_dtype) -> ops.Kernel:
    form = attrs.get("form", "identity")
    scale = attrs.get("scale")
    _check_range(attrs, dst_dtype, "MOVEMENT")
    if form == "identity" and src_dtype == "i8" and dst_dtype == "i32" and scale is None:
        return ops.to_copy_widen(src_shape)
    if src_dtype != "i8" or dst_dtype != "i8":
        raise IfaceDecline(f"MOVEMENT {src_dtype} -> {dst_dtype} is not implemented", "MOVEMENT", dst_shape)
    t = _tags(0, src_shape)
    fill = None
    if form == "identity":
        out = t
    elif form == "permute":
        out = np.transpose(t, attrs["perm"])
    elif form == "slice":
        out = t[tuple(slice(o, o + n * s, s) for o, n, s in zip(attrs["offsets"], attrs["sizes"], attrs["strides"]))]
    elif form == "pad":
        widths = list(zip(attrs["low"], attrs["high"]))
        mode = attrs["mode"]
        if mode == "constant":
            out = np.pad(t, widths, mode="constant", constant_values=-1)
            fill = np.full(out.shape, _requant_code(int(attrs["value_code"]), scale))
        else:
            out = np.pad(t, widths, mode="reflect" if mode == "reflect" else "edge")
    elif form == "broadcast":
        out = np.broadcast_to(t, tuple(dst_shape))
    elif form == "repeat":
        out = np.tile(t, attrs["repeats"])
    else:
        raise IfaceDecline(f"MOVEMENT form {form!r}", "MOVEMENT")
    if tuple(out.shape) != tuple(dst_shape):
        raise IfaceDecline("MOVEMENT result extent differs from the declared one", "MOVEMENT", dst_shape)
    lm = _gather_map(np.ascontiguousarray(out), {0: 1.0 if scale is None else f32(scale)}, fill)
    return exact_linear.kernel("MOVEMENT", lm, {0: exact_linear.Src("x0", tuple(src_shape), 1.0)}, 1.0)


def concat_kernel(attrs, src_shapes, dst_shape) -> ops.Kernel:
    _check_range(attrs, attrs.get("output_dtype", "i8"), "CONCAT")
    scales = attrs.get("scales")
    parts = [_tags(i, sh) for i, sh in enumerate(src_shapes)]
    out = np.concatenate(parts, axis=attrs["axis"])
    ratios = {i: 1.0 if scales is None else f32(scales[i]) for i in range(len(src_shapes))}
    srcs = {i: exact_linear.Src(f"x{i}", tuple(sh), 1.0) for i, sh in enumerate(src_shapes)}
    return exact_linear.kernel("CONCAT", _gather_map(out, ratios, None), srcs, 1.0)


def insert_slice_kernel(attrs, dest_shape, src_shape) -> ops.Kernel:
    out = _tags(0, dest_shape).copy()
    sl = tuple(slice(o, o + n * s, s) for o, n, s in zip(attrs["offsets"], src_shape, attrs["strides"]))
    out[sl] = _tags(1, src_shape)
    srcs = {0: exact_linear.Src("x0", tuple(dest_shape), 1.0), 1: exact_linear.Src("x1", tuple(src_shape), 1.0)}
    return exact_linear.kernel("INSERT_SLICE", _gather_map(out, {0: 1.0, 1: 1.0}, None), srcs, 1.0)


def fill_kernel(attrs, dst_shape) -> ops.Kernel:
    dt = attrs["output_dtype"]
    if dt != "i8":
        raise IfaceDecline(f"FILL {dt} is not implemented", "FILL", dst_shape)
    plan = mv.Plan()
    mv.full(plan, "y", int(np.prod(dst_shape)), int(attrs["value_code"]))
    return ops._plan_kernel("FILL", [Arg("y", "i8", tuple(dst_shape), "output")], plan)


def affine_kernel(attrs, x_shape, y_shape, dst_dtype) -> ops.Kernel:
    w = list(attrs["weights"])
    c = int(attrs["constant"])
    relu = attrs["activation"] == "relu"
    sc = float(attrs["readout_scale"])
    _check_range(attrs, dst_dtype, "AFFINE_MAP")
    x_shape = tuple(x_shape)
    n = int(np.prod(x_shape)) if x_shape else 1
    two = y_shape is not None
    args = [Arg("x", "i8", x_shape, "input")] + ([Arg("y", "i8", tuple(y_shape), "input")] if two else []) + \
        [Arg("dst", dst_dtype, x_shape, "output")]
    if dst_dtype == "i32":
        if sc != 1.0 or relu:
            raise IfaceDecline("AFFINE_MAP i32 has no readout", "AFFINE_MAP")
        if w == [1] and c == 0 and not two:
            return ops.to_copy_widen(x_shape)                      # args x -> y, exact widening
    bound = abs(c) + sum(128 * abs(v) for v in w)
    accumulator_bound(bound)
    if dst_dtype == 'i8':
        kernel = (ops.add_tensor('AFFINE_MAP',x_shape,y_shape,wa=w[0],alpha=w[1],c=c,s_a=1,s_b=1,s_out=Q(1)/Q(sc),relu=relu)
                  if two else ops.add_scalar('AFFINE_MAP',x_shape,c,w=w[0],s_x=1,s_out=Q(1)/Q(sc),relu=relu))
        return kernel
    s = ops._begin(args)
    if not two and w == [1] and c == 0 and not relu and dst_dtype == "i8":
        ew.unary_scale(s, "x", "dst", n, sc)                      # the identical single rounding
        return ops._end("AFFINE_MAP", s, form="mvin_scale")
    N = x_shape[-1] if x_shape else 1
    if two and tuple(y_shape) != x_shape:
        lead = len(x_shape) - len(y_shape)
        if tuple(x_shape[lead:]) != tuple(y_shape) or int(np.prod(y_shape)) == 0:
            raise IfaceDecline("AFFINE_MAP y broadcasts along leading dims only", "AFFINE_MAP")
        inner = int(np.prod(y_shape))
        rows, cols, xs, ys = n // inner, inner, inner, 0
    elif n % 64 == 0:
        rows, cols, xs, ys = n // 64, 64, 64, 64
    else:
        rows, cols, xs, ys = n // N, N, N, N
    terms = []
    if w[0]:
        terms.append(("x", xs, ops.parts127(w[0])))
    if two and w[1]:
        terms.append(("y", ys, ops.parts127(w[1])))
    if not terms:
        raise IfaceDecline("AFFINE_MAP with all-zero weights", "AFFINE_MAP")
    if relu and sc <= 0:
        raise IfaceDecline("AFFINE_MAP relu with a nonpositive readout_scale", "AFFINE_MAP")
    try:
        ew.lincomb(s, terms, "dst", rows, cols, sc, bias=c or None, relu=relu, out_i32=dst_dtype == "i32")
    except ValueError as exc:
        raise IfaceDecline(f"AFFINE_MAP: {exc}", "AFFINE_MAP")
    return ops._end("AFFINE_MAP", s, form="lincomb")


# --- WINDOW_SUM ------------------------------------------------------------------------------

def _pool_windows(size, kernel, stride, pad, ceil_mode, include_pad):
    span = size + 2 * pad - kernel
    out = (-(-span // stride) if ceil_mode else span // stride) + 1
    if ceil_mode and (out - 1) * stride >= size + pad:
        out -= 1
    res = []
    for index in range(out):
        start = index * stride - pad
        end = min(start + kernel, size + pad)
        lo, hi = max(start, 0), min(end, size)
        res.append((lo, hi, end - start if include_pad else hi - lo))
    return res


def _adaptive_windows(size, out):
    return [((i * size) // out, -(-((i + 1) * size) // out), -(-((i + 1) * size) // out) - (i * size) // out)
            for i in range(out)]


def window_sum_kernel(attrs, src_shape, dst_shape) -> ops.Kernel:
    if not src_shape or any(d <= 0 for d in (*src_shape, *dst_shape)):
        raise IfaceDecline("empty or dynamic window shape", "WINDOW_SUM", dst_shape)
    if 'divisor_override' in attrs and attrs['divisor_override'] <= 0:
        raise IfaceDecline("window divisor must be positive", "WINDOW_SUM", dst_shape)
    for key in ('kernel','stride','dilation'):
        if any(d <= 0 for d in attrs.get(key, [])):
            raise IfaceDecline("window extents must be positive", "WINDOW_SUM", dst_shape)
    mode = attrs["mode"]
    dt = attrs["output_dtype"]
    _check_range(attrs, dt, "WINDOW_SUM")
    if dt != "i8":
        raise IfaceDecline("WINDOW_SUM i32 readout is not implemented", "WINDOW_SUM", dst_shape)
    scale = Q(attrs["readout_scale"])
    src_shape, dst_shape = tuple(src_shape), tuple(dst_shape)
    divisor = attrs.get("divisor_override")
    if mode == "reduce":
        if not attrs["dims"] or any(d < -len(src_shape) or d >= len(src_shape) for d in attrs["dims"]):
            raise IfaceDecline("invalid reduction dimensions", "WINDOW_SUM", dst_shape)
        dims = sorted({d % len(src_shape) for d in attrs["dims"]})
        count = int(np.prod([src_shape[d] for d in dims])) if attrs["average"] else 1
        readout = scale / (divisor or count)
        try:
            return ops.mean_dim(src_shape, dims, len(dst_shape) == len(src_shape), s_x=1.0, s_out=1.0,
                                w=readout, op="WINDOW_SUM")
        except ValueError:
            pass                                            # non-contiguous dims: a general sum map
        red = np.ones(len(src_shape), bool)
        red[dims] = False
        cells = np.indices(src_shape).reshape(len(src_shape), -1)
        kept = [d for d in range(len(src_shape)) if red[d]]
        o = np.ravel_multi_index([cells[d] for d in kept], [src_shape[d] for d in kept]) if kept else \
            np.zeros(cells.shape[1], np.int64)
        i = np.arange(cells.shape[1], dtype=np.int64)
        W = np.ones(o.size, np.int64)
        rho = np.full(int(np.prod(dst_shape)), readout, dtype=object)
        lm = LinearMap(dst_shape, o, np.zeros(o.size, np.int64), i, W * rho[o], np.zeros(rho.size), {})
        return _sum(lm, src_shape, W, rho, dst_shape)
    terms_o, terms_i, terms_w = [], [], []
    rho = None
    if mode == "fold":
        kernel, stride, padding, dilation = (attrs[k] for k in ("kernel", "stride", "padding", "dilation"))
        n, c = dst_shape[:2]
        out_sizes = dst_shape[2:]
        blocks = [(sz + 2 * p - d * (k - 1) - 1) // s + 1
                  for sz, p, d, k, s in zip(out_sizes, padding, dilation, kernel, stride)]
        cols_shape = (n, c, *kernel, *blocks)
        if int(np.prod(cols_shape)) != int(np.prod(src_shape)):
            raise IfaceDecline("WINDOW_SUM fold extents", "WINDOW_SUM", dst_shape)
        for tap in np.ndindex(*kernel):
            for block in np.ndindex(*blocks):
                pos = [b * s - p + t * d for b, s, p, t, d in zip(block, stride, padding, tap, dilation)]
                if all(0 <= q < sz for q, sz in zip(pos, out_sizes)):
                    nn, cc = np.meshgrid(np.arange(n), np.arange(c), indexing="ij")
                    terms_o.append(np.ravel_multi_index((nn, cc, *[np.full_like(nn, q) for q in pos]), dst_shape))
                    terms_i.append(np.ravel_multi_index((nn, cc, *[np.full_like(nn, t) for t in tap],
                                                         *[np.full_like(nn, b) for b in block]), cols_shape))
        o = np.concatenate([t.reshape(-1) for t in terms_o])
        i = np.concatenate([t.reshape(-1) for t in terms_i])
        W = np.ones(o.size, np.int64)
        rho = np.full(int(np.prod(dst_shape)), scale, dtype=object)
    else:
        transpose = mode.endswith("_transpose")
        source_shape = dst_shape if transpose else src_shape
        window_shape = src_shape if transpose else dst_shape
        layout = attrs.get("layout", "nhwc")
        rank = len(src_shape)
        if mode.startswith("adaptive"):
            spatial = rank - 2
        else:
            spatial = len(attrs["kernel"])
        axes = list(range(rank - 1 - spatial, rank - 1)) if layout == "nhwc" else list(range(rank - spatial, rank))
        if mode.startswith("adaptive"):
            per_dim = [_adaptive_windows(source_shape[a], window_shape[a]) for a in axes]
        else:
            per_dim = [_pool_windows(source_shape[a], k, s, p, bool(attrs.get("ceil_mode", False)),
                                     attrs["count_include_pad"])
                       for a, k, s, p in zip(axes, attrs["kernel"], attrs["stride"], attrs["padding"])]
        if tuple(len(w) for w in per_dim) != tuple(window_shape[a] for a in axes):
            raise IfaceDecline("WINDOW_SUM pooled extent", "WINDOW_SUM", dst_shape)
        others = [d for d in range(rank) if d not in axes]
        other_idx = np.indices([source_shape[d] for d in others]).reshape(len(others), -1) if others else \
            np.zeros((0, 1), np.int64)
        grid = list(np.ndindex(*[len(w) for w in per_dim]))
        counts = {g: (divisor or int(np.prod([per_dim[k][g[k]][2] for k in range(spatial)]))) for g in grid}

        rho_full = np.full(int(np.prod(dst_shape)), Q(0), dtype=object)
        for g in grid:
            box = [range(per_dim[k][g[k]][0], per_dim[k][g[k]][1]) for k in range(spatial)]
            cells = np.array(list(np.ndindex(*[len(b) for b in box])), np.int64).reshape(-1, spatial)
            cells = cells + np.array([b.start for b in box], np.int64)
            for cell in cells:
                full_w = [0] * rank
                full_c = [0] * rank
                for k, a in enumerate(axes):
                    full_w[a] = np.full(other_idx.shape[1], g[k])
                    full_c[a] = np.full(other_idx.shape[1], cell[k])
                for j, d in enumerate(others):
                    full_w[d] = other_idx[j]
                    full_c[d] = other_idx[j]
                w_flat = np.ravel_multi_index(full_w, window_shape)
                c_flat = np.ravel_multi_index(full_c, source_shape)
                if transpose:
                    terms_o.append(c_flat)
                    terms_i.append(w_flat)
                    terms_w.append(np.full(w_flat.size, Q(1) / counts[g], dtype=object))
                else:
                    terms_o.append(w_flat)
                    terms_i.append(c_flat)
                    terms_w.append(np.ones(w_flat.size, np.int64))
            if not transpose:
                full_w = [0] * rank
                for k, a in enumerate(axes):
                    full_w[a] = np.full(other_idx.shape[1], g[k])
                for j, d in enumerate(others):
                    full_w[d] = other_idx[j]
                rho_full[np.ravel_multi_index(full_w, window_shape)] = scale / counts[g]
        o = np.concatenate(terms_o)
        i = np.concatenate(terms_i)
        W = np.concatenate(terms_w)
        rho = np.full(int(np.prod(dst_shape)), scale, dtype=object) if transpose else rho_full
    lm = LinearMap(dst_shape, o, np.zeros(o.size, np.int64), i, W * rho[o], np.zeros(int(np.prod(dst_shape))), {})
    return _sum(lm, src_shape, W, rho, dst_shape)


def _sum(lm, src_shape, W, rho, dst_shape) -> ops.Kernel:
    try:
        return exact_linear.kernel("WINDOW_SUM",lm,{0:exact_linear.Src("x0",src_shape,1)},1)
    except (Decline, ValueError) as exc:
        raise IfaceDecline(f"WINDOW_SUM: {exc}", "WINDOW_SUM", dst_shape)


# ---------------------------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------------------------

class Program:
    def __init__(self):
        self.tensors: dict[str, dict] = {}
        self.names: dict = {}
        self.commands: list[dict] = []
        self.kernels: list[tuple[ops.Kernel, dict[str, str], str]] = []    # (kernel, local->global, tag)

    def declare(self, name, shape, dtype, role):
        if name in self.tensors:
            raise IfaceDecline(f"tensor {name!r} declared twice")
        self.tensors[name] = dict(shape=list(shape), dtype=dtype, role=role)

    def stream(self) -> Stream:
        args = []
        for name, t in self.tensors.items():
            role = "scratch" if t["role"] == "intermediate" else _ROLE.get(t["role"], "input")
            args.append(Arg(name, t["dtype"], tuple(t["shape"]), role))
        s = Stream(args)
        s.fence()
        s.flush()
        for kernel, mapping, tag in self.kernels:
            if kernel is None:
                s.fence()
                continue
            ks = kernel.stream
            remap = {}
            for k, a in enumerate(ks.args):
                if a.role == "const":
                    cname = f"{tag}__{a.name}"
                    s.const(cname, ks.consts[a.name])
                    remap[k] = s._index[cname]
                else:
                    remap[k] = s._index[mapping[a.name]]
            start=len(s.ins)
            for ins in ks.ins:
                s.ins.append(ins if ins.arg is None else Ins(ins.kind, ins.attrs, remap[ins.arg], ins.off))
            from dataclasses import replace
            for r in ks.readouts:
                s.readouts.append(replace(r,out=mapping[r.out],terms=[(mapping[n],f) for n,f in r.terms],
                    permutations={mapping[n]:v for n,v in (r.permutations or {}).items()},
                    broadcasts={mapping[n]:v for n,v in (r.broadcasts or {}).items()},
                    after=start+(len(ks.ins) if r.after is None else r.after)))
        s.fence()
        return s


def build(module, text: str | None = None) -> tuple[dict, object, Stream]:
    """(command buffer, LLVM-dialect module defining @gemmini_kernel, the program stream)."""
    literals = {}
    prog = Program()
    target = getattr(module.attributes.get("merlin_iface.target"), "data", "gemmini")
    for op in module.walk():
        m = _mnemonic(op)
        if not m:
            continue
        a = {k: _py(v) for k, v in op.attributes.items()}
        if m == "tensor":
            shape, dtype = _shape_dtype(op.results[0])
            prog.declare(a["name"], shape, dtype, a["role"])
            prog.names[op.results[0]] = a["name"]
            continue
        if m == "fence":
            prog.commands.append({"opcode": "FENCE", "operands": {}})
            prog.kernels.append((None, {}, "fence"))
            continue
        if m not in EXTENSION | {"movement"}:
            raise IfaceDecline(f"merlin_iface.{m} mixed with extension opcodes is not implemented", m)
        name = a.pop("name")
        for key in list(a):
            if (name, key) in literals and isinstance(a[key], float) and f32(literals[(name, key)]) == f32(a[key]):
                a[key] = literals[(name, key)]
        dst_shape, dst_dtype = _shape_dtype(op.results[0])
        srcs = [prog.names[v] for v in op.operands]
        sshapes = [_shape_dtype(v) for v in op.operands]
        try:
            if m == "movement":
                k = movement_kernel(a, sshapes[0][0], sshapes[0][1], dst_shape, dst_dtype)
                opcode, operands = "MOVEMENT", {"src": srcs[0]}
            elif m == "concat":
                k = concat_kernel(a, [sh for sh, _ in sshapes], dst_shape)
                opcode, operands = "CONCAT", {f"src{i}": n for i, n in enumerate(srcs)}
            elif m == "insert_slice":
                k = insert_slice_kernel(a, sshapes[0][0], sshapes[1][0])
                opcode, operands = "INSERT_SLICE", {"dest": srcs[0], "src": srcs[1]}
            elif m == "fill":
                k = fill_kernel(a, dst_shape)
                opcode, operands = "FILL", {}
            elif m == "affine_map":
                k = affine_kernel(a, sshapes[0][0], sshapes[1][0] if len(sshapes) > 1 else None, dst_dtype)
                opcode, operands = "AFFINE_MAP", {"x": srcs[0], **({"y": srcs[1]} if len(srcs) > 1 else {})}
            else:
                k = window_sum_kernel(a, sshapes[0][0], dst_shape)
                opcode, operands = "WINDOW_SUM", {"src": srcs[0]}
        except (Decline, ValueError, NotImplementedError) as exc:
            raise IfaceDecline(f"{m}: {exc}", m, dst_shape) from exc
        ins_args = [x.name for x in k.stream.args if x.role == "input"]
        if len(ins_args) != len(srcs):
            raise IfaceDecline(f"{m}: kernel reads {len(ins_args)} operands, the op has {len(srcs)}", m)
        if any(_shape_dtype(v)[1] != x.dtype for v, x in
               zip(op.operands, [x for x in k.stream.args if x.role == "input"])):
            raise IfaceDecline(f"{m}: operand dtype differs from the kernel's", m)
        mapping = dict(zip(ins_args, srcs))
        prog.declare(name, dst_shape, dst_dtype, "output")
        prog.names[op.results[0]] = name
        out_arg = next(x.name for x in k.stream.args if x.role == "output")
        mapping[out_arg] = name
        for x in k.stream.args:
            if x.role == "scratch":
                g = f"{name}__{x.name}"
                prog.declare(g, x.shape, x.dtype, "intermediate")
                mapping[x.name] = g
        operands["dst"] = name
        prog.commands.append({"opcode": opcode, "operands": operands, "attributes": a})
        prog.kernels.append((k, mapping, name))
    if not prog.commands:
        raise IfaceDecline("no command")
    stream = prog.stream()
    from .lint import lint
    lint(stream)
    module_out, _ = build_module(stream, "gemmini_kernel")
    access = {"output": "write", "intermediate": "write"}
    cb = {"abi_version": "0.1", "target": target, "backend": "mlir_oot_xdsl_gemmini",
          "tensors": {n: dict(t) for n, t in prog.tensors.items()},
          "commands": prog.commands,
          "kernel_abi": {"kind": "whole_program",
                         "args": [{"tensor": n, "access": access.get(t["role"], "read")}
                                  for n, t in prog.tensors.items()],
                         "outputs": [n for n, t in prog.tensors.items() if t["role"] == "output"]}}
    return cb, module_out, stream
