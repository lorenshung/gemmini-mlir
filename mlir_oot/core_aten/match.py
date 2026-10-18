"""Structural matching of model2MLIR linalg-on-tensors forms onto Core ATen device kernels.

Matching is by semantics, never by provenance labels: ``prov.*`` attributes are only
recorded for traceability.  The recognised carriers are the forms model2MLIR emits
(``linalg.generic`` elementwise bodies, compare/select ReLU, ``linalg.matmul``, batched
contraction generics, the im2col convolution region, ``linalg.reduce`` followed by a
division, ``linalg.transpose``, ``tensor.concat``, ``tensor.insert_slice`` into a splat or
into a tensor, ``tensor.splat``, identity and cast generics), named or generic.

Numerics come from the quantized graph itself: a tensor input produced by
``quant_ext.dequantize_per_tensor`` (int8, zero point 0) carries its scale, and the
matched result must feed exactly one ``quant_ext.quantize_per_tensor`` (int8, zero point
0) whose scale is the output scale; the kernel then replaces DQ -> op -> Q.  Integer-
domain forms whose integer semantics equal the device's (byte movement, compare/select
ReLU, i8 -> i32 sign extension, i8 x i8 -> i32 contractions) match without a sandwich.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, TensorType
from xdsl.ir import Block, Operation, OpResult, SSAValue

from . import ops
from .recipes.conv import ConvGeom


class Decline(Exception):
    """The operation is recognised but this lowering cannot implement it exactly."""


# ---------------------------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------------------------

def name_of(op: Operation) -> str:
    n = getattr(op, "op_name", None)
    return n.data if n is not None else op.name


def shape(v: SSAValue) -> tuple[int, ...] | None:
    t = v.type
    if not isinstance(t, TensorType):
        return None
    return tuple(int(x) for x in t.get_shape())


def etype(v: SSAValue) -> str:
    t = v.type
    return str(t.get_element_type()) if isinstance(t, TensorType) else str(t)


def producer(v: SSAValue) -> Operation | None:
    return v.owner if isinstance(v, OpResult) else None


def users(v: SSAValue) -> list[Operation]:
    return [u.operation for u in v.uses]


def _scalar_const(op: Operation | None):
    """A scalar ``arith.constant`` value (float), else None."""
    if op is None or op.name != "arith.constant":
        return None
    attr = _attr_of(op, "value")
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        value = float(attr.value.data)
        return value if math.isfinite(value) else None
    return None


def splat_value(v: SSAValue):
    """Value of a ``tensor.splat`` of a scalar constant (or a 0-d/1-element dense constant)."""
    p = producer(v)
    if p is None:
        return None
    if p.name == "tensor.splat":
        return _scalar_const(producer(p.operands[0]))
    if p.name == "arith.constant":
        attr = _attr_of(p, "value")
        if isinstance(attr, DenseIntOrFPElementsAttr):
            vals = list(attr.get_values()) if hasattr(attr, "get_values") else None
            if vals and all(x == vals[0] for x in vals):
                return vals[0]
    return None


def dense_values(v: SSAValue) -> np.ndarray | None:
    p = producer(v)
    if p is None or p.name != "arith.constant":
        return None
    attr = _attr_of(p, "value")
    if not isinstance(attr, DenseIntOrFPElementsAttr):
        return None
    return np.array(list(attr.get_values()), dtype=np.float64).reshape(shape(v))


def _attr_of(op, key):
    a = op.properties.get(key)
    return op.attributes.get(key) if a is None else a


def _prop_int(op, key):
    a = _attr_of(op, key)
    return None if a is None else int(a.value.data)


@dataclass(frozen=True)
class QIn:
    value: SSAValue          # the integer tensor (kernel argument)
    scale: float
    dq: Operation | None     # the dequantize op consumed (None: integer domain)
    dtype: str
    zero_point: int = 0
    unsigned: bool = False


def exact_widening_source(p):
    """A closed, elementwise widening carrier changes no rational code value."""
    if p is not None and p.name == 'linalg.generic' and len(p.operands) == 2 and len(p.results) == 1:
        body = list(p.regions[0].blocks[0].ops)
        block = p.regions[0].blocks[0]
        if (len(body) == 2 and body[0].name == 'arith.extf' and body[1].name == 'linalg.yield'
                and list(body[0].operands) == [block.args[0]]
                and list(body[1].operands) == [body[0].results[0]]
                and shape(p.operands[0]) == shape(p.results[0])
                and all(identity_map(m) for m in generic_maps(p))
                and all(i == 'parallel' for i in iterators(p))):
            from .linmap import exact_float_widening
            if exact_float_widening(str(body[0].operands[0].type), str(body[0].results[0].type)):
                return p.operands[0]
    return None


def dequant(v: SSAValue) -> QIn | None:
    p = producer(v)
    widened = exact_widening_source(p)
    if widened is not None:
        return dequant(widened)
    if p is None or name_of(p) != "quant_ext.dequantize_per_tensor":
        return None
    from .quant_schema import attributes
    policy=attributes(p,"dequantize")
    if getattr(policy.get("host_only"),"data",None)=="true":
        raise Decline("host-only source dequantization")
    s = splat_value(p.operands[1])
    zp = splat_value(p.operands[2])
    if zp is not None and getattr(p.attributes.get('prov.quant.zero_point_dtype'),'data',None)=='uint8':
        zp=int(zp)&255
    src = p.operands[0]
    if s is None or zp is None or etype(src) not in ("i8", "i32") or not math.isfinite(s) or not s > 0:
        return None
    if 'prov.quant.scale_numerator' in p.attributes or 'prov.quant.scale_denominator' in p.attributes:
        from .quant_schema import attributes,ratios
        attrs=attributes(p,'dequantize')
        if getattr(attrs.get('host_only'),'data',None)=='true':
            raise Decline('host-only source dequantization')
        pair=ratios(attrs)
        if len(pair)!=1 or pair[0]!=Fraction(s):
            raise Decline('source DQ operands differ from exact scale provenance')
        s=pair[0]
    lo, hi = _prop_int(p, "quant_min"), _prop_int(p, "quant_max")
    if lo is None or hi is None or lo > hi:
        return None
    for key,expected in [('quant_min',lo),('quant_max',hi),('zero_point',zp)]:
        if key in policy and policy[key].value.data!=expected:
            raise Decline('DQ provenance differs from source parameters: '+key)
    if 'granularity' in policy and policy['granularity'].data!='tensor':
        raise Decline('per-tensor DQ has inconsistent granularity')
    storage=getattr(p.attributes.get('prov.quant.storage_dtype'),'data',None) or getattr(p.properties.get('input_dtype'),'data',None)
    declared=getattr(p.properties.get('input_dtype'),'data',None)
    if declared in ('int8','uint8') and storage in ('int8','uint8') and declared!=storage:
        raise Decline('DQ storage dtype differs from its source operand contract')
    unsigned=etype(src)=='i8' and storage=='uint8'
    bounds = (0,255) if unsigned else (-128,127)
    if etype(src)=="i8" and not (bounds[0]<=lo<=hi<=bounds[1] and bounds[0]<=zp<=bounds[1]):
        return None
    return QIn(src, float(s), p, etype(src),int(zp),unsigned)


@dataclass(frozen=True)
class QOut:
    op: Operation
    value: SSAValue
    scale: float
    qmin: int
    qmax: int
    zero_point: int = 0
    unsigned: bool = False


def quant_sink(v: SSAValue) -> QOut | None:
    us = users(v)
    if len(us) != 1 or name_of(us[0]) != "quant_ext.quantize_per_tensor":
        return None
    q = us[0]
    if q.operands[0] is not v:
        return None
    from .quant_schema import attributes
    policy=attributes(q,"quantize")
    if getattr(policy.get("host_only"),"data",None)=="true":
        raise Decline("host-only source quantization")
    s, zp = splat_value(q.operands[1]), splat_value(q.operands[2])
    if zp is not None and getattr(q.attributes.get('prov.quant.zero_point_dtype'),'data',None)=='uint8':
        zp=int(zp)&255
    if s is None or zp is None or etype(q.results[0]) != "i8" or not math.isfinite(s) or not s > 0:
        return None
    if 'prov.quant.scale_numerator' in q.attributes or 'prov.quant.scale_denominator' in q.attributes:
        from .quant_schema import attributes,ratios
        attrs=attributes(q,'quantize')
        if getattr(attrs.get('host_only'),'data',None)=='true':
            raise Decline('host-only source quantization')
        pair=ratios(attrs)
        if len(pair)!=1 or pair[0]!=Fraction(s):
            raise Decline('source Q operands differ from exact scale provenance')
        s=pair[0]
    lo, hi = _prop_int(q, "quant_min"), _prop_int(q, "quant_max")
    for key,expected in [('quant_min',lo),('quant_max',hi),('zero_point',zp)]:
        if key in policy and policy[key].value.data!=expected:
            raise Decline('Q provenance differs from source parameters: '+key)
    if 'granularity' in policy and policy['granularity'].data!='tensor':
        raise Decline('per-tensor Q has inconsistent granularity')
    storage=getattr(q.attributes.get('prov.quant.storage_dtype'),'data',None) or getattr(q.properties.get('output_dtype'),'data',None)
    declared=getattr(q.properties.get('output_dtype'),'data',None)
    if declared in ('int8','uint8') and storage in ('int8','uint8') and declared!=storage:
        raise Decline('Q storage dtype differs from its source result contract')
    unsigned=storage=='uint8'
    bounds=(0,255) if unsigned else (-128,127)
    if lo is None or hi is None or not bounds[0]<=lo<hi<=bounds[1] or not bounds[0]<=zp<=bounds[1]:
        return None
    return QOut(q,q.results[0],float(s),lo,hi,int(zp),unsigned)


def single_use(v: SSAValue) -> Operation | None:
    us = users(v)
    return us[0] if len(us) == 1 else None


# ---------------------------------------------------------------------------------------------
# Elementwise body analysis: y = act(sum_i w_i x_i + c), act in {none, relu}
# ---------------------------------------------------------------------------------------------

@dataclass
class Lin:
    terms: dict[int, float] = field(default_factory=dict)
    c: float = Fraction(0)

    def scaled(self, k):
        return Lin({i: w * k for i, w in self.terms.items()}, self.c * k)

    def plus(self, other, k=Fraction(1)):
        t = dict(self.terms)
        for i, w in other.terms.items():
            t[i] = t.get(i, Fraction(0)) + k * w
        return Lin({i: w for i, w in t.items() if w != 0}, self.c + k * other.c)


@dataclass
class Body:
    lin: Lin
    relu: bool = False
    widen: bool = False


def _is_zero(x):
    return isinstance(x, (float, Fraction)) and x == 0


def analyse_body(block: Block, n_in: int, consts: dict[int, float]) -> Body | None:
    """Symbolic evaluation of an elementwise linalg body (block args 0..n_in-1 are inputs)."""
    env: dict = {}
    for i, a in enumerate(block.args[:n_in]):
        env[a] = Fraction(consts[i]) if i in consts else Lin({i: Fraction(1)}, Fraction(0))
    relu_of = {}
    widen = False
    for op in block.ops:
        nm = op.name
        if nm == "linalg.yield":
            out = env.get(op.operands[0])
            if isinstance(out, (float, Fraction)):
                return Body(Lin({}, out), False, widen)
            if isinstance(out, Lin):
                return Body(out, False, widen)
            if isinstance(out, tuple) and out[0] == "relu":
                return Body(out[1], True, widen)
            return None
        vals = [env.get(v) for v in op.operands]
        if nm == "arith.constant":
            c = _scalar_const(op)
            if c is None:
                return None
            env[op.results[0]] = Fraction(c)
        elif nm == 'arith.extf':
            from .linmap import exact_float_widening
            if (not isinstance(vals[0], (Lin, float, Fraction))
                    or not exact_float_widening(str(op.operands[0].type), str(op.results[0].type))):
                return None
            env[op.results[0]] = vals[0]
        elif nm in ("arith.sitofp", "arith.uitofp"):
            if not isinstance(vals[0], (float, Fraction)):
                return None
            env[op.results[0]] = vals[0]
        elif nm == "arith.extsi":
            if not (isinstance(vals[0], Lin) and str(op.operands[0].type) == "i8"
                    and str(op.results[0].type) == "i32"):
                return None
            widen = True
            env[op.results[0]] = vals[0]
        elif nm in ("arith.addf", "arith.subf"):
            a, b = vals
            k = Fraction(1) if nm == "arith.addf" else Fraction(-1)
            if isinstance(a, (float, Fraction)) and isinstance(b, (float, Fraction)):
                env[op.results[0]] = a + k * b
            else:
                la = a if isinstance(a, Lin) else Lin({}, a) if isinstance(a, (float, Fraction)) else None
                lb = b if isinstance(b, Lin) else Lin({}, b) if isinstance(b, (float, Fraction)) else None
                if la is None or lb is None:
                    return None
                env[op.results[0]] = la.plus(lb, k)
        elif nm == "arith.mulf":
            a, b = vals
            if isinstance(a, (float, Fraction)) and isinstance(b, Lin):
                a, b = b, a
            if isinstance(a, Lin) and isinstance(b, (float, Fraction)):
                env[op.results[0]] = a.scaled(b)
            elif isinstance(a, (float, Fraction)) and isinstance(b, (float, Fraction)):
                env[op.results[0]] = a * b
            else:
                return None
        elif nm == "arith.divf":
            a, b = vals
            if isinstance(a, Lin) and isinstance(b, (float, Fraction)) and b != 0:
                env[op.results[0]] = a.scaled(Fraction(1) / b)
            elif isinstance(a, (float, Fraction)) and isinstance(b, (float, Fraction)) and b != 0:
                env[op.results[0]] = a / b
            else:
                return None
        elif nm == "arith.negf":
            if isinstance(vals[0], Lin):
                env[op.results[0]] = vals[0].scaled(Fraction(-1))
            elif isinstance(vals[0], (float, Fraction)):
                env[op.results[0]] = -vals[0]
            else:
                return None
        elif nm in ("arith.cmpf", "arith.cmpi"):
            pred = op.properties.get("predicate")
            p = int(pred.value.data) if pred is not None else None
            a, b = vals
            # cmpf: olt=4, ult=12, ogt=2, ugt=10 ; cmpi: slt=2, sgt=4
            lt = (nm == "arith.cmpf" and p in (4, 12)) or (nm == "arith.cmpi" and p == 2)
            gt = (nm == "arith.cmpf" and p in (2, 10)) or (nm == "arith.cmpi" and p == 4)
            if isinstance(a, Lin) and _is_zero(b) and (lt or gt):
                env[op.results[0]] = ("cmp0", a, "lt" if lt else "gt")
            else:
                return None
        elif nm == "arith.select":
            c, t, f = vals
            if not (isinstance(c, tuple) and c[0] == "cmp0"):
                return None
            x = c[1]
            if c[2] == "lt" and _is_zero(t) and isinstance(f, Lin) and f == x:
                env[op.results[0]] = ("relu", x)
            elif c[2] == "gt" and _is_zero(f) and isinstance(t, Lin) and t == x:
                env[op.results[0]] = ("relu", x)
            else:
                return None
        elif nm in ("arith.maximumf", "arith.maxnumf", "arith.maxsi"):
            a, b = vals
            if isinstance(a, Lin) and _is_zero(b):
                env[op.results[0]] = ("relu", a)
            elif isinstance(b, Lin) and _is_zero(a):
                env[op.results[0]] = ("relu", b)
            else:
                return None
        else:
            return None
    return None


def generic_maps(op: Operation):
    return [m.data for m in op.properties["indexing_maps"].data]


def iterators(op: Operation) -> list[str]:
    return [str(x.data.value if hasattr(x.data, "value") else x.data) for x in op.properties["iterator_types"].data]


def map_coeffs(amap, bounds=None) -> list[list[int]] | None:
    """Prove a zero-offset linear map, optionally over declared dimension bounds.

    Sampling the origin and basis vectors does not prove linearity of affine
    floor/mod expressions. Only structural algebra and range proofs grant it.
    """
    from xdsl.ir.affine import AffineDimExpr,AffineConstantExpr,AffineBinaryOpExpr,AffineBinaryOpKind as K
    n=amap.num_dims
    if amap.num_symbols: return None
    def form(expr):
        if isinstance(expr,AffineDimExpr): return [int(i==expr.position) for i in range(n)]+[0]
        if isinstance(expr,AffineConstantExpr): return [0]*n+[expr.value]
        if not isinstance(expr,AffineBinaryOpExpr): raise ValueError('nonlinear affine expression')
        a,b=form(expr.lhs),form(expr.rhs)
        if expr.kind==K.Add: return [x+y for x,y in zip(a,b)]
        if expr.kind==K.Mul:
            if not any(a[:-1]): return [a[-1]*v for v in b]
            if not any(b[:-1]): return [b[-1]*v for v in a]
            raise ValueError('product of dimensions')
        if any(b[:-1]) or b[-1]<=0: raise ValueError('unproved affine divisor')
        d=b[-1]
        if all(v%d==0 for v in a):
            return [0]*(n+1) if expr.kind==K.Mod else [v//d for v in a]
        if bounds is not None:
            lo=a[-1]+sum(min(0,c*(size-1)) for c,size in zip(a[:-1],bounds))
            hi=a[-1]+sum(max(0,c*(size-1)) for c,size in zip(a[:-1],bounds))
            if expr.kind in (K.FloorDiv,K.CeilDiv):
                f=(lambda x:x//d) if expr.kind==K.FloorDiv else (lambda x:-(-x//d))
                if f(lo)==f(hi): return [0]*n+[f(lo)]
        raise ValueError('quasi-affine map is not proved linear')
    try: rows=[form(e) for e in amap.results]
    except ValueError: return None
    return [r[:-1] for r in rows] if all(r[-1]==0 for r in rows) else None


def identity_map(amap) -> bool:
    return amap.num_dims == len(amap.results) and map_coeffs(amap) == [
        [1 if i == j else 0 for j in range(amap.num_dims)] for i in range(amap.num_dims)]


def trailing_block(amap) -> int | None:
    """k when the map is (d_k, ..., d_{n-1}) (a broadcast over the leading dims)."""
    co = map_coeffs(amap)
    if co is None:
        return None
    n = amap.num_dims
    k = n - len(co)
    for r, row in enumerate(co):
        if row != [1 if j == k + r else 0 for j in range(n)]:
            return None
    return k


# ---------------------------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------------------------

@dataclass
class Match:
    op: str                                     # Core ATen overload the region implements
    root: Operation                             # the op whose result the kernel produces
    consumed: list[Operation]                   # ops the kernel replaces (excluding shared producers)
    operands: list[tuple[str, SSAValue]]        # kernel argument -> source value (int8/int32 tensors)
    result: tuple[str, SSAValue]                # kernel output argument -> source value
    kernel: ops.Kernel
    layout: dict = field(default_factory=dict)
    numeric: dict = field(default_factory=dict)


def _sandwich_out(v: SSAValue):
    q = quant_sink(v)
    if q is None:
        raise Decline("result does not feed one per-tensor int8 quantize")
    return q


def _data_inputs(op: Operation, n_in: int):
    """Split generic inputs into quantized data inputs and splat constants."""
    consts, data = {}, {}
    for i, v in enumerate(op.operands[:n_in]):
        c = splat_value(v)
        if c is not None:
            consts[i] = c
            continue
        q = dequant(v)
        if q is not None:
            data[i] = q
            continue
        data[i] = None
    return consts, data


def match_elementwise(op: Operation) -> Match | None:
    if op.name != "linalg.generic":
        return None
    its = iterators(op)
    if any(i != "parallel" for i in its) or len(op.results) != 1:
        return None
    n_in = len(op.operands) - 1
    maps = generic_maps(op)
    if not identity_map(maps[-1]):
        return None
    block = op.regions[0].blocks[0]
    consts, data = _data_inputs(op, n_in)
    body = analyse_body(block, n_in, consts)
    if body is None:
        return None
    if list(block.args[n_in].uses):
        return None
    oshape = shape(op.results[0])
    used = sorted(body.lin.terms)
    integer = all(data.get(i) is None for i in used) and etype(op.results[0]) in ("i8", "i32")
    # --- integer domain: relu on i8, widening i8 -> i32, identity copies ---------------------
    if integer:
        if len(used) != 1 or body.lin.c or body.lin.terms[used[0]] != 1.0:
            return None
        src = op.operands[used[0]]
        if not identity_map(maps[used[0]]) or shape(src) != oshape:
            return None
        if body.widen and etype(src) == "i8" and etype(op.results[0]) == "i32" and not body.relu:
            return Match("aten._to_copy.default", op, [op], [("x", src)], ("y", op.results[0]),
                         ops.to_copy_widen(oshape))
        if etype(src) == "i8" and etype(op.results[0]) == "i8" and not body.widen:
            if body.relu:
                return Match("aten.relu.default", op, [op], [("x", src)], ("y", op.results[0]),
                             ops.relu(oshape, s_x=1.0, s_out=1.0))
            return Match("aten.clone.default", op, [op], [("x", src)], ("y", op.results[0]), ops.clone(oshape))
        return None
    # --- quantized sandwich -----------------------------------------------------------------
    if any(data.get(i) is None for i in used) or etype(op.results[0]) not in ('f16', 'bf16', 'f32', 'f64'):
        return None
    qout = quant_sink(op.results[0])
    if qout is None:
        return None
    s_o = qout.scale
    terms = [(i, body.lin.terms[i], data[i]) for i in used]
    c = body.lin.c + sum(Fraction(w)*Fraction(q.scale)*((128 if q.unsigned else 0)-q.zero_point) for _,w,q in terms)
    consumed = [op, qout.op] + [t[2].dq for t in terms]
    if len(terms) == 0:
        raise Decline("constant elementwise result")
    if len(terms) == 1:
        i, w, q = terms[0]
        if not identity_map(maps[i]) or shape(q.value) != oshape:
            raise Decline("broadcast single-input affine map")
        if c == 0 and not body.relu:
            if w == 1.0:
                aten, k = "aten.clone.default", ops.clone(oshape, s_x=q.scale, s_out=s_o)
            else:
                aten = "aten.neg.default" if w == -1.0 else "aten.mul.Scalar"
                k = ops.scale_map(aten, oshape, w, s_x=q.scale, s_out=s_o)
        elif c == 0 and body.relu:
            if w <= 0:
                pass # ReLU is evaluated exactly in the host epilogue
            aten, k = "aten.relu.default", ops.relu(oshape, s_x=Fraction(w) * Fraction(q.scale), s_out=s_o)
        else:
            aten = "aten.add.Scalar" if c > 0 else "aten.sub.Scalar"
            k = ops.add_scalar(aten, oshape, c, s_x=q.scale, s_out=s_o, w=w, relu=body.relu)
        _pack_inputs(k,[("x",q)])
        return Match(aten, op, consumed, [("x", q.value)], ("y", qout.value), k,
                     numeric=dict(w=w, c=c, s_x=q.scale, s_out=s_o))
    if len(terms) == 2:
        (ia, wa, qa), (ib, wb, qb) = terms
        ka, kb = trailing_block(maps[ia]), trailing_block(maps[ib])
        if ka is None or kb is None:
            raise Decline("non-trailing broadcast")
        if ka > kb:                     # the full operand first
            (ia, wa, qa), (ib, wb, qb), ka, kb = (ib, wb, qb), (ia, wa, qa), kb, ka
        if ka != 0 or shape(qa.value) != oshape:
            raise Decline("broadcast of the first operand")
        bshape = shape(qb.value)
        aten = "aten.add.Tensor" if wb > 0 else "aten.sub.Tensor"
        k = ops.add_tensor(aten, oshape, bshape, wa=wa, alpha=wb, s_a=qa.scale, s_b=qb.scale, s_out=s_o,
                           c=c, relu=body.relu, b_lead=kb)
        _pack_inputs(k,[("a",qa),("b",qb)])
        return Match(aten, op, consumed, [("a", qa.value), ("b", qb.value)], ("y", qout.value), k,
                     numeric=dict(wa=wa, wb=wb, c=c, s_a=qa.scale, s_b=qb.scale, s_out=s_o))
    raise Decline("more than two tensor operands")


# ---------------------------------------------------------------------------------------------
# Epilogue chains after a contraction or reduction: reshapes and affine generics, then Q
# ---------------------------------------------------------------------------------------------

RESHAPES = ("tensor.collapse_shape", "tensor.expand_shape")


@dataclass
class Chain:
    ops: list[Operation]
    final: SSAValue
    w: float = Fraction(1)               # result = act(w * value + bias + c)
    c: float = Fraction(0)
    relu: bool = False
    bias: tuple | None = None            # (QIn or constant ndarray, weight, trailing-block index, map coeffs)
    perm: list | None = None             # a final layout transpose before the quantize


def _scaled_dequant(x: SSAValue):
    """(QIn, multiplier, ops) for a dequantized tensor, possibly through single-input scale generics."""
    k, ops_ = Fraction(1), []
    for _ in range(4):
        q = dequant(x)
        if q is not None:
            return q, k, ops_
        p = producer(x)
        if p is None or p.name != "linalg.generic" or any(i != "parallel" for i in iterators(p)):
            return None, k, ops_
        n_in = len(p.operands) - 1
        consts = {i: c for i, v in enumerate(p.operands[:n_in]) if (c := splat_value(v)) is not None}
        body = analyse_body(p.regions[0].blocks[0], n_in, consts)
        maps = generic_maps(p)
        if body is None or body.relu or body.lin.c or len(body.lin.terms) != 1:
            return None, k, ops_
        (i, w), = body.lin.terms.items()
        if not identity_map(maps[i]) or not identity_map(maps[-1]) or len(list(p.results[0].uses)) != 1:
            return None, k, ops_
        k *= w
        ops_.append(p)
        x = p.operands[i]
    return None, k, ops_


def follow_chain(v: SSAValue) -> Chain:
    ch = Chain([], v)
    while True:
        u = single_use(ch.final)
        if u is None:
            raise Decline("contraction result has several users")
        if u.name in RESHAPES:
            ch.ops.append(u)
            ch.final = u.results[0]
            continue
        if name_of(u) == "quant_ext.quantize_per_tensor":
            return ch
        if exact_widening_source(u) is ch.final:
            ch.ops.append(u)
            ch.final = u.results[0]
            continue
        if u.name == "linalg.transpose" and ch.perm is None:
            ch.perm = [int(p) for p in u.properties["permutation"].iter_values()]
            ch.ops.append(u)
            ch.final = u.results[0]
            continue
        if ch.perm is not None:
            raise Decline("epilogue after a layout transpose")
        if u.name != "linalg.generic" or ch.relu:
            raise Decline(f"unsupported epilogue {name_of(u)}")
        if any(i != "parallel" for i in iterators(u)):
            raise Decline("reduction in the epilogue")
        n_in = len(u.operands) - 1
        maps = generic_maps(u)
        pos = [i for i, x in enumerate(u.operands[:n_in]) if x is ch.final]
        if len(pos) != 1 or not identity_map(maps[pos[0]]) or not identity_map(maps[-1]):
            raise Decline("epilogue does not map the value elementwise")
        consts = {}
        others = {}
        for i, x in enumerate(u.operands[:n_in]):
            if i == pos[0]:
                continue
            c = splat_value(x)
            if c is not None:
                consts[i] = c
            else:
                others[i] = x
        body = analyse_body(u.regions[0].blocks[0], n_in, consts)
        if body is None or set(body.lin.terms) - {pos[0], *others}:
            raise Decline("epilogue body is not affine")
        k = body.lin.terms.get(pos[0], 0.0)
        if k == 0:
            raise Decline("epilogue drops the contraction")
        ch.w *= k
        ch.c = ch.c * k + body.lin.c
        for i, x in others.items():
            if i not in body.lin.terms:
                continue
            if ch.bias is not None:
                raise Decline("more than one bias operand")
            q, k_b, extra = _scaled_dequant(x)
            src = q if q is not None else dense_values(x)
            if src is None:
                raise Decline("bias is neither dequantized nor a dense constant")
            ch.bias = (src, body.lin.terms[i] * k_b, maps[i])
            ch.ops.extend(extra)
        if ch.bias is not None and k != 1.0 and others:
            # a later scale also scales the bias term
            src, wb, m = ch.bias
            ch.bias = (src, wb, m)
        ch.relu = body.relu
        ch.ops.append(u)
        ch.final = u.results[0]


def _bias_operand(ch: Chain, s_acc: float, n: int, broadcast_dim_ok, full_ok=None):
    """('arg', QIn, repeat, full) for a runtime int32 bias whose weight is a small positive integer
    in accumulator units, ('const', int32 array, 1, full) for a compile-time bias, or None."""
    if ch.bias is None:
        return None
    src, wb, amap = ch.bias
    full = full_ok is not None and full_ok(amap)
    if not full and not broadcast_dim_ok(amap):
        raise Decline("bias broadcast does not follow the output channel")
    if isinstance(src, QIn):
        if src.dtype != 'i32':
            raise Decline('runtime bias is not int32')
        return ('arg',src,Fraction(wb)*Fraction(src.scale),full)
    vals = np.asarray(src,dtype=object)
    if vals.size != n and not full:
        raise Decline('bias extent')
    return ('const',np.array([Fraction(wb)*Fraction(float(v)) for v in vals.flat],dtype=object).reshape(vals.shape),1,full)


def _host_bias(kernel, bias, output_scale, convolution=False):
    if bias is None:
        return
    from .stream import Arg
    out = next(a for a in kernel.args if a.role == 'output')
    readout = kernel.stream.readouts[-1]
    bshape = (1,out.shape[1],1,1) if convolution else ((out.shape[-1],) if not bias[3] else out.shape)
    if bias[0] == 'const':
        readout.constant = np.broadcast_to(np.asarray(bias[1],dtype=object).reshape(bshape),out.shape)/Fraction(output_scale)
    else:
        value = bias[1].value
        arg = Arg('bias','i32',shape(value),'input')
        kernel.stream._index['bias'] = len(kernel.stream.args)
        kernel.stream.args.append(arg)
        readout.terms.append(('bias',Fraction(bias[2])/Fraction(output_scale)))
        if bias[1].zero_point:
            readout.constant=readout.constant-bias[1].zero_point*Fraction(bias[2])/Fraction(output_scale)
        readout.broadcasts = {'bias':bshape}



def match_contraction(op: Operation) -> Match | None:
    """mm / bmm / addmm: f32 matmul of two dequantized int8 tensors with an affine epilogue."""
    batched = False
    if op.name == "linalg.matmul":
        if len(op.operands) != 3:
            return None
    elif op.name == "linalg.generic":
        its = iterators(op)
        if its != ["parallel"] * 3 + ["reduction"]:
            return None
        maps = [map_coeffs(m) for m in generic_maps(op)]
        want = ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        if tuple(maps) != want:
            return None
        names = [o.name for o in op.regions[0].blocks[0].ops]
        if names not in (["arith.mulf", "arith.addf", "linalg.yield"], ["arith.extsi", "arith.extsi", "arith.muli",
                                                                         "arith.addi", "linalg.yield"]):
            return None
        block = op.regions[0].blocks[0]
        if names[0] == "arith.mulf":
            multiply, addition, yielded = list(block.ops)
            if not (set(multiply.operands) == set(block.args[:2])
                    and set(addition.operands) == {multiply.results[0], block.args[2]}
                    and list(yielded.operands) == [addition.results[0]]):
                return None
        else:
            from ..contraction_patterns import _canonical_body
            if not _canonical_body(op):
                return None
        batched = True
    else:
        return None
    a, b, init = op.operands[:3]
    if splat_value(init) != 0.0:
        return None
    sa, sb = shape(a), shape(b)
    if batched:
        Bn, M, K = sa
        N = sb[2]
    else:
        (M, K), N, Bn = sa, sb[1], 0
    # integer contraction: exact i32 readout
    if etype(a) == "i8" and etype(b) == "i8" and etype(op.results[0]) == "i32":
        if any(key.startswith('prov.quant.') for key in op.attributes):
            from .quant_schema import attributes
            attrs=attributes(op,'contraction')
            if getattr(attrs.get('host_only'),'data',None)=='true':
                raise Decline('host-only source integer contraction')
        k = ops.matmul("aten.bmm.default" if batched else "aten.mm.default", M, K, N, batch=Bn)
        return Match(k.op, op, [op], [("a", a), ("b", b)], ("out", op.results[0]), k)
    qa, qb = dequant(a), dequant(b)
    if qa is None or qb is None or qa.dtype != "i8" or qb.dtype != "i8":
        return None
    if _is_conv_cols(b) or _im2col_rows(a) is not None:
        return None           # the convolution matchers own im2col operands
    ch = follow_chain(op.results[0])
    if ch.perm is not None:
        raise Decline("contraction result is transposed")
    q = _sandwich_out(ch.final)
    s_acc = Fraction(qa.scale) * Fraction(qb.scale)
    out_shape = shape(q.value)
    bias = _bias_operand(ch, s_acc, N, lambda m: trailing_block(m) == m.num_dims - 1,
                         full_ok=(lambda m: identity_map(m)) if not batched else None)
    if ch.c:
        raise Decline("constant epilogue offset")
    aten = "aten.bmm.default" if batched else ("aten.addmm.default" if bias else "aten.mm.default")
    k = ops.matmul(aten, M, K, N, batch=Bn, s_a=qa.scale, s_b=qb.scale, s_out=q.scale, alpha=ch.w,
                   bias=False, relu=ch.relu,
                   bias_const=None,
                   bias_rows=False)
    _host_bias(k,bias,q.scale)
    ops.contraction_zero_points(k,qa,qb,M,K,N,Bn)
    operands = [("a", qa.value), ("b", qb.value)]
    if bias and bias[0] == "arg":
        operands.append(("bias", bias[1].value))
    consumed = [op, *ch.ops, q.op, qa.dq, qb.dq] + ([bias[1].dq] if bias and bias[0] == "arg" else [])
    return Match(aten, ch.final.owner if False else q.op, consumed, operands, ("out", q.value), k,
                 numeric=dict(s_a=qa.scale, s_b=qb.scale, s_out=q.scale, alpha=ch.w, relu=ch.relu))


# ---------------------------------------------------------------------------------------------
# Convolution: pad (insert_slice into a zero splat) -> im2col generic -> reshapes -> matmul(W, cols)
# ---------------------------------------------------------------------------------------------

def _through_reshapes(v: SSAValue) -> SSAValue:
    while True:
        p = producer(v)
        if p is None or p.name not in RESHAPES:
            return v
        v = p.operands[0]


def _im2col(v: SSAValue):
    p = producer(_through_reshapes(v))
    if p is None or p.name != "linalg.generic" or any(i != "parallel" for i in iterators(p)):
        return None
    if len(p.operands) != 2 or [o.name for o in p.regions[0].blocks[0].ops] != ["linalg.yield"]:
        return None
    if p.regions[0].blocks[0].ops.first.operands[0] is not p.regions[0].blocks[0].args[0]:
        return None
    maps = generic_maps(p)
    if not identity_map(maps[1]) or len(maps[0].results) != 4 or maps[0].num_dims != 6:
        return None
    return p


def _is_conv_cols(v: SSAValue) -> bool:
    return _im2col(v) is not None


def _im2col_rows(v: SSAValue):
    p = producer(_through_reshapes(v))
    if p is None or p.name != "linalg.generic" or any(i != "parallel" for i in iterators(p)):
        return None
    if len(p.operands) != 2 or [o.name for o in p.regions[0].blocks[0].ops] != ["linalg.yield"]:
        return None
    if p.regions[0].blocks[0].ops.first.operands[0] is not p.regions[0].blocks[0].args[0]:
        return None
    maps = generic_maps(p)
    if not identity_map(maps[1]) or len(maps[0].results) != 4 or maps[0].num_dims != 7:
        return None
    return p


def match_convolution_rows(op: Operation) -> Match | None:
    """Weight-RHS rows form: im2col rows [N*OH*OW, G*C*KH*KW] x W^T -> [pixels, F], bias row, NCHW view."""
    if op.name != "linalg.matmul" or len(op.operands) != 3:
        return None
    im = _im2col_rows(op.operands[0])
    if im is None or splat_value(op.operands[2]) != 0.0:
        return None
    co = map_coeffs(generic_maps(im)[0])
    N_, OH, OW, G, C_, KH, KW = shape(im.results[0])
    if G != 1:
        raise Decline("grouped convolution rows form")
    # (d0, d3*C + d4, d1*sh + d5*dh, d2*sw + d6*dw)
    if co is None or co[0] != [1, 0, 0, 0, 0, 0, 0] or co[1][4] != 1 or any(co[1][j] for j in (0, 1, 2, 5, 6)):
        raise Decline("im2col rows dims")
    if any(co[2][j] for j in (0, 2, 3, 4, 6)) or any(co[3][j] for j in (0, 1, 3, 4, 5)):
        raise Decline("im2col window couples spatial axes")
    sh, dh, sw, dw = co[2][1], co[2][5], co[3][2], co[3][6]
    xsrc, ph, pw, pad_op = _unpad(im.operands[0])
    tr = producer(op.operands[1])
    if tr is None or tr.name != "linalg.transpose" or [int(p) for p in tr.properties["permutation"].iter_values()] != [1, 0]:
        raise Decline("weight operand is not W^T")
    qx, qw = dequant(xsrc), dequant(_through_reshapes(tr.operands[0]))
    if qx is None or qw is None or qx.dtype != "i8" or qw.dtype != "i8":
        return None
    N, C, H, W = shape(qx.value)
    F = shape(qw.value)[0]
    if shape(qw.value) != (F, C, KH, KW) or (N, C) != (N_, C_):
        raise Decline("convolution operand extents")
    g = ConvGeom(N, H, W, C, F, KH, KW, sh, sw, ph, pw, dh, dw, 1)
    if (g.oh, g.ow) != (OH, OW):
        raise Decline("convolution output extent")
    ch = follow_chain(op.results[0])
    q = _sandwich_out(ch.final)
    if ch.perm != [0, 3, 1, 2] or shape(q.value) != (N, F, OH, OW):
        raise Decline("convolution result is not viewed NCHW")
    s_acc = Fraction(qx.scale) * Fraction(qw.scale)
    bias = _bias_operand(ch, s_acc, F, lambda m: trailing_block(m) == m.num_dims - 1)
    if ch.c:
        raise Decline("constant epilogue offset")
    k = ops.convolution_nchw(g, s_x=qx.scale, s_w=qw.scale, s_out=q.scale, alpha=ch.w, bias=False,
                             relu=ch.relu, bias_const=None,
                             bias_repeat=1)
    _host_bias(k,bias,q.scale,convolution=True)
    _pack_inputs(k,[("x",qx),("w",qw)])
    ops.convolution_zero_points(k,("x",qx),("w",qw),g)
    operands = [("x", qx.value), ("w", qw.value)]
    if bias and bias[0] == "arg":
        operands.append(("bias", bias[1].value))
    consumed = [op, im, tr, *ch.ops, q.op, qx.dq, qw.dq] + ([pad_op] if pad_op else []) + \
        ([bias[1].dq] if bias and bias[0] == "arg" else [])
    return Match("aten.convolution.default", q.op, consumed, operands, ("y", q.value), k,
                 layout=dict(source="NCHW,OIHW", device="NHWC,HWIO", transposes="on-device"),
                 numeric=dict(s_x=qx.scale, s_w=qw.scale, s_out=q.scale, geometry=g.__dict__))


def match_convolution(op: Operation) -> Match | None:
    if op.name != "linalg.matmul" or len(op.operands) != 3:
        return None
    im = _im2col(op.operands[1])
    if im is None or splat_value(op.operands[2]) != 0.0:
        return None
    co = map_coeffs(generic_maps(im)[0])
    if co is None:
        raise Decline("non-linear im2col map")
    C_, KH, KW, N_, OH, OW = shape(im.results[0])
    # expected (d3, d0, d4*sh + d1*dh, d5*sw + d2*dw)
    if co[0] != [0, 0, 0, 1, 0, 0] or co[1] != [1, 0, 0, 0, 0, 0]:
        raise Decline("im2col dims are not (C, KH, KW, N, OH, OW)")
    if any(co[2][j] for j in (0, 2, 3, 5)) or any(co[3][j] for j in (0, 1, 3, 4)):
        raise Decline("im2col window couples spatial axes")
    sh, dh, sw, dw = co[2][4], co[2][1], co[3][5], co[3][2]
    xp = im.operands[0]
    ph = pw = 0
    pad_op = producer(xp)
    consumed = [op, im]
    if pad_op is not None and pad_op.name == "tensor.insert_slice":
        if splat_value(pad_op.operands[1]) != 0.0:
            raise Decline("non-zero convolution padding")
        offs = list(pad_op.properties["static_offsets"].iter_values())
        strides_ = list(pad_op.properties["static_strides"].iter_values())
        if offs[:2] != [0, 0] or any(s != 1 for s in strides_):
            raise Decline("padding outside the spatial dims")
        ph, pw = offs[2], offs[3]
        full = shape(pad_op.results[0])
        src = pad_op.operands[0]
        if full[2] - shape(src)[2] != 2 * ph or full[3] - shape(src)[3] != 2 * pw:
            raise Decline("asymmetric convolution padding")
        consumed.append(pad_op)
        xp = src
    qx = dequant(xp)
    w_src = _through_reshapes(op.operands[0])
    qw = dequant(w_src)
    if qx is None or qw is None or qx.dtype != "i8" or qw.dtype != "i8":
        return None
    N, C, H, W = shape(qx.value)
    F = shape(qw.value)[0]
    if shape(qw.value) != (F, C, KH, KW) or N != N_ or C != C_:
        raise Decline("convolution operand extents")
    g = ConvGeom(N, H, W, C, F, KH, KW, sh, sw, ph, pw, dh, dw, 1)
    if (g.oh, g.ow) != (OH, OW):
        raise Decline("convolution output extent")
    ch = follow_chain(op.results[0])
    # the epilogue must see [F, N*OH*OW] reshaped to NCHW with N == 1 (a pure view)
    if N != 1:
        raise Decline("batched im2col output needs a transpose")
    q = _sandwich_out(ch.final)
    if shape(q.value) != (N, F, OH, OW):
        raise Decline("convolution result is not NCHW")
    s_acc = Fraction(qx.scale) * Fraction(qw.scale)
    bias = _bias_operand(ch, s_acc, F, lambda m: map_coeffs(m) == [[0, 1, 0, 0]])
    if ch.c:
        raise Decline("constant epilogue offset")
    k = ops.convolution_nchw(g, s_x=qx.scale, s_w=qw.scale, s_out=q.scale, alpha=ch.w, bias=False,
                             relu=ch.relu, bias_const=None,
                             bias_repeat=1)
    _host_bias(k,bias,q.scale,convolution=True)
    _pack_inputs(k,[("x",qx),("w",qw)])
    ops.convolution_zero_points(k,("x",qx),("w",qw),g)
    operands = [("x", qx.value), ("w", qw.value)]
    if bias and bias[0] == "arg":
        operands.append(("bias", bias[1].value))
    consumed += [*ch.ops, q.op, qx.dq, qw.dq] + ([bias[1].dq] if bias and bias[0] == "arg" else [])
    return Match("aten.convolution.default", q.op, consumed, operands, ("y", q.value), k,
                 layout=dict(source="NCHW,OIHW", device="NHWC,HWIO", transposes="on-device"),
                 numeric=dict(s_x=qx.scale, s_w=qw.scale, s_out=q.scale, geometry=g.__dict__))


# ---------------------------------------------------------------------------------------------
# mean.dim: linalg.reduce(addf) over dims -> divide -> Q
# ---------------------------------------------------------------------------------------------

def match_mean(op: Operation) -> Match | None:
    """linalg.reduce(addf) over dims, optionally scaled (mean.dim: 1 / L; a sum: any w)."""
    if op.name != "linalg.reduce":
        return None
    block = op.regions[0].blocks[0]
    names = [o.name for o in block.ops]
    if names != ["arith.addf", "linalg.yield"] or splat_value(op.operands[1]) != 0.0:
        return None
    addition, yielded = list(block.ops)
    if set(addition.operands) != set(block.args) or list(yielded.operands) != [addition.results[0]]:
        return None
    qx = dequant(op.operands[0])
    if qx is None or qx.dtype != "i8":
        return None
    dims = [int(d) for d in op.properties["dimensions"].iter_values()]
    xshape = shape(qx.value)
    L = int(np.prod([xshape[d] for d in dims]))
    ch = follow_chain(op.results[0])
    if ch.bias is not None or ch.c or ch.relu or ch.perm is not None:
        raise Decline("reduction epilogue beyond a scale")
    q = _sandwich_out(ch.final)
    keepdim = len(shape(q.value)) == len(xshape)
    is_mean = Fraction(ch.w) * L == 1
    aten = "aten.mean.dim" if is_mean else (getattr(op.attributes.get("prov.aten"), "data", "") or "aten.sum.dim_IntList")
    try:
        k = ops.mean_dim(xshape, dims, keepdim, s_x=qx.scale, s_out=q.scale, w=ch.w, op=aten)
    except ValueError as exc:
        raise Decline(str(exc))
    if int(np.prod(k.args[-1].shape)) != int(np.prod(shape(q.value))):
        raise Decline("reduction output view")
    k.args[-1]
    return Match(aten, q.op, [op, *ch.ops, q.op, qx.dq], [("x", qx.value)], ("y", q.value), k,
                 numeric=dict(s_x=qx.scale, s_out=q.scale, dims=dims, w=ch.w))


# ---------------------------------------------------------------------------------------------
# convolution_backward: grad_weight (a 3-reduction contraction) and grad_input (a gather loop)
# ---------------------------------------------------------------------------------------------

def _unpad(v: SSAValue):
    """(source, pad_h, pad_w, consumed op) for a zero insert_slice pad, else (v, 0, 0, None)."""
    p = producer(v)
    if p is not None and p.name == "tensor.insert_slice" and splat_value(p.operands[1]) == 0.0:
        offs = list(p.properties["static_offsets"].iter_values())
        st = list(p.properties["static_strides"].iter_values())
        if offs[:2] == [0, 0] and all(x == 1 for x in st):
            return p.operands[0], offs[2], offs[3], p
    return v, 0, 0, None


def match_conv_wgrad(op: Operation) -> Match | None:
    if op.name != "linalg.generic" or iterators(op) != ["parallel"] * 4 + ["reduction"] * 3:
        return None
    if [o.name for o in op.regions[0].blocks[0].ops] != ["arith.mulf", "arith.addf", "linalg.yield"]:
        return None
    if len(op.operands) != 3 or splat_value(op.operands[2]) != 0.0:
        return None
    xsrc, ph, pw, pad = _unpad(op.operands[0])
    qx, qg = dequant(xsrc), dequant(op.operands[1])
    if qx is None or qg is None:
        return None
    maps = generic_maps(op)
    N, C, H, W = shape(qx.value)
    Ng, F, OH, OW = shape(qg.value)
    _, _, KH, KW = shape(op.results[0])
    bounds=(F,C,KH,KW,N,OH,OW)
    cx,cg,co=(map_coeffs(m,bounds) for m in maps)
    if cx is None or cg is None or co != [[1 if j == i else 0 for j in range(7)] for i in range(4)]:
        raise Decline("weight-gradient maps")
    block=op.regions[0].blocks[0]
    multiply,addition,yielded=list(block.ops)
    if not (set(multiply.operands)==set(block.args[:2]) and
            set(addition.operands)=={multiply.results[0],block.args[2]} and
            list(yielded.operands)==[addition.results[0]]):
        raise Decline('weight-gradient body is not a closed MAC')
    # x: (d4, d1, d5*sh + d2*dh, d6*sw + d3*dw); grad: (d4, d0, d5, d6)
    if cg != [[0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]:
        raise Decline("grad map")
    if cx[0] != [0, 0, 0, 0, 1, 0, 0] or cx[1] != [0, 1, 0, 0, 0, 0, 0]:
        raise Decline("input map")
    if maps[0].eval([F - 1, 0, 0, 0, 0, 0, 0], [])[1] != 0:
        raise Decline("grouped weight gradient")
    sh, dh, sw, dw = cx[2][5], cx[2][2], cx[3][6], cx[3][3]
    g = ConvGeom(N, H, W, C, F, KH, KW, sh, sw, ph, pw, dh, dw, 1)
    if (g.oh, g.ow, Ng) != (OH, OW, N):
        raise Decline("weight-gradient extents")
    q = _sandwich_out(op.results[0])
    k = ops.conv_backward_weight(g, s_g=qg.scale, s_x=qx.scale, s_out=q.scale)
    _pack_inputs(k,[("g",qg),("x",qx)])
    ops.convolution_zero_points(k,("g",qg),("x",qx),g,"weight")
    cons = [op, q.op, qx.dq, qg.dq] + ([pad] if pad else [])
    aten = getattr(op.attributes.get("prov.aten"), "data", "") or "aten.convolution_backward.default"
    return Match(aten, q.op, cons, [("g", qg.value), ("x", qx.value)], ("y", q.value), k,
                 layout=dict(source="NCHW,NCHW->OIHW", transposes="on-device"),
                 numeric=dict(output="grad_weight", s_g=qg.scale, s_x=qx.scale, s_out=q.scale))


def _dq_leaves(root: SSAValue, limit: int = 4000) -> list[QIn]:
    """Dequantized tensors the producers of ``root`` read (stopping at dequantize ops)."""
    seen, out, stack = set(), [], [root]
    while stack and len(seen) < limit:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        q = dequant(v)
        if q is not None:
            if all(x.value is not q.value for x in out):
                out.append(q)
            continue
        p = producer(v)
        if p is None:
            continue
        for o in p.walk():
            stack.extend(o.operands)
    return out


def _per_tap_contraction(root: SSAValue, qg: QIn, qw: QIn):
    """The generic T[n, g, c, kh, kw, oh, ow] = sum_f g[n, g, f, oh, ow] w[g, f, c, kh, kw] in root's producers."""
    seen, stack = set(), [root]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        p = producer(v)
        if p is None or dequant(v) is not None:
            continue
        if (p.name == "linalg.generic" and iterators(p) == ["parallel"] * 7 + ["reduction"]
                and [o.name for o in p.regions[0].blocks[0].ops] == ["arith.mulf", "arith.addf", "linalg.yield"]):
            return p
        stack.extend(p.operands)
    return None


def _conv_input_two_stage(op, q0, root, qg, qw):
    from . import linmap
    t = _per_tap_contraction(root, qg, qw)
    if t is None:
        raise Decline("grad_input region too large and not a per-tap contraction")
    N, C, H, W = shape(root)
    _, F, OH, OW = shape(qg.value)
    _, _, KH, KW = shape(qw.value)
    srcs = [_through_reshapes(x) for x in t.operands[:2]]
    if not {id(s_) for s_ in srcs} <= {id(qw.dq.results[0]), id(qg.dq.results[0])} or splat_value(t.operands[2]) != 0.0:
        raise Decline("per-tap contraction operands")
    cos = [map_coeffs(m) for m in generic_maps(t)]
    wmap = cos[0] if srcs[0] is qw.dq.results[0] else cos[1]
    gmap = cos[1] if srcs[0] is qw.dq.results[0] else cos[0]
    e = lambda d: [1 if j == d else 0 for j in range(8)]
    if (wmap != [e(1), e(7), e(2), e(3), e(4)] or gmap != [e(0), e(1), e(7), e(5), e(6)]
            or cos[2] != [e(d) for d in range(7)]):
        raise Decline("per-tap contraction maps")
    block = t.regions[0].blocks[0]
    multiply,addition,yielded = list(block.ops)
    if not (set(multiply.operands)==set(block.args[:2]) and
            set(addition.operands)=={multiply.results[0],block.args[2]} and
            list(yielded.operands)==[addition.results[0]]):
        raise Decline('per-tap contraction body is not closed MAC')
    tshape = shape(t.results[0])
    if tshape != (N, 1, C, KH, KW, OH, OW):
        raise Decline("per-tap contraction extents")

    def leaf(v):
        return (0, Fraction(1)) if v is t.results[0] else None

    lm, region = linmap.extract(root, leaf)
    if np.any(lm.coef != 1.0) or np.any(lm.const != 0):
        raise Decline("shift-add chain is not a pure scatter-add")
    n_, _, c_, kh_, kw_, oh_, ow_ = np.unravel_index(lm.idx, tshape)
    on, oc, oy, ox = np.unravel_index(lm.out, (N, C, H, W))
    if not (np.array_equal(on, n_) and np.array_equal(oc, c_)):
        raise Decline("scatter mixes channels")
    def axis_geometry(position, tap, output):
        # Solve the exact affine identity output = stride*position +
        # dilation*tap - padding. Rank-deficient axes get unit defaults.
        rows = []
        for a,b,c in sorted(set(zip(position.tolist(),tap.tolist(),output.tolist()))):
            row = [Fraction(a),Fraction(b),Fraction(1),Fraction(c)]
            for pivot,prior in rows:
                factor = row[pivot]
                row = [x-factor*y for x,y in zip(row,prior)]
            pivot = next((j for j in range(3) if row[j]),None)
            if pivot is None:
                if row[3]: raise Decline('non-affine transposed convolution scatter')
                continue
            row = [x/row[pivot] for x in row]
            rows.append((pivot,row)); rows.sort()
        answer = [Fraction(1),Fraction(1),Fraction(0)]
        for pivot,row in reversed(rows):
            answer[pivot] = row[3]-sum(row[j]*answer[j] for j in range(pivot+1,3))
        if any(v.denominator != 1 for v in answer):
            raise Decline('noninteger transposed convolution geometry')
        stride,dilation,offset = map(int,answer)
        return stride,dilation,-offset
    sh,dh,ph = axis_geometry(oh_,kh_,oy)
    sw,dw,pw = axis_geometry(ow_,kw_,ox)
    found = ConvGeom(N,H,W,C,F,KH,KW,sh,sw,ph,pw,dh,dw,1)
    found.validate()
    if (found.oh,found.ow) != (OH,OW):
        raise Decline('transposed convolution extents')
    iy = np.arange(OH)[:,None]*sh-ph+np.arange(KH)[None]*dh
    ix = np.arange(OW)[:,None]*sw-pw+np.arange(KW)[None]*dw
    expected = int(((iy>=0)&(iy<H)).sum()*((ix>=0)&(ix<W)).sum())*N*C
    pairs = np.stack((lm.out,lm.idx),axis=1)
    if expected != lm.out.size or len(np.unique(pairs,axis=0)) != expected:
        raise Decline('transposed scatter does not cover each contribution exactly once')
    if found is None:
        raise Decline("scatter is not a transposed convolution of a square geometry")
    k = ops.conv_backward_input(found, s_g=qg.scale, s_w=qw.scale, s_out=q0.scale)
    _pack_inputs(k,[("g",qg),("w",qw)])
    ops.convolution_zero_points(k,("g",qg),("w",qw),found,"input")
    aten = getattr(t.attributes.get("prov.aten"), "data", "") or "aten.convolution_backward.default"
    return Match(aten, q0.op, [*region, t, q0.op, qg.dq, qw.dq], [("g", qg.value), ("w", qw.value)],
                 ("y", q0.value), k, layout=dict(source="NCHW,OIHW->NCHW", transposes="on-device"),
                 numeric=dict(output="grad_input", s_g=qg.scale, s_w=qw.scale, s_out=q0.scale,
                              geometry=found.__dict__, verification="structural contraction + extracted scatter"))


def match_conv_input(op):
    if name_of(op) != 'quant_ext.quantize_per_tensor': return None
    q = quant_sink(op.operands[0])
    root = op.operands[0]
    if q is None or len(shape(root) or ()) != 4: return None
    leaves = [v for v in _dq_leaves(root) if v.dtype=='i8' and len(shape(v.value))==4]
    if len(leaves)!=2: return None
    N,C,H,W = shape(root)
    for qg,qw in (leaves,leaves[::-1]):
        gs,ws=shape(qg.value),shape(qw.value)
        if gs[0]==N and ws[0]==gs[1] and ws[1]==C and _per_tap_contraction(root,qg,qw) is not None:
            return _conv_input_two_stage(op,q,root,qg,qw)
    return None


# ---------------------------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------------------------

def _q_or_int(v: SSAValue):
    """A movement operand: dequantized int8 (scale) or a plain int8 tensor (scale None)."""
    q = dequant(v)
    if q is not None and q.dtype == "i8":
        return q
    if etype(v) == "i8":
        return QIn(v, 1.0, None, "i8")
    return None


def _movement_out(op: Operation, ins: list[QIn]):
    """(output value, output scale, consumed quantize op or None)."""
    res = op.results[0]
    if all(q.dq is None for q in ins) and etype(res) == "i8":
        return res, 1.0, None
    if any(q.dq is None for q in ins) or etype(res) not in ('f16', 'bf16', 'f32', 'f64'):
        return None
    qo = quant_sink(res)
    if qo is None:
        return None
    return qo.value, qo.scale, qo.op


def match_movement(op: Operation) -> Match | None:
    nm = op.name
    if nm == "linalg.transpose":
        q = _q_or_int(op.operands[0])
        if q is None:
            return None
        out = _movement_out(op, [q])
        if out is None:
            return None
        perm = [int(p) for p in op.properties["permutation"].iter_values()]
        k = ops.permute(shape(q.value), perm, s_x=q.scale, s_out=out[1])
        return Match("aten.permute.default", op, [op] + ([out[2], q.dq] if out[2] else []), [("x", q.value)],
                     ("y", out[0]), k)
    if nm == "tensor.cast" and shape(op.operands[0]) == shape(op.results[0]) and shape(op.results[0]) is not None:
        # a no-op cast materialized between DQ and Q (e.g. constant_pad_nd with zero pads): a copy
        q = _q_or_int(op.operands[0])
        if q is None:
            return None
        out = _movement_out(op, [q])
        if out is None:
            return None
        k = ops.clone(shape(q.value), s_x=q.scale, s_out=out[1])
        return Match("aten.clone.default", op, [op] + ([out[2], q.dq] if out[2] else []), [("x", q.value)],
                     ("y", out[0]), k, numeric=dict(form="identity cast"))
    if nm == "tensor.concat":
        qs = [_q_or_int(v) for v in op.operands]
        if any(q is None for q in qs):
            return None
        out = _movement_out(op, qs)
        if out is None:
            return None
        dim = int(op.properties["dim"].value.data)
        k = ops.cat([shape(q.value) for q in qs], dim, s_in=[q.scale for q in qs], s_out=out[1])
        cons = [op] + ([out[2]] + [q.dq for q in qs] if out[2] else [])
        return Match("aten.cat.default", op, cons, [(f"x{i}", q.value) for i, q in enumerate(qs)], ("y", out[0]), k)
    if nm == "tensor.insert_slice":
        src, dest = op.operands[0], op.operands[1]
        offs = list(op.properties["static_offsets"].iter_values())
        sizes = list(op.properties["static_sizes"].iter_values())
        strides_ = list(op.properties["static_strides"].iter_values())
        if len(op.operands) != 2:
            return None
        value = splat_value(dest)
        if value is not None:
            # constant_pad_nd: src (optionally an extract_slice crop) placed into a constant tensor
            if any(s != 1 for s in strides_):
                return None
            crop = producer(src)
            lo_crop = [0] * len(offs)
            base = src
            cons = [op]
            if crop is not None and crop.name == "tensor.extract_slice" and len(crop.operands) == 1:
                if any(s != 1 for s in crop.properties["static_strides"].iter_values()):
                    return None
                lo_crop = list(crop.properties["static_offsets"].iter_values())
                base = crop.operands[0]
                cons.append(crop)
            q = _q_or_int(base)
            if q is None:
                return None
            out = _movement_out(op, [q])
            if out is None:
                return None
            xs, os_ = shape(base), shape(op.results[0])
            pad = []
            for d in reversed(range(len(xs))):
                lo = offs[d] - lo_crop[d]
                hi = os_[d] - xs[d] - lo
                pad += [lo, hi]
            val = value if q.dq is not None else value
            k = ops.constant_pad_nd(xs, pad, val, s_x=q.scale, s_out=out[1])
            if out[2]:
                cons += [out[2], q.dq]
            return Match("aten.constant_pad_nd.default", op, cons, [("x", q.value)], ("y", out[0]), k,
                         numeric=dict(value=value, pad=pad))
        # slice_scatter: src into a data tensor along one strided dim
        qd, qs_ = _q_or_int(dest), _q_or_int(src)
        if qd is None or qs_ is None:
            return None
        out = _movement_out(op, [qd, qs_])
        if out is None:
            return None
        dshape = shape(dest)
        diff = [d for d in range(len(dshape)) if sizes[d] != dshape[d] or offs[d] != 0 or strides_[d] != 1]
        if len(diff) > 1:
            raise Decline("multi-dimensional scatter")
        d = diff[0] if diff else 0
        k = ops.slice_scatter(dshape, shape(src), d, offs[d], offs[d] + (sizes[d] - 1) * strides_[d] + 1,
                              strides_[d], s_self=qd.scale, s_src=qs_.scale, s_out=out[1])
        cons = [op] + ([out[2], qd.dq, qs_.dq] if out[2] else [])
        return Match("aten.slice_scatter.default", op, cons, [("self", qd.value), ("src", qs_.value)],
                     ("y", out[0]), k)
    if nm == "tensor.splat":
        value = _scalar_const(producer(op.operands[0]))
        if value is None:
            return None
        res = op.results[0]
        if etype(res) == "i8":
            if not users(res) or any(u.name == "linalg.generic" and res in u.operands[:len(u.operands) - 1]
                                     for u in users(res)):
                return None
            k = ops.full("aten.full.default", shape(res), value, s_out=1.0)
            return Match("aten.full.default", op, [op], [], ("y", res), k)
        qo = quant_sink(res)
        if qo is None:
            return None
        k = ops.full("aten.full.default", shape(res), value, s_out=qo.scale)
        return Match("aten.full.default", op, [op, qo.op], [], ("y", qo.value), k, numeric=dict(value=value))
    return None


def _region_closed(ops_, root_users) -> bool:
    inside = set(id(o) for o in ops_)
    for o in ops_:
        for r in o.results:
            for u in r.uses:
                if id(u.operation) not in inside and u.operation not in root_users:
                    return False
    return True


def match_linear(op: Operation) -> Match | None:
    """A region linear in its quantized (or plain int8) inputs, exactly extracted (``linmap``)."""
    from . import linmap, linmatch
    nm = name_of(op)
    if nm == "quant_ext.quantize_per_tensor":
        q = quant_sink(op.operands[0])
        if q is None or q.op is not op:
            return None
        root, out_val, s_out, integer = op.operands[0], op.results[0], q.scale, False
    elif op.name == "func.return":
        cands = [v for v in op.operands if etype(v) == "i8" and producer(v) is not None
                 and producer(v).name in ("linalg.generic", "scf.for", "tensor.cast", "tensor.insert_slice",
                                          "tensor.concat", "tensor.collapse_shape", "tensor.expand_shape")]
        if len(cands) != 1:
            return None
        root, out_val, s_out, integer = cands[0], cands[0], 1.0, True
        if 'gemmini.limb' in producer(root).attributes:
            return None  # A refused explicit component contract has no fallback.
    else:
        return None
    srcs: dict[int, linmatch.Src] = {}
    by_value: dict = {}

    def leaf(v):
        if integer:
            if isinstance(v, OpResult) or etype(v) != "i8":
                return None
            src = v
            scale = 1.0
        else:
            dq = dequant(v)
            if dq is None or dq.dtype != "i8":
                return None
            src, scale = dq.value, dq.scale
        if src not in by_value:
            k = len(by_value)
            by_value[src] = (k, dq.dq if not integer else None)
            srcs[k] = linmatch.Src(f"x{k}", shape(src), scale, dq.unsigned if not integer else False)
        return by_value[src][0], Fraction(scale), Fraction(scale)*((128 if dq.unsigned else 0)-dq.zero_point) if not integer else Fraction(0)

    try:
        lm, region = linmap.extract(root, leaf)
    except linmap.NotLinear as exc:
        raise Decline(f"not an exactly extractable linear region: {exc}")
    if not region:
        return None
    if not _region_closed(region, {op} | set(users(out_val))):
        raise Decline("region values escape the region")
    aten = getattr(region[-1].attributes.get("prov.aten"), "data", "") or "linear_region"
    kernel = linmatch.kernel_for(aten, lm, srcs, s_out)
    operands = [(srcs[k].name, v) for v, (k, _) in sorted(by_value.items(), key=lambda kv: kv[1][0])]
    consumed = list(region) + ([op] if not integer else []) + [d for _, (k, d) in by_value.items() if d is not None]
    return Match(aten, op if not integer else producer(root), consumed, operands, ("y", out_val), kernel,
                 numeric=dict(s_out=s_out, terms=int(lm.out.size), form=kernel.meta.get("form")))


MATCHERS = (match_convolution, match_convolution_rows, match_conv_wgrad, match_conv_input, match_contraction, match_mean, match_elementwise,
            match_movement, match_linear)


def _pack_inputs(kernel, inputs):
    from .stream import Arg,Ins
    s=kernel.stream
    for name,q in inputs:
        if not q.unsigned: continue
        old=s._index[name]; packed=name+'__signed'; new=len(s.args)
        s.args.append(Arg(packed,'i8',shape(q.value),'scratch'));s._index[packed]=new
        s.packs.append((name,packed))
        s.ins=[Ins(i.kind,i.attrs,new if i.arg==old else i.arg,i.off) for i in s.ins]


def _configure_quant(match):
    output = producer(match.result[1])
    if output is not None and name_of(output)=='quant_ext.quantize_per_tensor':
        q = quant_sink(output.operands[0])
        if q is not None:
            for r in match.kernel.stream.readouts:
                if r.out==match.result[0]:
                    r.zero_point,r.quant_min,r.quant_max=q.zero_point,q.qmin,q.qmax


def _bind_dense_layouts(match):
    """Bind the pointer ABI to source descriptor ranks after proven dense views."""
    sources = dict(match.operands)
    sources[match.result[0]] = match.result[1]
    for index, arg in enumerate(match.kernel.stream.args):
        if arg.name not in sources:
            continue
        value = sources[arg.name]
        actual = shape(value)
        if actual is None or etype(value) != arg.dtype or math.prod(actual) != math.prod(arg.shape):
            raise Decline("source tensor layout/type differs from the dense kernel ABI")
        match.kernel.stream.args[index] = replace(arg, shape=actual)


def match_all(module) -> tuple[list[Match], list[dict]]:
    """Every matched region in program order, plus recorded declines (with reasons)."""
    from .integer_regions import match_integer_region
    from .bitwise import match as match_bitwise
    from .limbs import match as match_limb
    matches, declines = [], []
    quantized_module = any(name_of(o).startswith('quant_ext.') for o in module.walk())
    generalized_quant = any(name_of(o).startswith('quant_ext.') and
        (splat_value(o.operands[2]) != 0 or _prop_int(o,'quant_min') != -128 or _prop_int(o,'quant_max') != 127)
        for o in module.walk() if len(o.operands)>=3 and etype(o.operands[0])!='i32')
    taken: set[int] = set()
    for ordinal, op in enumerate(module.walk()):
        if id(op) in taken:
            continue
        if any(any(d <= 0 for d in shape(v)) for v in (*op.operands, *op.results) if shape(v) is not None):
            declines.append(dict(ordinal=ordinal, op=op.name, reason="empty or dynamic tensor shape", aten=""))
            continue
        selected_matchers = (match_integer_region,match_convolution,match_convolution_rows,match_conv_wgrad,match_conv_input,match_contraction,match_elementwise,match_linear) if generalized_quant else (match_integer_region,*MATCHERS)
        if not quantized_module:
            selected_matchers = (match_bitwise, *selected_matchers)
        if 'gemmini.limb' in op.attributes:
            selected_matchers = (match_limb,)
        for fn in selected_matchers:
            try:
                m = fn(op)
                if m is not None:
                    if id(m.root) in taken:
                        break
                    _configure_quant(m)
                    _bind_dense_layouts(m)
            except (Decline, ValueError, OverflowError, ZeroDivisionError) as exc:
                declines.append(dict(ordinal=ordinal, op=op.name, reason=str(exc),
                                     aten=getattr(op.attributes.get("prov.aten"), "data", "")))
                m = None
                break
            if m is not None:
                matches.append(m)
                taken.update(id(x) for x in m.consumed if x is not None)
                break
        if name_of(op).startswith("quant_ext.quantize_") and id(op) not in taken and not any(d["ordinal"] == ordinal for d in declines):
            declines.append(dict(ordinal=ordinal, op=name_of(op), reason="unsupported or unproved quantization scheme; device requires exact per-tensor signed int8 readout", aten=""))
    if not quantized_module:
        arithmetic = {'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.negf',
                      'arith.addi', 'arith.subi', 'arith.muli', 'arith.fptosi', 'arith.sitofp',
                      'arith.extf', 'arith.truncf', 'arith.extsi', 'arith.extui', 'arith.trunci',
                      'arith.uitofp', 'arith.fptoui', 'arith.divsi', 'arith.divui',
                      'arith.andi', 'arith.ori', 'arith.xori', 'arith.cmpi'}
        reported = {d['ordinal'] for d in declines}
        for ordinal, candidate in enumerate(module.walk()):
            if id(candidate) in taken or ordinal in reported or any(k.startswith('prov.quant.') for k in candidate.attributes):
                continue
            if candidate.results and all(not list(r.uses) for r in candidate.results):
                continue
            if candidate.name in ('linalg.matmul', 'linalg.batch_matmul') or (
                    candidate.name == 'linalg.generic' and any(o.name in arithmetic for o in candidate.walk())):
                declines.append(dict(ordinal=ordinal, op=candidate.name,
                    reason='no exact integer schedule for unquantized data arithmetic or numeric conversion',
                    aten=getattr(candidate.attributes.get('prov.aten'), 'data', '')))
    return matches, declines
