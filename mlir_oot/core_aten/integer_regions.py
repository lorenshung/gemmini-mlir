"""Source-bound seeded integer MAC with full-width readout and retained host quantization.

Only a checked closed body grants semantics. Provenance describes bindings but
never substitutes for that check. Unbounded runtime seeds stay on the host:
i32 SRAM cannot implement an infinite-precision seeded sum without overflow.
"""
from fractions import Fraction as Q
import math

import numpy as np

from ..contraction_patterns import match_integer_gemm, _zero_init
from . import ops
from .exactness import require
from .match import (Match, Decline, producer, shape, dense_values, splat_value,
                    _attr_of, _scalar_const, identity_map, generic_maps, iterators)


def readout(op):
    """Identify the source readout; retain its exact host implementation.

    The backend binds ONLY the closed integer producer. It does not reinterpret
    an explicit float bias or substitute a rounded product for the rational pair.
    """
    from .quant_schema import attributes, ratios
    uses = list(op.results[0].uses)
    for use in uses:
        reader = use.operation
        if reader.name != 'linalg.generic' or len(reader.results) != 1:
            continue
        if getattr(reader.attributes.get('prov.quant.arithmetic'),'data',None) != 'exact':
            continue
        attrs = attributes(reader,'integer_readout')
        require(getattr(attrs.get('host_only'),'data',None) != 'true','host-only source readout')
        require(getattr(attrs.get('accumulator'),'data',None)=='i32','readout accumulator is not i32')
        require(getattr(attrs.get('numerics'),'data',None)=='integer_readout','readout numerics are not exact integer')
        factors = ratios(attrs)
        require(all(s != 0 for s in factors),'zero exact readout coefficient')
        require(len(factors) in (1,shape(op.results[0])[-1]),'readout scale channel extent')
        if any(k.startswith('prov.quant.') for k in op.attributes):
            producer_attrs = attributes(op,'contraction')
            require(getattr(producer_attrs.get('host_only'),'data',None) != 'true','host-only source contraction')
        return reader,factors,False
    return None


def has_integer_readout(module):
    return any(match_integer_gemm(op, require_zero=False) is not None and
               any(getattr(use.operation.attributes.get('prov.quant.arithmetic'),'data',None)=='exact'
                   for use in op.results[0].uses)
               for op in module.walk())


def seed_guard(value):
    """Recognize the source's retained i32 range assertion, with all SSA links checked."""
    guard = producer(value)
    if guard is None or guard.name != 'linalg.generic' or len(guard.operands) != 2:
        return None
    if any(i != 'parallel' for i in iterators(guard)) or not all(identity_map(m) for m in generic_maps(guard)):
        return None
    block = guard.regions[0].blocks[0]
    body = list(block.ops)
    if [x.name for x in body] != ['arith.extsi','arith.constant','arith.cmpi','arith.constant','arith.cmpi','arith.andi','cf.assert','linalg.yield']:
        return None
    ext,low,ge,high,le,both,assertion,yielded = body
    if not (str(block.args[0].type) == 'i32' and str(ext.results[0].type) == 'i64'
            and list(ext.operands) == [block.args[0]]
            and list(ge.operands) == [ext.results[0],low.results[0]] and ge.properties['predicate'].value.data == 5
            and list(le.operands) == [ext.results[0],high.results[0]] and le.properties['predicate'].value.data == 3
            and list(both.operands) == [ge.results[0],le.results[0]]
            and list(assertion.operands) == [both.results[0]] and list(yielded.operands) == [block.args[0]]):
        return None
    lower,upper = _scalar_const(low),_scalar_const(high)
    require(lower is not None and upper is not None and -(1<<31) <= lower <= upper < (1<<31), 'invalid seed guard range')
    return guard, guard.operands[0], (int(lower),int(upper))


def match_integer_region(op):
    dims = match_integer_gemm(op, require_zero=False)
    if dims is None:
        return None
    selected = readout(op)
    if selected is None:
        return None
    reader, scale, relu = selected
    rank = len(shape(op.operands[0]))
    batch = dims.batch if rank == 3 else 0
    seed = op.operands[2]
    consumed = [op]
    bias = not _zero_init(seed)
    bias_rows = True
    values = None
    guarded_range = None
    guard_op = None
    runtime_seed = None
    if bias:
        seed_op = producer(seed)
        if seed_op is not None and seed_op.name == 'linalg.broadcast':
            dimensions = list(seed_op.properties['dimensions'].iter_values())
            require(dimensions == list(range(rank - 1)) and shape(seed_op.operands[0]) == (dims.n,), 'non-channel seed broadcast')
            consumed.append(seed_op)
            seed = seed_op.operands[0]
            bias_rows = False
        checked = seed_guard(seed)
        if checked is not None:
            guard_op, original_seed, guarded_range = checked
            values = dense_values(original_seed)
            if values is not None:
                require(np.all(values >= guarded_range[0]) and np.all(values <= guarded_range[1]), 'constant seed violates its range guard')
                consumed.append(guard_op)
                seed = original_seed
        else:
            values = dense_values(seed)
        if values is None:
            value = splat_value(seed)
            if value is not None:
                values = np.full(shape(seed), value, np.int64)
        require(values is not None or guarded_range is not None, 'runtime i32 seed has no nonoverflow range proof')
        if values is not None:
            require(np.all(np.isfinite(values)) and np.all(values == np.trunc(values)), 'noninteger seed')
            values = np.asarray(values, np.int32)
        else:
            runtime_seed = seed
    kernel = ops.matmul('aten.bmm.default' if batch else 'aten.addmm.default' if bias else 'aten.mm.default', dims.m, dims.k, dims.n,
                        batch=batch, s_out=None, bias=bias, bias_const=values,
                        bias_rows=bias_rows, relu=relu, bias_range=guarded_range)
    kernel.meta.update(integer_readout=True, dimensions=dict(batch=dims.batch,m=dims.m,n=dims.n,k=dims.k),
                       seed=('guarded_runtime' if runtime_seed is not None else 'constant') if bias else 'zero')
    kernel.meta.update(readout="full_i32", requantization="retained_source_exact_integer",
                       scale_numerator=[str(v.numerator) for v in scale],scale_denominator=[str(v.denominator) for v in scale])
    return Match(kernel.op, op, consumed, [('a', op.operands[0]), ('b', op.operands[1])] + ([('bias',runtime_seed)] if runtime_seed is not None else []),
                 ('out', op.results[0]), kernel,
                 numeric=dict(integer_readout=True, scale=scale, relu=relu, seed=None if values is None else values.tolist(), runtime_seed=runtime_seed is not None))
