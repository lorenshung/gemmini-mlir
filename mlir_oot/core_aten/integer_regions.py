"""Source-bound seeded integer MAC plus exact Gemmini narrowing readout.

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
    """Return a structurally checked full-range symmetric scalar readout."""
    uses = list(op.results[0].uses)
    if len(uses) != 1:
        return None
    reader = uses[0].operation
    if reader.name != 'linalg.generic' or len(reader.operands) != 2 or len(reader.results) != 1:
        return None
    if str(reader.results[0].type.get_element_type()) != 'i8':
        return None
    if any(i != 'parallel' for i in iterators(reader)) or not all(identity_map(m) for m in generic_maps(reader)):
        return None
    if shape(reader.results[0]) != shape(op.results[0]):
        return None
    block = reader.regions[0].blocks[0]
    body = list(block.ops)
    attrs = {k.removeprefix('prov.quant.'): v for k, v in reader.attributes.items() if k.startswith('prov.quant.')}
    for key,expected in [('rounding','half_to_even'),('saturate','true'),('accumulator','i32'),('granularity','tensor')]:
        if key in attrs:
            require(getattr(attrs[key],'data',None) == expected, 'unsupported readout policy: ' + key)
    if getattr(attrs.get('arithmetic'), 'data', None) == 'exact':
        allowed = {'scale','granularity','zero_point','quant_min','quant_max','rounding','saturate','relu','accumulator','arithmetic'}
        require(not (attrs.keys() - allowed), 'unknown prov.quant.* readout parameters')
        number = lambda key: getattr(getattr(attrs.get(key), 'value', None), 'data', None)
        factor = number('scale')
        require(isinstance(factor, (int,float)) and math.isfinite(factor) and factor > 0 and number('zero_point') == 0
                and number('quant_min') == -128 and number('quant_max') == 127,
                'integer readout requires full signed range and positive per-tensor scale')
        relu = getattr(attrs.get('relu'), 'data', None)
        require(relu in ('true','false'), 'missing exact readout activation')
        # Delegate the canonical integer quantization algorithm to its source
        # owner, rather than duplicating its arbitrary-width arithmetic here.
        try:
            from m2m.ir.exact_quant import build_readout
        except ImportError:
            raise Decline('exact arithmetic: canonical source readout verifier is unavailable')
        generated, result = build_readout(op.results[0], [factor], output_scale=1.0, relu=relu == 'true')
        expected = result.owner
        equivalent = reader.regions[0].is_structurally_equivalent(expected.regions[0])
        for temporary in reversed(generated):
            temporary.drop_all_references()
        require(equivalent, 'exact readout body differs from its canonical source semantics')
        return reader, factor, relu == 'true'
    # This is model2MLIR's scalar, zp=0, RNE + saturated narrowing body.
    names = ['arith.sitofp', 'arith.constant', 'arith.mulf', 'math.roundeven',
             'arith.constant', 'arith.constant', 'arith.maximumf', 'arith.minimumf',
             'arith.fptosi', 'linalg.yield']
    if [x.name for x in body] != names:
        return None
    cast, scale, mul, rounding, low, high, maximum, minimum, narrow, yielded = body
    if not (list(cast.operands) == [block.args[0]] and str(cast.results[0].type) == 'f32'
            and list(mul.operands) == [cast.results[0], scale.results[0]]
            and list(rounding.operands) == [mul.results[0]]
            and list(maximum.operands) == [rounding.results[0], low.results[0]]
            and list(minimum.operands) == [maximum.results[0], high.results[0]]
            and list(narrow.operands) == [minimum.results[0]]
            and list(yielded.operands) == [narrow.results[0]]):
        return None
    lo, hi, factor = _scalar_const(low), _scalar_const(high), _scalar_const(scale)
    if lo not in (-128, 0) or hi != 127 or factor is None or factor <= 0:
        raise Decline('exact arithmetic: integer readout requires full signed range and positive per-tensor scale')
    allowed = {'scale', 'granularity', 'zero_point', 'quant_min', 'quant_max', 'rounding', 'saturate', 'relu', 'accumulator'}
    attrs = {k.removeprefix('prov.quant.'): v for k, v in reader.attributes.items() if k.startswith('prov.quant.')}
    require(not (attrs.keys() - allowed), 'unknown prov.quant.* readout parameters')
    # If supplied, metadata must describe the checked body exactly.
    for key, expected in [('scale', factor), ('zero_point', 0), ('quant_min', -128), ('quant_max', 127)]:
        if key in attrs:
            actual = getattr(getattr(attrs[key], 'value', None), 'data', None)
            require(actual == expected, 'readout metadata differs from its body: ' + key)
    if 'relu' in attrs:
        require(getattr(attrs['relu'], 'data', None) == ('true' if lo == 0 else 'false'), 'ReLU metadata differs from body')
    return reader, factor, lo == 0


def has_integer_readout(module):
    return any(match_integer_gemm(op, require_zero=False) is not None and
               any(use.operation.name == 'linalg.generic' and
                   str(use.operation.results[0].type.get_element_type()) == 'i8'
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
    require(dims.batch == 1 and len(shape(op.operands[0])) == 2, 'seeded readout batching is not implemented')
    seed = op.operands[2]
    consumed = [op, reader]
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
            require(dimensions == [0] and shape(seed_op.operands[0]) == (dims.n,), 'non-channel seed broadcast')
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
    kernel = ops.matmul('aten.addmm.default' if bias else 'aten.mm.default', dims.m, dims.k, dims.n,
                        s_out=Q(1) / Q(scale), bias=bias, bias_const=values,
                        bias_rows=bias_rows, relu=relu, bias_range=guarded_range)
    kernel.meta.update(integer_readout=True, dimensions=dict(batch=1,m=dims.m,n=dims.n,k=dims.k),
                       seed=('guarded_runtime' if runtime_seed is not None else 'constant') if bias else 'zero')
    return Match(kernel.op, reader, consumed, [('a', op.operands[0]), ('b', op.operands[1])] + ([('bias',runtime_seed)] if runtime_seed is not None else []),
                 ('out', reader.results[0]), kernel,
                 numeric=dict(integer_readout=True, scale=scale, relu=relu, seed=None if values is None else values.tolist(), runtime_seed=runtime_seed is not None))
