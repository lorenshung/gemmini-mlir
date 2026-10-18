"""Closed quantized selectors with identical source/destination code contracts.

The proof is on the selector body and Q/DQ parameters, never provenance names.
Only literal elements undergo quantization, once at compile time using rationals.
"""

from fractions import Fraction
from math import prod

import numpy as np

from . import linmap, ops
from .bitwise import Selectors
from .match import (Match, dequant, name_of, producer, quant_sink,
                    shape, _prop_int, _region_closed)
from .quant_schema import attributes, ratios
from .recipes import movement as mv
from .stream import Arg


class QuantizedSelectors(Selectors):
    def _scalar(self, op, vals, env, ivs, mask):
        # Same-qparam dtype copies may contain a lossless float widening.
        if op.name == 'arith.extf' and linmap.exact_float_widening(
                str(op.operands[0].type), str(op.results[0].type)):
            return vals[0]
        return super()._scalar(op, vals, env, ivs, mask)


def _contract(q, kind, ordinal):
    op = q.op if kind == 'quantize' else q.dq
    attrs = attributes(op, kind)
    scale = ratios(attrs)[0] if 'scale_numerator' in attrs else Fraction(q.scale)
    return dict(operation_ordinal=ordinal, scale_numerator=str(scale.numerator),
                scale_denominator=str(scale.denominator), zero_point=q.zero_point,
                quant_min=_prop_int(op, 'quant_min'), quant_max=_prop_int(op, 'quant_max'),
                storage_dtype='uint8' if q.unsigned else 'int8',
                source_quant_attributes={'prov.quant.' + k: str(v) for k, v in attrs.items()})


def match(op, ordinals):
    if name_of(op) != 'quant_ext.quantize_per_tensor':
        return None
    qout = quant_sink(op.operands[0])
    if qout is None:
        return None
    root = op.operands[0]
    # Bound the static selector map just as the native-byte selector does.
    if any(d <= 0 for d in shape(root)) or prod(shape(root)) > 1 << 23:
        return None
    sources, ids = [], {}

    def leaf(value):
        if producer(value) is None or name_of(producer(value)) != 'quant_ext.dequantize_per_tensor':
            return None
        q = dequant(value)
        if q is None or q.dtype != 'i8':
            raise linmap.NotLinear('selector input has no byte quantization contract')
        if any(d <= 0 for d in shape(q.value)) or prod(shape(q.value)) > 1 << 23:
            raise linmap.NotLinear('selector input exceeds static code-map capacity')
        if value not in ids:
            ids[value] = len(sources)
            sources.append(q)
        return ids[value], 1

    region = QuantizedSelectors(leaf)
    try:
        value = region.tensor(root)
    except linmap.NotLinear:
        return None
    if not region.ops or not _region_closed(region.ops, {op}):
        return None
    output = _contract(qout, 'quantize', ordinals[op])
    inputs = {f'x{i}': _contract(q, 'dequantize', ordinals[q.dq]) for i, q in enumerate(sources)}
    fields = ('scale_numerator', 'scale_denominator', 'zero_point',
              'quant_min', 'quant_max', 'storage_dtype')
    if any(any(record[k] != output[k] for k in fields) for record in inputs.values()):
        return None  # Preserve the existing full-i32 + exact host requant path.
    if not isinstance(value, linmap.Sym):
        value = linmap.Sym([], np.asarray(value, dtype=object))
    value = value.bshape(shape(root))
    n = prod(shape(root))
    which, indices = np.full(n, -1, np.int64), np.zeros(n, np.int64)
    for sid, index, coefficient in value.terms:
        c = np.asarray(coefficient).reshape(-1)
        active = c != 0
        if not (np.all(c[active] == 1) and np.all(which[active] == -1)):
            return None
        which[active], indices[active] = sid, np.asarray(index).reshape(-1)[active]
    constants = np.asarray(value.const, dtype=object).reshape(-1)
    if any(v != 0 for v in constants[which >= 0]):
        return None
    s = ops._begin([Arg(f'x{i}', 'i8', shape(q.value), 'input') for i, q in enumerate(sources)] +
                   [Arg('y', 'i8', shape(root), 'output')])
    if np.any(which < 0):
        scale = Fraction(int(output['scale_numerator']), int(output['scale_denominator']))
        codes = np.zeros(n, np.uint8)
        for p in np.flatnonzero(which < 0):
            code = min(qout.qmax, max(qout.qmin, round(Fraction(constants[p]) / scale) + qout.zero_point))
            codes[p] = code & 255
        s.const('literal', codes.view(np.int8))
    mv.gather_elements(s, which, indices, 1, 'y', list(inputs), 'literal')
    proof = dict(method='closed_static_selectors_identical_qparams', inputs=inputs, output=output,
                 code_identity='clamp(RNE((code-zero_point)*scale/scale)+zero_point)=code',
                 input_codes='within_declared_quant_range', literal_quantization='exact_rational_RNE_once')
    numeric = dict(family='movement', readout='storage_codes', exact_arithmetic=True,
                   requantization='none', identical_qparam_proof=proof)
    kernel = ops._end('quantized_movement', s, **numeric)
    kernel.meta['device_readout'] = 'storage_codes'
    consumed = [producer(root), *(o for o in region.ops if o is not producer(root)), op,
                *(q.dq for q in sources)]
    return Match('quantized_movement', op, consumed,
                 [(f'x{i}', q.value) for i, q in enumerate(sources)], ('y', qout.value), kernel,
                 numeric=numeric)
