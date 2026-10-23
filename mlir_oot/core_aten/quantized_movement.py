"""Closed quantized selectors with identical source/destination code contracts.

The proof is on the selector body and Q/DQ parameters, never provenance names.
Only literal elements undergo quantization, once at compile time using rationals.
"""

from fractions import Fraction
from math import isfinite, prod

import numpy as np

from . import linmap, ops
from .exactness import code_offset, quantized
from .bitwise import Selectors
from .match import (Decline, Match, dequant, name_of, producer, quant_sink,
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


def _literal(value):
    """A fill literal as a rational. -inf, inf and NaN have no code; the normal matchers record the decline."""
    if not isfinite(value):
        raise Decline('non-finite fill literal')
    return Fraction(value)


def _contract(q, kind, ordinal):
    op = q.op if kind == 'quantize' else q.dq
    attrs = attributes(op, kind)
    scale = ratios(attrs)[0] if 'scale_numerator' in attrs else Fraction(q.scale)
    return dict(operation_ordinal=ordinal, scale_numerator=str(scale.numerator),
                scale_denominator=str(scale.denominator), zero_point=q.zero_point,
                quant_min=_prop_int(op, 'quant_min'), quant_max=_prop_int(op, 'quant_max'),
                storage_dtype='uint8' if q.unsigned else 'int8',
                source_quant_attributes={'prov.quant.' + k: str(v) for k, v in attrs.items()})


def structural(op, qout, ordinals):
    from .layout_match import describe, literal
    from . import layout
    from .requant import Readout
    root_value = op.operands[0]
    root = producer(root_value)
    inputs, leaves, nodes, plans, visited = [], {}, [], {}, set()
    def visit(value):
        if value in visited:
            return True
        owner = producer(value)
        if owner is not None and name_of(owner)=='quant_ext.dequantize_per_tensor':
            q = dequant(value)
            if q is None or q.dtype!='i8':
                return False
            leaves[value] = len(inputs)
            inputs.append(q)
        elif owner is not None:
            constant = literal(owner)
            plan = ([],[]) if constant is not None else describe(owner,quantized=True)
            if constant is not None and owner.name=='linalg.generic':
                plan = (list(owner.operands[:-1]),[])
            if plan is None or not all(visit(v) for v in plan[0]):
                return False
            plans[value] = (plan,constant)
            nodes.append(value)
        else:
            return False
        visited.add(value)
        return True
    if not visit(root_value) or not nodes or not _region_closed([producer(v) for v in nodes],{op}):
        return None
    output = _contract(qout,'quantize',ordinals[op])
    records = {f'x{i}':_contract(q,'dequantize',ordinals[q.dq]) for i,q in enumerate(inputs)}
    fields = ('scale_numerator','scale_denominator','zero_point','quant_min','quant_max','storage_dtype')
    identical = all(all(record[k]==output[k] for k in fields) for record in records.values())
    so = Fraction(int(output['scale_numerator']),int(output['scale_denominator']))
    s = ops._begin([Arg(f'x{i}','i8',shape(q.value),'input') for i,q in enumerate(inputs)]+
                   [Arg('y','i8',shape(root_value),'output')])
    names = {}
    def scratch(name,dtype,sh):
        s._index[name] = len(s.args)
        s.args.append(Arg(name,dtype,tuple(sh),'scratch'))
        return name
    for value,i in leaves.items():
        q,record = inputs[i],records[f'x{i}']
        source = f'x{i}'
        if any(record[k]!=output[k] for k in fields):
            # Exact pointwise quantization commutes with a closed selector.
            # Widen each source once, retain the rational host boundary, then
            # execute the entire layout DAG in destination code units.
            if q.unsigned:
                source = s.signed_scratch(source, tuple(shape(q.value)))
            raw = scratch(f'raw_{i}','i32',shape(q.value))
            converted = scratch(f'converted_{i}','i8',shape(q.value))
            layout.widen(s,source,raw,shape(q.value));s.fence()
            factor = Fraction(int(record['scale_numerator']),int(record['scale_denominator']))/so
            s.readouts.append(Readout(converted,[(raw,factor)],constant=code_offset(q)*factor,
                                      zero_point=qout.zero_point,quant_min=qout.qmin,quant_max=qout.qmax,after=len(s.ins)))
            source=converted
        names[value]=source
    for number,value in enumerate(nodes):
        (values,maps),constant = plans[value]
        if (value is not root_value and producer(value).name in
                ('tensor.cast','tensor.reshape','tensor.collapse_shape','tensor.expand_shape')):
            names[value]=names[values[0]]
            continue
        target = 'y' if value is root_value else scratch(f'layout_{number}','i8',shape(value))
        if constant is not None:
            code=quantized(_literal(constant)/so,qout.qmin,qout.qmax,qout.zero_point)
            layout.fill(s,target,prod(shape(value)),code)
        for v,mapping in zip(values,maps):
            sh,ss,offset,ds,*source_offset=mapping
            layout.move(s,(names[v],source_offset[0] if source_offset else 0),(target,offset),sh,ss,destination_strides=ds)
            s.fence()
        names[value]=target
        s.fence()
    if identical:
        proof = dict(method='closed_static_selectors_identical_qparams',inputs=records,output=output,
                     code_identity='clamp(RNE((code-zero_point)*scale/scale)+zero_point)=code',
                     input_codes='within_declared_quant_range',literal_quantization='exact_rational_RNE_once')
        numeric = dict(family='movement',readout='storage_codes',requantization='none',exact_arithmetic=True,
                       identical_qparam_proof=proof)
    else:
        numeric = dict(family='movement',readout='full_i32',requantization='exact_rational_host_epilogue',
                       exact_arithmetic=True,source_qparams=records,output_qparams=output)
    kernel = ops._end('quantized_movement',s,**numeric,form='exact_movement_dma',
                      accumulator_min=-128,accumulator_max=127,selector_representation='shape_stride_map')
    kernel.meta['device_readout']=numeric['readout']
    return Match('quantized_movement',op,[root,*(producer(v) for v in nodes if producer(v) is not root),op,*(q.dq for q in inputs)],
                 [(f'x{i}',q.value) for i,q in enumerate(inputs)],('y',qout.value),kernel,numeric=numeric)


def match(op, ordinals):
    if name_of(op) != 'quant_ext.quantize_per_tensor':
        return None
    qout = quant_sink(op.operands[0])
    if qout is None:
        return None
    direct = structural(op,qout,ordinals)
    if direct is not None:
        return direct
    root = op.operands[0]
    # Bound the static selector map just as the native-byte selector does.
    if any(d <= 0 for d in shape(root)) or prod(shape(root)) > 1 << 16:
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
    identical = all(all(record[k] == output[k] for k in fields) for record in inputs.values())
    found = linmap.selector(value, shape(root))
    if found is None:
        return None
    which, indices, constants = found
    n = which.size
    if any(v != 0 for v in constants[which >= 0]):
        return None
    s = ops._begin([Arg(f'x{i}', 'i8', shape(q.value), 'input') for i, q in enumerate(sources)] +
                   [Arg('y', 'i8', shape(root), 'output')])
    from . import layout
    from .requant import Readout
    target='y'
    names=list(inputs)
    if not identical:
        target='codes'
        for name,dtype in [('codes','i8'),('raw_i32','i32')]:
            s._index[name]=len(s.args);s.args.append(Arg(name,dtype,shape(root),'scratch'))
        for i,q in enumerate(sources):
            if q.unsigned:
                names[i]=s.signed_scratch(names[i],shape(q.value),f'x{i}__signed')
    if np.any(which < 0):
        scale = Fraction(int(output['scale_numerator']), int(output['scale_denominator']))
        codes = np.zeros(n, np.uint8)
        for p in np.flatnonzero(which < 0):
            code = quantized(_literal(constants[p]) / scale, qout.qmin, qout.qmax, qout.zero_point)
            codes[p] = code & 255
        s.const('literal', codes.view(np.int8))
        if not identical:
            layout.move(s,('literal',0),('y',0),(n,),(1,))
            s.fence()
    mv.gather_elements(s, which, indices, 1, target, names, 'literal')
    if not identical:
        s.fence();layout.widen(s,'codes','raw_i32',shape(root))
        so=Fraction(int(output['scale_numerator']),int(output['scale_denominator']))
        for sid,q in enumerate(sources):
            positions=np.flatnonzero(which==sid)
            record=inputs[f'x{sid}']
            factor=Fraction(int(record['scale_numerator']),int(record['scale_denominator']))/so
            start=0
            while start<len(positions):
                end=start+1
                step=int(positions[end]-positions[start]) if end<len(positions) else 1
                while end<len(positions) and positions[end]-positions[end-1]==step:
                    end+=1
                s.readouts.append(Readout('y',[('raw_i32',factor)],
                    constant=code_offset(q)*factor,
                    placement=(end-start,1,int(positions[start]),step)))
                start=end
    proof = dict(method='closed_static_selectors_identical_qparams', inputs=inputs, output=output,
                 code_identity='clamp(RNE((code-zero_point)*scale/scale)+zero_point)=code',
                 input_codes='within_declared_quant_range', literal_quantization='exact_rational_RNE_once')
    numeric = dict(family='movement', readout='storage_codes', exact_arithmetic=True,
                   requantization='none', identical_qparam_proof=proof)
    if not identical:
        numeric=dict(family='movement',readout='full_i32',exact_arithmetic=True,
                     requantization='exact_rational_host_epilogue',source_qparams=inputs,output_qparams=output)
    kernel = ops._end('quantized_movement', s, **numeric,form='exact_movement_dma',
                      accumulator_min=-128,accumulator_max=127)
    kernel.meta['device_readout'] = numeric['readout']
    consumed = [producer(root), *(o for o in region.ops if o is not producer(root)), op,
                *(q.dq for q in sources)]
    return Match('quantized_movement', op, consumed,
                 [(f'x{i}', q.value) for i, q in enumerate(sources)], ('y', qout.value), kernel,
                 numeric=numeric)
