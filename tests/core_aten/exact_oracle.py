"""Independent rational source oracle for admitted Core ATen regions (test support)."""
from fractions import Fraction as F

import numpy as np

from mlir_oot.core_aten import linmap
from mlir_oot.core_aten.match import dequant, producer, name_of


def quantize(values, scale=1, lo=-128, hi=127):
    values = np.asarray(values, dtype=object)
    return np.array([min(hi, max(lo, round(F(v) / F(scale)))) for v in values.flat], dtype=np.int8).reshape(values.shape)


def evaluate(match, inputs):
    numeric = match.numeric
    result_name, result_value = match.result
    output = next(a for a in match.kernel.args if a.role == 'output')
    if numeric.get('integer_readout'):
        result = inputs['a'].astype(object) @ inputs['b'].astype(object)
        if numeric.get('runtime_seed'):
            result += inputs['bias'].astype(object)
        if numeric['seed'] is not None:
            result += np.asarray(numeric['seed'], dtype=object)
        if numeric['relu']:
            result = np.maximum(result, 0)
        return {result_name: quantize(result * F(numeric['scale']))}
    if 'wa' in numeric:
        a = inputs['a'].astype(object) * F(numeric['wa']) * F(numeric['s_a'])
        b = inputs['b'].astype(object) * F(numeric['wb']) * F(numeric['s_b'])
        return {result_name: quantize(a + b + F(numeric['c']), numeric['s_out'])}
    if 's_a' in numeric:
        a, b = inputs['a'].astype(object), inputs['b'].astype(object)
        values = a @ b
        scale = F(numeric['alpha']) * F(numeric['s_a']) * F(numeric['s_b']) / F(numeric['s_out'])
        assert 'bias' not in inputs
        return {result_name: quantize(values * scale)}
    if 'dims' in numeric:
        values = np.sum(inputs['x'].astype(object), axis=tuple(numeric['dims']))
        scale = F(numeric['w']) * F(numeric['s_x']) / F(numeric['s_out'])
        return {result_name: quantize(values * scale).reshape(output.shape)}
    if 's_x' in numeric and 'w' in numeric:
        values = inputs['x'].astype(object) * F(numeric['w']) * F(numeric['s_x']) + F(numeric['c'])
        if match.op == 'aten.relu.default':
            values = np.maximum(values, 0)
        return {result_name: quantize(values, numeric['s_out'])}
    if 's_a' not in numeric and 'wa' in numeric:
        raise AssertionError('affine oracle needs operand scales')
    # Structural movement only: extract source indices and literal constants.
    # Its coefficients are unchanged source binary32 scales, exactly representable
    # in binary64. No float arithmetic is allowed in the extracted region.
    sources = {v: i for i, (_, v) in enumerate(match.operands)}
    scales = {}
    def leaf(value):
        dq = dequant(value)
        src, scale = (dq.value, dq.scale) if dq else (value, 1)
        if src not in sources:
            return None
        scales[sources[src]] = scale
        return sources[src], scale
    root = result_value
    q = producer(root)
    s_out = 1
    if q is not None and name_of(q) == 'quant_ext.quantize_per_tensor':
        from mlir_oot.core_aten.match import splat_value
        s_out = splat_value(q.operands[1])
        root = q.operands[0]
    lm, region = linmap.extract(root, leaf)
    assert not any(x.name in ('arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'linalg.reduce')
                   for op in region for x in op.walk()), 'oracle requires rational arithmetic dispatch'
    values = np.array([F(float(c)) for c in lm.const], dtype=object)
    arrays = [inputs[name].reshape(-1) for name, _ in match.operands]
    for out, sid, idx, coefficient in zip(lm.out, lm.src, lm.idx, lm.coef):
        values[out] += F(float(coefficient)) * int(arrays[sid][idx])
    if output.dtype == 'i32':
        return {result_name: np.asarray(values, np.int32).reshape(output.shape)}
    return {result_name: quantize(values, s_out).reshape(output.shape)}


def command_reference(cb, inputs):
    """Slow rational opcode semantics, independent of instruction recipes."""
    import itertools
    values = {k: np.asarray(v, dtype=object).reshape(cb['tensors'][k]['shape']) for k, v in inputs.items()}
    outputs = {}
    for command in cb['commands']:
        op, args, attrs = command['opcode'], command['operands'], command.get('attributes', {})
        if op == 'FENCE':
            continue
        dst = args['dst']
        shape = tuple(cb['tensors'][dst]['shape'])
        dtype = cb['tensors'][dst]['dtype']
        if op == 'AFFINE_MAP':
            result = np.full(shape, int(attrs['constant']), dtype=object)
            for key, weight in zip(('x', 'y'), attrs['weights']):
                result += weight * values[args[key]]
            if attrs['activation'] == 'relu':
                result = np.maximum(result, 0)
            if dtype == 'i8':
                result = quantize(result * F(float(np.float32(attrs['readout_scale']))))
        elif op == 'FILL':
            result = np.full(shape, attrs['value_code'], dtype=object)
        elif op == 'INSERT_SLICE':
            result = values[args['dest']].copy()
            src = values[args['src']]
            result[tuple(slice(o, o + n*s, s) for o, n, s in zip(attrs['offsets'], src.shape, attrs['strides']))] = src
        elif op == 'CONCAT':
            arrays = [values[args[k]] for k in sorted(args) if k != 'dst']
            scales = attrs.get('scales', [1] * len(arrays))
            arrays = [quantize(a * F(float(np.float32(s)))) for a, s in zip(arrays, scales)]
            result = np.concatenate(arrays, axis=attrs['axis'])
        elif op == 'MOVEMENT':
            src = values[args['src']]
            form = attrs.get('form', 'identity')
            if form == 'identity':
                result = src.copy()
            elif form == 'permute':
                result = np.transpose(src, attrs['perm'])
            elif form == 'slice':
                result = src[tuple(slice(o, o+n*s, s) for o, n, s in zip(attrs['offsets'], attrs['sizes'], attrs['strides']))]
            elif form == 'broadcast':
                result = np.broadcast_to(src, shape).copy()
            elif form == 'repeat':
                result = np.tile(src, attrs['repeats'])
            elif form == 'pad':
                mode = attrs['mode']
                result = np.pad(src, list(zip(attrs['low'], attrs['high'])),
                    mode='edge' if mode == 'replicate' else mode,
                    **({'constant_values': attrs['value_code']} if mode == 'constant' else {}))
            else:
                raise AssertionError(form)
            if dtype == 'i8':
                result = quantize(result * F(float(np.float32(attrs.get('scale', 1)))))
        elif op == 'WINDOW_SUM':
            src = values[args['src']]
            mode = attrs['mode']
            scale = F(float(np.float32(attrs['readout_scale'])))
            result = np.full(shape, F(0), dtype=object)
            if mode == 'reduce':
                dims = tuple(d % src.ndim for d in attrs['dims'])
                count = int(np.prod([src.shape[d] for d in dims])) if attrs['average'] else 1
                result = np.sum(src, axis=dims).reshape(shape) * scale / (attrs.get('divisor_override') or count)
            elif mode == 'fold':
                kernel, stride, pad, dilation = (attrs[k] for k in ('kernel', 'stride', 'padding', 'dilation'))
                blocks = [(size+2*p-d*(k-1)-1)//s+1 for size,p,d,k,s in zip(shape[2:], pad,dilation,kernel,stride)]
                columns = src.reshape(shape[:2] + tuple(kernel) + tuple(blocks))
                for index in np.ndindex(columns.shape):
                    n,c = index[:2]
                    taps, positions = index[2:2+len(kernel)], index[2+len(kernel):]
                    spatial = tuple(pos*s-p+tap*d for pos,s,p,tap,d in zip(positions,stride,pad,taps,dilation))
                    if all(0 <= i < limit for i,limit in zip(spatial,shape[2:])):
                        result[(n,c,*spatial)] += int(columns[index]) * scale
            else:
                transposed = mode.endswith('_transpose')
                full_shape, pooled_shape = (shape, src.shape) if transposed else (src.shape, shape)
                axes = tuple(range(2,len(shape))) if attrs.get('layout','nhwc') == 'nchw' else tuple(range(1,len(shape)-1))
                for pooled in np.ndindex(pooled_shape):
                    spans, denominator = [], 1
                    for j,axis in enumerate(axes):
                        if mode.startswith('adaptive'):
                            start = pooled[axis] * full_shape[axis] // pooled_shape[axis]
                            end = -(-((pooled[axis]+1)*full_shape[axis]) // pooled_shape[axis])
                            denominator *= end-start
                        else:
                            k,s,p = (attrs[key][j] for key in ('kernel','stride','padding'))
                            start = pooled[axis]*s-p
                            end = start+k
                            if attrs['count_include_pad']:
                                denominator *= min(end,full_shape[axis]+p)-max(start,-p)
                            else:
                                denominator *= min(end,full_shape[axis])-max(start,0)
                        spans.append(range(max(start,0), min(end,full_shape[axis])))
                    denominator = attrs.get('divisor_override') or denominator
                    for spatial in itertools.product(*spans):
                        full = list(pooled)
                        for axis,index in zip(axes,spatial):
                            full[axis] = index
                        if transposed:
                            result[tuple(full)] += int(src[pooled]) * scale / denominator
                        else:
                            result[pooled] += int(src[tuple(full)]) * scale / denominator
            result = quantize(result)
        else:
            raise AssertionError(op)
        values[dst] = np.asarray(result, dtype=object)
        outputs[dst] = np.asarray(result, dtype=np.int32 if dtype == 'i32' else np.int8).reshape(-1)
    return outputs


def host_guard_values(text, inputs):
    """Evaluate source seed assertions before direct kernel tests; no device arithmetic here."""
    from mlir_oot.frontend.parse import parse_module
    from xdsl.ir import BlockArgument
    module=parse_module(text)
    ordinals={op:i for i,op in enumerate(module.walk())}
    function=next(op for op in module.walk() if op.name=='func.func')
    arguments=list(function.body.blocks[0].args)
    key=lambda v: ('arg',arguments.index(v)) if isinstance(v,BlockArgument) and v in arguments else ('op',ordinals[v.owner],list(v.owner.results).index(v))
    values={('arg',int(name[3:])):array for name,array in inputs.items()}
    outputs={}
    for generic in module.walk():
        if generic.name!='linalg.generic' or not any(op.name=='cf.assert' for op in generic.walk()):
            continue
        if len(generic.operands)!=2 or key(generic.operands[0]) not in values:
            continue
        block=generic.regions[0].blocks[0]
        env={block.args[0]:values[key(generic.operands[0])].astype(np.int64)}
        for op in block.ops:
            operands=[env[v] for v in op.operands]
            if op.name=='arith.extsi': result=operands[0]
            elif op.name=='arith.constant': result=op.value.value.data
            elif op.name=='arith.cmpi':
                pred=op.predicate.value.data
                assert pred in (3,5)
                result=(operands[0]<=operands[1]) if pred==3 else (operands[0]>=operands[1])
            elif op.name=='arith.andi': result=operands[0] & operands[1]
            elif op.name=='cf.assert':
                assert np.all(operands[0]),'source seed guard failed'
                continue
            elif op.name=='linalg.yield':
                result=operands[0].astype(np.int32)
                values[key(generic.results[0])]=result
                outputs[key(generic.results[0])]=result
                continue
            else: raise AssertionError(op.name)
            env[op.results[0]]=result
    return outputs
