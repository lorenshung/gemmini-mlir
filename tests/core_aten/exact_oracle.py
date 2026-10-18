"""Independent rational source oracle for admitted Core ATen regions (test support)."""
from fractions import Fraction as F

import numpy as np

from mlir_oot.core_aten import linmap



def quantize(values, scale=1, lo=-128, hi=127):
    values = np.asarray(values, dtype=object)
    return np.array([min(hi, max(lo, round(F(v) / F(scale)))) for v in values.flat], dtype=np.int8).reshape(values.shape)


class SourceOracle(linmap.Region):
    """Concrete rational interpretation of source SSA, independent of matching.

    The interpreter reads Q/DQ operands and operation attributes itself. It
    never reads match.numeric, kernel metadata or extracted coefficient maps.
    The gather-loop executor supplies control flow only; concrete data values
    and each quantization decision remain exact rationals.
    """
    def __init__(self, inputs):
        super().__init__(lambda value: inputs.get(value),max_points=1 << 26)

    def _eval_op(self, op, result):
        from fractions import Fraction
        name = getattr(getattr(op,'op_name',None),'data',op.name)
        if name.startswith('quant_ext.'):
            values,scale,zp = [self.tensor(v) for v in op.operands[:3]]
            scale,zp = Fraction(np.asarray(scale).item()),int(np.asarray(zp).item())
            if getattr(op.attributes.get('prov.quant.zero_point_dtype'),'data',None)=='uint8': zp=zp%256
            low = op.properties['quant_min'].value.data
            high = op.properties['quant_max'].value.data
            storage=getattr(op.attributes.get('prov.quant.storage_dtype'),'data',None) or getattr(op.properties.get('output_dtype'),'data',None) or getattr(op.properties.get('input_dtype'),'data',None)
            unsigned=storage=='uint8'
            if 'dequantize' in name:
                unsigned = str(op.operands[0].type.get_element_type())=='i8' and unsigned
                codes = np.asarray(values)
                if unsigned: codes=codes.astype(np.int64).astype(np.uint8)
                return (codes.astype(object)-zp)*scale
            dt = np.uint8 if unsigned else np.int8
            return np.array([min(high,max(low,round(Fraction(x)/scale)+zp)) for x in np.asarray(values).flat],dtype=dt).reshape(np.shape(values))
        if name in ('linalg.matmul','linalg.batch_matmul'):
            a,b,init=[self.tensor(v).astype(object) for v in op.operands]
            return a@b+init
        if name=='linalg.reduce':
            value,init=[self.tensor(v).astype(object) for v in op.operands]
            dims=tuple(op.properties['dimensions'].iter_values())
            return value.sum(axis=dims)+init
        if name=='linalg.fill':
            return np.full(op.results[0].type.get_shape(),np.asarray(self.tensor(op.operands[0])).item(),dtype=object)
        return super()._eval_op(op,result)

    def _generic(self,op):
        from xdsl.ir.affine import AffineDimExpr
        maps=[m.data for m in op.properties['indexing_maps']]
        kinds=[getattr(t.data,'value',t.data) for t in op.properties['iterator_types']]
        dims=[None]*len(kinds)
        for value,mapping in zip(op.operands,maps):
            for size,expr in zip(value.type.get_shape(),mapping.results):
                if isinstance(expr,AffineDimExpr): dims[expr.position]=int(size)
        assert all(d is not None for d in dims),'unbounded source iteration'
        ivs=list(np.indices(dims,dtype=np.int64))
        block=op.regions[0].blocks[0]
        env={}
        n_in=len(op.operands)-len(op.results)
        reductions=tuple(i for i,k in enumerate(kinds) if k=='reduction')
        for ordinal,(value,mapping,arg) in enumerate(zip(op.operands,maps,block.args)):
            indices=tuple(mapping.eval(ivs,[]))
            if reductions and ordinal>=n_in:
                env[arg]=np.zeros(dims,dtype=object)
            else:
                env[arg]=np.asarray(self.tensor(value),dtype=object)[indices]
        result=self._block(block,env,ivs,np.ones(dims,bool))[0]
        result=np.broadcast_to(result,dims)
        if reductions:
            result=result.sum(axis=reductions)+self.tensor(op.operands[n_in])
        return np.asarray(result,dtype=object)

    def _scalar(self,op,vals,env,ivs,mask):
        name=op.name
        if name=='arith.bitcast':
            import struct
            source_type=op.operands[0].type
            target_type=op.results[0].type
            def cast(value):
                if str(source_type) in ('f32','f64'):
                    width=32 if str(source_type)=='f32' else 64
                    raw=int.from_bytes(struct.pack('<f' if width==32 else '<d',float(value)),'little')
                    return raw-(1<<width) if raw >= (1<<(width-1)) else raw
                width=32 if str(target_type)=='f32' else 64
                return F(struct.unpack('<f' if width==32 else '<d',(int(value)%(1<<width)).to_bytes(width//8,'little'))[0])
            return np.array([cast(v) for v in np.asarray(vals[0]).flat],dtype=object).reshape(np.shape(vals[0]))
        if name=='arith.extui':
            width=op.operands[0].type.width.data
            return np.asarray(vals[0],dtype=object)%(1<<width)
        if name=='arith.trunci':
            width=op.results[0].type.width.data
            result=np.asarray(vals[0],dtype=object)%(1<<width)
            return np.where(result >= (1<<(width-1)),result-(1<<width),result)
        if name in ('arith.shli','arith.shrsi','arith.shrui'):
            a,b=np.asarray(vals[0],dtype=object),np.asarray(vals[1],dtype=object)
            if name=='arith.shrui':a=a%(1<<op.operands[0].type.width.data)
            return a << b if name=='arith.shli' else a >> b
        if name=='math.roundeven':
            return np.array([round(F(x)) for x in np.asarray(vals[0]).flat],dtype=object).reshape(np.shape(vals[0]))
        if name=='cf.assert':
            assert np.all(vals[0]),'source assertion failed'
            return []
        return super()._scalar(op,vals,env,ivs,mask)


def evaluate(match, inputs):
    sources={value:np.asarray(inputs[name],dtype=object) for name,value in match.operands}
    result_name,result_value=match.result
    values=SourceOracle(sources).tensor(result_value)
    # Public tensor byte signedness is carried by Q, not signless MLIR i8.
    output=next(a for a in match.kernel.args if a.role=='output')
    dtype=np.int32 if output.dtype=='i32' else np.int8
    return {result_name:np.asarray(values,dtype=np.int64).astype(dtype).reshape(output.shape)}


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
