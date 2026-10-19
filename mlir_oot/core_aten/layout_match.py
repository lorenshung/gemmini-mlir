"""Recognize closed structural layout operations without interpreting elements."""
from math import prod

from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr

from . import layout, ops
from .match import Match, shape, etype, _attr_of
from .storage import itemsize
from .stream import Arg, Stream


def literal(op):
    from .match import _scalar_const, producer, splat_value
    if op.name == "arith.constant" and len(op.results) == 1:
        return splat_value(op.results[0])
    if op.name in ('tensor.splat','linalg.fill'):
        return _scalar_const(producer(op.operands[0]))
    if op.name=='linalg.generic':
        body=list(op.regions[0].blocks[0].ops)
        if (len(body)==2 and body[-1].name=='linalg.yield'
                and list(body[-1].operands)==list(body[0].results)):
            return _scalar_const(body[0])
    return None


def describe(op, *, quantized=False):
    """Return source values and rectangular maps in element units, or no match."""
    if len(op.results) != 1 or shape(op.results[0]) is None:
        return None
    out = shape(op.results[0])
    if any(d <= 0 for d in out):
        return None
    ds = layout.dense_strides(out)
    name = op.name
    if name == 'linalg.transpose':
        src = op.operands[0]
        perm = tuple(_attr_of(op,'permutation').iter_values())
        if sorted(perm) != list(range(len(shape(src)))) or out != tuple(shape(src)[p] for p in perm):
            return None
        strides = layout.dense_strides(shape(src))
        return [src], [(out,tuple(strides[p] for p in perm),0,ds)]
    if name in ('tensor.cast','tensor.reshape','tensor.collapse_shape','tensor.expand_shape'):
        from .views import reindex_shape
        try:
            if reindex_shape(op) != out or prod(shape(op.operands[0])) != prod(out):
                return None
        except ValueError:
            return None
        return [op.operands[0]], [(out,layout.dense_strides(out),0,ds)]
    if name == 'tensor.concat':
        dim = _attr_of(op,'dim').value.data
        inputs = list(op.operands)
        if not 0 <= dim < len(out):
            return None
        offset, maps = 0, []
        for value in inputs:
            sh = shape(value)
            if len(sh)!=len(out) or any(sh[d]!=out[d] for d in range(len(out)) if d!=dim):
                return None
            maps.append((sh,layout.dense_strides(sh),offset*ds[dim],ds))
            offset += sh[dim]
        if offset != out[dim]:
            return None
        return inputs,maps
    if name=='tensor.extract_slice' and len(op.operands)==1:
        src=op.operands[0]
        offsets=tuple(_attr_of(op,'static_offsets').iter_values())
        steps=tuple(_attr_of(op,'static_strides').iter_values())
        sizes=tuple(_attr_of(op,'static_sizes').iter_values())
        if sizes!=out or len(offsets)!=len(shape(src)) or any(v<0 for v in offsets) or any(v<=0 for v in steps):
            return None
        if any(o+(n-1)*st>=d for o,n,st,d in zip(offsets,sizes,steps,shape(src))):
            return None
        ss=layout.dense_strides(shape(src))
        return [src],[(out,tuple(v*st for v,st in zip(ss,steps)),0,ds,sum(o*v for o,v in zip(offsets,ss)))]
    if name=='tensor.insert_slice' and len(op.operands)==2:
        src,dest=op.operands
        offsets=tuple(_attr_of(op,'static_offsets').iter_values())
        steps=tuple(_attr_of(op,'static_strides').iter_values())
        sizes=tuple(_attr_of(op,'static_sizes').iter_values())
        if shape(dest)!=out or shape(src)!=sizes or len(offsets)!=len(out) or any(v<0 for v in offsets) or any(v<=0 for v in steps):
            return None
        if any(o+(n-1)*st>=d for o,n,st,d in zip(offsets,sizes,steps,out)):
            return None
        return [dest,src],[(out,ds,0,ds),(sizes,layout.dense_strides(sizes),sum(o*v for o,v in zip(offsets,ds)),tuple(v*st for v,st in zip(ds,steps)))]
    if name == 'linalg.broadcast':
        dims = set(_attr_of(op,'dimensions').iter_values())
        src = op.operands[0]
        ss = iter(layout.dense_strides(shape(src)))
        remaining = tuple(out[i] for i in range(len(out)) if i not in dims)
        if remaining != shape(src):
            return None
        return [src],[(out,tuple(0 if i in dims else next(ss) for i in range(len(out))),0,ds)]
    if name != 'linalg.generic':
        return None
    maps = [m.data for m in _attr_of(op,'indexing_maps')]
    if any(t.data.value != 'parallel' for t in _attr_of(op,'iterator_types')):
        return None
    if (maps[-1].num_dims!=len(out) or
            tuple(maps[-1].results)!=tuple(AffineDimExpr(i) for i in range(len(out)))):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.ops)
    if not body or body[-1].name != 'linalg.yield' or len(body[-1].operands)!=1:
        return None
    yielded = body[-1].operands[0]
    if len(body)==2 and len(body[0].operands)==1:
        cast = body[0]
        from .linmap import exact_float_widening
        if (cast.name=='arith.extf' and quantized and
                exact_float_widening(str(cast.operands[0].type),str(cast.results[0].type))):
            pass
        elif cast.name=='arith.bitcast' and itemsize(str(cast.operands[0].type))==itemsize(str(cast.results[0].type)):
            pass
        else:
            return None
        if yielded is not cast.results[0]:
            return None
        yielded = cast.operands[0]
    elif len(body)!=1:
        return None
    if yielded not in block.args[:-1]:
        return None
    index = list(block.args).index(yielded)
    src = op.operands[index]
    sh = shape(src)
    ss = layout.dense_strides(sh)
    stride = [0]*len(out)
    seen = set()
    for extent,step,expr in zip(sh,ss,maps[index].results):
        if isinstance(expr,AffineDimExpr):
            if expr.position in seen or out[expr.position]!=extent:
                return None
            seen.add(expr.position)
            stride[expr.position] = step
        elif not isinstance(expr,AffineConstantExpr) or expr.value!=0 or extent!=1:
            return None
    return [src],[(out,tuple(stride),0,ds)]


def native(op):
    if op.name == "arith.constant":
        return None  # Literal storage is already materialized by the host ABI.
    value = literal(op)
    plan = ([],[]) if value is not None else describe(op)
    if plan is None:
        return None
    sources,maps = plan
    root = op.results[0]
    width = itemsize(etype(root))
    if any(itemsize(etype(v)) != width for v in sources):
        return None
    args = [Arg(f'x{i}',etype(v),shape(v),'input') for i,v in enumerate(sources)]
    s = Stream(args+[Arg('y',etype(root),shape(root),'output')],bitwise=True)
    s.fence();s.flush()
    if value is not None:
        from .storage import literal_bytes
        layout.fill_pattern(s,'y',prod(shape(root))*width,literal_bytes(value,etype(root)))
    for i,mapping in enumerate(maps):
        sh,ss,off,ds,*source_offset=mapping
        if i:
            s.fence()
        layout.move(s,(f'x{i}',(source_offset[0] if source_offset else 0)*width),('y',off*width),sh,tuple(v*width for v in ss),width=width,
                    destination_strides=tuple(v*width for v in ds))
    numeric = dict(family='F1',numerics='bitwise_movement',element_bytes=width,
                   operand_element_bytes={f'x{i}':width for i in range(len(sources))},byte_order='little',
                   quantization='none',exact_arithmetic=True)
    kernel = ops._end('bitwise_movement',s,**numeric,readout='scratchpad_bytes',proof='closed_static_element_selectors',
                      selector_representation='shape_stride_map')
    return Match('bitwise_movement',op,[op],[(f'x{i}',v) for i,v in enumerate(sources)],('y',root),kernel,numeric=numeric)
