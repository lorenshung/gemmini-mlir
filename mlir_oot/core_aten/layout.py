"""Shape/stride-driven DMA schedules; no element index or coefficient arrays."""
from math import prod

from .stream import ACC, SP, DIM, CompactInstructions, InstructionLoop
from .exactness import require


def repeat(s, count, strides, body):
    if not count:
        return
    if not isinstance(s.ins, CompactInstructions):
        s.ins = CompactInstructions(s.ins)
    parent = s.ins
    s.ins = CompactInstructions()
    try:
        body()
        nodes = tuple(s.ins.nodes)
    finally:
        s.ins = parent
    if count == 1:
        parent.nodes.extend(nodes)
    else:
        parent.append(InstructionLoop(count, nodes, tuple((s._index[k], v) for k,v in strides.items())))


def rectangle(s, src, dst, rows, cols, sld, dld, *, widen=False):
    """Copy a rectangle in <=16x16 tiles with explicit CPU loops and tails."""
    require(rows > 0 and cols > 0, 'empty DMA rectangle')
    require(src[0]!=dst[0],'movement DMA requires nonaliasing source and destination arguments')
    s.config_ld(0, sld, 1, shrink=widen)
    s.config_st(dld)
    output_width = 4 if widen else 1
    def tile(r, c, rh, cw):
        local = ACC(0) if widen else SP(0)
        s.mvin(s.at(src[0], src[1] + r*sld + c), local, cw, rh)
        s.mvout(s.at(dst[0], dst[1] + r*dld + c*output_width),
                ACC(0, full=True) if widen else local, cw, rh)
    for nr, r, rh in ((rows//DIM, 0, DIM), (bool(rows%DIM), rows//DIM*DIM, rows%DIM)):
        if not nr:
            continue
        def row(r=r, rh=rh):
            repeat(s, cols//DIM, {src[0]:DIM, dst[0]:DIM*output_width}, lambda:tile(r,0,rh,DIM))
            if cols%DIM:
                tile(r,cols//DIM*DIM,rh,cols%DIM)
        repeat(s, nr, {src[0]:DIM*sld, dst[0]:DIM*dld}, row)


def dense_strides(shape, width=1):
    result = []
    for dim in reversed(shape):
        result.append(width)
        width *= dim
    return tuple(reversed(result))


def move(s, src, dst, shape, source_strides, *, width=1, destination_strides=None):
    """An affine selector in output coordinates, in ordered width-byte groups."""
    shape = tuple(shape)
    require(all(d > 0 for d in shape), 'empty or dynamic movement shape')
    target = dense_strides(shape, width) if destination_strides is None else tuple(destination_strides)
    strides = tuple(source_strides)
    require(len(shape) == len(strides) == len(target), 'movement stride rank mismatch')
    require(all(v >= 0 for v in strides) and all(v > 0 for v in target), 'negative movement stride')
    for (name, offset), step in ((src, strides), (dst, target)):
        require(offset >= 0 and offset + sum((d-1)*v for d,v in zip(shape,step)) + width <= s.args[s._index[name]].nbytes,
                'movement exceeds declared buffer extent')
    # Collapse a shared contiguous suffix without inspecting any tensor values.
    # A singleton coordinate is always zero, independent of its recorded
    # stride. Remove it before deciding which byte rectangles are contiguous.
    active = [i for i, d in enumerate(shape) if d != 1]
    axes = [shape[i] for i in active]
    ss, ds = [strides[i] for i in active], [target[i] for i in active]
    while len(axes)>1 and ss[-2] == ss[-1]*axes[-1] and ds[-2] == ds[-1]*axes[-1]:
        axes[-2:] = [axes[-2]*axes[-1]]
        ss[-2:] = [ss[-1]]
        ds[-2:] = [ds[-1]]
    if axes and ss[-1] == ds[-1] == width:
        cols = axes.pop()*width
        ss.pop(); ds.pop()
    else:
        cols = width
    if axes:
        rows = axes.pop(); sld = ss.pop(); dld = ds.pop()
    else:
        rows, sld, dld = 1, cols, cols
    def outer(axis):
        if axis == len(axes):
            if rows == 1 and cols > DIM:
                rectangle(s,src,dst,cols//DIM,DIM,DIM,DIM)
                if cols%DIM:
                    offset = cols//DIM*DIM
                    rectangle(s,(src[0],src[1]+offset),(dst[0],dst[1]+offset),1,cols%DIM,DIM,DIM)
            else:
                rectangle(s,src,dst,rows,cols,sld,dld)
        else:
            repeat(s,axes[axis],{src[0]:ss[axis],dst[0]:ds[axis]},lambda:outer(axis+1))
    outer(0)


def widen(s, src, dst, shape):
    n = prod(shape)
    full, tail = divmod(n,DIM)
    if full:
        rectangle(s,(src,0),(dst,0),full,DIM,DIM,DIM*4,widen=True)
    if tail:
        rectangle(s,(src,full*DIM),(dst,full*DIM*4),1,tail,tail,tail*4,widen=True)


def fill(s, target, n, code):
    fill_pattern(s,target,n,bytes([code & 255]))


def fill_pattern(s, target, n, pattern):
    import numpy as np
    require(DIM%len(pattern)==0 and n%len(pattern)==0,'literal pattern storage width')
    require(0<n<=s.args[s._index[target]].nbytes,'literal fill exceeds destination buffer')
    name = 'layout_fill_'+pattern.hex()
    if name not in s._index:
        s.const(name,np.frombuffer(pattern*(DIM//len(pattern)),dtype=np.int8).copy())
    s.config_ld(0,0,1)
    s.mvin(s.at(name),SP(64),DIM,DIM)
    s.config_st(DIM)
    repeat(s,n//(DIM*DIM),{target:DIM*DIM},lambda:s.mvout(s.at(target),SP(64),DIM,DIM))
    offset = n//(DIM*DIM)*(DIM*DIM)
    if n-offset>=DIM:
        s.mvout(s.at(target,offset),SP(64),DIM,(n-offset)//DIM)
    offset = n//DIM*DIM
    if n%DIM:
        s.mvout(s.at(target,offset),SP(64),n%DIM,1)


def quantized(op, shapes, output_shape, maps, factors, *, biases=None, unsigned=None, force_requant=False, placements=None, names=None, fill_code=None, staged=False):
    from . import ops
    from .stream import Arg
    from .requant import Readout
    biases = biases or [0]*len(shapes)
    unsigned = unsigned or [False]*len(shapes)
    exact = force_requant or any(f != 1 or b != 0 for f,b in zip(factors,biases))
    names = names or [f'x{i}' for i in range(len(shapes))]
    s = ops._begin([Arg(name,'i8',tuple(sh),'input') for name,sh in zip(names,shapes)]+
                   [Arg('y','i8',tuple(output_shape),'output')])
    def scratch(name,dtype):
        s._index[name] = len(s.args)
        s.args.append(Arg(name,dtype,tuple(output_shape),'scratch'))
    target = 'codes' if exact else 'y'
    if exact:
        scratch(target,'i8')
    if fill_code is not None:
        fill(s,'y',prod(output_shape),fill_code)
        if exact:
            fill(s,target,prod(output_shape),0)
        s.fence()
    for i,mapping in enumerate(maps):
        copy_shape,ss,offset,ds,*source_offset = mapping
        source = names[i]
        if exact and unsigned[i]:
            source = s.signed_scratch(source, tuple(shapes[i]))
        if i and staged:
            s.fence()
        move(s,(source,source_offset[0] if source_offset else 0),(target,offset),copy_shape,ss,destination_strides=ds)
    if exact:
        s.fence()
        scratch('raw_i32','i32')
        widen(s,target,'raw_i32',output_shape)
        if placements is None:
            placements = [None]
        for factor,bias,placement in zip(factors,biases,placements):
            s.readouts.append(Readout('y',[('raw_i32',factor)],constant=bias,placement=placement))
    return ops._end(op,s,form='exact_movement_dma',readout='full_i32' if exact else 'storage_codes',
                    accumulator_min=-128,accumulator_max=127,selector_proof='shape_stride_map')
