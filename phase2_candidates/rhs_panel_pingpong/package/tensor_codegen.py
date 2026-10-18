"""Bit-preserving tensor layout lowering for statically shaped xDSL IR.

Generated loops manipulate addresses and copy bytes. No tensor element is
converted or calculated by these layout operations. A staging buffer makes
results independent of any aliasing between the provided tensor pointers.
"""
from math import prod
import re


def _name(op):
    if op.name == 'builtin.unregistered':
        return op.attributes['op_name__'].data
    return op.name


def _attrs(op):
    return {**op.attributes, **dict(getattr(op, 'properties', {}))}


def _shape(value):
    typ = value.type
    if not hasattr(typ, 'get_shape'):
        raise TypeError('tensor layout operation requires ranked tensors')
    shape = tuple(int(v) for v in typ.get_shape())
    if any(v < 0 for v in shape):
        raise NotImplementedError('dynamic tensor layout shapes')
    return shape


def _width(value):
    typ = value.type.element_type
    text = str(typ)
    if text == 'index':
        return 8
    bits = getattr(typ, 'bitwidth', None)
    if bits is None:
        widths = {'bf16': 16, 'f16': 16, 'f32': 32, 'f64': 64}
        bits = widths.get(text)
    if bits is None:
        match = re.fullmatch(r'[us]?i(\d+)', text)
        if match:
            bits = int(match.group(1))
    if bits is None:
        raise NotImplementedError(f'tensor layout element type {typ}')
    return max(1, (int(bits) + 7) // 8)


def _integer(attr):
    value = getattr(attr, 'value', attr)
    return int(getattr(value, 'data', value))


def _array(attr):
    if hasattr(attr, 'get_values'):
        return [int(v) for v in attr.get_values()]
    return [_integer(v) for v in attr]


def _strides(shape):
    return [prod(shape[i + 1:]) for i in range(len(shape))]


def _coord(flat, shape, i):
    if shape[i] <= 1:
        return '0'
    stride = prod(shape[i + 1:])
    return f'(({flat} / {stride}UL) % {shape[i]}UL)'


def _slice_params(op, attrs, rank, initial):
    """Read mixed static/constant-SSA offsets, sizes and strides in ABI order."""
    dynamic = iter(op.operands[initial:])
    values = []
    for name in ('static_offsets', 'static_sizes', 'static_strides'):
        array = _array(attrs[name])
        if len(array) != rank:
            raise ValueError(f'{name} rank mismatch')
        resolved = []
        for v in array:
            if v < 0:
                operand = next(dynamic, None)
                owner = getattr(operand, 'owner', None)
                value = _attrs(owner).get('value') if getattr(owner, 'name', '') == 'arith.constant' else None
                if value is None:
                    raise NotImplementedError('nonconstant dynamic tensor slice geometry')
                v = _integer(value)
            resolved.append(v)
        values.append(resolved)
    return values


def _finish(tag, dst, nbytes, body):
    """Commit only after every input read, allowing arbitrary pointer aliases."""
    return '\n'.join([
        '{',
        f'  static unsigned char {tag}[{max(1, nbytes)}];',
        *['  ' + line for line in body],
        f'  for (unsigned long b = 0; b < {nbytes}UL; ++b)',
        f'    ((unsigned char *)({dst}))[b] = {tag}[b];',
        '}',
    ])


def emit_tensor(op, operands, results, uid):
    """Emit real data movement for supported tensor/layout operations.

    Operand/result strings are pointer expressions in the operation's SSA
    order. Shapes, permutation, slice geometry and concat dimension are read
    directly from the parsed operation and its ranked tensor types.
    """
    name = _name(op)
    if name == 'tensor.empty':
        return '/* tensor.empty has no initialized elements. */'
    supported = {'linalg.transpose', 'tensor.collapse_shape', 'tensor.expand_shape',
                 'tensor.extract_slice', 'tensor.insert_slice', 'tensor.concat', 'tensor.cast'}
    if name not in supported:
        raise NotImplementedError(f'tensor layout operation {name}')
    if len(results) != 1 or len(op.results) != 1:
        raise ValueError(f'{name} requires exactly one result')
    attrs = _attrs(op)
    outshape = _shape(op.results[0])
    width = _width(op.results[0])
    nbytes = prod(outshape) * width
    tag = 'tensor_tmp_' + re.sub(r'[^A-Za-z0-9_]', '_', str(uid))
    if any(_width(v) != width for v in op.operands if hasattr(v.type, 'get_shape')):
        raise ValueError('tensor layout operation must preserve element storage type')
    body = []
    if name in {'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'}:
        if prod(_shape(op.operands[0])) != prod(outshape):
            raise ValueError('reshape must preserve the number of elements')
        body += [f'for (unsigned long b = 0; b < {nbytes}UL; ++b)',
                 f'  {tag}[b] = ((const unsigned char *)({operands[0]}))[b];']
    elif name == 'linalg.transpose':
        shape = _shape(op.operands[0])
        permutation = _array(attrs['permutation'])
        if sorted(permutation) != list(range(len(shape))):
            raise ValueError('invalid transpose permutation')
        if outshape != tuple(shape[i] for i in permutation):
            raise ValueError('transpose result shape does not match permutation')
        strides = _strides(shape)
        address = ' + '.join(f'{_coord("i", outshape, d)} * {strides[p]}UL'
                             for d, p in enumerate(permutation)) or '0'
        body += [f'for (unsigned long i = 0; i < {prod(outshape)}UL; ++i) {{',
                 f'  unsigned long s = {address};',
                 f'  for (unsigned long b = 0; b < {width}UL; ++b)',
                 f'    {tag}[i * {width}UL + b] = ((const unsigned char *)({operands[0]}))[s * {width}UL + b];',
                 '}']
    elif name in {'tensor.extract_slice', 'tensor.insert_slice'}:
        insert = name == 'tensor.insert_slice'
        target_shape = outshape if insert else _shape(op.operands[0])
        offsets, sizes, steps = _slice_params(op, attrs, len(target_shape), 2 if insert else 1)
        if any(x < 0 for x in offsets + sizes) or any(x <= 0 for x in steps):
            raise ValueError('invalid tensor slice geometry')
        if any(sz and off + (sz - 1) * st >= dim for off, sz, st, dim in zip(offsets, sizes, steps, target_shape)):
            raise ValueError('tensor slice exceeds tensor extent')
        slice_shape = _shape(op.operands[0]) if insert else outshape
        if prod(slice_shape) != prod(sizes):
            raise ValueError('tensor slice element count mismatch')
        # Rank reduction can only drop unit dimensions. Flattening the complete
        # static sizes preserves the order of the resulting reduced-rank data.
        reduced = [d for d in sizes if d != 1]
        if [d for d in slice_shape if d != 1] != reduced:
            raise ValueError('tensor slice has invalid rank-reduced shape')
        strides = _strides(target_shape)
        address = ' + '.join(f'({offsets[d]}UL + {_coord("i", sizes, d)} * {steps[d]}UL) * {strides[d]}UL'
                             for d in range(len(target_shape))) or '0'
        if insert:
            if _shape(op.operands[1]) != outshape:
                raise ValueError('tensor.insert_slice destination/result mismatch')
            body += [f'for (unsigned long b = 0; b < {nbytes}UL; ++b)',
                     f'  {tag}[b] = ((const unsigned char *)({operands[1]}))[b];']
        body += [f'for (unsigned long i = 0; i < {prod(sizes)}UL; ++i) {{',
                 f'  unsigned long s = {address};',
                 f'  for (unsigned long b = 0; b < {width}UL; ++b)']
        if insert:
            body += [f'    {tag}[s * {width}UL + b] = ((const unsigned char *)({operands[0]}))[i * {width}UL + b];']
        else:
            body += [f'    {tag}[i * {width}UL + b] = ((const unsigned char *)({operands[0]}))[s * {width}UL + b];']
        body += ['}']
    elif name == 'tensor.concat':
        axis = _integer(attrs.get('dim', attrs.get('axis')))
        if axis < 0:
            axis += len(outshape)
        if axis not in range(len(outshape)):
            raise ValueError('invalid tensor.concat dimension')
        shapes = [_shape(v) for v in op.operands]
        if any(len(s) != len(outshape) or any(s[d] != outshape[d] for d in range(len(s)) if d != axis) for s in shapes):
            raise ValueError('tensor.concat non-concatenated dimensions differ')
        if sum(s[axis] for s in shapes) != outshape[axis]:
            raise ValueError('tensor.concat result extent mismatch')
        inner = prod(outshape[axis + 1:]) * width
        outer = prod(outshape[:axis])
        axis_offset = 0
        for pointer, shape in zip(operands, shapes):
            chunk = shape[axis] * inner
            body += [f'for (unsigned long o = 0; o < {outer}UL; ++o)',
                     f'  for (unsigned long b = 0; b < {chunk}UL; ++b)',
                     f'    {tag}[o * {outshape[axis] * inner}UL + {axis_offset * inner}UL + b] = ((const unsigned char *)({pointer}))[o * {chunk}UL + b];']
            axis_offset += shape[axis]
    return _finish(tag, results[0], nbytes, body)
