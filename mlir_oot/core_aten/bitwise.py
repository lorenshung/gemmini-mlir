"""Prove tensor selectors and implement them with element-width byte DMA.

No data arithmetic is accepted, including floating identities such as x+0.
Indices may use the existing exact static index interpreter. Provenance is
traceability only; the proof checks the actual body and element storage types.
"""

import numpy as np
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, TensorType
from . import linmap, ops
from .exactness import require
from .match import Match, etype, shape, producer, _attr_of
from .recipes import movement as mv
from .storage import itemsize, literal_bytes
from .stream import Arg, Stream


CARRIERS = {'tensor.splat', 'tensor.concat', 'tensor.insert_slice', 'tensor.extract_slice',
            'tensor.reshape', 'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast', 'linalg.generic',
            'linalg.broadcast', 'linalg.transpose', 'linalg.fill'}


class Selectors(linmap.Region):
    def _eval_op(self, op, result):
        if op.name == 'arith.constant':
            attr = _attr_of(op, 'value')
            if isinstance(attr, FloatAttr):
                return np.asarray(attr.value.data)  # retain negative zero
            if isinstance(attr, DenseIntOrFPElementsAttr):
                return np.asarray(list(attr.get_values()), dtype=object).reshape(shape(op.results[0]))
        if op.name == 'linalg.fill':
            value = self.tensor(op.operands[0])
            return np.full(shape(op.results[0]), np.asarray(value).item(), dtype=object)
        if op.name == 'scf.for':
            raise linmap.NotLinear('loop-carried data is not proved bitwise movement')
        if op.name not in CARRIERS and op.name != 'arith.constant':
            raise linmap.NotLinear('not a tensor selector: ' + op.name)
        return super()._eval_op(op, result)

    def _generic(self, op):
        if any(t.data.value != 'parallel' for t in _attr_of(op, 'iterator_types')):
            raise linmap.NotLinear('reductions are not bitwise movement')
        return super()._generic(op)

    def _scalar(self, op, vals, env, ivs, mask):
        if any(isinstance(v, linmap.Sym) for v in vals):
            if op.name == 'tensor.extract' and any(isinstance(v, linmap.Sym) for v in vals[1:]):
                raise linmap.NotLinear('data-dependent movement index')
            if op.name == 'arith.select' and isinstance(vals[0], linmap.Sym):
                raise linmap.NotLinear('data-dependent movement predicate')
            if op.name == 'arith.bitcast':
                if itemsize(str(op.operands[0].type)) != itemsize(str(op.results[0].type)):
                    raise linmap.NotLinear('bitcast changes element storage width')
                return vals[0]
            if op.name not in ('tensor.extract', 'arith.select'):
                raise linmap.NotLinear('data arithmetic/conversion is not bitwise movement: ' + op.name)
        if op.name in ('arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.negf',
                       'arith.extf', 'arith.truncf', 'arith.sitofp', 'arith.fptosi'):
            raise linmap.NotLinear('floating computation is not bitwise movement: ' + op.name)
        if op.name == 'arith.constant' and isinstance(_attr_of(op, 'value'), FloatAttr):
            return np.asarray(_attr_of(op, 'value').value.data)
        return super()._scalar(op, vals, env, ivs, mask)


def match(op):
    if any(k.startswith('prov.quant.') for k in op.attributes):
        return None
    if op.name not in CARRIERS or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
        return None
    from .layout_match import native
    structural = native(op)
    if structural is not None:
        return structural
    root = op.results[0]
    width = itemsize(etype(root))
    n = int(np.prod(shape(root))) if shape(root) else 1
    require(n <= 1 << 16 and n * width <= 1 << 24, 'interpreted selector exceeds bounded byte-map capacity; use a structural layout')
    sources, ids = [], {}

    def leaf(value):
        if value is root or not isinstance(value.type, TensorType):
            return None
        owner = producer(value)
        if owner is not None and owner.name in ('arith.constant', 'tensor.empty'):
            return None
        require(int(np.prod(shape(value))) * itemsize(etype(value)) <= 1 << 24,
                'bitwise source exceeds static byte-map capacity')
        if value not in ids:
            ids[value] = len(sources)
            sources.append(value)
        return ids[value], 1

    reg = Selectors(leaf)
    try:
        value = reg.tensor(root)
    except linmap.NotLinear:
        return None
    if not isinstance(value, linmap.Sym):
        value = linmap.Sym([], np.asarray(value, dtype=object))
    value = value.bshape(shape(root))
    which = np.full(n, -1, dtype=np.int64)
    indices = np.zeros(n, dtype=np.int64)
    for sid, index, coefficient in value.terms:
        c = np.asarray(coefficient).reshape(-1)
        active = c != 0
        require(np.all(c[active] == 1) and np.all(which[active] == -1), 'selector combines data elements')
        require(itemsize(etype(sources[sid])) == width, 'selector changes element storage width')
        which[active], indices[active] = sid, np.asarray(index).reshape(-1)[active]
    constants = np.asarray(value.const, dtype=object).reshape(-1)
    require(all(v == 0 for v in constants[which >= 0]), 'selector adds a data constant')
    args = [Arg(f'x{i}', etype(v), shape(v), 'input') for i, v in enumerate(sources)]
    args.append(Arg('y', etype(root), shape(root), 'output'))
    s = Stream(args, bitwise=True)
    s.fence(); s.flush()
    # Only unselected constant elements are encoded; input payloads stay opaque.
    if np.any(which < 0):
        data = bytearray(n * width)
        for p in np.flatnonzero(which < 0):
            data[p * width:(p + 1) * width] = literal_bytes(constants[p], etype(root))
        s.const('literal', np.frombuffer(data, dtype=np.int8).copy())
    mv.gather_elements(s, which, indices, width, 'y', [f'x{i}' for i in range(len(sources))], 'literal')
    kernel = ops._end('bitwise_movement', s, family='F1', numerics='bitwise_movement',
                     element_bytes=width, quantization='none', readout='scratchpad_bytes',
                     byte_order='little', proof='closed_static_element_selectors',
                     operand_element_bytes={f'x{i}': itemsize(etype(v)) for i, v in enumerate(sources)})
    return Match('bitwise_movement', op, [op], [(f'x{i}', v) for i, v in enumerate(sources)],
                 ('y', root), kernel, numeric=dict(family='F1', numerics='bitwise_movement', element_bytes=width,
                    operand_element_bytes=kernel.meta['operand_element_bytes'], byte_order='little',
                    quantization='none', exact_arithmetic=True))
