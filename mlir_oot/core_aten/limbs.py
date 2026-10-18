"""Explicit F2/F4 component producers; no native-to-int8 approximation.

The representation plan is descriptive. Exactness follows from the closed
integer source body and full-range partial-sum proofs, never from that label.
"""
import json
import numpy as np
from xdsl.dialects.builtin import StringAttr

from ..contraction_patterns import match_integer_gemm
from . import linmap, ops
from .exactness import require
from .match import Match, Decline, shape, etype, producer, generic_maps, identity_map, iterators
from .stream import Arg


def has_limb_plan(module):
    return any('gemmini.limb' in o.attributes for o in module.walk())


def byte_source(value):
    """Accept the exact sign-extension carrier emitted around i8 boundaries."""
    if etype(value) == 'i8':
        return value
    op = producer(value)
    if (etype(value) != 'i32' or op is None or op.name != 'linalg.generic' or
            len(op.operands) != 2 or len(op.results) != 1 or
            not all(identity_map(m) for m in generic_maps(op)) or
            any(it != 'parallel' for it in iterators(op))):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.ops)
    if (len(body) == 2 and body[0].name == 'arith.extsi' and body[1].name == 'linalg.yield'
            and etype(op.operands[0]) == 'i8' and shape(op.operands[0]) == shape(value)
            and list(body[0].operands) == [block.args[0]] and str(body[0].results[0].type) == 'i32'
            and list(body[1].operands) == [body[0].results[0]]
            and not any(k.startswith('prov.quant.') for k in op.attributes)):
        return op.operands[0]
    return None


def plan(op):
    attr = op.attributes.get('gemmini.limb')
    require(isinstance(attr, StringAttr), 'gemmini.limb must be a JSON StringAttr')
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate key ' + key)
                result[key] = value
            return result
        p = json.loads(attr.data, object_pairs_hook=unique)
    except (ValueError, TypeError) as exc:
        raise Decline('invalid gemmini.limb JSON: ' + str(exc)) from exc
    require(isinstance(p, dict), 'limb plan must be an object')
    keys = {'version', 'family', 'radix', 'inputs'}
    require(p.get('family') in ('F2', 'F4'), 'unsupported limb family')
    if p['family'] == 'F4':
        keys.add('component')
    require(set(p) == keys, 'unknown or missing limb plan keys')
    require(type(p['version']) is int and p['version'] == 1 and
            type(p['radix']) is int and p['radix'] == 256, 'unsupported limb plan version/radix')
    require(isinstance(p['inputs'], list) and p['inputs'], 'limb inputs must be a nonempty list')
    seen = set()
    for entry in p['inputs']:
        require(isinstance(entry, dict) and set(entry) == {'operand', 'index', 'count', 'exponent'},
                'unknown or missing limb input keys')
        require(all(type(v) is int for v in entry.values()), 'limb input fields must be integers')
        operand, index, count = (entry[k] for k in ('operand', 'index', 'count'))
        require(0 <= operand < len(op.operands) and operand not in seen, 'duplicate or invalid limb operand')
        require(0 <= index < count, 'invalid limb index/count')
        require(byte_source(op.operands[operand]) is not None,
                'limb input must be signed byte or its exact i32 sign extension')
        seen.add(operand)
    require(seen == {i for i, v in enumerate(op.operands) if byte_source(v) is not None},
            'limb plan does not cover every signed-byte operand')
    return p


def _bound(value):
    """Order-independent absolute bound, without cancellation assumptions."""
    if not isinstance(value, linmap.Sym):
        return np.abs(np.asarray(value, object))
    out = np.abs(np.asarray(value.const, object))
    for _, _, coefficient in value.terms:
        out = out + 128*np.abs(coefficient)
    return out


class IntegerRegion(linmap.Region):
    sum_operations = ('arith.addi',)

    def _eval_op(self, op, result):
        require(op.name != 'tensor.empty', 'uninitialized limb reduction seed')
        if op.name == 'linalg.fill':
            scalar = self.tensor(op.operands[0])
            require(not isinstance(scalar, linmap.Sym), 'nonconstant limb seed')
            return np.full(shape(op.results[0]), scalar, dtype=object)
        value = super()._eval_op(op, result)
        if isinstance(value, linmap.Sym):
            require(np.all(_bound(value) < (1 << 31)), 'unproved i32 limb intermediate overflow bound')
        return value

    def _scalar(self, op, vals, env, ivs, mask):
        symbolic = any(isinstance(v, linmap.Sym) for v in vals)
        nm = op.name
        if nm == 'arith.extsi' and symbolic:
            require(str(op.operands[0].type) in ('i8', 'i32') and str(op.results[0].type) in ('i32', 'i64'),
                    'limb extension is not signed byte/i32 widening')
            return vals[0]
        if symbolic and nm in ('arith.addi', 'arith.subi', 'arith.muli'):
            require(str(op.results[0].type) == 'i32', 'limb linear arithmetic must produce i32')
            a, b = vals
            if nm == 'arith.muli':
                require(not (isinstance(a, linmap.Sym) and isinstance(b, linmap.Sym)),
                        'nonlinear limb product needs a canonical contraction')
                value = a.scale(np.asarray(b, object)) if isinstance(a, linmap.Sym) else b.scale(np.asarray(a, object))
            else:
                value = (a if isinstance(a, linmap.Sym) else linmap.Sym([], np.asarray(a, object))).add(b, -1 if nm == 'arith.subi' else 1)
            require(np.all(_bound(value) < (1 << 31)), 'unproved i32 limb intermediate overflow bound')
            return value
        if symbolic:
            require(nm in ('tensor.extract', 'arith.select', 'scf.for', 'scf.if'),
                    'unsupported limb data operation: ' + nm)
            if nm == 'tensor.extract':
                require(not any(isinstance(v, linmap.Sym) for v in vals[1:]), 'data-dependent limb index')
            if nm in ('arith.select', 'scf.if'):
                require(not isinstance(vals[0], linmap.Sym), 'data-dependent limb predicate')
        require(not any(str(v.type).startswith(('f', 'bf')) for v in (*op.operands, *op.results)),
                'floating arithmetic in limb producer')
        value = super()._scalar(op, vals, env, ivs, mask)
        if op.results and not isinstance(value, (list, linmap.Sym)):
            dtype = str(op.results[0].type)
            if dtype in ('i8', 'i32', 'i64'):
                bits = int(dtype[1:])
                data = np.asarray(value, object)
                require(np.all(data >= -(1 << (bits-1))) and np.all(data < (1 << (bits-1))),
                        'unproved constant integer overflow in limb body')
        return value


def match(op):
    if 'gemmini.limb' not in op.attributes:
        return None
    p = plan(op)
    require(not any(k.startswith('prov.quant.') for k in op.attributes),
            'limb producer cannot also claim a quantized readout contract')
    require(len(op.results) == 1 and etype(op.results[0]) == 'i32', 'limb producer must return one i32 tensor')
    dims = match_integer_gemm(op)
    exponents = [8*e['index'] + e['exponent'] for e in p['inputs']]
    if dims is not None:
        require({e['operand'] for e in p['inputs']} == {0, 1}, 'bilinear limb operand plan')
        kernel = ops.matmul('limb_bilinear', dims.m, dims.k, dims.n,
                            batch=dims.batch if len(shape(op.operands[0])) == 3 else 0, s_out=None)
        inputs = [('a', op.operands[0]), ('b', op.operands[1])]
        bound, primitive_k = dims.k*16384, min(16, dims.k)
        binary_exponent = sum(exponents)
        consumed = [op]
        form = 'bilinear'
        kernel.meta['dimensions'] = dict(batch=dims.batch, m=dims.m, n=dims.n, k=dims.k)
    else:
        require(op.name in ('linalg.generic', 'linalg.reduce'), 'unsupported limb producer; use a closed integer generic/sum or canonical GEMM')
        require(len(set(exponents)) == 1, 'linear limb inputs have different binary exponents; retain separate host terms')
        ids = {byte_source(op.operands[e['operand']]): e['operand'] for e in p['inputs']}
        require(len(ids) == len(p['inputs']), 'duplicate limb SSA input; use one operand and explicit integer coefficients')
        def leaf(value):
            return (ids[value], 1) if value in ids else None
        try:
            lm, consumed = linmap.extract(op.results[0], leaf, region_type=IntegerRegion)
        except linmap.NotLinear as exc:
            raise Decline('unsupported exact limb linear map: ' + str(exc)) from exc
        require(set(lm.sources) <= set(ids.values()), 'unbound limb source')
        require(all(int(c) == c for c in (*lm.coef, *lm.const)), 'noninteger limb linear coefficient/seed')
        work = sum((abs(int(c))+126)//127 for c in lm.coef)
        require(work <= 1 << 18, 'limb linear instruction-part capacity exceeded')
        bounds = np.abs(np.asarray(lm.const, object))
        for out, c in zip(lm.out, lm.coef):
            bounds[int(out)] += 128*abs(int(c))
        bound = max((int(v) for v in bounds), default=0)
        require(bound < (1 << 31), 'unproved i32 limb accumulator overflow bound')
        inputs = [(f'x{k}', v) for v, k in sorted(ids.items(), key=lambda kv: kv[1])]
        args = [Arg(name, 'i8', shape(v), 'input') for name, v in inputs]
        args.append(Arg('out', 'i32', shape(op.results[0]), 'output'))
        stream = ops._begin(args)
        from .recipes.limbs import linear
        linear(stream, lm, {k: f'x{k}' for k in ids.values()})
        kernel = ops._end('limb_linear', stream)
        binary_exponent, primitive_k, form = exponents[0], 1, 'linear'
    if p['family'] == 'F4':
        require(p['component'] in (('rr', 'ii', 'ri', 'ir') if form == 'bilinear' else ('real', 'imag')),
                'invalid real/imag limb component')
    partial = primitive_k*(16384 if form == 'bilinear' else 128*127)
    proof = dict(method='full_signed_byte_absolute_partial_sums', input_range=[-128, 127],
                 primitive_k=primitive_k, max_mesh_partial_sum=partial,
                 required_signed_mac_bits=partial.bit_length()+1,
                 accumulator_min=-bound, accumulator_max=bound)
    numeric = dict(family=p['family'], limb_plan=p, form=form, binary_exponent=binary_exponent,
                   exact_arithmetic=True, readout='full_i32',
                   recombination='retained_source_host_exact_integer', bound_proof=proof)
    kernel.meta.update(numeric)
    return Match(kernel.op, op, consumed, inputs, ('out', op.results[0]), kernel, numeric=numeric)
