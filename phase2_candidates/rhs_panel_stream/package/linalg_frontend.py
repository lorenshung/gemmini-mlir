"""Structural xDSL frontend and exact integer contraction lowering.

No textual SSA matching: all tensor edges, indexing maps and scalar operations
are read from the parsed IR. ``inventory`` is also useful to mixed-program
lowering, because its operation indices include initializers and constants.
"""
from __future__ import annotations

from math import prod


def _name(op):
    from xdsl.dialects.builtin import StringAttr
    if op.name == 'builtin.unregistered':
        attr = op.attributes.get('op_name__')
        if isinstance(attr, StringAttr):
            return attr.data
    return op.name


def _attrs(op):
    return {**op.attributes, **dict(getattr(op, 'properties', {}))}


def _shape(value_type):
    from xdsl.dialects.builtin import TensorType
    return list(value_type.get_shape()) if isinstance(value_type, TensorType) else []


def _dtype(value_type):
    from xdsl.dialects.builtin import TensorType
    return str(value_type.element_type) if isinstance(value_type, TensorType) else str(value_type)


def _constant(value):
    op = getattr(value, 'owner', None)
    if getattr(op, 'name', '') == 'tensor.splat':
        return _constant(op.operands[0])
    if getattr(op, 'name', '') != 'arith.constant':
        return None
    attr = _attrs(op).get('value')
    data = getattr(getattr(attr, 'value', None), 'data', None)
    return data if isinstance(data, (int, float)) else None


def _context():
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.tensor import Tensor
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.affine import Affine
    from xdsl.dialects.memref import MemRef
    from xdsl.ir import Dialect
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.dialects.builtin import StringAttr

    # Upstream's custom generic parser mistakes a parenthesized multiple-result
    # list for a function type. Read that one grammar production explicitly.
    class Generic(GenericOp):
        @classmethod
        def parse(cls, parser):
            attrs = parser.parse_optional_attr_dict()
            if 'indexing_maps' not in attrs or 'iterator_types' not in attrs:
                parser.raise_error('linalg.generic requires maps and iterators')
            maps = attrs.pop('indexing_maps')
            its = [IteratorTypeAttr(IteratorType(x.data)) if isinstance(x, StringAttr) else x
                   for x in attrs.pop('iterator_types')]
            doc, library = attrs.pop('doc', None), attrs.pop('library_call', None)
            groups = []
            for keyword in ('ins', 'outs'):
                if parser.parse_optional_characters(keyword):
                    pos = parser.pos
                    parser.parse_punctuation('(')
                    unresolved = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_unresolved_operand)
                    parser.parse_punctuation(':')
                    types = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
                    parser.parse_punctuation(')')
                    groups.append(parser.resolve_operands(unresolved, types, pos))
                else:
                    groups.append([])
            extra = {}
            if parser.parse_optional_keyword('attrs'):
                parser.parse_punctuation('=')
                extra = parser.parse_optional_attr_dict()
            body = parser.parse_region()
            results = []
            if parser.parse_optional_punctuation('->'):
                if parser.parse_optional_punctuation('('):
                    if not parser.parse_optional_punctuation(')'):
                        results = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
                        parser.parse_punctuation(')')
                else:
                    results = parser.parse_comma_separated_list(parser.Delimiter.NONE, parser.parse_type)
            op = cls(groups[0], groups[1], body, maps, its, results, doc, library)
            op.attributes.update(attrs)
            op.attributes.update(extra)
            return op

    ctx = Context(allow_unregistered=True)
    lin = Dialect(Linalg.name, [Generic if op is GenericOp else op for op in Linalg.operations], list(Linalg.attributes))
    for dialect in (Builtin, Func, Arith, Tensor, lin, Math, Scf, Cf, Affine, MemRef):
        ctx.load_dialect(dialect)
    return ctx


def read_module(text):
    """Return the real xDSL module and its unique defined entry function."""
    from xdsl.parser import Parser
    module = Parser(_context(), text).parse_module()
    definitions = [op for op in module.walk() if op.name == 'func.func' and op.body.blocks]
    forward = [op for op in definitions if _attrs(op)['sym_name'].data == 'forward']
    if len(forward) == 1:
        fn = forward[0]
    elif len(definitions) == 1:
        fn = definitions[0]
    else:
        raise ValueError('expected one defined entry function')
    if len(fn.body.blocks) != 1:
        raise ValueError('entry control flow requires mixed-program lowering')
    return module, fn


def inventory(text):
    """JSON-compatible full source inventory with stable SSA edges and scalar DAGs."""
    _, fn = read_module(text)
    block = fn.body.blocks[0]
    operations = [op for op in block.ops if op.name != 'func.return']
    ids = {value: ('op', i, j) for i, op in enumerate(operations) for j, value in enumerate(op.results)}
    ids.update({value: ('arg', i) for i, value in enumerate(block.args)})

    def operand(value):
        rec = {'source': list(ids.get(value, ('unknown',))), 'shape': _shape(value.type), 'dtype': _dtype(value.type)}
        const = _constant(value)
        if const is not None:
            rec['const_value'] = const
        return rec

    records = []
    for i, op in enumerate(operations):
        attrs = _attrs(op)
        ins = list(getattr(op, 'inputs', []))
        outs = list(getattr(op, 'outputs', []))
        if _name(op) == 'linalg.reduce':
            ins, outs = list(op.operands[:1]), list(op.operands[1:])
        if not ins and not outs:
            ins = list(op.operands)
        body = None
        if op.regions and _name(op).startswith('linalg.'):
            rb = op.regions[0].blocks[0]
            bodyids = {v: f'arg:{k}' for k, v in enumerate(rb.args)}
            bodyids.update({v: f'op:{j}:{k}' for j, inner in enumerate(rb.ops) for k, v in enumerate(inner.results)})
            captures = []
            capids = {}
            def edge(v):
                if v in bodyids:
                    return bodyids[v]
                if v not in capids:
                    capids[v] = len(captures)
                    captures.append(operand(v))
                return f'capture:{capids[v]}'
            bodyops, yields = [], []
            for inner in rb.ops:
                refs = [edge(v) for v in inner.operands]
                if _name(inner) == 'linalg.yield':
                    yields = refs
                else:
                    bodyops.append({'op': _name(inner), 'operands': refs,
                                    'results': [str(v.type) for v in inner.results],
                                    'attributes': {k: str(v) for k, v in _attrs(inner).items() if not k.startswith('prov.')}})
            body = {'arguments': [str(v.type) for v in rb.args], 'captures': captures, 'operations': bodyops, 'yields': yields}
        records.append({'id': i, 'source_op_index': i, 'kind': _name(op),
                        'prov': {k[5:]: getattr(v, 'data', str(v)) for k, v in attrs.items() if k.startswith('prov.')},
                        'ins': [operand(v) for v in ins], 'outs': [operand(v) for v in outs],
                        'results': [{'shape': _shape(v.type), 'dtype': _dtype(v.type)} for v in op.results],
                        'attributes': {k: str(v) for k, v in attrs.items() if not k.startswith('prov.')},
                        'scalar_body': body})
    terminator = block.ops.last
    if terminator is None or terminator.name != 'func.return':
        raise ValueError('entry has no return')
    return {'entry': _attrs(fn)['sym_name'].data,
            'args': [{'index': i, 'shape': _shape(v.type), 'dtype': _dtype(v.type)} for i, v in enumerate(block.args)],
            'ops': records, 'returns': [operand(v) for v in terminator.operands]}


def _canonical_multiply_accumulate(op):
    """Check the scalar SSA DAG, including signed widening and yield identity."""
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    if len(block.args) != 3:
        return False
    expr = {block.args[0]: ('a',), block.args[1]: ('b',), block.args[2]: ('c',)}
    for inner in block.ops:
        name = _name(inner)
        args = [expr.get(v) for v in inner.operands]
        if name == 'linalg.yield':
            return args == [('add', ('c',), ('mul', ('a',), ('b',)))] or args == [('add', ('mul', ('a',), ('b',)), ('c',))]
        if len(inner.results) != 1 or any(a is None for a in args):
            return False
        if name == 'arith.extsi' and len(args) == 1 and str(inner.results[0].type) == 'i32':
            value = args[0]
        elif name == 'arith.muli' and len(args) == 2:
            value = ('mul', *args)
        elif name == 'arith.addi' and len(args) == 2:
            value = ('add', *args)
        else:
            return False
        expr[inner.results[0]] = value
    return False


def parse(text):
    """Lower exact integer matmul/batched-matmul functions to the public buffer ABI.

    Initializers are proved zero and generic contractions must have the exact
    canonical index maps and multiply-accumulate DAG. Other source computation
    requires a mixed-program lowering and is diagnosed explicitly.
    """
    from xdsl.ir.affine import AffineDimExpr
    _, fn = read_module(text)
    block = fn.body.blocks[0]
    names = {v: f'arg{i}' for i, v in enumerate(block.args)}
    tensors = {names[v]: {'shape': _shape(v.type), 'dtype': _dtype(v.type), 'role': 'input'} for v in block.args}
    commands = []
    returns = list(block.ops.last.operands)
    return_names = {v: f'Y{i}' for i, v in enumerate(returns)}
    serial = 0
    for op in block.ops:
        kind = _name(op)
        if kind == 'func.return':
            continue
        # Unobservable scalar constants / destination initializers are consumed
        # structurally by the contraction. Tensor constants may be observable.
        if kind in ('arith.constant', 'tensor.splat', 'tensor.empty', 'linalg.fill'):
            if any(r in returns for r in op.results):
                raise ValueError(f'{kind}: observable initialization needs an emitted tensor kernel')
            continue
        if kind not in ('linalg.generic', 'linalg.matmul', 'linalg.batch_matmul', 'linalg.matmul_transpose_b'):
            raise ValueError(f'{kind}: no integer contraction lowering')
        ins, outs = list(getattr(op, 'inputs', [])), list(getattr(op, 'outputs', []))
        if len(ins) != 2 or len(outs) != 1 or len(op.results) != 1:
            raise ValueError(f'{kind}: expected two operands and one accumulator')
        a, b = ins
        shape_a, shape_b, shape_y = _shape(a.type), _shape(b.type), _shape(op.results[0].type)
        if _dtype(a.type) != 'i8' or _dtype(b.type) != 'i8' or _dtype(op.results[0].type) != 'i32':
            raise ValueError(f'{kind}: requires signed i8 operands and i32 accumulation')
        if _constant(outs[0]) != 0:
            owner = outs[0].owner
            if _name(owner) != 'linalg.fill' or not owner.operands or _constant(owner.operands[0]) != 0:
                raise ValueError(f'{kind}: accumulator is not initialized to zero')
        if a not in names or b not in names:
            raise ValueError(f'{kind}: operand producer has no emitted tensor')
        transpose = kind == 'linalg.matmul_transpose_b'
        batched = len(shape_a) >= 3
        if kind == 'linalg.generic':
            if not _canonical_multiply_accumulate(op):
                raise ValueError('linalg.generic: scalar body is not signed integer multiply-accumulate')
            attrs = _attrs(op)
            maps = attrs.get('indexing_maps')
            its = attrs.get('iterator_types')
            ndim = len(shape_a) + 1
            expected = [tuple(range(ndim - 3)) + (ndim-3, ndim-1),
                        tuple(range(ndim - 3)) + (ndim-1, ndim-2),
                        tuple(range(ndim - 1))]
            actual = []
            for attr in maps:
                m = attr.data
                if m.num_dims != ndim or m.num_symbols or any(not isinstance(x, AffineDimExpr) for x in m.results):
                    raise ValueError('linalg.generic: noncanonical contraction indexing')
                actual.append(tuple(x.position for x in m.results))
            if actual != expected or len(its) != ndim or any('reduction' not in str(it) if j == ndim - 1 else 'parallel' not in str(it) for j, it in enumerate(its)):
                raise ValueError('linalg.generic: unsupported contraction indexing maps')
        if len(shape_a) < 2 or len(shape_b) != len(shape_a) or len(shape_y) != len(shape_a):
            raise ValueError(f'{kind}: incompatible contraction ranks')
        expected_y = shape_a[:-1] + [shape_b[-2] if transpose else shape_b[-1]]
        if shape_y != expected_y or shape_a[-1] != shape_b[-1 if transpose else -2] or shape_a[:-2] != shape_b[:-2]:
            raise ValueError(f'{kind}: incompatible contraction shapes')
        dst = return_names.get(op.results[0], f'tmp{serial}')
        tensors[dst] = {'shape': shape_y, 'dtype': 'i32', 'role': 'output' if op.results[0] in returns else 'intermediate'}
        names[op.results[0]] = dst
        if transpose:
            if batched:
                raise ValueError('batched transpose-B contraction requires explicit packing')
            commands.append({'opcode': 'ATTENTION_QK', 'operands': {'q': names[a], 'k': names[b], 'dst': dst}, 'attributes': {'output_dtype': 'i32', 'epilogue': []}})
        elif batched:
            commands.append({'opcode': 'BATCHED_MATMUL', 'operands': {'a': names[a], 'w': names[b], 'dst': dst}, 'attributes': {}})
        else:
            acc = f'acc{serial}'
            commands.extend([{'opcode': 'MATMUL', 'operands': {'lhs': names[a], 'rhs': names[b], 'dst': acc}, 'attributes': {}},
                             {'opcode': 'COMMIT', 'operands': {'src': acc, 'dst': dst}, 'attributes': {'output_dtype': 'i32', 'epilogue': []}}])
        serial += 1
    if not commands or any(v not in names or names[v] != return_names[v] for v in returns):
        raise ValueError('entry result has no emitted contraction producer')
    return {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': commands,
            'compiler_pointer_abi': {'version': 1, 'arguments': [f'arg{i}' for i in range(len(block.args))] + [return_names[v] for v in returns]}}
