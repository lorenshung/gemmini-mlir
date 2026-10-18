"""Agent-authored xDSL frontend for the frozen merlin_iface v0.1 grammar.

The custom parsers below build an SSA xDSL module. Lowering walks that module,
so operand resolution and type checking are performed before command emission.
"""
from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr, Builtin, FloatAttr, FunctionType, IntegerAttr, StringAttr, TensorType,
)
from xdsl.ir import Dialect, Operation
from xdsl.irdl import traits_def
from xdsl.parser import Parser
from epilogue_validation import validate_acc_scale
from bias_validation import validate_bias_epilogue

_OPERANDS = {
    'tensor': (), 'resident_pack': ('src',), 'matmul': ('lhs', 'rhs'),
    'commit': ('src',), 'evict': ('handle',), 'conv2d': ('ifm', 'weight'),
    'movement': ('src',), 'attention_qk': ('q', 'k'),
    'attention_pv': ('p', 'v'), 'bias_add': ('src', 'bias'),
    'matmul_batched': ('a', 'w'), 'residual_add': ('lhs', 'rhs'),
    'rmsnorm': ('src', 'gamma'), 'rope': ('src',), 'softmax': ('src',),
}
_OPCODES = {
    'resident_pack': 'RES_PACK', 'matmul': 'MATMUL_RESIDENT',
    'commit': 'COMMIT', 'evict': 'EVICT', 'conv2d': 'CONV2D',
    'movement': 'MOVEMENT', 'attention_qk': 'ATTENTION_QK',
    'attention_pv': 'ATTENTION_PV', 'bias_add': 'BIAS_ADD',
    'matmul_batched': 'BATCHED_MATMUL', 'residual_add': 'RESIDUAL_ADD',
    'rmsnorm': 'RMSNORM', 'rope': 'ROPE', 'softmax': 'SOFTMAX',
}


def _operation(mnemonic):
    """Register custom assembly syntax with xDSL, rather than matching SSA text."""
    class InterfaceOperation(Operation):
        name = 'merlin_iface.' + mnemonic
        traits = traits_def()

        @classmethod
        def parse(cls, parser):
            operands = []
            for index, _ in enumerate(_OPERANDS[mnemonic]):
                if index:
                    parser.parse_punctuation(',')
                operands.append(parser.parse_operand())
            attributes = parser.parse_optional_attr_dict()
            parser.parse_punctuation(':')
            if mnemonic == 'tensor':
                result_types = [parser.parse_type()]
            else:
                signature = parser.parse_function_type()
                if len(signature.inputs) != len(operands):
                    parser.raise_error('operand type count does not match operand count')
                for operand, expected in zip(operands, signature.inputs):
                    if operand.type != expected:
                        parser.raise_error('operand type does not match its SSA definition')
                result_types = list(signature.outputs)
            expected_count = 0 if mnemonic == 'evict' else 1
            if len(result_types) != expected_count:
                parser.raise_error(f'{cls.name} requires {expected_count} result(s)')
            return cls.create(operands=operands, result_types=result_types,
                              attributes=attributes)
    InterfaceOperation.__name__ = ''.join(s.title() for s in mnemonic.split('_')) + 'Op'
    return InterfaceOperation


_INTERFACE_DIALECT = Dialect('merlin_iface', [_operation(k) for k in _OPERANDS])


def parse_module(text):
    """Parse an interface module with xDSL and verify SSA structural invariants."""
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(_INTERFACE_DIALECT)
    module = Parser(context, text).parse_module()
    module.verify()
    version = module.attributes.get('merlin_iface.version')
    if not isinstance(version, StringAttr) or version.data != '0.1':
        raise ValueError('unsupported merlin_iface.version (expected 0.1)')
    target = module.attributes.get('merlin_iface.target')
    if not isinstance(target, StringAttr) or target.data != 'gemmini':
        raise ValueError('unsupported merlin_iface.target (expected gemmini)')
    abi_version = module.attributes.get('merlin_iface.abi_version')
    if not isinstance(abi_version, StringAttr) or abi_version.data != '0.1':
        raise ValueError('unsupported merlin_iface.abi_version (expected 0.1)')
    return module


def _value(attribute):
    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, IntegerAttr):
        return attribute.value.data
    if isinstance(attribute, FloatAttr):
        return attribute.value.data
    if isinstance(attribute, ArrayAttr):
        return [_value(a) for a in attribute]
    raise ValueError(f'unsupported interface attribute {attribute}')


def _tensor(type_, role):
    if not isinstance(type_, TensorType):
        raise ValueError(f'expected ranked tensor, received {type_}')
    shape = list(type_.get_shape())
    if not shape or any(d <= 0 for d in shape):
        raise ValueError('interface tensors require a nonempty positive static shape')
    return {'shape': shape, 'dtype': str(type_.element_type), 'role': role}


def _validate_biases(tensors, commands):
    """Recover contraction channel extents for declared bias vectors."""
    residents, accumulators = {}, {}
    for command in commands:
        opcode = command['opcode']
        operands, attrs = command['operands'], command['attributes']
        if opcode == 'RES_PACK':
            residents[operands['dst']] = operands['src']
        elif opcode in {'MATMUL', 'MATMUL_RESIDENT'}:
            accumulators[operands['dst']] = residents.get(operands['rhs'], operands['rhs'])
        active = 'bias_add' in attrs.get('epilogue', []) or 'bias' in attrs.get('epilogue', [])
        if not active and opcode != 'BIAS_ADD':
            continue
        if opcode == 'COMMIT':
            source = accumulators.get(operands['src'])
            if source not in tensors:
                raise ValueError('bias commit requires a computed contraction accumulator')
            channels = tensors[source]['shape'][-1]
        elif opcode == 'CONV2D':
            kernel = attrs.get('kernel')
            if not isinstance(kernel, (list, tuple)) or len(kernel) != 4:
                raise ValueError('bias convolution requires four kernel dimensions')
            channels = kernel[3]
        elif opcode in {'ATTENTION_QK', 'ATTENTION_PV'}:
            source = operands['k'] if opcode == 'ATTENTION_QK' else operands['v']
            shape = tensors[source]['shape']
            if len(shape) < 2:
                raise ValueError('bias attention requires a rank-two contraction operand')
            channels = shape[-2] if opcode == 'ATTENTION_QK' else shape[-1]
        elif opcode == 'BIAS_ADD':
            channels = tensors[operands['src']]['shape'][-1]
            attrs = {**attrs, 'epilogue': ['bias_add']}
        else:
            continue
        validate_bias_epilogue(attrs, operands, tensors, channels)


def parse(text):
    """Return a command-buffer v0.1 dictionary, preserving logical tensor names."""
    module = parse_module(text)
    tensors = {}
    commands = []
    names = {}
    for number, operation in enumerate(module.body.block.ops):
        mnemonic = operation.name.removeprefix('merlin_iface.')
        if mnemonic not in _OPERANDS:
            # Unregistered xDSL operations carry their spelling in op_name__.
            original = operation.attributes.get('op_name__')
            raise ValueError(f'unsupported interface operation {original or operation.name}')
        attrs = {k: _value(v) for k, v in operation.attributes.items()}
        if mnemonic in {'commit', 'conv2d', 'attention_qk', 'attention_pv'}:
            scale_attr = operation.attributes.get('acc_scale')
            if ('acc_scale' in attrs.get('epilogue', []) and
                    isinstance(scale_attr, IntegerAttr) and str(scale_attr.type) == 'i1'):
                raise ValueError('acc_scale must be a numeric scalar, not a boolean')
            validate_acc_scale(attrs)
        if len(operation.operands) != len(_OPERANDS[mnemonic]):
            raise ValueError(f'{operation.name} has wrong operand count')
        result = operation.results[0] if operation.results else None
        if mnemonic == 'tensor':
            name = attrs.get('name')
            role = attrs.get('role')
            if not isinstance(name, str) or not name:
                raise ValueError('tensor requires a nonempty logical name')
            if role not in {'input', 'weight', 'bias', 'scale'}:
                raise ValueError(f'invalid leaf tensor role {role}')
            if name in tensors:
                raise ValueError(f'duplicate tensor name {name}')
            tensors[name] = _tensor(result.type, role)
            names[result] = name
            continue
        try:
            operands = {k: names[v] for k, v in zip(_OPERANDS[mnemonic], operation.operands)}
        except KeyError as error:
            raise ValueError(f'{operation.name} uses an unresolved logical value') from error
        if result is not None:
            if mnemonic in {'resident_pack', 'matmul'}:
                name = result.name_hint or f'compiler_value_{number}'
                if name in tensors or name in names.values():
                    name = f'compiler_value_{number}'
            else:
                name = attrs.pop('name', None)
                if not isinstance(name, str) or not name:
                    raise ValueError(f'{operation.name} requires a nonempty logical name')
                if name in tensors:
                    raise ValueError(f'duplicate tensor name {name}')
                tensors[name] = _tensor(result.type, 'output')
                if 'output_dtype' in attrs and attrs['output_dtype'] != tensors[name]['dtype']:
                    raise ValueError(f'{operation.name} output_dtype does not match its result type')
            names[result] = name
            operands['dst'] = name
        commands.append({'opcode': _OPCODES[mnemonic], 'operands': operands, 'attributes': attrs})
    if not commands:
        raise ValueError('interface module declares no executable operations')
    _validate_biases(tensors, commands)
    return {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': commands}
