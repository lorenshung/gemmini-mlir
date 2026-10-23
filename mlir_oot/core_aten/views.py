"""Legality of dense, element-order-preserving tensor shape changes."""
import math

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType


def reindex_shape(op):
    """Return the static destination shape, or refuse an unproved shape operand.

    Tensor reshape/collapse/expand preserve flattened element order by their IR
    semantics. Static types alone do not prove a dynamic reshape shape operand.
    """
    source, target = op.operands[0].type, op.results[0].type
    if not isinstance(source, TensorType) or not isinstance(target, TensorType):
        raise ValueError('reindex requires ranked tensors')
    a, b = tuple(source.get_shape()), tuple(target.get_shape())
    if (source.element_type != target.element_type or any(d <= 0 for d in (*a, *b))
            or math.prod(a) != math.prod(b)):
        raise ValueError('reindex must preserve element type and static nonempty extent')
    if op.name == 'tensor.expand_shape' and len(op.operands) != 1:
        raise ValueError('expand requires static output dimensions')
    if op.name == 'tensor.reshape':
        owner = op.operands[1].owner
        attr = owner.properties.get('value') if hasattr(owner, 'properties') else None
        if attr is None and hasattr(owner, 'attributes'):
            attr = owner.attributes.get('value')
        if (getattr(owner, 'name', None) != 'arith.constant'
                or not isinstance(attr, DenseIntOrFPElementsAttr)
                or tuple(attr.get_values()) != b):
            raise ValueError('reshape requires a literal shape matching the static result')
    return b
