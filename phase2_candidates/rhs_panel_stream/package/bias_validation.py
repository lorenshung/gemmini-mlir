"""Validation of explicit accumulator-domain bias epilogue bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_bias_epilogue(
    attrs: Mapping[str, Any],
    operands: Mapping[str, Any],
    tensors: Mapping[str, Any],
    channels: int,
) -> str | None:
    """Return the validated bias tensor name for an active bias stage.

    The frozen i8 contraction datapath adds a length-N i32 bias vector to
    every accumulator row.  Either explicit bias binding is accepted; when
    both are supplied they must agree.  Inactive stages return ``None``
    without inspecting bindings or changing the supplied mappings.
    """
    stages = attrs.get('epilogue', ())
    if 'bias_add' not in stages and 'bias' not in stages:
        return None

    explicit = []
    for label, container in (('operand', operands), ('attribute', attrs)):
        if 'bias' not in container:
            continue
        name = container['bias']
        if not isinstance(name, str) or not name:
            raise ValueError(f'bias epilogue requires a nonempty bias tensor name in its {label}')
        explicit.append(name)
    if not explicit:
        raise ValueError('bias epilogue requires an explicit bias tensor name')
    if len(explicit) == 2 and explicit[0] != explicit[1]:
        raise ValueError('bias epilogue has conflicting operand and attribute tensor names')
    name = explicit[0]
    if name not in tensors:
        raise ValueError(f'bias epilogue names undeclared tensor {name!r}')
    tensor = tensors[name]
    if not isinstance(tensor, Mapping):
        raise ValueError(f'bias tensor {name!r} requires a tensor declaration')
    if tensor.get('role') != 'bias':
        raise ValueError(f'bias tensor {name!r} must have role bias')
    if tensor.get('dtype') != 'i32':
        raise ValueError(f'bias tensor {name!r} must have accumulator dtype i32')
    shape = tensor.get('shape')
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels <= 0
        or not isinstance(shape, (list, tuple))
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] != channels
    ):
        raise ValueError(f'bias tensor {name!r} must have exact one-dimensional shape [{channels}]')
    return name
