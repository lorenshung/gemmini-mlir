"""Validation for the frozen accumulator-scale epilogue attribute."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import Any


def validate_acc_scale(attrs: Mapping[str, Any]) -> Any:
    """Return the existing scale, validating it when its stage is requested.

    An active ``acc_scale`` stage requires an explicit integer or float scalar,
    excluding booleans, whose conversion to IEEE binary32 remains finite.  The
    returned float retains the input's precision: the existing target emission
    performs binary32 rounding, so validation does not change valid constants.

    When the stage is absent, retain the emitter's prior lookup behavior,
    including its identity default and any explicitly supplied attribute.  This
    function never mutates ``attrs``.
    """
    if 'acc_scale' not in attrs.get('epilogue', ()):
        return attrs.get('acc_scale', 1.0)
    if 'acc_scale' not in attrs:
        raise ValueError('acc_scale epilogue requires an explicit acc_scale multiplier')
    value = attrs['acc_scale']
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('acc_scale multiplier must be a numeric scalar, not a boolean')
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError('acc_scale multiplier must be finite and representable as f32') from error
    if not math.isfinite(normalized):
        raise ValueError('acc_scale multiplier must be finite and representable as f32')
    try:
        binary32 = struct.unpack('!f', struct.pack('!f', normalized))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError('acc_scale multiplier must be finite and representable as f32') from error
    if not math.isfinite(binary32):
        raise ValueError('acc_scale multiplier must be finite and representable as f32')
    return normalized
