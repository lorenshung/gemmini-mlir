"""Static geometry checks shared by native pooling epilogues."""
import math


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must contain integers')
    return value


def _vector(attrs, name, length, default=None):
    value = attrs.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must have {length} integer entries')
    return tuple(_integer(v, name) for v in value)


def validate_pool(attrs, rows, channels, dst_shape, output_dtype,
                  expected_in_dims=None):
    """Return gm_pool operands after proving its buffers and geometry agree.

    The native reducer consumes already scaled signed bytes. A nonzero pad
    requires an explicit signed-byte value, as required by the interface.
    """
    if output_dtype != 'i8':
        raise ValueError('maxpool requires i8 readout')
    epilogue = list(attrs.get('epilogue', []))
    if epilogue.count('maxpool') != 1 or epilogue[-1] != 'maxpool':
        raise ValueError('maxpool must occur once as the final epilogue stage')
    h, w = _vector(attrs, 'pool_in_dims', 2)
    ph, pw = _vector(attrs, 'pool_size', 2)
    sh, sw = _vector(attrs, 'pool_stride', 2)
    pt, pl, pb, pr = _vector(attrs, 'pool_padding', 4, [0, 0, 0, 0])
    if min(rows, channels, h, w, ph, pw, sh, sw) <= 0:
        raise ValueError('maxpool extents and strides must be positive')
    if min(pt, pl, pb, pr) < 0:
        raise ValueError('maxpool padding must be nonnegative')
    if expected_in_dims is not None and (h, w) != tuple(expected_in_dims):
        raise ValueError('pool_in_dims disagrees with convolution output extent')
    if rows % (h * w):
        raise ValueError('pool_in_dims must divide the contraction row count')
    if any((pt, pl, pb, pr)) and 'pool_pad_value' not in attrs:
        raise ValueError('nonzero pool_padding requires pool_pad_value')
    pad = _integer(attrs.get('pool_pad_value', 0), 'pool_pad_value')
    if not -128 <= pad <= 127:
        raise ValueError('pool_pad_value must fit signed i8')
    oh = (h + pt + pb - ph) // sh + 1
    ow = (w + pl + pr - pw) // sw + 1
    batch = rows // (h * w)
    if min(oh, ow) <= 0:
        raise ValueError('maxpool has empty output geometry')
    shape = tuple(dst_shape)
    if len(shape) < 2 or shape[-1] != channels or math.prod(shape) != batch * oh * ow * channels:
        raise ValueError('maxpool destination shape disagrees with geometry')
    return (batch, h, w, ph, pw, sh, sw, pt, pl, pb, pr, pad)
