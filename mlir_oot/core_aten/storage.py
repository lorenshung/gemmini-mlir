"""Byte storage of dense target tensor ABI elements (little-endian RISC-V)."""

SIZES = {'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
         'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8, 'complex<f32>': 8}


def itemsize(dtype):
    from .exactness import require
    require(dtype in SIZES, 'unsupported dense element storage: ' + dtype)
    return SIZES[dtype]


def numpy_storage(dtype):
    import numpy as np
    return np.dtype({'i1': '?', 'i8': 'i1', 'i16': '<i2', 'i32': '<i4', 'i64': '<i8',
                     'f16': '<f2', 'bf16': '<u2', 'f32': '<f4', 'f64': '<f8',
                     'complex<f32>': '<c8'}[dtype])


def literal_bytes(value, dtype):
    """Encode a source literal once at its declared IEEE/integer storage width.

    Integer/rational RNE avoids binary64 -> binary32 -> bf16 double rounding.
    Runtime payloads, including NaNs and signed zeros, never enter this routine.
    """
    from fractions import Fraction
    import math
    from .exactness import require
    width = itemsize(dtype)
    if dtype.startswith('i'):
        if dtype == 'i1':
            return bytes([int(value != 0)])
        require(value == int(value), 'nonintegral integer movement literal')
        return (int(value) % (1 << (width * 8))).to_bytes(width, 'little')
    require(dtype in ('f16', 'bf16', 'f32', 'f64'), 'unsupported movement literal: ' + dtype)
    require(not math.isnan(value), 'NaN movement literal has no preserved payload proof')
    exponent_bits, fraction_bits = {'f16': (5, 10), 'bf16': (8, 7),
                                   'f32': (8, 23), 'f64': (11, 52)}[dtype]
    sign = int(math.copysign(1, value) < 0)
    if math.isinf(value):
        bits = (sign << (exponent_bits + fraction_bits)) | (((1 << exponent_bits)-1) << fraction_bits)
        return bits.to_bytes(width, 'little')
    q = abs(Fraction(value))
    if not q:
        bits = sign << (exponent_bits + fraction_bits)
    else:
        bias = (1 << (exponent_bits - 1)) - 1
        power = lambda e: Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)
        exponent = q.numerator.bit_length() - q.denominator.bit_length()
        if q < power(exponent):
            exponent -= 1
        exponent = max(1 - bias, exponent)
        mantissa = round(q / power(exponent - fraction_bits))
        if mantissa == 1 << (fraction_bits + 1):
            mantissa >>= 1
            exponent += 1
        if exponent > bias:
            field, fraction = (1 << exponent_bits) - 1, 0
        elif mantissa < 1 << fraction_bits:
            field, fraction = 0, mantissa
        else:
            field, fraction = exponent + bias, mantissa - (1 << fraction_bits)
        bits = (sign << (exponent_bits + fraction_bits)) | (field << fraction_bits) | fraction
    return bits.to_bytes(width, 'little')


def module_text(module):
    """Emit target ABI literal/global storage without boxed scalar expansion.

    xDSL's default dense printer unpacks all values even for hex output.
    Hex preserves every payload bit and uses memory proportional to the bytes.
    """
    from io import StringIO
    from xdsl.printer import Printer
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    class StoragePrinter(Printer):
        def print_attribute(self, attribute):
            if isinstance(attribute, DenseIntOrFPElementsAttr):
                self.print_string('dense<"0x'+attribute.data.data.hex().upper()+'"> : ')
                super().print_attribute(attribute.type)
            else:
                super().print_attribute(attribute)
    output = StringIO()
    StoragePrinter(stream=output).print_op(module)
    return output.getvalue()
