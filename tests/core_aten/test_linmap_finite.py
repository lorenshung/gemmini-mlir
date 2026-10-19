"""Finite-value proofs must cover every signed and unsigned storage byte."""
from fractions import Fraction as Q

import numpy as np
import pytest

from mlir_oot.core_aten.linmap import Sym, _finite_byte_form


@pytest.mark.parametrize('dtype,precision,exponent', [
    ('f16', 11, 15), ('bf16', 8, 127), ('f32', 24, 127), ('f64', 53, 1023),
])
@pytest.mark.parametrize('sign', [-1, 1])
def test_unsigned_leaf_near_float_max_over_128(dtype, precision, exponent, sign):
    maximum = (Q(2) - Q(1, 1 << (precision - 1))) * (1 << exponent)
    index = np.array([0])

    def form(coefficient, constant=0):
        return Sym([(0, index, np.array([coefficient], object))], constant)

    # The former 128 bound admitted this, although uint8(255) overflows.
    coefficient = sign * maximum / 128
    assert abs(coefficient * 128) == maximum
    assert abs(coefficient * 255) > maximum
    assert not _finite_byte_form(form(coefficient), dtype)

    # The actual byte-domain endpoint is legal; even a tiny excess is not.
    coefficient = sign * maximum / 255
    assert _finite_byte_form(form(coefficient), dtype)
    assert not _finite_byte_form(form(coefficient, sign * Q(1, 2)), dtype)


def test_byte_leaf_bounds_include_every_term_and_lane():
    maximum = Q(int(np.finfo(np.float32).max))
    index = np.array([0, 1])
    coefficient = maximum / 510
    form = Sym([(0, index, np.array([coefficient, -coefficient], object)),
                (1, index, np.array([-coefficient, coefficient], object))])
    assert _finite_byte_form(form, 'f32')
    form.const = np.array([0, Q(1, 2)], object)
    assert not _finite_byte_form(form, 'f32')
