"""Device integer affine work and the exact rational host boundary."""
from fractions import Fraction as Q

import numpy as np
import pytest

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.lint import lint
from mlir_oot.core_aten.stream import simulate, ACC, Arg, Stream


def quant(values):
    return np.array([min(127, max(-128, round(Q(v)))) for v in values.flat], np.int8).reshape(values.shape)


@pytest.mark.parametrize('weight', [1, 2, -1, -3])
@pytest.mark.parametrize('broadcast', [False, True])
def test_integer_ratio_tensor_terms_compute_in_one_accumulator(weight, broadcast):
    shape = (19, 23)
    a = np.resize(np.arange(-128, 128, dtype=np.int16).astype(np.int8), shape)
    b = np.resize(np.array([127, -128, -1, 0, 1], np.int8), (23,) if broadcast else shape)
    kernel = ops.add_tensor('affine', shape, b.shape, alpha=weight, s_a=Q(1,32), s_b=Q(1,16), s_out=Q(3,64))
    expected = quant((a.astype(object)+2*weight*b.astype(object))*Q(2,3))
    assert np.array_equal(simulate(kernel.stream, dict(a=a, b=b))['y'], expected)
    lint(kernel.stream)
    assert len(kernel.stream.readouts[0].terms) == 1
    assert any(i.kind == 'compute' for i in kernel.stream.ins) or kernel.meta['arithmetic'] == 'integer_accumulating_mvin'
    assert all(i.a('scale') == 1 for i in kernel.stream.ins if i.kind == 'config_ld')


@pytest.mark.parametrize('constant,device', [(Q(5,32), True), (Q(-7,32), True), (Q(1,10), False)])
def test_scalar_constant_is_seeded_only_when_integral_in_code_units(constant, device):
    x = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    kernel = ops.add_scalar('affine', x.shape, constant, s_x=Q(1,32), s_out=Q(3,64))
    assert np.array_equal(simulate(kernel.stream, dict(x=x))['y'], quant(x.astype(object)*Q(2,3)+constant/Q(3,64)))
    assert kernel.meta['device_constant'] is device
    readout = kernel.stream.readouts[0]
    assert bool(np.any(np.asarray(readout.constant, object))) == (not device)
    lint(kernel.stream)


@pytest.mark.parametrize('factor', [-1, -3, 4, Q(3,2), Q(1,3)])
def test_signed_scalar_numerator_on_device(factor):
    x = np.resize(np.arange(-128, 128, dtype=np.int16).astype(np.int8), (17,19))
    kernel = ops.scale_map('scalar', x.shape, factor, s_x=Q(1,32), s_out=Q(1,16))
    assert np.array_equal(simulate(kernel.stream, dict(x=x))['y'], quant(x.astype(object)*factor/2))
    assert any(i.kind == 'compute' for i in kernel.stream.ins) == (factor != Q(1,3))
    assert kernel.stream.readouts[0].terms[0][1].flat[0] > 0
    lint(kernel.stream)


def test_irregular_scale_ratio_retains_separate_exact_terms():
    sa = Q(float(np.float32(.03933464363217354)))
    sb = Q(float(np.float32(.049954403191804886)))
    a = np.array([-128, -1, 0, 1, 127], np.int8)
    b = a[::-1].copy()
    kernel = ops.add_tensor('affine', a.shape, b.shape, s_a=sa, s_b=sb, s_out=Q(1,16))
    assert kernel.meta['arithmetic'] == 'separate_rational_terms_host_combination'
    assert len(kernel.stream.readouts[0].terms) == 2
    assert np.array_equal(simulate(kernel.stream, dict(a=a, b=b))['y'], quant((a.astype(object)*sa+b.astype(object)*sb)*16))
    lint(kernel.stream)


def test_byte_relu_is_device_arithmetic_with_full_i32_readout():
    x = np.resize(np.arange(-128,128,dtype=np.int16).astype(np.int8), (17,19))
    kernel = ops.relu(x.shape, s_x=Q(1,32), s_out=Q(3,64))
    assert kernel.meta['device_relu'] and not kernel.stream.readouts[0].relu
    assert np.array_equal(simulate(kernel.stream, dict(x=x))['y'], quant(np.maximum(x.astype(object),0)*Q(2,3)))
    assert kernel.meta['execute_activation_proof']['clipping_is_identity']
    lint(kernel.stream)


def test_relu_of_wider_affine_accumulator_keeps_exact_host_clamp():
    x = np.arange(-128,128,dtype=np.int16).astype(np.int8)
    kernel = ops.add_scalar('affine_relu', x.shape, 3, s_x=1, s_out=2, relu=True)
    assert not kernel.meta['device_relu'] and kernel.stream.readouts[0].relu
    assert kernel.meta['device_constant']
    assert np.array_equal(simulate(kernel.stream, dict(x=x))['y'], quant(np.maximum(x.astype(object)+3,0)/Q(2)))


def test_full_i32_readout_bypasses_store_relu_on_stock_datapath():
    # This is why CONFIG_ST ReLU cannot implement the requested raw-i32 clamp.
    s = Stream([Arg('x','i32',(3,),'input'), Arg('out','i32',(3,),'output')])
    s.config_ld(2,12)
    s.mvin(s.at('x'), ACC(0), 3, 1, 2)
    s.config_st(12, 1, 1)
    s.mvout(s.at('out'), ACC(0,full=True), 3, 1)
    x = np.array([-1000,0,1000],np.int32)
    assert np.array_equal(simulate(s,dict(x=x))['out'], x)
