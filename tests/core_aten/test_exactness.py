"""Exhaustive signed-byte and adversarial rational-vs-device qualification."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.exactness import prove_readout
from mlir_oot.core_aten.match import Decline
from mlir_oot.core_aten.stream import simulate
from exact_oracle import quantize


@pytest.mark.parametrize('scale', [F(1), F(-1), F(1, 2), F(3, 4), F(7, 8), F(1, 16)])
def test_all_byte_codes_and_rounding_ties(scale):
    x = np.arange(-128, 128, dtype=np.int8)
    kernel = ops.scale_map('aten.mul.Scalar', x.shape, scale, s_x=1, s_out=1)
    assert np.array_equal(simulate(kernel.stream, {'x': x})['y'], quantize(x.astype(object) * scale))


def test_exhaustive_two_byte_affine():
    a, b = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128), indexing='ij')
    kernel = ops.add_tensor('aten.add.Tensor', a.shape, b.shape, s_a=1, s_b=1, s_out=2)
    expected = quantize((a.astype(object) + b.astype(object)) / F(2))
    got = simulate(kernel.stream, {'a': a.astype(np.int8), 'b': b.astype(np.int8)})['y']
    assert np.array_equal(got, expected)


def test_mean_review_repro_uses_full_width_and_exact_host_rounding():
    x=np.zeros((37,50),np.int8);x[:,:2]=[-62,87]
    kernel=ops.mean_dim(x.shape,[1],False,s_x=.03,s_out=.01)
    expected=quantize(x.astype(object).sum(1)*F(.03)/(50*F(.01)))
    assert np.array_equal(simulate(kernel.stream,{'x':x})['y'],expected)
    assert kernel.stream.readouts
    assert all(i.a('acc_scale')==1 for i in kernel.stream.ins if i.kind=='config_st')


@pytest.mark.parametrize('length', [1, 2, 4, 16, 32, 64, 128])
def test_random_and_tie_means(length):
    rng = np.random.default_rng(749 + length)
    x = rng.integers(-128, 128, size=(37, length), dtype=np.int8)
    if length >= 2:
        x[0] = 0
        x[0, 0] = length // 2  # an exact halfway case
    kernel = ops.mean_dim(x.shape, [1], False, s_x=.03125, s_out=.03125)
    expected = quantize(np.sum(x.astype(object), axis=1) / F(length))
    assert np.array_equal(simulate(kernel.stream, {'x': x})['y'], expected)


def test_random_readout_proof_never_accepts_a_counterexample():
    rng = np.random.default_rng(9831)
    scales = [F(1, 10), F(1, 3), F(3, 50), F(1, 2) + F(1, 1 << 40)]
    scales += [F(float(rng.uniform(-3, 3))) for _ in range(50)]
    for scale in scales:
        lo, hi = -4096, 4096
        try:
            prove_readout(scale, lo, hi)
        except Decline:
            continue
        x = np.arange(lo, hi + 1, dtype=np.int64)
        hw = np.clip(np.rint(x.astype(np.float32) * np.float32(float(scale))), -128, 127).astype(np.int8)
        assert np.array_equal(hw, quantize(x.astype(object) * scale))



def test_eager_float_divergence_is_informational(capsys):
    # Binary32 source scales .03 and .01 have ratio exactly 3.
    import torch
    x = torch.zeros(37, 50, dtype=torch.int8)
    x[:, :2] = torch.tensor([-62, 87], dtype=torch.int8)
    si, so = float(np.float32(.03)), float(np.float32(.01))
    kernel = ops.mean_dim(x.shape, [1], False, s_x=si, s_out=so)
    exact = quantize(np.sum(x.numpy().astype(object), axis=1) * F(si) / (50 * F(so)))
    assert np.array_equal(simulate(kernel.stream, {'x': x.numpy()})['y'], exact)
    eager = torch.round((x.float() * si).mean(1) / so).clamp(-128, 127).to(torch.int8).numpy()
    print('eager-float divergence (informational):', np.count_nonzero(eager != exact))
    assert 'informational' in capsys.readouterr().out


@pytest.mark.parametrize('scale',[F(1,2),F(1,1024),F(3,8)])
def test_full_i32_shortcut_is_refused_even_when_saturation_hides_rounding(scale):
    with pytest.raises(Decline,match='conversion'):
        prove_readout(scale,-(1<<31),(1<<31)-1)
