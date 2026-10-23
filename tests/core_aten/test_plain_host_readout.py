"""Native exact rational host readout, channel parameters and placements."""
from fractions import Fraction as F
import numpy as np
import pytest

from conftest import native_readout


@pytest.mark.parametrize("axis,shape", [(-1,(129,7)),(1,(2,7,3,5))])
def test_native_per_channel_bias_zero_point_and_multiterm(tmp_path,axis,shape):
    from mlir_oot.core_aten.stream import Stream, Arg
    from mlir_oot.core_aten.requant import Readout
    rng=np.random.default_rng(2231)
    x=rng.integers(-(1 << 31),1 << 31,shape,dtype=np.int32)
    z=np.resize(np.arange(-256,257,dtype=np.int32),shape).copy()
    x.reshape(-1)[::3]=0
    factors=np.array([F(1),F(1,2),F(-3,8),F(17,31),F(1,1 << 149),F(1 << 127),F(1,1 << 80)],object)
    bias=np.array([F(0),F(1,2),F(-1,3),F(11,7),F(1,2),F(-7,3),F(0)],object)
    zps=np.array([0,1,7,123,128,254,255])
    s=Stream([Arg('x','i32',shape,'input'),Arg('z','i32',shape,'input'),Arg('y','i8',shape,'output')])
    s.readouts.append(Readout('y',[('x',factors),('z',np.array([F(0),F(0),F(2,7),F(3,5),F(0),F(0),F(1,2)],object))],bias,
                              zero_point=zps,quant_min=0,quant_max=255,axis=axis))
    fn,meta=native_readout(s,'channel_readout',tmp_path)
    assert meta['host_readout_contract'] == 'exact_rational_RNE_v1'
    assert 'host_requantization_proofs' not in meta
    assert 'readout_pipeline' not in meta
    out=np.zeros(shape,np.uint8);fn(x.ctypes.data,z.ctypes.data,out.ctypes.data)
    expected=np.array([min(255,max(0,round(int(x[idx])*factors[idx[axis]]+int(z[idx])*s.readouts[0].terms[1][1][idx[axis]]+bias[idx[axis]])+int(zps[idx[axis]])))
                       for idx in np.ndindex(shape)],np.uint8).reshape(shape)
    assert np.array_equal(out,expected)

@pytest.mark.parametrize('placement', [
    (12, 3, 2, 7),
    dict(shape=(2, 2, 3), strides=(15, 5, 1), offset=1),
])
@pytest.mark.parametrize('factor', [F(3, 8), F(-17, 31)])
@pytest.mark.parametrize('relu', [False, True])
def test_native_structural_placement_preserves_exact_readout_and_unwritten_bytes(tmp_path, placement, factor, relu):
    from mlir_oot.core_aten.stream import Stream, Arg, simulate
    from mlir_oot.core_aten.requant import Readout
    shape = (2, 3, 5)
    raw = np.arange(-15, 15, dtype=np.int32).reshape(shape)
    result = np.full(shape, -99, np.int8)
    stream = Stream([Arg('x', 'i32', shape, 'input'), Arg('y', 'i8', shape, 'output')])
    stream.readouts.append(Readout('y', [('x', factor)], constant=F(1, 2),
                                  zero_point=-7, placement=placement, relu=relu))
    fn, _ = native_readout(stream, 'placed_readout', tmp_path)
    fn(raw.ctypes.data, result.ctypes.data)
    expected = np.full(shape, -99, np.int8)
    if isinstance(placement, dict):
        positions = [placement['offset']+sum(i*s for i,s in zip(index, placement['strides']))
                     for index in np.ndindex(placement['shape'])]
    else:
        count, block, offset, stride = placement
        positions = [i//block*stride+i%block+offset for i in range(count)]
    for p in positions:
        value = int(raw.flat[p])*factor+F(1, 2)
        expected.flat[p] = min(127, max(-128, round(max(0, value) if relu else value)-7))
    assert np.array_equal(result, expected)
    simulated = simulate(stream, dict(x=raw))['y']
    assert np.array_equal(simulated.flat[positions], expected.flat[positions])
