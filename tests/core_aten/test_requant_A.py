"""Option A: complete boundary proofs, exact fallbacks and channel bindings."""
from fractions import Fraction as F
import copy
import random
import numpy as np
import pytest
from mlir_oot.core_aten.requant_a import plan, apply, validate, exact


def test_unit_ratio_is_unit_multiply():
    k = plan(F(1))['constants']
    assert (k['M'], k['B'], k['S']) == ('1', '0', 0)


@pytest.mark.parametrize('f,c,zp', [(F(1,2),F(0),0), (F(3,8),F(1,2),-7),
                                  (F(1),F(1,2),123)])
def test_alternating_ties_use_exact_table(f,c,zp):
    p = plan(f,c,zp,0,255) if zp >= 0 else plan(f,c,zp)
    assert p['algorithm'] == 'threshold_table'
    assert len(p['thresholds']) <= 255
    for x in range(-2048,2049):
        assert apply(x,p) == exact(x,f,c,zp,p['quant_min'],p['quant_max'])


def test_random_and_adversarial_full_i32_boundaries():
    rng = random.Random(9517)
    ratios = [F(1, 1 << 149), F(1 << 127), F(1,3), F(17,31), F(0xFFFFFF,1 << 24)]
    ratios += [F(rng.randrange(1,1 << 24), rng.randrange(1,1 << 24)) for _ in range(100)]
    for f in ratios:
        c = F(rng.randrange(-500,500),rng.randrange(1,40))
        zp = rng.randrange(1,255)
        p = plan(f,c,zp,1,255,bool(rng.randrange(2)))
        assert validate(p) == p['proof']
        values = [-(1 << 31),(1 << 31)-1,0]+[rng.randrange(-(1 << 31),1 << 31) for _ in range(50)]
        if p['algorithm'] == 'threshold_table':
            values += [v for t in map(int,p['thresholds']) for v in (t-1,t) if -(1 << 31)<=v<1 << 31]
        for x in values:
            assert apply(x,p) == exact(x,f,c,zp,1,255,p['relu'])


def test_exhaustive_small_domains_every_tie_both_signs():
    for f in [F(1,2),F(3,8),F(5,7),F(1),F(2),F(17,31)]:
        for c in [F(0),F(-1,2),F(7,3)]:
            p=plan(f,c,123,1,255,False,-1024,1024)
            assert all(apply(x,p)==exact(x,f,c,123,1,255) for x in range(-1024,1025))


def test_tampered_certificates_are_refused():
    p=plan(F(17,31),F(2,3),123,1,255)
    for key,value in [('M',str(1 << 63)),('B',str(1 << 63)),('S',63)]:
        bad=copy.deepcopy(p); bad['constants'][key]=value
        with pytest.raises(ValueError): validate(bad)
    bad=copy.deepcopy(p);bad['proof']['checked_points_sha256']='0'*64
    with pytest.raises(ValueError): validate(bad)
    bad=copy.deepcopy(p);bad['rounding']='toward_zero'
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):plan(F(1),zp=F(1,2))


@pytest.mark.parametrize("axis,shape", [(-1,(129,7)),(1,(2,7,3,5))])
def test_native_per_channel_bias_zero_point_and_multiterm(tmp_path,axis,shape):
    import ctypes, os, subprocess
    from pathlib import Path
    from mlir_oot.core_aten.stream import Stream, Arg
    from mlir_oot.core_aten.requant import Readout, apply as reference
    from mlir_oot.core_aten.emit import build_module
    bindir=Path(os.environ.get('GEMMINI_LLVM_BIN','/nonexistent'))
    if not (bindir/'mlir-translate').exists():pytest.skip('LLVM tools unavailable')
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
    meta={};module,_=build_module(s,'channel_readout',meta=meta)
    assert len(meta['host_requantization_proofs'])==7
    assert max(p['normal_form_width'] for p in meta['host_requantization_proofs']) > 64
    assert {p['algorithm'] for p in meta['host_requantization_proofs']}=={'threshold_table','clamp_mul_add_shift'}
    src=tmp_path/'channels.mlir';src.write_text(str(module)+'\n')
    ir,obj,shared=(src.with_suffix(e) for e in ('.ll','.o','.so'))
    subprocess.run([str(bindir/'mlir-translate'),'--mlir-to-llvmir',str(src),'-o',str(ir)],check=True)
    subprocess.run([str(bindir/'llc'),'-mtriple=x86_64-unknown-linux-gnu','-relocation-model=pic','-O2','-filetype=obj',str(ir),'-o',str(obj)],check=True)
    subprocess.run(['cc','-shared',str(obj),'-o',str(shared)],check=True)
    fn=ctypes.CDLL(str(shared)).channel_readout;fn.argtypes=[ctypes.c_void_p]*3
    out=np.zeros(shape,np.uint8);fn(x.ctypes.data,z.ctypes.data,out.ctypes.data)
    expected=np.array([min(255,max(0,round(int(x[idx])*factors[idx[axis]]+int(z[idx])*s.readouts[0].terms[1][1][idx[axis]]+bias[idx[axis]])+int(zps[idx[axis]])))
                       for idx in np.ndindex(shape)],np.uint8).reshape(shape)
    assert np.array_equal(out,expected)


def test_double_buffer_overlap_tails_and_serial_refusal():
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.stream import simulate
    from mlir_oot.core_aten.readout_pipeline import prepare
    from mlir_oot.core_aten.lint import lint
    k=ops.matmul('aten.mm.default',133,19,130,s_a=F(1,7),s_out=F(1))
    rng=np.random.default_rng(782)
    inputs={'a':rng.integers(-128,128,(133,19),dtype=np.int8),'b':rng.integers(-128,128,(19,130),dtype=np.int8)}
    expected=simulate(k.stream,inputs)['out'].copy()
    prepare(k.stream,k.meta)
    p=k.meta['readout_pipeline'];assert p['status']=='enabled'
    assert p['buffers']==2
    assert np.array_equal(simulate(k.stream,inputs)['out'],expected)
    lint(k.stream)
    for a,b in zip(p['tiles'],p['tiles'][1:]):
        assert a['ready_after'] < a['host_after'] < b['ready_after']
        assert a['slot'] != b['slot']
    small=ops.mean_dim((2,3),[1],False,s_x=F(1,3),s_out=F(1))
    before=list(small.stream.ins);prepare(small.stream,small.meta)
    assert small.meta['readout_pipeline']['status']=='serial'
    assert small.stream.ins==before


def test_scalar_fp32_parameters_preserve_the_canonical_value():
    value=np.float32(0.03)
    p=plan(value,np.float32(0.01))
    assert p['ratio']==dict(numerator=str(F(float(value)).numerator),denominator=str(F(float(value)).denominator))
    assert validate(p)==p['proof']


def test_fractional_recorded_accumulator_bounds_are_refused():
    from mlir_oot.core_aten.readout_bounds import interval
    from mlir_oot.core_aten.requant import Readout
    with pytest.raises(TypeError):
        interval(Readout('y',[('x',F(1))]),[1,0],dict(accumulator_min=-128.5,accumulator_max=127))


def test_malformed_boolean_or_constant_fields_are_refused():
    p=plan(F(1,3))
    bad=copy.deepcopy(p);bad['relu']='false'
    with pytest.raises(ValueError):validate(bad)
    bad=copy.deepcopy(p);bad['constants']['out_clamp']='false'
    with pytest.raises(ValueError):validate(bad)
    bad=copy.deepcopy(p);bad['constants']['S']=1.0
    with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):plan(F(1),relu='false')


@pytest.mark.parametrize('placement', [
    (12, 3, 2, 7),
    dict(shape=(2, 2, 3), strides=(15, 5, 1), offset=1),
])
@pytest.mark.parametrize('factor', [F(3, 8), F(-17, 31)])
@pytest.mark.parametrize('relu', [False, True])
def test_native_structural_placement_preserves_exact_readout_and_unwritten_bytes(tmp_path, placement, factor, relu):
    import ctypes, os, subprocess
    from pathlib import Path
    from mlir_oot.core_aten.stream import Stream, Arg, simulate
    from mlir_oot.core_aten.requant import Readout
    from mlir_oot.core_aten.emit import build_module
    bindir = Path(os.environ.get('GEMMINI_LLVM_BIN', '/nonexistent'))
    if not (bindir/'mlir-translate').exists():
        pytest.skip('LLVM tools unavailable')
    shape = (2, 3, 5)
    raw = np.arange(-15, 15, dtype=np.int32).reshape(shape)
    result = np.full(shape, -99, np.int8)
    stream = Stream([Arg('x', 'i32', shape, 'input'), Arg('y', 'i8', shape, 'output')])
    stream.readouts.append(Readout('y', [('x', factor)], constant=F(1, 2),
                                  zero_point=-7, placement=placement, relu=relu))
    meta = {}
    module, _ = build_module(stream, 'placed_readout', meta=meta)
    certificate, = meta['host_requantization_proofs']
    assert certificate['output_placement'] == placement
    assert certificate['elements'] == 12
    validate(certificate)
    source, ir, obj, shared = (tmp_path/('placed'+suffix) for suffix in ('.mlir', '.ll', '.o', '.so'))
    source.write_text(str(module)+'\n')
    subprocess.run([str(bindir/'mlir-translate'), '--mlir-to-llvmir', str(source), '-o', str(ir)], check=True)
    subprocess.run([str(bindir/'llc'), '-mtriple=x86_64-unknown-linux-gnu', '-relocation-model=pic',
                    '-O2', '-filetype=obj', str(ir), '-o', str(obj)], check=True)
    subprocess.run(['cc', '-shared', str(obj), '-o', str(shared)], check=True)
    fn = ctypes.CDLL(str(shared)).placed_readout
    fn.argtypes = [ctypes.c_void_p]*2
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
