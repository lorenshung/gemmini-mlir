"""Model-sized layout builds measured in isolated, memory-limited subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def source(kind='transpose', dtype='i8', shape=(1,256,56,56), scheme='native'):
    def tensor(sh,dt):
        return 'tensor<'+'x'.join(map(str,(*sh,dt)))+'>'
    output = tuple(shape[p] for p in (0,2,3,1)) if kind in ('permute','transpose') else shape
    input_shape = (shape[0],shape[1]//2,*shape[2:]) if kind=='cat' else shape
    quantized = scheme in ('same','requant')
    element = 'f32' if quantized else dtype
    it,ot = tensor(input_shape,element),tensor(output,element)
    code_in,code_out = tensor(input_shape,'i8'),tensor(output,'i8')
    count = 2 if kind=='cat' else 1
    arguments = ', '.join(f'%arg{i}: {code_in if quantized else it}' for i in range(count))
    lines = [f'builtin.module {{ func.func @forward({arguments}) -> {code_out if quantized else ot} {{']
    if quantized or scheme=='integer_in_quantized_module':
        lines += ['%s = arith.constant dense<0.03125> : tensor<f64>',
                  '%so = arith.constant dense<0.0625> : tensor<f64>',
                  '%z = arith.constant dense<0> : tensor<i64>']
    values=[]
    for i in range(count):
        value=f'%arg{i}'
        if quantized:
            lines += [f'%dq{i} = "quant_ext.dequantize_per_tensor"({value}, %s, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}}> : ({code_in}, tensor<f64>, tensor<i64>) -> {it}']
            value=f'%dq{i}'
        values.append(value)
    if scheme=='integer_in_quantized_module':
        lines += ['%f = arith.constant dense<0.0> : tensor<1xf32>',
                  '%unused = "quant_ext.quantize_per_tensor"(%f, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<1xf32>, tensor<f64>, tensor<i64>) -> tensor<1xi8>']
    if kind=='transpose':
        lines += [f'%init = tensor.empty() : {ot}',
                  f'%y = linalg.transpose ins({values[0]} : {it}) outs(%init : {ot}) permutation = [0, 2, 3, 1]']
    elif kind=='cat':
        lines += [f'%y = tensor.concat dim(1) {", ".join(values)} : ({it}, {it}) -> {ot}']
    else:
        mapping = '(d0, d3, d1, d2)' if kind=='permute' else '(d0, d1, d2, d3)'
        lines += [f'%init = tensor.empty() : {ot}',
          f'%y = linalg.generic {{indexing_maps = [affine_map<(d0,d1,d2,d3) -> {mapping}>, affine_map<(d0,d1,d2,d3) -> (d0,d1,d2,d3)>], iterator_types = ["parallel","parallel","parallel","parallel"]}} ins({values[0]} : {it}) outs(%init : {ot}) {{',
          f'^bb0(%a: {element}, %b: {element}): linalg.yield %a : {element}',f'}} -> {ot}']
    ret = '%y'
    if quantized:
        scale = '%so' if scheme=='requant' else '%s'
        lines += [f'%q = "quant_ext.quantize_per_tensor"(%y, {scale}, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> : ({ot}, tensor<f64>, tensor<i64>) -> {code_out}']
        ret='%q'
    lines += [f'func.return {ret} : {code_out if quantized else ot}','} }']
    return '\n'.join(lines)


def asymmetric_cat_source(axis=1):
    text=source('cat','i8',(1,18,3,5),'requant')
    text=text.replace('%z = arith.constant dense<0> : tensor<i64>',
        '%z = arith.constant dense<0> : tensor<i64>\n%s1 = arith.constant dense<0.09375> : tensor<f64>\n%z1 = arith.constant dense<128> : tensor<i64>')
    text=text.replace('(%arg1, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}>',
        '(%arg1, %s1, %z1) <{quant_min = 0 : i64, quant_max = 255 : i64, input_dtype = "uint8"}>')
    if axis==3:
        text=text.replace('tensor.concat dim(1)','tensor.concat dim(3)').replace('tensor<1x18x3x5','tensor<1x9x3x10')
    return text


def worker(kind,dtype,scheme):
    import resource
    import time
    resource.setrlimit(resource.RLIMIT_AS,(768<<20,768<<20))
    # Includes source parsing, liveness, matching, hazard audit and LLVM module emission.
    from mlir_oot.core_aten.catalog import build
    started=time.monotonic()
    result={}
    try:
        module,manifest,matches=build(source(kind,dtype,scheme=scheme))
        assert len(matches)==1,manifest['declines']
        stream=matches[0].kernel.stream
        compact = hasattr(stream.ins,'templates')
        result.update(status='ok',llvm_operations=sum(1 for _ in module.walk()),
                      instructions=len(stream.ins),compact=compact,
                      templates=sum(1 for _ in stream.ins.templates()) if compact else len(stream.ins),
                      readout=matches[0].kernel.meta.get('readout'),
                      source_representation=matches[0].kernel.meta.get('selector_representation'))
    except Exception as exc:
        result.update(status=type(exc).__name__,reason=str(exc)[:500])
    hwm = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))
    result.update(seconds=time.monotonic()-started,peak_rss_kib=hwm)
    print(json.dumps(result),flush=True)


def measure(kind,dtype,scheme='native',root=None,timeout=45):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    repo=Path(__file__).resolve().parents[2] if root is None else Path(root)
    env['PYTHONPATH']=str(repo)+':/scratch/loren/core-aten-ref/merlin/src'
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),kind,dtype,scheme],env=env,
                     capture_output=True,text=True,timeout=timeout)
    assert p.returncode==0,p.stderr[-2000:]
    return json.loads(p.stdout.splitlines()[-1])


@pytest.mark.parametrize('kind',['transpose','permute','copy','cat'])
@pytest.mark.parametrize('dtype',['i8','f32'])
def test_model_sized_native_layout_has_bounded_build_resources(kind,dtype):
    record=measure(kind,dtype)
    assert record['status']=='ok',record
    assert record['peak_rss_kib'] < 384*1024,record
    assert record['seconds'] < 30,record
    assert record['compact'] and record['templates'] < 64 and record['llvm_operations'] < 600,record
    assert record['source_representation']=='shape_stride_map',record


@pytest.mark.parametrize('kind',['transpose','permute','copy','cat'])
@pytest.mark.parametrize('scheme',['same','requant'])
def test_model_sized_quantized_layout_has_no_dense_map(kind,scheme):
    record=measure(kind,'i8',scheme)
    assert record['status']=='ok',record
    assert record['peak_rss_kib'] < 384*1024 and record['seconds'] < 30,record
    assert record['templates'] < 64 and record['llvm_operations'] < 600,record
    assert record['source_representation']=='shape_stride_map',record
    assert record['readout']==('storage_codes' if scheme=='same' else 'full_i32'),record


def test_integer_layout_in_a_quantized_module_does_not_enter_dense_linear_lowering():
    record=measure('transpose','i8','integer_in_quantized_module')
    assert record['status']=='ok' and record['seconds']<30 and record['peak_rss_kib']<384*1024,record
    assert record['compact'] and record['templates']<64,record


@pytest.mark.parametrize('kind',['transpose','permute','copy','cat'])
@pytest.mark.parametrize('dtype,scheme',[('i8','native'),('f32','native'),('i8','same'),('i8','requant'),
                                       ('i8','integer_in_quantized_module')])
def test_proved_layouts_never_construct_sparse_or_dense_element_maps(monkeypatch,kind,dtype,scheme):
    from mlir_oot.core_aten import ops,linmap,exact_linear
    from mlir_oot.core_aten.catalog import build
    def forbidden(*args,**kwargs):
        pytest.fail('pure layout entered element-map arithmetic lowering')
    monkeypatch.setattr(ops,'_dense_linear',forbidden)
    monkeypatch.setattr(exact_linear,'kernel',forbidden)
    monkeypatch.setattr(linmap,'extract',forbidden)
    _,catalog,matches=build(source(kind,dtype,(1,18,3,5),scheme))
    assert len(matches)==1,catalog['declines']


def test_compact_loop_slices_and_encoding_are_lossless_and_stable():
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.emit import encode,instruction_tree
    from mlir_oot.core_aten.reroll import iter_expand
    from mlir_oot.core_aten.stream import CompactInstructions
    for length in (1,15,16,17,255,256,257):
        kernel=ops.clone((length,),s_out=2)
        instructions=kernel.stream.ins
        assert isinstance(instructions,CompactInstructions)
        assert list(iter_expand(instruction_tree(instructions)))==[encode(i) for i in instructions]
        assert repr(instructions)==repr(ops.clone((length,),s_out=2).stream.ins)
        assert instructions.counts()==kernel.stream.counts()


def test_oversize_layout_declines_before_allocating_payload_or_instruction_maps():
    import time
    from mlir_oot.core_aten.catalog import build
    started=time.monotonic()
    _,catalog,matches=build(source('transpose','f32',(1,256,5600,5600)))
    assert not matches
    assert any('bounded compiler' in d['reason'] for d in catalog['declines'])
    assert time.monotonic()-started<2


@pytest.mark.parametrize('axis',[1,3])
def test_independent_unsigned_qparams_commute_with_concat_exactly(axis):
    import numpy as np
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate
    _,catalog,matches=build(asymmetric_cat_source(axis))
    match,=matches
    arrays=[np.resize(np.arange(-128,128,dtype=np.int16),(1,9,3,5)).astype(np.int8),
            np.resize(np.arange(127,-129,-1,dtype=np.int16),(1,9,3,5)).astype(np.int8)]
    feed={name:arrays[i] for i,(name,_) in enumerate(match.operands)}
    assert len(match.kernel.stream.readouts)==2,catalog
    assert np.array_equal(simulate(match.kernel.stream,feed)['y'],evaluate(match,feed)['y'])


if __name__=='__main__':
    worker(*sys.argv[1:])
