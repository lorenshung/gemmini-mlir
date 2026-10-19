"""Literal DMA specialization and many-layout catalog resource regressions."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def tensor(shape, dtype):
    return 'tensor<'+'x'.join(map(str,(*shape,dtype)))+'>'


def constant_source(dtype='i8', shape=(2,3,5,7), perm=(0,2,3,1), *, consumer=False):
    width=4 if dtype=='f32' else 1
    raw=np.random.default_rng(614).integers(0,256,(*shape,width),dtype=np.uint8)
    it=tensor(shape,dtype);ot=tensor(tuple(shape[p] for p in perm),dtype)
    args=f'%x: {ot}' if consumer else ''
    lines=[f'builtin.module {{ func.func @forward({args}) -> {ot} {{',
        f'%c = arith.constant dense<"0x{raw.tobytes().hex()}"> : {it}',
        f'%init = tensor.empty() : {ot}',
        f'%p = linalg.transpose ins(%c : {it}) outs(%init : {ot}) permutation = {list(perm)}']
    if consumer:
        lines += [f'%y = tensor.concat dim(0) %p, %x : ({ot}, {ot}) -> {tensor((shape[perm[0]]*2,*[shape[p] for p in perm[1:]]),dtype)}']
        # Give the consumer its correct result type.
        lines[0]=lines[0].replace(f'-> {ot}',f'-> {tensor((shape[perm[0]]*2,*[shape[p] for p in perm[1:]]),dtype)}')
    lines += [f'func.return {"%y" if consumer else "%p"} : '+(tensor((shape[perm[0]]*2,*[shape[p] for p in perm[1:]]),dtype) if consumer else ot),'} }']
    expected=raw.transpose((*perm,4)).copy().tobytes()
    return '\n'.join(lines),expected


@pytest.mark.parametrize('dtype',['i8','f32'])
@pytest.mark.parametrize('consumer',[False,True])
def test_literal_transpose_is_stored_constant_with_source_bound_proof(dtype,consumer,monkeypatch):
    from mlir_oot.core_aten import ops,linmap
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.offload import rewrite
    from mlir_oot.frontend.parse import parse_module
    def forbidden(*a,**kw):
        pytest.fail('constant movement entered arithmetic element maps')
    monkeypatch.setattr(ops,'_dense_linear',forbidden)
    monkeypatch.setattr(linmap,'extract',forbidden)
    source,expected=constant_source(dtype,consumer=consumer)
    _,manifest,matches=build(source)
    assert len(manifest['constant_folds'])==1
    assert len(matches)==int(consumer)
    rewritten=parse_module(rewrite(source,manifest))
    constants=[op for op in rewritten.walk() if op.name=='arith.constant']
    assert len(constants)==1
    assert constants[0].value.data.data==expected
    assert not any(op.name=='linalg.transpose' for op in rewritten.walk())
    if consumer:
        assert len(manifest['bindings'])==1
        fold=manifest['constant_folds'][0]
        assert manifest['bindings'][0]['operands']['x0']['op_ordinal']==fold['root_ordinal']
    manifest['constant_folds'][0]['storage_sha256']='0'*64
    with pytest.raises(ValueError,match='proof differs'):
        rewrite(source,manifest)


def pattern_source(runtime=280, constants=40):
    # Many live nodes, two model-sized storage widths, shared large immutable
    # weight data. All outputs escape: DCE cannot make the budget vacuous.
    lines=['builtin.module { func.func @forward(%x: tensor<1x256x56x56xi8>, %f: tensor<1x256x56x56xf32>) -> (']
    body=[];outputs=[];types=[]
    for i in range(runtime):
        dt='i8' if i%2==0 else 'f32';arg='%x' if dt=='i8' else '%f'
        it=tensor((1,256,56,56),dt)
        ot=tensor((1,56,56,256),dt) if i%4<2 else it
        body.append(f'%init{i} = tensor.empty() : {ot}')
        if i%4<2:
            body.append(f'%r{i} = linalg.transpose ins({arg} : {it}) outs(%init{i} : {ot}) permutation = [0,2,3,1]')
        else:
            body.append(f'%r{i} = linalg.generic {{indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>,affine_map<(a,b,c,d)->(a,b,c,d)>], iterator_types = ["parallel","parallel","parallel","parallel"]}} ins({arg} : {it}) outs(%init{i} : {ot}) {{ ^bb0(%a: {dt}, %b: {dt}): linalg.yield %a : {dt} }} -> {ot}')
        outputs.append(f'%r{i}');types.append(ot)
    # 2,359,296 nonuniform bytes. No element-valued Python list is generated.
    raw=bytes(range(256))*(2359296//256)
    body.append(f'%w = arith.constant dense<"0x{raw.hex()}"> : tensor<256x1024x3x3xi8>')
    for i in range(constants):
        body += [f'%ci{i} = tensor.empty() : tensor<3x3x1024x256xi8>',
                 f'%c{i} = linalg.transpose ins(%w : tensor<256x1024x3x3xi8>) outs(%ci{i} : tensor<3x3x1024x256xi8>) permutation = [2,3,1,0]']
        outputs.append(f'%c{i}');types.append('tensor<3x3x1024x256xi8>')
    lines[0]+=', '.join(types)+') {'
    return '\n'.join([*lines,*body,'func.return '+', '.join(outputs)+' : '+', '.join(types),'} }'])


def worker():
    import resource,time
    resource.setrlimit(resource.RLIMIT_AS,(768<<20,768<<20))
    from mlir_oot.core_aten.catalog import build
    source=pattern_source()
    start=time.monotonic()
    try:
        device,manifest,matches=build(source)
        result=dict(status='ok',bindings=len(matches),folds=len(manifest.get('constant_folds',[])),
                    kernels=manifest['unique_kernels'],llvm_operations=sum(1 for _ in device.walk()),
                    templates=sum(sum(1 for _ in m.kernel.stream.ins.templates()) for m in matches))
    except Exception as exc:
        result=dict(status=type(exc).__name__,error=str(exc)[:500])
    hwm=next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))
    result.update(seconds=time.monotonic()-start,peak_rss_kib=hwm)
    print(json.dumps(result),flush=True)


def test_many_movement_catalog_build_budget():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker'],env=env,text=True,capture_output=True,timeout=90)
    assert p.returncode==0,p.stderr[-2000:]
    record=json.loads(p.stdout.splitlines()[-1])
    assert record['status']=='ok',record
    assert record['bindings']==280 and record['folds']==40,record
    assert record['kernels']==4 and record['llvm_operations']<400,record
    assert record['seconds']<60 and record['peak_rss_kib']<384*1024,record
    assert record['templates']<3000,record


@pytest.mark.parametrize('dtype',['i8','f32'])
def test_constant_copy_cat_slice_insert_chain_preserves_storage(dtype,monkeypatch):
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.offload import rewrite
    from mlir_oot.frontend.parse import parse_module
    width=4 if dtype=='f32' else 1
    data=np.random.default_rng(7).integers(0,256,(3,5,width),dtype=np.uint8)
    if width==4:
        # Signed zero and two distinct NaN payloads are opaque storage.
        data[0,:3]=np.frombuffer(bytes.fromhex('000000800100c07f0200c0ff'),np.uint8).reshape(3,4)
    a=tensor((3,5),dtype);b=tensor((5,3),dtype);c=tensor((10,3),dtype)
    d=tensor((4,3),dtype);out=tensor((12,5),dtype)
    text=f'''builtin.module {{ func.func @forward() -> {out} {{
      %a = arith.constant dense<"0x{data.tobytes().hex()}"> : {a}
      %init = tensor.empty() : {b}
      %p = linalg.transpose ins(%a : {a}) outs(%init : {b}) permutation = [1,0]
      %copy = linalg.generic {{indexing_maps = [affine_map<(a,b)->(a,b)>,affine_map<(a,b)->(a,b)>], iterator_types = ["parallel","parallel"]}} ins(%p : {b}) outs(%init : {b}) {{ ^bb0(%aa: {dtype}, %bb: {dtype}): linalg.yield %aa : {dtype} }} -> {b}
      %cat = tensor.concat dim(0) %copy, %p : ({b}, {b}) -> {c}
      %slice = "tensor.extract_slice"(%cat) <{{static_offsets = array<i64: 1,0>, static_sizes = array<i64: 4,3>, static_strides = array<i64: 2,1>, operandSegmentSizes = array<i32: 1,0,0,0>}}> : ({c}) -> {d}
      %zero = arith.constant {"-0.0" if dtype=='f32' else "0"} : {dtype}
      %dest = tensor.splat %zero : {out}
      %insert = "tensor.insert_slice"(%slice, %dest) <{{static_offsets = array<i64: 1,1>, static_sizes = array<i64: 4,3>, static_strides = array<i64: 2,1>, operandSegmentSizes = array<i32: 1,1,0,0,0>}}> : ({d}, {out}) -> {out}
      func.return %insert : {out}
    }} }}'''
    def forbidden(*a,**k):
        pytest.fail('dense literal unpacked into per-element Python objects')
    monkeypatch.setattr(DenseIntOrFPElementsAttr,'get_values',forbidden)
    _,manifest,matches=build(text)
    assert not matches and len(manifest['constant_folds'])==5,manifest['declines']
    module=parse_module(rewrite(text,manifest))
    returned=next(op for op in module.walk() if op.name=='func.return').operands[0].owner
    raw=returned.value.data.data
    pattern=bytes.fromhex('00000080') if width==4 else b'\0'
    expected=np.tile(np.frombuffer(pattern,np.uint8),(12,5,1))
    p=data.transpose(1,0,2)
    expected[1:9:2,1:4]=np.concatenate([p,p],axis=0)[1:9:2]
    assert raw==expected.tobytes()
    assert not any(op.name.startswith('linalg.') or op.name in ('tensor.insert_slice','tensor.extract_slice','tensor.concat') for op in module.walk())


def test_constant_fold_only_catalog_compiles_an_empty_object_for_provider(tmp_path,monkeypatch):
    from mlir_oot.core_aten import catalog
    source,_=constant_source()
    path=tmp_path/'source.mlir';path.write_text(source)
    def compile_empty(device,workdir,**kw):
        assert len(list(device.walk()))==1
        (workdir/'core_aten.o').write_bytes(b'empty test object')
        return dict(status='test')
    monkeypatch.setattr(catalog,'compile_to_object',compile_empty)
    manifest=catalog.compile_catalog(path,tmp_path)
    assert manifest['status']=='compiled' and not manifest['kernels'] and manifest['constant_folds']
    _,obj=catalog.merlin_builder(tmp_path)(path,tmp_path)
    assert obj.is_file()


@pytest.mark.parametrize('kind',['permute','transpose','copy','cat'])
def test_native_float_layout_in_quantized_module_never_uses_element_maps(kind,monkeypatch):
    from test_layout_scalability import source
    from mlir_oot.core_aten import ops,linmap,exact_linear
    from mlir_oot.core_aten.catalog import build
    def forbidden(*a,**kw):
        pytest.fail('native float layout entered interpreted arithmetic')
    monkeypatch.setattr(ops,'_dense_linear',forbidden)
    monkeypatch.setattr(linmap,'extract',forbidden)
    monkeypatch.setattr(exact_linear,'kernel',forbidden)
    _,manifest,matches=build(source(kind,'f32',(1,18,3,5),'integer_in_quantized_module'))
    assert len(matches)==1,manifest['declines']
    assert matches[0].kernel.meta['selector_representation']=='shape_stride_map'


def test_constant_transpose_fold_is_not_confused_by_parameter_provenance():
    from test_layout_scalability import source
    from mlir_oot.core_aten.catalog import build
    # A function argument remains mutable by its caller. Labels are traceability.
    from mlir_oot.frontend.parse import parse_module
    from xdsl.dialects.builtin import StringAttr
    module=parse_module(source('transpose','i8',(1,18,3,5)))
    op=next(op for op in module.walk() if op.name=='linalg.transpose')
    op.attributes['prov.role']=StringAttr('CONSTANT')
    op.attributes['prov.aten']=StringAttr('weight.transpose')
    text=str(module)
    _,manifest,matches=build(text)
    assert len(matches)==1 and not manifest['constant_folds']


@pytest.mark.parametrize('kind',['forward','input','weight'])
def test_internal_convolution_literal_layout_has_zero_runtime_dma(kind):
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.recipes.conv import ConvGeom
    from mlir_oot.core_aten.stream import simulate
    rng=np.random.default_rng(71)
    g=ConvGeom(1,4,5,2,3,2,2)
    x=rng.integers(-3,4,(1,2,4,5),dtype=np.int8)
    w=rng.integers(-3,4,(3,2,2,2),dtype=np.int8)
    grad=rng.integers(-3,4,(1,3,g.oh,g.ow),dtype=np.int8)
    if kind=='forward':
        factory=lambda **kw: ops.convolution_nchw(g,s_x=1,s_w=1,s_out=1,**kw)
        feed={'x':x,'w':w};constants={'w':w}
    elif kind=='input':
        factory=lambda **kw: ops.conv_backward_input(g,s_g=1,s_w=1,s_out=1,**kw)
        feed={'g':grad,'w':w};constants={'w':w}
    else:
        factory=lambda **kw: ops.conv_backward_weight(g,s_g=1,s_x=1,s_out=1,**kw)
        feed={'g':grad,'x':x};constants={'x':x}
    ordinary=factory();folded=factory(constants=constants)
    live_feed={name:array for name,array in feed.items() if name not in constants}
    assert np.array_equal(simulate(ordinary.stream,feed)['y'],simulate(folded.stream,live_feed)['y'])
    proof,=folded.meta['constant_layout_proofs']
    assert proof['runtime_instructions']==0
    target=folded.stream._index[proof['destination']]
    assert not any(i.kind=='mvout' and i.arg==target for i in folded.stream.ins)
    assert len(folded.stream.ins)<len(ordinary.stream.ins)
    from mlir_oot.core_aten.lint import lint
    lint(folded.stream)


@pytest.mark.parametrize('unsigned',[False,True])
def test_internal_convolution_binding_pins_literal_source_and_abi(unsigned):
    from fixture_source import read
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.offload import rewrite
    from mlir_oot.frontend.parse import parse_module
    # Preserve an actual captured convolution body and replace only the weight
    # operand with literal code storage. No model-name selector is introduced.
    path=Path(__file__).parents[1]/'fixtures/core_aten/convolution/model.mlir'
    if not path.exists():
        pytest.fail('real convolution fixture missing')
    module=parse_module(read(path))
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import BytesAttr,DenseIntOrFPElementsAttr
    from mlir_oot.core_aten.match import name_of
    dqs=[op for op in module.walk() if name_of(op)=='quant_ext.dequantize_per_tensor']
    weight=dqs[1].operands[0]
    shape=weight.type.get_shape()
    raw=bytes((i%256 for i in range(int(np.prod(shape)))))
    if unsigned:
        from xdsl.dialects.builtin import IntegerAttr,StringAttr,i64
        dqs[1].properties['quant_min']=IntegerAttr(0,i64)
        dqs[1].properties['quant_max']=IntegerAttr(255,i64)
        dqs[1].properties['input_dtype']=StringAttr('uint8')
    _,original_manifest,original_matches=build(str(module))
    original=next(m for m in original_matches if m.op=='aten.convolution.default')
    weight_data=np.frombuffer(raw,np.int8).reshape(shape)
    original_feed={name:(weight_data if name=='w' else np.ones(value.type.get_shape(),np.int8))
                   for name,value in original.operands}
    from mlir_oot.core_aten.stream import simulate
    expected=simulate(original.kernel.stream,original_feed)['y']
    constant=arith.ConstantOp(DenseIntOrFPElementsAttr(weight.type,BytesAttr(raw)))
    dqs[1].parent_block().insert_op_before(constant,dqs[1])
    operands=list(dqs[1].operands);operands[0]=constant.result;dqs[1].operands=operands
    text=str(module)
    _,manifest,matches=build(text)
    conv=[m for m in matches if 'constant_layout_proofs' in m.kernel.meta]
    assert len(conv)==1,manifest['declines']
    binding=next(b for b in manifest['bindings'] if b['op']==conv[0].op)
    assert 'w' not in binding['operands'] and 'w' in binding['constant_operands']
    actual=simulate(conv[0].kernel.stream,{name:data for name,data in original_feed.items() if name!='w'})['y']
    assert np.array_equal(actual,expected)
    proof=conv[0].kernel.meta['constant_code_packing']['w']['proof']
    assert proof==('uint8_xor_0x80_to_signed_units' if unsigned else 'identity_storage_codes')
    parse_module(rewrite(text,manifest)).verify()


if __name__=='__main__':
    worker()
