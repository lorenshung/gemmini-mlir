"""Literal operands retain ordinary exact device DMA and source bindings."""
from pathlib import Path

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


@pytest.mark.parametrize('kind',['forward','input','weight'])
def test_internal_convolution_literal_layout_uses_exact_runtime_dma(kind):
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
    assert 'constant_layout_proofs' not in folded.meta
    assert any(i.kind=='mvout' for i in folded.stream.ins)
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
    conv=[m for m in matches if m.op=='aten.convolution.default']
    assert len(conv)==1,manifest['declines']
    binding=next(b for b in manifest['bindings'] if b['op']==conv[0].op)
    assert 'w' not in binding['operands'] and 'w' in binding['constant_operands']
    actual=simulate(conv[0].kernel.stream,{name:data for name,data in original_feed.items() if name!='w'})['y']
    assert np.array_equal(actual,expected)
    proof=conv[0].kernel.meta['constant_code_packing']['w']['proof']
    assert proof==('uint8_xor_0x80_to_signed_units' if unsigned else 'identity_storage_codes')
    parse_module(rewrite(text,manifest)).verify()


@pytest.mark.parametrize('dtype', ['i8', 'f32'])
def test_literal_transpose_runs_on_device_without_constant_folding(dtype):
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from mlir_oot.core_aten.offload import rewrite
    from mlir_oot.frontend.parse import parse_module
    source, expected = constant_source(dtype)
    _, manifest, matches = build(source)
    assert 'constant_folds' not in manifest
    assert len(matches) == 1, manifest['declines']
    match, = matches
    feed = {name: np.frombuffer(value.owner.value.data.data, np.uint8)
            for name, value in match.operands}
    actual = np.ascontiguousarray(simulate(match.kernel.stream, feed)[match.result[0]])
    assert actual.view(np.uint8).tobytes() == expected
    assert any(i.kind == 'mvin' for i in match.kernel.stream.ins)
    parse_module(rewrite(source, manifest)).verify()
