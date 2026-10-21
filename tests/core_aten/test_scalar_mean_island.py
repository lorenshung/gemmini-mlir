"""Closed quantized reductions retain an exact numerator through dense views."""
from fractions import Fraction as Q
import json
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build, compile_catalog, merlin_builder, parse_module
from mlir_oot.core_aten.lint import lint
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate
from conftest import PACKAGE, POLICY, provider

FIXTURES = Path(__file__).parent/'fixtures/scalar_mean'


def source(name='minimal'):
    return (FIXTURES/(name+'.mlir')).read_text()


def reshape_variant(text, kind):
    old = '%c0_25 = tensor.reshape %c0_19(%c0_24) : (tensor<f32>, tensor<2xi64>) -> tensor<1x1xf32>'
    # The supported xDSL verifier rejects rank-zero collapse/expand. Use a
    # literal scalar-to-vector reshape before testing ordinary reassociations.
    prefix = '''%shape1 = arith.constant dense<1> : tensor<1xi64>
    %flat = tensor.reshape %c0_19(%shape1) : (tensor<f32>, tensor<1xi64>) -> tensor<1xf32>
    '''
    if kind == 'expand':
        new = prefix+'%c0_25 = tensor.expand_shape %flat [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>'
    elif kind == 'collapse_expand':
        new = prefix+'''%expanded = tensor.expand_shape %flat [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
    %c0_25 = tensor.expand_shape %collapsed [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>'''
    else:
        return text
    assert old in text
    return text.replace(old, new)


def check_kernel(text, codes):
    _, manifest, matches = build(text)
    assert len(matches) == len(manifest['kernels']) == 1, manifest
    match, = matches
    name, = [a.name for a in match.kernel.args if a.role == 'input']
    values = simulate(match.kernel.stream, {name: codes})
    numerator = sum(int(v) for v in codes.flat)
    factor = Q(0.0097962627187371254)/6/Q(0.0019607844296842813)
    expected = np.array([[min(127, max(-128, round(numerator*factor)))]], np.int8)
    assert np.array_equal(values['y'], expected)
    raw = match.kernel.stream.readouts[0].terms[0][0]
    assert np.asarray(values[raw]).item() == numerator
    lint(match.kernel.stream)
    assert all(i.a('scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_ld')
    assert all(i.a('acc_scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_st')
    assert all(i.a('local') & (1 << 29) for i in match.kernel.stream.ins if i.kind == 'mvout')
    return match, manifest


@pytest.mark.parametrize('name', ['minimal', 'normalized', 'generic'])
def test_real_scalar_islands_have_complete_source_bound_numeric_proofs(name):
    text = source(name)
    _, manifest = check_kernel(text, np.array([[-128,127,-5],[7,4,-1]], np.int8))
    api = provider()
    proof = api.admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert proof['bindings'] == manifest['bindings']
    assert all(k['exact_arithmetic'] and k['schedule']['readout'] == 'full_i32' for k in proof['kernels'])
    rewritten = rewrite(text, manifest)
    parse_module(rewritten).verify()
    assert rewritten.count('func.call @ca_') == 1
    assert 'arith.divf' not in rewritten and 'linalg.reduce' not in rewritten


@pytest.mark.parametrize('kind', ['reshape', 'expand', 'collapse_expand'])
def test_shape_only_epilogue_chain_preserves_numerator(kind):
    text = reshape_variant(source(), kind)
    parse_module(text).verify()
    check_kernel(text, np.full((2,3), -128, np.int8))


def test_all_reachable_six_byte_sums_use_one_exact_rational_rounding():
    _, _, matches = build(source('generic'))
    k = matches[0].kernel
    name = next(a.name for a in k.args if a.role == 'input')
    factor = Q(0.0097962627187371254)/6/Q(0.0019607844296842813)
    for total in range(-768, 763):
        remaining = total+768
        codes = np.full(6,-128,np.int16)
        for i in range(6):
            delta = min(remaining,255)
            codes[i] += delta
            remaining -= delta
        actual = simulate(k.stream, {name: codes.astype(np.int8).reshape(2,3)})
        assert actual['acc_combined'].item() == total
        assert actual['y'].item() == min(127,max(-128,round(total*factor)))


@pytest.mark.parametrize('target', [(1,2), (2,1)])
def test_nonscalar_reduction_epilogue_reindexes_every_output(target):
    text = source()
    start, end = text.index('    %c0_10 ='), text.index('    %c0_26 =')
    result = 'x'.join(map(str,target))
    middle = f'''    %zero = arith.constant 0.0 : f32
    %init = tensor.splat %zero : tensor<2xf32>
    %sum = linalg.reduce ins(%c0_9:tensor<2x3xf32>) outs(%init:tensor<2xf32>) dimensions = [1]
    (%a: f32, %b: f32) {{
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
    }}
    %three = arith.constant 3.0 : f32
    %divisor = tensor.splat %three : tensor<f32>
    %empty = tensor.empty() : tensor<2xf32>
    %div = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->()>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
      ins(%sum, %divisor : tensor<2xf32>, tensor<f32>) outs(%empty : tensor<2xf32>) {{
      ^bb(%a: f32, %b: f32, %c: f32):
        %v = arith.divf %a, %b : f32
        linalg.yield %v : f32
      }} -> tensor<2xf32>
    %shape = arith.constant dense<[{target[0]}, {target[1]}]> : tensor<2xi64>
    %c0_25 = tensor.reshape %div(%shape) : (tensor<2xf32>, tensor<2xi64>) -> tensor<{result}xf32>
'''
    text = (text[:start]+middle+text[end:]).replace('tensor<1x1x',f'tensor<{result}x')
    _, manifest, matches = build(text)
    assert len(matches) == 1, manifest
    match, = matches
    values = np.array([[-128,20,9],[127,15,-7]], np.int8)
    name = match.operands[0][0]
    actual = simulate(match.kernel.stream,{name:values})
    factor = Q(0.0097962627187371254)/3/Q(0.0019607844296842813)
    expected = np.array([min(127,max(-128,round(int(v)*factor))) for v in values.astype(np.int32).sum(1)],np.int8).reshape(target)
    assert np.array_equal(actual['y'],expected)
    assert provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY)['status'] == 'proven'


def test_nonliteral_reshape_shape_is_not_assumed_from_static_result():
    text = source().replace('%arg0: tensor<2x3xf32>', '%arg0: tensor<2x3xf32>, %shape: tensor<2xi64>', 1)
    text = text.replace('%c0_19(%c0_24)', '%c0_19(%shape)')
    _, manifest, _ = build(text)
    assert not manifest['kernels']
    assert any('literal shape' in d['reason'] for d in manifest['declines'])
    assert provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)['status'] == 'declined'


def test_partial_island_proof_does_not_hide_live_unsupported_arithmetic():
    text = source().replace('func.return %c0_35, %arg0', '''%e = tensor.empty() : tensor<2x3xf32>
    %squared = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2x3xf32>) outs(%e : tensor<2x3xf32>) {
      ^bb(%x: f32, %y: f32):
        %z = arith.mulf %x, %x : f32
        linalg.yield %z : f32
      } -> tensor<2x3xf32>
    func.return %c0_35, %squared''')
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'declined'
    assert proof['reason'] == 'provider did not cover every live quantized arithmetic producer'


def test_layout_prelude_does_not_intercept_a_complete_legacy_gemm_proof(tmp_path):
    from mlir_oot.golden_device_catalog import build_catalog
    from types import SimpleNamespace
    text = source('host_quantized_gemm')
    _, region_catalog, _ = build(text)
    assert region_catalog['kernels']  # Partial byte-layout matches triggered the regression.
    _, legacy = build_catalog(text)
    assert legacy['coverage_complete'] and legacy['covered_contractions'] == 1
    api = provider()
    proof = api.admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert proof['bindings'] == legacy['bindings']
    assert proof['kernels'][0]['schedule']['output_dtype'] == 'i32'
    (tmp_path/'model.mlir').write_text(text)
    route = api.routing(tmp_path,target='gemmini',package=PACKAGE,facts=None,
                        eligible=[(None,SimpleNamespace(dtypes=('i8','i8','i32')))])
    assert route.catalog_builder is not None and route.region_offload is None


def test_empty_catalog_never_claims_an_object_or_rewritten_artifact(tmp_path):
    path = tmp_path/'source.mlir'
    path.write_text('builtin.module { func.func @forward(%x: tensor<2xi32>) -> tensor<2xi32> { func.return %x : tensor<2xi32> } }')
    direct = compile_catalog(path, tmp_path/'direct')
    assert direct['status'] == 'declined' and direct['reason'] == 'no exact device kernel produced'
    assert 'compilation' not in direct
    for stage, invoke in [('provider', lambda w: provider().core_aten_catalog(path,w,package=PACKAGE)),
                          ('builder', lambda w: merlin_builder(Path('/unused'))(path,w))]:
        work = tmp_path/stage
        work.mkdir()
        # A stale object from an earlier invocation is no evidence of coverage.
        (work/'core_aten.o').write_bytes(b'stale')
        manifest, obj = invoke(work)
        assert obj is None and json.loads(manifest.read_text())['status'] == 'declined'
    model, shim, obj, manifest = provider().core_aten_offload(path,tmp_path/'offload',package=PACKAGE)
    assert model is shim is obj is None and manifest.is_file()


def test_uncovered_seeded_region_admission_also_returns_a_clean_refusal():
    from test_integer_regions import source as seeded
    proof = provider().admit_numeric_schedule(seeded(runtime=True,guarded=False),package=PACKAGE,numeric_policy=POLICY)
    assert proof['status'] == 'declined' and not proof['kernels'] and not proof['bindings']
    assert proof['reason'] == 'provider did not cover any exact seeded readout region'
    assert any('runtime i32 seed' in d['reason'] for d in proof['declines'])
