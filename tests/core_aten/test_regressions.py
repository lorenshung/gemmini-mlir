"""ABI and shape regressions from the exactness review."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.linmap import LinearMap
from mlir_oot.core_aten.linmatch import copy_kernel
from mlir_oot.core_aten.match import Decline
from mlir_oot.core_aten.offload import shim_source


def test_catalog_declines_a_hazardous_candidate_without_publishing_a_binding(monkeypatch):
    from mlir_oot.core_aten import catalog
    from mlir_oot.core_aten.stream import SP
    from fixture_source import read
    original = catalog.match_all
    def hazardous_candidate(module, **kwargs):
        matches, declines = original(module)
        stream = matches[0].kernel.stream
        stream.config_ld(0, 1)
        stream.mvin(stream.at('a'), SP(0), 1, 1)
        stream.mvin(stream.at('a'), SP(0), 1, 1)
        return matches, declines
    monkeypatch.setattr(catalog, 'match_all', hazardous_candidate)
    text = read(Path(__file__).parents[1] / 'fixtures/core_aten/mm/model.mlir')
    _, manifest, matches = catalog.build(text)
    assert not matches and not manifest['bindings'] and not manifest['kernels']
    assert any('unsafe device schedule:' in d['reason'] and 'HW3' in d['reason'] for d in manifest['declines'])


@pytest.mark.parametrize('error', [ValueError, OverflowError, ZeroDivisionError])
def test_selector_island_prepass_surfaces_errors_other_than_decline(monkeypatch, error):
    from mlir_oot.core_aten import catalog, quantized_movement
    from fixture_source import read
    text = read(Path(__file__).parents[1] / 'fixtures/core_aten/mm/model.mlir')
    def raise_(exc):
        raise exc
    monkeypatch.setattr(quantized_movement, 'match', lambda op, ordinals: raise_(Decline('no selector island')))
    assert catalog.build(text)[1]['bindings']     # a decline falls through to the normal matchers
    monkeypatch.setattr(quantized_movement, 'match', lambda op, ordinals: raise_(error('matcher bug')))
    with pytest.raises(error, match='matcher bug'):
        catalog.build(text)


@pytest.mark.parametrize('name', ['mean_dim', 'avg_pool2d', 'col2im', 'convolution_backward',
                                  '_adaptive_avg_pool2d_backward'])
def test_catalog_does_not_read_aten_provenance(name):
    """Matches, kernels and declines follow IR structure; prov.aten is only passed through to bindings."""
    import re
    from mlir_oot.core_aten import catalog
    from fixture_source import read
    text = read(Path(__file__).parents[1] / 'fixtures/core_aten' / name / 'model.mlir')
    renamed = re.sub(r'prov\.aten = "aten\.[^"]*"', 'prov.aten = "aten.renamed.default"', text)
    assert renamed != text
    def placement(source):
        manifest = catalog.build(source)[1]
        bindings = [{k: v for k, v in b.items() if k != 'aten'} for b in manifest['bindings']]
        return bindings, manifest['kernels'], manifest['declines']
    assert placement(renamed) == placement(text)


@pytest.mark.parametrize('kind', ['config_ld', 'config_st', 'narrow'])
def test_kernel_boundary_declines_unproved_device_scaling_or_narrowing(kind):
    from mlir_oot.core_aten.stream import ACC, Arg, Stream
    s = Stream([Arg('out', 'i8', (1,), 'output')])
    if kind == 'config_ld':
        s.config_ld(0, 1, 2)
    elif kind == 'config_st':
        s.config_st(1, scale=0.5)
    else:
        s.mvout(s.at('out'), ACC(0), 1, 1)
    with pytest.raises(Decline, match='nonunit device scale|full int32 width'):
        ops._end('regression', s)


@pytest.mark.parametrize('code', [-128, -127, -1, 0, 1, 126, 127])
def test_fill_literal_codes_use_only_unit_scales(code):
    from mlir_oot.core_aten.lint import lint
    from mlir_oot.core_aten.stream import simulate
    kernel = ops.full('aten.full.default', (17, 19), code, s_out=1)
    lint(kernel.stream)
    assert np.array_equal(simulate(kernel.stream, {})['y'], np.full((17, 19), code, np.int8))
    assert not kernel.meta['fp32_product_proofs']
    assert all(i.a('scale') == 1 for i in kernel.stream.ins if i.kind == 'config_ld')
    assert all(i.a('acc_scale') == 1 for i in kernel.stream.ins if i.kind == 'config_st')


@pytest.mark.parametrize('unsigned,zp', [(False, -7), (True, 0), (True, 123)])
@pytest.mark.parametrize('batch,m,k,n', [(2, 3, 4, 2), (3, 17, 33, 19)])
def test_bmm_zero_point_reductions_fence_shared_local_rows(unsigned, zp, batch, m, k, n):
    from fractions import Fraction
    from types import SimpleNamespace
    from mlir_oot.core_aten.lint import lint
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import quantize
    q = SimpleNamespace(unsigned=unsigned, zero_point=zp)
    kernel = ops.matmul('aten.bmm.default', m, k, n, batch=batch, s_out=1024)
    ops.contraction_zero_points(kernel, q, q, m, k, n, batch)
    lint(kernel.stream)  # K=4 leaves unused rows of the first reduction's ones tile.
    rng = np.random.default_rng(1234)
    low, high = (0, 256) if unsigned else (-128, 128)
    a = rng.integers(low, high, (batch, m, k), dtype=np.int32)
    b = rng.integers(low, high, (batch, k, n), dtype=np.int32)
    expected = quantize(((a.astype(object) - zp) @ (b.astype(object) - zp)) * Fraction(1, 1024))
    actual = simulate(kernel.stream, dict(a=a.astype(np.int8), b=b.astype(np.int8)))['out']
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize('channel_seed', [False, True])
def test_batched_integer_seeds_keep_batch_offsets(channel_seed):
    from mlir_oot.core_aten.lint import lint
    from mlir_oot.core_aten.stream import simulate
    rng = np.random.default_rng(128)
    batch, m, k, n = 3, 17, 33, 19
    a = rng.integers(-128, 128, (batch, m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (batch, k, n), dtype=np.int8)
    seed = rng.integers(-1000, 1000, (n,) if channel_seed else (batch, m, n), dtype=np.int32)
    kernel = ops.matmul('aten.bmm.default', m, k, n, batch=batch,
                        bias=True, bias_const=seed, bias_rows=not channel_seed)
    lint(kernel.stream)
    assert np.array_equal(simulate(kernel.stream, dict(a=a, b=b))['out'],
                          a.astype(np.int64) @ b.astype(np.int64) + seed)


@pytest.mark.parametrize('shape,dims', [((3, 0), [1]), ((0, 3), [0]), ((2, 3), []), ((), []), ((3,), [1]), ((3,), [0, -1])])
def test_reduction_declines_invalid_dimensions(shape, dims):
    with pytest.raises(Decline):
        ops.mean_dim(shape, dims, False, s_x=1, s_out=1)


def test_empty_gather_declines_before_stride_division():
    z = np.zeros(0, dtype=np.int64)
    lm = LinearMap((2, 0), z, z, z, z.astype(float), z.astype(float), {})
    with pytest.raises(Decline, match='empty'):
        copy_kernel('gather', lm, {}, 1)


def test_nonzero_memref_offsets_are_elements_for_i32_inputs_and_outputs(tmp_path):
    args = [dict(name='bias', dtype='i32', shape=[2], role='input'),
            dict(name='out', dtype='i32', shape=[2], role='output')]
    source = shim_source({'kernels': [dict(symbol='offset_kernel', args=args)]})
    source += r'''
#include <stdlib.h>
void ca_abort(const char *s) { abort(); }
void offset_kernel(void *b, void *o) {
  int32_t *bias=b, *out=o;
  out[0]=bias[0]+10; out[1]=bias[1]+20;
}
int main(void) {
  int32_t b[]={-111, -222, 31, 47, -333}, o[]={-7,-7,-7,-7,-7,-7};
  ca_memref_1 bias={b,b,2,{2},{1}}, out={o,o,3,{2},{1}}, result;
  _mlir_ciface_ca_offset_kernel(&result,&bias,&out);
  return !(o[0]==-7 && o[2]==-7 && o[3]==41 && o[4]==67 && o[5]==-7 && result.offset==3 && result.aligned==o);
}
'''
    path = tmp_path / 'offset.c'
    path.write_text(source)
    subprocess.run(['cc', '-std=c99', str(path), '-o', str(tmp_path / 'offset')], check=True)
    subprocess.run([str(tmp_path / 'offset')], check=True)


def test_ceil_divisor_case_is_always_generated_with_partial_windows():
    path = Path(__file__).parent / 'iface_ext/ext_cases.py'
    spec = importlib.util.spec_from_file_location('ext_cases_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for representative in (False, True):
        cb = module.cases(representative)['ws_avg_pool2d_ceil_div']
        assert cb['tensors']['A']['shape'][1] % 2 == 1
        attrs = cb['commands'][0]['attributes']
        assert attrs['ceil_mode'] and attrs['divisor_override'] == 3


def test_contraction_readout_view_keeps_source_descriptor_rank():
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate
    fixture=Path(__file__).parents[1]/'fixtures/core_aten/mm'
    text=(fixture/'model.mlir').read_text()
    text=text.replace('    %15 = arith.constant',
        '    %flat = tensor.collapse_shape %14 [[0, 1]] : tensor<37x45xf32> into tensor<1665xf32>\n    %15 = arith.constant',1)
    lines=text.splitlines()
    for index,line in enumerate(lines):
        if '%19 = "quant_ext.quantize_per_tensor"' in line:
            lines[index]=line.replace('(%14,','(%flat,').replace('tensor<37x45xf32>','tensor<1665xf32>')
    text='\n'.join(lines).replace('tensor<37x45xi8>','tensor<1665xi8>')
    from fixture_source import source_text
    _,manifest,matches=build(source_text(text))
    assert len(matches)==1,manifest['declines']
    match,=matches
    output=next(arg for arg in match.kernel.args if arg.role=='output')
    assert output.shape==(1665,)
    assert 'ca_memref_1 *result' in shim_source(manifest)
    inputs=dict(np.load(fixture/'inputs.npz'))
    feed={'a':inputs['arg0'],'b':inputs['arg1']}
    assert np.array_equal(simulate(match.kernel.stream,feed)['out'],evaluate(match,feed)['out'])


def test_quasi_affine_maps_are_not_proved_by_basis_samples():
    from xdsl.ir.affine import AffineExpr as A,AffineMap
    from mlir_oot.core_aten.match import map_coeffs,identity_map
    from mlir_oot.core_aten.linmap import _affine
    d=A.dimension(0)
    mapping=AffineMap(1,0,(d+d//2,))
    assert map_coeffs(mapping) is None
    assert not identity_map(mapping)
    points=np.arange(9)
    assert np.array_equal(_affine(mapping,[points],(9,))[0],points+points//2)
    bounded=AffineMap(1,0,(d+(d//24)*20,))
    assert map_coeffs(bounded,(24,))==[[1]]
    assert map_coeffs(bounded,(25,)) is None
