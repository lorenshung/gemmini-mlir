"""Code identity, qparam differences, and semantic selector refusals."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import ACC_BIT, simulate
from exact_oracle import evaluate


def source(*, low=-128, high=127, zp=0, storage='int8', output=None, arithmetic=False):
    inp = (Fraction(float(np.float32(.03))), low, high, zp, storage)
    out = output or inp
    def params(p, kind):
        scale, lo, hi, zero, dtype = p
        attrs = (f'prov.quant.granularity = "tensor", prov.quant.storage_dtype = "{dtype}", '
                 f'prov.quant.quant_min = {lo} : i64, prov.quant.quant_max = {hi} : i64, '
                 f'prov.quant.zero_point = {zero} : i64, '
                 f'prov.quant.scale_numerator = ["{scale.numerator}"], '
                 f'prov.quant.scale_denominator = ["{scale.denominator}"]')
        if kind == 'q':
            attrs += ', prov.quant.rounding = "half_to_even"'
        return attrs
    body = ('%c = arith.constant 0.0 : f32\n %v = arith.addf %a, %c : f32\n linalg.yield %v : f32'
            if arithmetic else 'linalg.yield %a : f32')
    return f'''builtin.module {{
      func.func @forward(%x: tensor<256xi8>) -> tensor<256xi8> {{
        %s0 = arith.constant dense<{float(inp[0])}> : tensor<f64>
        %s1 = arith.constant dense<{float(out[0])}> : tensor<f64>
        %z0 = arith.constant dense<{inp[3]}> : tensor<i64>
        %z1 = arith.constant dense<{out[3]}> : tensor<i64>
        %dq = "quant_ext.dequantize_per_tensor"(%x, %s0, %z0)
          <{{quant_min = {inp[1]} : i64, quant_max = {inp[2]} : i64, input_dtype = "{inp[4]}"}}>
          {{{params(inp, 'dq')}}} : (tensor<256xi8>, tensor<f64>, tensor<i64>) -> tensor<256xf32>
        %empty = tensor.empty() : tensor<256xf32>
        %copy = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}} ins(%dq : tensor<256xf32>) outs(%empty : tensor<256xf32>) {{
          ^bb0(%a: f32, %b: f32): {body}
        }} -> tensor<256xf32>
        %q = "quant_ext.quantize_per_tensor"(%copy, %s1, %z1)
          <{{quant_min = {out[1]} : i64, quant_max = {out[2]} : i64, output_dtype = "{out[4]}"}}>
          {{{params(out, 'q')}}} : (tensor<256xf32>, tensor<f64>, tensor<i64>) -> tensor<256xi8>
        func.return %q : tensor<256xi8>
      }}
    }}'''


@pytest.mark.parametrize('low,high,zp,storage', [(-128,127,0,'int8'), (-127,127,-7,'int8'),
                                               (0,255,128,'uint8'), (1,254,129,'uint8')])
def test_every_legal_storage_code_is_moved_without_accumulator(low, high, zp, storage):
    text = source(low=low, high=high, zp=zp, storage=storage)
    _, manifest, matches = build(text)
    match, = matches
    binding, = manifest['bindings']
    proof = binding['numeric']['identical_qparam_proof']
    assert binding['numeric']['readout'] == 'storage_codes'
    assert proof['inputs']['x0']['source_quant_attributes']
    assert proof['inputs']['x0']['scale_numerator'] == proof['output']['scale_numerator']
    assert not match.kernel.stream.readouts and not match.kernel.stream.packs
    assert not any(i.a('local') & ACC_BIT for i in match.kernel.stream.ins if i.kind in ('mvin','mvout'))
    codes = np.resize(np.arange(low,high+1,dtype=np.int64),256).astype(np.uint8).view(np.int8)
    actual = simulate(match.kernel.stream, dict(x0=codes))['y']
    assert np.array_equal(actual, codes)
    assert np.array_equal(actual, evaluate(match,dict(x0=codes))['y'])
    assert 'quant_ext.' not in rewrite(text, manifest)


@pytest.mark.parametrize('field,value', [(0,Fraction(1,16)), (1,-127), (2,126), (3,7), (4,'uint8')])
def test_each_qparam_difference_preserves_exact_host_requantization(field, value):
    out = [Fraction(float(np.float32(.03))), -128,127,0,'int8']
    out[field] = value
    if field == 4:
        out[1:4] = [0,255,128]
    _, manifest, matches = build(source(output=tuple(out)))
    assert matches, manifest['declines']
    assert all(m.numeric.get('readout') != 'storage_codes' and m.kernel.stream.readouts for m in matches)
    for m in matches:
        feed = {name: np.arange(-128,128,dtype=np.int8) for name,_ in m.operands}
        assert np.array_equal(simulate(m.kernel.stream,feed)['y'],evaluate(m,feed)['y'])


def test_arithmetic_identity_is_not_licensed_by_equal_qparams():
    _, _, matches = build(source(arithmetic=True))
    assert matches and all(m.numeric.get('readout') != 'storage_codes' for m in matches)


def test_unknown_storage_dtype_cannot_be_proved_as_signed_bytes():
    _, catalog, matches = build(source().replace('prov.quant.storage_dtype = "int8"',
                                                'prov.quant.storage_dtype = "int4"'))
    assert not matches
    assert any('storage dtype' in d['reason'] for d in catalog['declines'])


@pytest.mark.parametrize('literal', [.015, -.015, 1.5, -100.0])
def test_literal_selector_is_quantized_once_with_rational_rne(literal):
    text = source().replace('linalg.yield %a : f32',
        f'%literal = arith.constant {literal} : f32\n linalg.yield %literal : f32')
    _, catalog, matches = build(text)
    match, = matches
    assert match.numeric['readout'] == 'storage_codes', catalog['declines']
    feed = {name: np.arange(-128,128,dtype=np.int8) for name,_ in match.operands}
    assert np.array_equal(simulate(match.kernel.stream,feed)['y'],evaluate(match,feed)['y'])
    assert not match.kernel.stream.readouts


def test_equal_rationals_need_not_have_identical_numerator_spelling():
    text = source()
    n, d = Fraction(float(np.float32(.03))).as_integer_ratio()
    # Only the input pair is rescaled; equality is on the reduced rational.
    text = text.replace(f'scale_numerator = ["{n}"]',f'scale_numerator = ["{2*n}"]',1)
    text = text.replace(f'scale_denominator = ["{d}"]',f'scale_denominator = ["{2*d}"]',1)
    _, _, matches = build(text)
    assert matches[0].numeric['readout'] == 'storage_codes'


def test_provider_authenticates_byte_stream_without_a_mac_bound():
    path = Path(__file__).parents[2] / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('movement_provider',path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    proof = provider.admit_numeric_schedule(source(), package=path.parents[2], numeric_policy={})
    assert proof['status'] == 'proven', proof
    kernel, = proof['kernels']
    assert kernel['device_readout'] == kernel['schedule']['readout'] == 'storage_codes'
    assert kernel['requantization'] == []
