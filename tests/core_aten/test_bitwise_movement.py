"""Opaque payloads, dtype-width DMA, constants and typed descriptor offsets."""
import subprocess

import numpy as np
import pytest

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.lint import lint
from mlir_oot.core_aten.offload import shim_source
from mlir_oot.core_aten.recipes.movement import gather_elements
from mlir_oot.core_aten.storage import itemsize, literal_bytes
from mlir_oot.core_aten.stream import Arg, Stream, simulate


@pytest.mark.parametrize('dtype', ['i1', 'i8', 'f16', 'bf16', 'i32', 'i64', 'f32', 'f64', 'complex<f32>'])
def test_element_selectors_preserve_opaque_payloads(dtype):
    width = itemsize(dtype)
    # Reverse, broadcast, stride, and tile tails; the byte oracle sees no values.
    indices = np.r_[np.arange(34)[::-1], np.zeros(19, dtype=int), np.arange(1, 34, 3)]
    source = np.arange(34 * width, dtype=np.uint8)
    if dtype == 'i1':
        source %= 2
    s = Stream([Arg('x', dtype, (34,), 'input'), Arg('y', dtype, (len(indices),), 'output')], bitwise=True)
    gather_elements(s, np.zeros(len(indices), dtype=int), indices, width, 'y', ['x'], 'unused')
    kernel = ops._end('selectors', s)
    lint(s)
    actual = np.ascontiguousarray(simulate(s, {'x': source})['y']).reshape(-1).view(np.uint8)
    assert np.array_equal(actual, source.reshape(-1, width)[indices].reshape(-1))
    assert not kernel.meta['fp32_product_proofs']


@pytest.mark.parametrize('dtype,value,expected', [
    ('bf16', -0.0, '0080'), ('f16', -0.0, '0080'),
    ('i64', (1 << 60) + 3, '0300000000000010'), ('i1', -1, '01'),
    ('bf16', 1.00390625, '803f'),  # exact halfway rounds to even
    ('bf16', 1.01171875, '823f'),
    ('bf16', float('inf'), '807f'), ('f16', -float('inf'), '00fc'),
])
def test_literal_storage_is_exact(dtype, value, expected):
    assert literal_bytes(value, dtype).hex() == expected


def test_nan_literal_declines_without_a_payload_proof():
    from mlir_oot.core_aten.match import Decline
    with pytest.raises(Decline, match='NaN movement literal has no preserved payload proof'):
        literal_bytes(float('nan'), 'f32')


@pytest.mark.parametrize('dtype', ['i1','i8','f16','bf16','i32','i64','f32','f64','complex<f32>'])
@pytest.mark.parametrize('rank', [0,1,5])
def test_literal_reshape_preserves_bytes_including_scalar_shape_descriptors(dtype, rank):
    text = reshape_source(dtype, rank)
    _, manifest, matches = build(text)
    assert len(matches) == 1, manifest
    match, = matches
    assert match.numeric['family'] == 'F1'
    assert not manifest['declines'], manifest['declines']
    data = np.arange(itemsize(dtype),dtype=np.uint8)+1
    if dtype == 'i1': data[:] = 1
    actual = np.ascontiguousarray(simulate(match.kernel.stream,{'x0':data})['y']).view(np.uint8).reshape(-1)
    assert np.array_equal(actual,data)
    lint(match.kernel.stream)


def reshape_source(dtype, rank):
    target = 'x'.join(['1']*rank+[dtype])
    descriptor = f'dense<1> : tensor<{rank}xi64>' if rank else 'dense<> : tensor<0xi64>'
    return f'''builtin.module {{
      func.func @forward(%x: tensor<1x{dtype}>) -> tensor<{target}> {{
        %shape = arith.constant {descriptor}
        %y = tensor.reshape %x(%shape) : (tensor<1x{dtype}>, tensor<{rank}xi64>) -> tensor<{target}>
        func.return %y : tensor<{target}>
      }}
    }}'''


def test_float_add_zero_is_not_bitwise_movement():
    text = '''builtin.module {
      func.func @forward(%x: tensor<3xf32>) -> tensor<3xf32> {
        %e = tensor.empty() : tensor<3xf32>
        %y = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
          ins(%x : tensor<3xf32>) outs(%e : tensor<3xf32>) {
          ^bb0(%a: f32, %b: f32):
            %z = arith.constant 0.0 : f32
            %v = arith.addf %a, %z : f32
            linalg.yield %v : f32
          } -> tensor<3xf32>
        func.return %y : tensor<3xf32>
      }
    }'''
    _, manifest, matches = build(text)
    assert not matches and not manifest['bindings']


def test_native_numeric_widening_has_a_clean_refusal_reason():
    text = '''builtin.module {
      func.func @forward(%x: tensor<3xf32>) -> tensor<3xf64> {
        %e = tensor.empty() : tensor<3xf64>
        %y = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
          ins(%x : tensor<3xf32>) outs(%e : tensor<3xf64>) {
          ^bb0(%a: f32, %b: f64):
            %wide = arith.extf %a : f32 to f64
            linalg.yield %wide : f64
          } -> tensor<3xf64>
        func.return %y : tensor<3xf64>
      }
    }'''
    _, manifest, matches = build(text)
    assert not matches
    assert any(d['reason'] == 'no exact integer schedule for unquantized data arithmetic or numeric conversion'
               for d in manifest['declines'])


@pytest.mark.parametrize('dtype', ['i8', 'f16', 'i64'])
def test_native_selector_is_bound_once_including_byte_elements(dtype):
    text = f'''builtin.module {{
      func.func @forward(%x: tensor<3x{dtype}>, %y: tensor<2x{dtype}>) -> tensor<5x{dtype}> {{
        %z = tensor.concat dim(0) %x, %y : (tensor<3x{dtype}>, tensor<2x{dtype}>) -> tensor<5x{dtype}>
        func.return %z : tensor<5x{dtype}>
      }}
    }}'''
    _, manifest, matches = build(text)
    assert len(matches) == 1, manifest
    assert manifest['bindings'][0]['numeric']['family'] == 'F1'


@pytest.mark.parametrize('dtype', ['bf16', 'i64', 'complex<f32>'])
def test_shim_offsets_and_strides_remain_in_logical_elements(dtype, tmp_path):
    width = itemsize(dtype)
    args = [dict(name='x', dtype=dtype, shape=[2, 3], role='input'),
            dict(name='y', dtype=dtype, shape=[2, 3], role='output')]
    source = shim_source({'kernels': [dict(symbol='copy_bytes', args=args)]})
    source += f'''
#include <stdlib.h>
#include <string.h>
void ca_abort(const char *s) {{ abort(); }}
void copy_bytes(void *x, void *y) {{ memcpy(y, x, {6 * width}); }}
int main(void) {{
  unsigned char x[128], y[128];
  for(int i=0;i<128;i++) {{ x[i]=(unsigned char)i; y[i]=0x5a; }}
  ca_memref_2 a={{x,x,3,{{2,3}},{{3,1}}}}, b={{y,y,5,{{2,3}},{{3,1}}}}, result;
  _mlir_ciface_ca_copy_bytes(&result,&a,&b);
  for(int i=0;i<128;i++) {{
    unsigned char want=(i>={5 * width} && i<{11 * width}) ? x[i-{2 * width}] : 0x5a;
    if(y[i]!=want) return 1;
  }}
  return result.offset!=5 || result.aligned!=y;
}}
'''
    path = tmp_path / 'offset.c'
    path.write_text(source)
    subprocess.run(['cc', '-std=c99', str(path), '-o', str(tmp_path / 'offset')], check=True)
    subprocess.run([str(tmp_path / 'offset')], check=True)
