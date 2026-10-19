"""Do not materialize source views or a retained exact host epilogue with DMA."""
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import Arg, Stream, simulate
from mlir_oot.core_aten.lint import lint
from test_scalar_mean_island import provider, PACKAGE, POLICY


@pytest.mark.parametrize('width', [1, 2, 4, 8])
def test_singleton_strides_do_not_fragment_contiguous_byte_groups(width):
    from mlir_oot.core_aten.layout import move
    count = 15*width
    stream = Stream([Arg('x', 'i8', (count,), 'input'), Arg('y', 'i8', (count,), 'output')], bitwise=True)
    stream.fence(); stream.flush()
    move(stream, ('x', 0), ('y', 0), (1, 3, 1, 5, 1),
         (131071, 5*width, 8191, width, 997), width=width)
    stream.fence()
    lint(stream)
    data = np.arange(count, dtype=np.uint8).view(np.int8)
    assert np.array_equal(simulate(stream, dict(x=data))['y'], data)
    assert sum(ins.kind == 'mvin' for ins in stream.ins) <= 2
    assert sum(ins.kind == 'mvout' for ins in stream.ins) <= 2


def test_pure_byte_movement_keeps_its_complete_internal_view_schedule():
    source = '''builtin.module {
      func.func @forward(%x: tensor<1x1x5x5xi8>) -> tensor<1x3x3x1xi8> {
        %a = "tensor.extract_slice"(%x) <{static_offsets = array<i64: 0,0,1,0>, static_sizes = array<i64: 1,1,3,5>, static_strides = array<i64: 1,1,1,1>, operandSegmentSizes = array<i32: 1,0,0,0>}> : (tensor<1x1x5x5xi8>) -> tensor<1x1x3x5xi8>
        %b = "tensor.extract_slice"(%a) <{static_offsets = array<i64: 0,0,0,1>, static_sizes = array<i64: 1,1,3,3>, static_strides = array<i64: 1,1,1,1>, operandSegmentSizes = array<i32: 1,0,0,0>}> : (tensor<1x1x3x5xi8>) -> tensor<1x1x3x3xi8>
        %e = tensor.empty() : tensor<1x3x3x1xi8>
        %y = linalg.transpose ins(%b : tensor<1x1x3x3xi8>) outs(%e : tensor<1x3x3x1xi8>) permutation = [0,2,3,1]
        func.return %y : tensor<1x3x3x1xi8>
      }
    }'''
    from mlir_oot.core_aten.layout_match import internal_view
    module = parse_module(source)
    views = [op for op in module.walk() if op.name == 'tensor.extract_slice']
    assert len(views) == 2 and all(internal_view(op) for op in views)
    # Pure F1 has no retained integer epilogue: all three byte movements remain.
    _, manifest, matches = build(source)
    assert len(matches) == len(manifest['bindings']) == 3
    rewritten = parse_module(rewrite(source, manifest))
    assert sum(op.name == 'func.call' for op in rewritten.walk()) == 3


def test_retained_readout_layouts_stay_host_and_unrelated_movement_stays_device():
    text = (Path(__file__).parent/'fixtures/mixed_integer/tiny.mlir').read_text()
    text = text.replace('-> (tensor<2x2xi32>, tensor<2x2xf32>)',
                        '-> (tensor<2x2xi32>, tensor<2x2xf32>, tensor<2x2xi8>)')
    text = text.replace('    %zero =', '''    %a_flat = tensor.collapse_shape %a [[0,1]] : tensor<2x2xi8> into tensor<4xi8>
    %a_matrix = tensor.expand_shape %a_flat [[0,1]] output_shape [2,2] : tensor<4xi8> into tensor<2x2xi8>
    %zero =''')
    text = text.replace('ins(%a, %b :', 'ins(%a_matrix, %b :')
    epilogue = '''
    %flat = tensor.collapse_shape %out [[0,1]] : tensor<2x2xf32> into tensor<4xf32>
    %matrix = tensor.expand_shape %flat [[0,1]] output_shape [2,2] : tensor<4xf32> into tensor<2x2xf32>
    %e_tr = tensor.empty() : tensor<2x2xf32>
    %tr = linalg.transpose ins(%matrix : tensor<2x2xf32>) outs(%e_tr : tensor<2x2xf32>) permutation = [1,0]
    %e_other = tensor.empty() : tensor<2x2xi8>
    %other = linalg.transpose ins(%a : tensor<2x2xi8>) outs(%e_other : tensor<2x2xi8>) permutation = [1,0]
    func.return %raw, %tr, %other : tensor<2x2xi32>, tensor<2x2xf32>, tensor<2x2xi8>
    '''
    text = text.replace('func.return %raw, %out : tensor<2x2xi32>, tensor<2x2xf32>', epilogue)
    _, manifest, matches = build(text)
    assert len(matches) == 3, manifest['declines']
    movements = [m for m in matches if m.numeric.get('family') == 'F1']
    movement, = movements
    assert str(movement.result[1].type) == 'tensor<2x2xi8>'
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert proof['bindings'] == manifest['bindings']
    rewritten = parse_module(rewrite(text, manifest))
    assert sum(op.name == 'func.call' for op in rewritten.walk()) == 3
    assert any(op.name == 'arith.sitofp' for op in rewritten.walk())
    assert sum(op.name == 'linalg.transpose' for op in rewritten.walk()) == 1
    assert sum(op.name == 'tensor.collapse_shape' for op in rewritten.walk()) == 2
    assert sum(op.name == 'tensor.expand_shape' for op in rewritten.walk()) == 2
