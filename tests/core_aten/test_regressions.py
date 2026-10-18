"""ABI and shape regressions from the exactness review."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten import ops
from mlir_oot.core_aten.linmap import LinearMap
from mlir_oot.core_aten.linmatch import gather_jobs
from mlir_oot.core_aten.match import Decline
from mlir_oot.core_aten.offload import shim_source


@pytest.mark.parametrize('shape,dims', [((3, 0), [1]), ((0, 3), [0]), ((2, 3), []), ((), []), ((3,), [1]), ((3,), [0, -1])])
def test_reduction_declines_invalid_dimensions(shape, dims):
    with pytest.raises(Decline):
        ops.mean_dim(shape, dims, False, s_x=1, s_out=1)


def test_empty_gather_declines_before_stride_division():
    z = np.zeros(0, dtype=np.int64)
    lm = LinearMap((2, 0), z, z, z, z.astype(float), z.astype(float), {})
    with pytest.raises(Decline, match='empty'):
        gather_jobs(lm, {}, 1, 'y')


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
    _,manifest,matches=build(text)
    assert len(matches)==1,manifest['declines']
    match,=matches
    output=next(arg for arg in match.kernel.args if arg.role=='output')
    assert output.shape==(1665,)
    assert 'ca_memref_1 *result' in shim_source(manifest)
    inputs=dict(np.load(fixture/'inputs.npz'))
    feed={'a':inputs['arg0'],'b':inputs['arg1']}
    assert np.array_equal(simulate(match.kernel.stream,feed)['out'],evaluate(match,feed)['out'])
