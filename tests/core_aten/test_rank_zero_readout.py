"""Exact scalar parameter extraction and rank-zero closed readout regression."""
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest

from mlir_oot.core_aten.exactness import rational
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.stream import simulate
from conftest import provider, PACKAGE, POLICY


@pytest.mark.parametrize('value', [Q(1, 2), Q(-17, 31), Q(1, 1 << 149),
                                   (1 << 200)+1, np.int64(-7), 0])
def test_zero_dimensional_object_parameter_is_extracted_exactly(value):
    assert rational(np.array(value, dtype=object)) == Q(value)


@pytest.mark.parametrize('shape', [(1,), (1, 1), (2,), (0,)])
def test_non_scalar_parameter_is_not_silently_squeezed(shape):
    with pytest.raises(ValueError, match='must be scalar'):
        rational(np.full(shape, Q(1, 2), dtype=object))


def test_real_rank_zero_linear_island_has_exact_catalog_and_admission():
    text = (Path(__file__).parent/'fixtures/rank_zero/div.mlir').read_text()
    _, manifest, matches = build(text)
    assert len(matches) == 1, manifest['declines']
    match, = matches
    proof = provider().admit_numeric_schedule(text, package=PACKAGE, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert proof['bindings'] == manifest['bindings']
    for code in range(-128, 128):
        feed = {name: np.array(code, np.int8) for name, _ in match.operands}
        assert simulate(match.kernel.stream, feed)[match.result[0]].item() == round(Q(code, 2))


@pytest.mark.parametrize('shape', [(), (7,)])
def test_native_readout_accepts_scalar_object_arrays_in_all_parameter_roles(tmp_path, shape):
    import ctypes, os, subprocess
    from mlir_oot.core_aten.stream import Stream, Arg
    from mlir_oot.core_aten.requant import Readout
    from mlir_oot.core_aten.emit import build_module
    bindir = Path(os.environ.get('GEMMINI_LLVM_BIN', '/nonexistent'))
    if not (bindir/'mlir-translate').is_file():
        pytest.skip('LLVM tools unavailable')
    def scalar(v): return np.array(v, dtype=object)
    stream = Stream([Arg('x', 'i32', shape, 'input'), Arg('y', 'i8', shape, 'output')])
    stream.readouts.append(Readout('y', [('x', scalar(Q(-3, 8)))],
                                  constant=scalar(Q(1, 2)), zero_point=scalar(-7),
                                  quant_min=scalar(-128), quant_max=scalar(127)))
    meta = {}
    module, _ = build_module(stream, 'scalar_parameters', meta=meta)
    source, ir, obj, shared = (tmp_path/('scalar'+ext) for ext in ('.mlir', '.ll', '.o', '.so'))
    source.write_text(str(module)+'\n')
    subprocess.run([str(bindir/'mlir-translate'), '--mlir-to-llvmir', str(source), '-o', str(ir)], check=True)
    subprocess.run([str(bindir/'llc'), '-mtriple=x86_64-unknown-linux-gnu', '-relocation-model=pic',
                    '-O2', '-filetype=obj', str(ir), '-o', str(obj)], check=True)
    subprocess.run(['cc', '-shared', str(obj), '-o', str(shared)], check=True)
    fn = ctypes.CDLL(str(shared)).scalar_parameters
    fn.argtypes = [ctypes.c_void_p]*2
    for value in (-(1 << 31), -7, -3, 0, 1, 7, (1 << 31)-1):
        raw = np.full(shape, value, np.int32)
        result = np.zeros(shape, np.int8)
        fn(raw.ctypes.data, result.ctypes.data)
        expected = min(127, max(-128, round(Q(value)*Q(-3, 8)+Q(1, 2))-7))
        assert np.all(result == expected)
