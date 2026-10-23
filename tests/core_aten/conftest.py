"""Shared Core ATen test setup: helper-script imports and the execution provider under test."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
for helpers in ('device', 'e2e', 'iface_ext'):
    sys.path.insert(0, str(HERE / helpers))

POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32', internal_arithmetic=dict(
    signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
    full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def provider():
    """A fresh load of merlin-support/core_aten/execution_provider.py."""
    path = PACKAGE / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('core_aten_execution_provider', path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api



def synthetic_route(text, tmp_path, dtypes=('i8', 'i8', 'i32'), api=None):
    """The provider's region route for ``text`` under the synthetic Core ATen facts fixture."""
    from types import SimpleNamespace
    from merlin.targetgen.core_aten_device import selected_facts
    (tmp_path/'model.mlir').write_text(text)
    with selected_facts('synthetic_core_aten', PACKAGE/'merlin-support/tests/fixtures/core_aten_device/facts.json') as facts:
        return (api or provider()).routing(tmp_path, target='synthetic_core_aten', package=PACKAGE, facts=facts,
                                           eligible=[(None, SimpleNamespace(dtypes=dtypes))])


def native_readout(stream, symbol, tmp_path):
    """Compile ``build_module(stream, symbol)`` for this host; return its ctypes function and metadata."""
    import ctypes, os, subprocess, tempfile
    import pytest
    from mlir_oot.core_aten.emit import build_module
    bindir = Path(os.environ.get('GEMMINI_LLVM_BIN', '/nonexistent'))
    if not (bindir/'mlir-translate').exists():
        pytest.skip('LLVM tools unavailable')
    meta = {}
    module, _ = build_module(stream, symbol, meta=meta)
    work = Path(tempfile.mkdtemp(dir=tmp_path))
    source, ir, obj, shared = (work/(symbol+ext) for ext in ('.mlir', '.ll', '.o', '.so'))
    source.write_text(str(module)+'\n')
    subprocess.run([str(bindir/'mlir-translate'), '--mlir-to-llvmir', str(source), '-o', str(ir)], check=True)
    subprocess.run([str(bindir/'llc'), '-mtriple=x86_64-unknown-linux-gnu', '-relocation-model=pic',
                    '-O2', '-filetype=obj', str(ir), '-o', str(obj)], check=True)
    subprocess.run(['cc', '-shared', str(obj), '-o', str(shared)], check=True)
    fn = getattr(ctypes.CDLL(str(shared)), symbol)
    fn.argtypes = [ctypes.c_void_p]*len(stream.args)
    return fn, meta
