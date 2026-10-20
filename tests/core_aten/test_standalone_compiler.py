"""Standalone compiler preserves the public entrypoints without harness imports."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def exporter():
    spec = importlib.util.spec_from_file_location('phase1_export', ROOT/'tools/export_phase1_backend.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_vendored_helpers_are_exact_extractions():
    from mlir_oot._vendor import pure_tensor_dce, llvm_loop_metadata
    snapshot = Path(os.environ.get('MERLIN_INTEGRITY_SNAPSHOT', '/scratch/loren/core-aten-ref/merlin-wtB/out/codex-dev/run13/snapshot/merlin'))
    if not snapshot.is_dir():
        pytest.skip('frozen provenance source unavailable')
    base = snapshot/'src/merlin/llvmlower'
    original = ast.parse((base/'entry_weight_projection.py').read_text())
    local = ast.parse(Path(pure_tensor_dce.__file__).read_text())
    for name in ('_pure_result_operation', '_pure_dce'):
        a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in local.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b)
    assert ast.dump(ast.parse((base/'llvm_loop_metadata.py').read_text())) == ast.dump(ast.parse(Path(llvm_loop_metadata.__file__).read_text()))


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_COMPILER_PYTHON', os.environ.get('MERLIN_COMPILER_PYTHON', sys.executable))
    return exporter().export(tmp_path/'compiler')


def test_export_records_interpreter_and_venv_context(package):
    import hashlib
    import yaml
    metadata = yaml.safe_load((package/'manifest.yaml').read_text())['python_interpreter']
    selected = os.environ['MERLIN_COMPILER_PYTHON']
    assert metadata['path'] == selected
    assert metadata['sha256'] == hashlib.sha256(Path(selected).read_bytes()).hexdigest()
    probe = subprocess.check_output([selected, '-I', '-c',
        'import json,sys; print(json.dumps([sys.version,sys.prefix,sys.base_prefix]))'], text=True)
    assert [metadata[k] for k in ('version', 'prefix', 'base_prefix')] == json.loads(probe)
    assert metadata['flags'] == ['-I']
    assert (package/'gemmini-opt').read_text().splitlines()[0] == f'#!{selected} -I'
    assert json.loads((package/'source-manifest.json').read_text())['python_interpreter'] == metadata


@pytest.mark.parametrize('selected', [None, 'python3'])
def test_export_requires_explicit_absolute_interpreter(tmp_path, monkeypatch, selected):
    monkeypatch.delenv('MERLIN_COMPILER_PYTHON', raising=False)
    if selected is not None:
        monkeypatch.setenv('MERLIN_COMPILER_PYTHON', selected)
    with pytest.raises(ValueError, match='explicitly name an absolute'):
        exporter().export(tmp_path/'compiler')
    assert not (tmp_path/'compiler').exists()


def test_export_runs_from_clean_environment_with_chipyard_python_first(package, tmp_path, matmul_input):
    # The actual failing deployment environment, or a dependency-free Python
    # stand-in on installations without Chipyard. Never invoke the tool through
    # sys.executable: that would bypass the shebang this regression exercises.
    conda = Path('/scratch2/loren/chipyard/.conda-env/bin')
    if not (conda/'python3').exists():
        conda = tmp_path/'wrong-python'; conda.mkdir()
        (conda/'python3').write_text('#!/bin/sh\nexit 99\n')
        (conda/'python3').chmod(0o755)
    env = {'PATH': str(conda)+':/usr/bin:/bin'}
    wrong = subprocess.run(['python3', '-c', 'import xdsl'], env=env, capture_output=True)
    assert wrong.returncode != 0, 'regression environment must lack xdsl on PATH'
    cb = tmp_path/'commands.json'
    args = ['--convert-iface-to-gemmini', '--emit-target-artifact',
            f'--emit-command-buffer={cb}', str(matmul_input)]
    run = subprocess.run([str(package/'gemmini-opt'), *args], env=env, cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    baseline = subprocess.run([os.environ['MERLIN_COMPILER_PYTHON'], '-I',
                               str(package/'gemmini-opt'), *args], env=env, cwd=tmp_path,
                              capture_output=True, text=True)
    assert baseline.returncode == 0, baseline.stderr
    assert run.stdout == baseline.stdout
    assert 'llvm.func' in run.stdout
    assert json.loads(cb.read_text())['commands']
    assert not list(package.rglob('__pycache__'))


def test_export_copies_complete_import_closure_without_rewriting(package):
    import hashlib
    receipt=json.loads((package/'source-manifest.json').read_text())
    assert receipt['source_rewriting'] is False
    for path,sha in receipt['files'].items():
        assert (package/path).read_bytes() == (ROOT/path).read_bytes()
        assert hashlib.sha256((package/path).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError,match='must not exist'):
        exporter().export(package)


@pytest.mark.parametrize('name', ['full', 'clone', 'add_Tensor', 'mean_dim', 'convolution'])
def test_public_cli_artifacts_and_command_buffers_are_byte_identical(package,tmp_path,name):
    # The same public Pipeline, over actual captured fixtures, through both CLIs.
    from fixture_source import read
    source=tmp_path/'input.mlir'; source.write_text(read(ROOT/'tests/fixtures/core_aten'/name/'model.mlir'))
    results=[]
    for label,tool in [('host',ROOT/'gemmini-opt'),('standalone',package/'gemmini-opt')]:
        cb=tmp_path/(label+'.json')
        run=subprocess.run([sys.executable,str(tool),'--convert-iface-to-gemmini','--emit-target-artifact',f'--emit-command-buffer={cb}',str(source)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
        results.append((run.stdout,cb.read_bytes()))
    assert results[0]==results[1]


def test_standalone_native_catalog_and_cli_do_not_import_merlin(package,tmp_path):
    from fixture_source import read
    source=tmp_path/'input.mlir';source.write_text(read(ROOT/'tests/fixtures/core_aten/mean_dim/model.mlir'))
    code='''import sys,importlib.abc,runpy
class NoHarness(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname == "merlin" or fullname.startswith("merlin."):
   raise RuntimeError("compiler attempted harness import: "+fullname)
sys.meta_path.insert(0,NoHarness())
sys.path.insert(0,sys.argv[1])
from mlir_oot.core_aten.catalog import build
_,manifest,matches=build(open(sys.argv[2]).read())
assert matches,manifest['declines']
sys.argv=[sys.argv[1]+"/gemmini-opt","--verify-diagnostics",sys.argv[2]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    run=subprocess.run([sys.executable,'-c',code,str(package),str(source)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_pipeline_refactor_is_identical_to_vcs_engine_head():
    old = subprocess.check_output(['git', '--git-dir='+str(ROOT/'.codex-side.git'),
        '--work-tree='+str(ROOT), 'show', '16a3db2:mlir_oot/gemmini_opt.py'], text=True)
    before=ast.parse(old)
    after=ast.parse((ROOT/'mlir_oot/compiler_cli.py').read_text())
    for name in ('Pipeline', '_ExtensionPlan', '_print'):
        a=next(n for n in before.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
        b=next(n for n in after.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
        assert ast.dump(a)==ast.dump(b)


@pytest.fixture
def matmul_input(tmp_path):
    source=tmp_path/'matmul.mlir'
    source.write_text('''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
 %a = merlin_iface.tensor {name = "a", role = "input"} : tensor<16x16xi8>
 %b = merlin_iface.tensor {name = "b", role = "weight"} : tensor<16x16xi8>
 %r = merlin_iface.resident_pack %b {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
 %acc = merlin_iface.matmul %a, %r {name = "acc"} : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
 %y = merlin_iface.commit %acc {name = "y", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi32>
 merlin_iface.evict %r : (!merlin_iface.resident) -> ()
}''')
    return source


def test_standalone_compiles_general_single_tile_matmul_interface(package,tmp_path,matmul_input):
    cb=tmp_path/'matmul.json'
    run=subprocess.run([sys.executable,str(package/'gemmini-opt'),'--convert-iface-to-gemmini','--emit-target-artifact',f'--emit-command-buffer={cb}',str(matmul_input)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    commands=json.loads(cb.read_text())['commands']
    assert any(c['opcode']=='MATMUL_RESIDENT' for c in commands)
    assert 'llvm.func' in run.stdout


def test_export_passes_unmodified_frozen_integrity_policy(package):
    snapshot=Path(os.environ.get('MERLIN_INTEGRITY_SNAPSHOT','/scratch/loren/core-aten-ref/merlin-wtB/out/codex-dev/run13/snapshot/merlin'))
    if not snapshot.is_dir():
        pytest.skip('frozen integrity policy unavailable')
    env=dict(os.environ,PYTHONPATH=str(snapshot/'src')+os.pathsep+str(snapshot))
    code='''import sys
from pathlib import Path
from merlin.targetgen.package_runtime import load_package,integrity_scan,CertFailure
p=load_package(Path(sys.argv[1]))
assert p.integrity_exempt is False
integrity_scan(p)
probe=p.directory/"bad_import.py"
try:
 probe.write_text("import merlin.llvmlower\\n")
 try:integrity_scan(p)
 except CertFailure:pass
 else:raise AssertionError("forbidden import was accepted")
finally:probe.unlink()
integrity_scan(p)
'''
    run=subprocess.run([sys.executable,'-c',code,str(package)],env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
