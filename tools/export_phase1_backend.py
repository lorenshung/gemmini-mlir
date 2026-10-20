#!/usr/bin/env python3
"""Materialize the standalone compiler contract, including its complete local import closure.

Host orchestration (golden export/build hooks and simulator/oracle adapters) is
not a compiler entrypoint. No source is filtered or rewritten during export.
The grader scans the entire resulting package under its unchanged policy.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ('mlir_oot.compiler_cli', 'mlir_oot.core_aten.catalog')
COMMANDS = ('parse', 'lower_interface_to_target', 'emit_command_buffer', 'lower_target_to_llvm')


def compiler_interpreter():
    """Pin the selected venv invocation, not its symlink's system-Python target."""
    selected = os.environ.get('MERLIN_COMPILER_PYTHON', '')
    path = Path(selected)
    if not selected or not path.is_absolute():
        raise ValueError('MERLIN_COMPILER_PYTHON must explicitly name an absolute Python executable')
    # Resolving this path would discard the venv context when bin/python is a
    # symlink. The Linux shebang has one optional argument and a 256-byte limit.
    shebang = f'#!{path} -I\n'
    if any(c.isspace() for c in str(path)) or len(shebang.encode()) > 255:
        raise ValueError('MERLIN_COMPILER_PYTHON cannot be represented by an absolute Python shebang')
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('MERLIN_COMPILER_PYTHON is not an executable file')
    try:
        probe = subprocess.run([str(path), '-I', '-c',
            'import json, sys, xdsl; print(json.dumps(dict(version=sys.version, '
            'prefix=sys.prefix, base_prefix=sys.base_prefix)))'],
            check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('MERLIN_COMPILER_PYTHON must run Python with xdsl installed in its environment') from exc
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                flags=['-I'], **json.loads(probe.stdout))


def compiler_sources(root=ROOT):
    pending = list(ROOTS)
    selected = set()
    while pending:
        name = pending.pop()
        candidate = root.joinpath(*name.split('.'))
        path = candidate.with_suffix('.py')
        if not path.is_file():
            path = candidate/'__init__.py'
        if not path.is_file() or path in selected:
            continue  # An imported symbol, not a local module.
        selected.add(path)
        parts = name.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1, len(parts)))
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name('.'*node.level+(node.module or ''), package)
                        if node.level else node.module or '')
                names = [base]+[base+'.'+a.name for a in node.names]
            pending.extend(n for n in names if n == 'mlir_oot' or n.startswith('mlir_oot.'))
    return sorted(selected)


def export(destination, root=ROOT):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('export destination must not exist; never merge with stale package files')
    interpreter = compiler_interpreter()
    sources = compiler_sources(root)
    destination.mkdir(parents=True)
    hashes = {}
    for source in [*sources, root/'merlin-support/quant_schema.json', root/'merlin-support/quant_schema.lock.json', root/'mlir_oot/_vendor/LICENSE.Merlin']:
        relative = source.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    shim = destination/'gemmini-opt'
    shim.write_text(f'#!{interpreter["path"]} -I\nimport sys\nfrom pathlib import Path\nsys.dont_write_bytecode = True\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom mlir_oot.compiler_cli import main\nraise SystemExit(main())\n')
    shim.chmod(0o755)
    original = yaml.safe_load((root/'manifest.yaml').read_text())
    manifest = {key: original[key] for key in ('artifact_type', 'target', 'language', 'commands')}
    manifest['authoring'] = dict(mode='deterministic_generated_from_spec', author='Loren Hung')
    manifest['package_id'] = 'gemmini_phase1_compiler_v1'
    manifest['entrypoints'] = {'tool': 'gemmini-opt'}
    manifest['python_interpreter'] = interpreter
    manifest['commands'] = {key: manifest['commands'][key] for key in COMMANDS}
    owned = ['gemmini-opt', 'manifest.yaml', 'source-manifest.json', *hashes]
    manifest['components'] = {key: owned for key in COMMANDS}
    manifest['integrity_exempt'] = False
    (destination/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    (destination/'source-manifest.json').write_text(json.dumps(dict(
        interface='four_entrypoint_compiler_and_native_catalog', roots=ROOTS,
        files=hashes, source_rewriting=False, python_interpreter=interpreter), indent=2)+'\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(export(parser.parse_args().destination))
