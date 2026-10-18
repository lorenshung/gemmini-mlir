"""Replay normalized scalar reduction captures through region offload and Spike.

Usage: python scalar_mean_islands.py OUT SHARD_DIR [SHARD_DIR ...]
Each shard supplies model.mlir and inputs.npz. No Verilator is invoked.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent/'device'))
from run_offload import build_case
import runner
from test_scalar_mean_island import provider, PACKAGE, POLICY
from mlir_oot.frontend.parse import parse_module


@contextmanager
def upstream_yield_serialization():
    # Older upstream MLIR rejects xDSL's attributes-before-operands yield
    # spelling. Only provenance on the no-result terminator is removed, after
    # source binding/rewrite. Arithmetic, values, types and numeric metadata on
    # every producer remain unchanged.
    original = subprocess.run
    def run(argv, *args, **kwargs):
        if Path(argv[0]).name == 'mlir-opt':
            path = Path(argv[1])
            module = parse_module(path.read_text())
            for op in module.walk():
                if op.name == 'linalg.yield':
                    for key in list(op.attributes):
                        if key.startswith('prov.'):
                            del op.attributes[key]
            path.write_text(str(module)+'\n')
        return original(argv, *args, **kwargs)
    with patch('subprocess.run', run):
        yield


def run(shards, out):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    api = provider()
    for i, shard in enumerate(shards):
        source = (shard/'model.mlir').read_text()
        work = out/str(i)
        case = work/'capture'
        case.mkdir(parents=True, exist_ok=True)
        path = case/'model.mlir'
        path.write_text(source)
        proof = api.admit_numeric_schedule(source, package=PACKAGE, numeric_policy=POLICY)
        assert proof['status'] == 'proven', proof
        manifest, obj = api.core_aten_catalog(path, work/'provider', package=PACKAGE)
        assert obj is not None and obj.is_file()
        data = np.load(shard/'inputs.npz')
        def position(name):
            return int(name.removeprefix('arg').removeprefix('in'))
        names = sorted(data.files, key=position)
        np.savez(case/'inputs.npz', **{f'arg{k}': data[n] for k,n in enumerate(names)})
        with upstream_yield_serialization():
            elf = build_case(case, work/'build')
        result = runner.run_spike(elf)
        (work/'spike.log').write_text(result.log)
        records.append(dict(source=str(shard/'model.mlir'),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(), proof=proof,
            manifest=str(manifest), object=str(obj), elf=str(elf), status=result.status,
            passed=len(result.passed), failed=result.failed))
        print(shard.name, result.status, len(result.passed), flush=True)
    (out/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    return all(r['status'] == 'PASS' for r in records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('shards', type=Path, nargs='+')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.shards, args.out) else 1)
