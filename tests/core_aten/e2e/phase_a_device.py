"""Replay source-bound kernels of the 38 Phase-A captures with an exact oracle.

This test driver selects captured source files only. Production scheduling never
uses these operation labels, inputs, or expected values.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import ModuleOp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'device'))
from exact_oracle import SourceOracle, evaluate
import runner
from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.stream import simulate
from mlir_oot.golden_device_catalog import build_catalog, final_elf_audit
from mlir_oot.core_aten.emit import compile_to_object


def run(captures, support, out):
    out.mkdir(parents=True, exist_ok=True)
    combined, symbols = ModuleOp([]), set()
    buffers, calls, outputs, checks, records = {}, [], set(), [], []
    for ordinal, aten in enumerate(support.read_text().splitlines()):
        if not aten.strip():
            continue
        slug = aten.replace('.', '__')
        files = list(captures.glob(f'part*/shards/{slug}/capture-bundle/model.mlir'))
        assert len(files) == 1, (aten, files)
        path, = files
        text = path.read_text()
        source = parse_module(text)
        operations = list(source.walk())
        fn = next(op for op in operations if op.name == 'func.func')
        data = np.load(path.with_name('inputs.npz'))
        oracle = SourceOracle({arg: np.asarray(data[f'in{i}'], object) for i, arg in enumerate(fn.body.block.args)})
        module, manifest, matches = build(text)
        from mlir_oot.core_aten.integer_regions import has_integer_readout
        golden_module, golden_manifest = build_catalog(text)
        use_golden = golden_manifest['covered_contractions'] and not has_integer_readout(source)
        record = dict(aten=aten, source=str(path), schema=manifest['schema'], kernels=manifest['kernels'],
                      declines=manifest['declines'])
        if matches and not use_golden:
            for k, (binding, match) in enumerate(zip(manifest['bindings'], matches)):
                feed, argv = {}, []
                for arg in match.kernel.args:
                    name = f'c{ordinal}_{k}_{arg.name}'
                    dt = np.int32 if arg.dtype == 'i32' else np.int8
                    if arg.role == 'input':
                        where = binding['operands'][arg.name]
                        value = fn.body.block.args[where['func_arg']] if 'func_arg' in where else operations[where['op_ordinal']].results[where['result']]
                        feed[arg.name] = np.asarray(oracle.tensor(value), dtype=dt).reshape(arg.shape)
                        buffers[name] = feed[arg.name]
                    else:
                        buffers[name] = np.zeros(arg.shape, dt)
                        outputs.add(name)
                    argv.append(name)
                actual = simulate(match.kernel.stream, feed)
                exact = evaluate(match, feed)
                for name, expected in exact.items():
                    assert np.array_equal(actual[name], expected), (aten, name, actual[name], expected)
                for arg in match.kernel.args:
                    if arg.role != 'input':
                        expected = exact.get(arg.name, actual[arg.name])
                        checks.append(runner.Check(f'c{ordinal}_{k}_{arg.name}', np.asarray(expected).astype(buffers[f'c{ordinal}_{k}_{arg.name}'].dtype), 0,
                                                   f'{slug}_{k}_{arg.name}'))
                calls.append((binding['symbol'], argv))
        else:
            # The same contraction catalog used for convolution backwards.
            module, manifest = golden_module, golden_manifest
            assert manifest['covered_contractions'] > 0, (aten, manifest)
            record.update(schema=manifest['schema'], kernels=manifest['kernels'])
            for k, binding in enumerate(manifest['bindings']):
                op = operations[binding['source_operation_ordinal']]
                a, b = [np.asarray(oracle.tensor(v), np.int8) for v in op.operands[:2]]
                expected = a.astype(np.int32)@b.astype(np.int32)
                argv = [f'c{ordinal}_{k}_{name}' for name in ('a','b','out')]
                buffers.update(zip(argv, (a, b, np.zeros(expected.shape, np.int32))))
                outputs.add(argv[-1])
                checks.append(runner.Check(argv[-1], expected, 0, slug+str(k)))
                calls.append((binding['symbol'], argv))
        arithmetic = any(k.get('counts',{}).get('compute',0) or k.get('meta',{}).get('arithmetic') == 'integer_accumulating_mvin' or
                         k.get('op','') in ('aten.mean.dim',) for k in record['kernels'])
        if record['schema'] == 'gemmini_golden_device_catalog_v1':
            arithmetic = True
        record['classification'] = 'device arithmetic' if arithmetic else 'device movement + exact host epilogue'
        records.append(record)
        for op in list(module.body.block.ops):
            symbol = op.sym_name.data
            if symbol not in symbols:
                symbols.add(symbol)
                op.detach()
                combined.body.block.add_op(op)
        print(aten, record['classification'], flush=True)
    from mlir_oot.golden_device_lower import lower
    receipt = compile_to_object(lower(combined), out/'build')
    test = runner.DeviceTest('phase_a', [out/'build/kernel.o'], buffers, calls, checks, outputs)
    elf = runner.build_elf(test, out/'build')
    final_elf_audit(elf)
    result = runner.run_spike(elf)
    (out/'spike.log').write_text(result.log)
    report = dict(cases=records, elf=str(elf.resolve()), compilation=receipt,
                  spike=dict(status=result.status, passed=len(result.passed), failed=result.failed))
    (out/'results.json').write_text(json.dumps(report, indent=2))
    print(report['spike'])
    return result.ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path)
    parser.add_argument('support', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.captures, args.support, args.out) else 1)
