"""Replay real f32/int8-recipe Phase 0 movement capsules, with per-case Spike costs.

Run this same driver with PYTHONPATH pointing at a baseline checkout to compare
identical source/input bytes. All source selection belongs to this experiment.
Host input quantization is evaluated by the independent rational source oracle;
the ELF calls each source-bound device kernel, including its generated epilogue.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'device'))
import runner
from exact_oracle import SourceOracle, evaluate
from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.core_aten.offload import rewrite, shim_source
from mlir_oot.core_aten.stream import ACC_BIT, simulate
from mlir_oot.golden_device_catalog import final_elf_audit

MOVEMENTS = ['clone', 'cat', 'constant_pad_nd', 'reflection_pad1d', 'reflection_pad2d',
             'reflection_pad3d', 'replication_pad2d', 'replication_pad3d', 'slice_scatter',
             'select_scatter', 'repeat', '_to_copy', 'full', 'full_like', 'fill']


def run(captures, out, design_rule, expect_codes):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in MOVEMENTS:
        paths = list(captures.glob(f'part*/shards/aten__{name}__*/capture-bundle/model.mlir'))
        assert len(paths) == 1, (name, paths)
        path, = paths
        text = path.read_text()
        wd = out / name
        wd.mkdir(exist_ok=True)
        (wd / 'source.mlir').write_text(text)
        module, manifest, matches = build(text)
        assert matches, manifest['declines']
        (wd / 'catalog.json').write_text(json.dumps(manifest, indent=2))
        (wd / 'offloaded.mlir').write_text(rewrite(text, manifest))
        (wd / 'shim.c').write_text(shim_source(manifest))
        source = parse_module(text)
        operations = list(source.walk())
        fn = next(o for o in operations if o.name == 'func.func')
        data = np.load(path.with_name('inputs.npz'))
        assert all(data[f'in{i}'].dtype == np.float32 for i in range(len(fn.body.block.args)))
        oracle = SourceOracle({arg: np.asarray(data[f'in{i}'], object)
                               for i, arg in enumerate(fn.body.block.args)})
        def value(where):
            return (fn.body.block.args[where['func_arg']] if 'func_arg' in where else
                    operations[where['op_ordinal']].results[where['result']])
        buffers, calls, checks, outputs = {}, [], [], set()
        for index, (binding, match) in enumerate(zip(manifest['bindings'], matches)):
            stream = match.kernel.stream
            if expect_codes:
                assert binding['numeric']['readout'] == 'storage_codes'
                assert not stream.readouts and not stream.packs
                assert not any(i.a('local') & ACC_BIT for i in stream.ins if i.kind in ('mvin', 'mvout'))
            feed, argv = {}, []
            for arg in match.kernel.args:
                key = f'b{index}_{arg.name}'
                dt = np.int32 if arg.dtype == 'i32' else np.int8
                if arg.role == 'input':
                    array = np.asarray(oracle.tensor(value(binding['operands'][arg.name])), np.int64).astype(dt)
                    feed[arg.name] = array
                else:
                    array = np.zeros(arg.shape, dt)
                buffers[key] = array
                argv.append(key)
                if arg.role == 'output':
                    outputs.add(key)
                    expected = np.asarray(oracle.tensor(value(binding['result'][arg.name])), np.int64).astype(dt)
                    checks.append(runner.Check(key, expected, 0, name + '_exact_codes'))
            actual = simulate(stream, feed)
            for key, expected in evaluate(match, feed).items():
                assert np.array_equal(actual[key], expected), name
            calls.append((binding['symbol'], argv))
        receipt = compile_to_object(module, wd, stem='kernel')
        # The audit's explicit ISA omits Zicntr; read the machine counter in
        # this M-mode harness so the exact mandated audit invocation works.
        harness = (HERE / 'device/ca_harness.h').read_text().replace('"rdcycle %0"', '"csrr %0, mcycle"')
        (wd / 'ca_harness.h').write_text(harness)
        test = runner.DeviceTest(name, [wd / 'kernel.o'], buffers, calls, checks, outputs)
        elf = runner.build_elf(test, wd)
        final_elf_audit(elf)
        result = runner.run_spike(elf)
        (wd / 'spike.log').write_text(result.log)
        assert result.ok, result.log[-2000:]
        subprocess.run(['python3', '-I', str(design_rule), str(elf), '268435456', str(wd / 'design_rule.json')], check=True)
        audit = json.loads((wd / 'design_rule.json').read_text())
        assert audit['design_rule'] == 'OK_full_i32', audit
        if expect_codes:
            assert audit['mvout']['acc_full'] == audit['mvout']['acc_narrow'] == 0, audit
        record = dict(name=name, source=str(path), source_sha256=manifest['source_sha256'],
                      inputs_sha256=hashlib.sha256(path.with_name('inputs.npz').read_bytes()).hexdigest(),
                      bindings=len(matches), storage_codes=all(m.numeric.get('readout') == 'storage_codes' for m in matches),
                      stream_instructions=sum(len(m.kernel.stream.ins) for m in matches),
                      spike_cycles=result.perf[name], gemmini_instructions=audit['gemmini_instructions'],
                      mvout=audit['mvout'], design_rule=audit['design_rule'], bit_exact=True,
                      elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(), compilation=receipt)
        records.append(record)
        (out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print(name, record['spike_cycles'], record['gemmini_instructions'], flush=True)
    return records


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('captures', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--design-rule', type=Path, required=True)
    ap.add_argument('--expect-storage-codes', action='store_true')
    args = ap.parse_args()
    run(args.captures, args.out.resolve(), args.design_rule, args.expect_storage_codes)
