"""Paired host-only timing and full-kernel correctness for captured Phase 0 IR.

Use a baseline host_readout.py extracted from the base revision, without adding
an algorithm selector to production. Frozen sources and exact scale/shape
contracts select the kernels; generated deterministic codes are benchmark data.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from statistics import median
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tests/core_aten/device'), str(ROOT/'tests/core_aten')]
import runner
from exact_oracle import evaluate as oracle
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import build_module, compile_to_object
from mlir_oot.core_aten.stream import simulate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rtl', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('baseline_host_readout', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    objects, buffers, calls, checks, outputs = [], {}, [], [], set()
    records = []
    for ordinal, (name, part) in enumerate([('aten__mean__dim', 7), ('aten__avg_pool2d__default', 7), ('aten__add__Tensor', 4)]):
        source = args.source_root/f'part{part}/shards/{name}/model.mlir'
        module, manifest, matches = build(source.read_text())
        assert len(matches) == 1, manifest['declines']
        kernel = matches[0].kernel
        rng = np.random.default_rng(173 + ordinal)
        inputs = {a.name: rng.integers(-128, 128, a.shape, dtype=np.int8)
                  for a in kernel.args if a.role == 'input'}
        model = simulate(kernel.stream, inputs)
        expected = oracle(matches[0], inputs)
        assert all(np.array_equal(model[n], v) for n, v in expected.items())
        record = dict(case=name, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      meta=manifest['kernels'][0]['meta'], elements=sum(v.size for v in expected.values()))
        records.append(record)
        for variant in ('full', 'baseline', 'magic'):
            symbol = f'e{ordinal}_{variant}'
            stream = kernel.stream if variant == 'full' else replace(kernel.stream, ins=[], packs=[],
                       readouts=[replace(r, after=None) for r in kernel.stream.readouts])
            if variant == 'baseline':
                with patch('mlir_oot.core_aten.host_readout.emit', baseline.emit):
                    mod, _ = build_module(stream, symbol, meta=dict(kernel.meta))
            else:
                mod, _ = build_module(stream, symbol, meta=dict(kernel.meta))
            obj = compile_to_object(mod, args.out, stem=symbol)['object']
            objects.append(Path(obj))
            # Check the emitted epilogue functions independently of any runtime.
            dis = subprocess.check_output([str(Path(__import__('os').environ['GEMMINI_LLVM_BIN'])/'llvm-objdump'), '-d', obj], text=True)
            (args.out/f'{symbol}.dis').write_text(dis)
            mnemonics = re.findall(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2,8}\s+)+([a-z][a-z0-9.]*)', dis, re.M)
            assert mnemonics and not any(m.startswith('f') and m not in ('fence', 'fence.i') for m in mnemonics)
            bnames = []
            for a in kernel.args:
                bn = f'{symbol}_{a.name}'
                bnames.append(bn)
                dt = np.int32 if a.dtype == 'i32' else np.int8
                buffers[bn] = (model[a.name] if variant != 'full' else inputs.get(a.name, np.zeros(a.shape, dt))).copy()
                if a.role == 'output':
                    outputs.add(bn)
                    checks.append(runner.Check(bn, expected[a.name].reshape(-1), 0, f'{symbol}_exact'))
            calls.append((symbol, bnames))
    test = runner.DeviceTest('host_requant', objects, buffers, calls, checks, outputs)
    elf = runner.build_elf(test, args.out)
    # Replace the harness's aggregate timing with warm, repeated paired timings.
    cpath = args.out/'main.c'
    text = cpath.read_text()
    # Machine-mode counters also work when Spike's ISA omits Zicntr.
    text = text.replace('#include "ca_harness.h"', '#include "ca_harness.h"\n'
                        'static uint64_t host_cycles(void) { uint64_t x; '
                        'asm volatile("csrr %0, mcycle" : "=r"(x)); return x; }')
    begin = text.index('  uint64_t t0 = ca_rdcycle();')
    end = text.index('  ca_check_', begin)
    timing = []
    for sym, bn in calls:
        call = f'{sym}({", ".join("ca_"+n for n in bn)});'
        timing.append('  '+call)
        if not sym.endswith('_full'):
            timing += ['  for (int trial=0; trial<5; ++trial) {',
                       '    uint64_t t0=host_cycles();',
                       f'    for (int rep=0; rep<100; ++rep) {{ {call} }}',
                       '    ca_fence(); uint64_t t1=host_cycles();',
                       f'    printf("PERF {sym} cycles=%lu\\n", (unsigned long)(t1-t0));', '  }']
    text = text[:begin]+'\n'.join(timing)+'\n'+text[end:]
    # build_elf generates sources, so preserve the generated driver with an
    # explicit link-command wrapper for the second link.
    def write(*unused):
        cpath.write_text(text)
        return cpath, args.out/'blobs.S'
    with patch.object(runner, 'write_sources', write):
        elf = runner.build_elf(test, args.out)
    report = dict(cases=records, repetitions=100, trials=5, elf=str(elf),
                  elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  measurement='host epilogues only; generated reachable codes; warmed 100 calls/trial; full kernels checked once')
    for engine, fn in [('spike', runner.run_spike)] + ([('rtl', runner.run_rtl)] if args.rtl else []):
        result = fn(elf)
        (args.out/f'{engine}.log').write_text(result.log)
        timings = {}
        for sym, cycles in re.findall(r'PERF (\S+) cycles=(\d+)', result.log):
            timings.setdefault(sym, []).append(int(cycles))
        savings = []
        for ordinal, record in enumerate(records):
            old, new = (median(timings[f'e{ordinal}_{v}'])/100 for v in ('baseline', 'magic'))
            savings.append(dict(case=record['case'], baseline_cycles_per_call=old,
                                magic_cycles_per_call=new, cycles_saved_per_call=old-new,
                                reduction_percent=100*(old-new)/old))
        report[engine] = dict(status=result.status, passed=result.passed, failed=result.failed,
                              seconds=result.seconds, cycles_per_100_calls=timings, savings=savings)
        (args.out/'measurements.json').write_text(json.dumps(report, indent=2)+'\n')
        print(engine, report[engine], flush=True)
        assert result.ok, result.log[-3000:]


if __name__ == '__main__':
    main()
