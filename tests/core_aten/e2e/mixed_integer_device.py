"""Replay mixed raw/readout producers, including batching and extrema, on Spike."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from xdsl.dialects.builtin import ModuleOp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'device'))
from test_mixed_integer_regions import source
from exact_oracle import evaluate
import runner
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.golden_device_catalog import final_elf_audit


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    combined, symbols = ModuleOp([]), set()
    buffers, calls, checks, outputs = {}, [], [], set()
    for batch in (False, True):
        device, manifest, matches = build(source(batch))
        assert len(matches) == 2 and not manifest['declines']
        for pattern_id, (left, right) in enumerate(((-128, -128), (-128, 127),
                                                   (127, -128), (127, 127))):
            for ordinal, (binding, match) in enumerate(zip(manifest['bindings'], matches)):
                label = f'b{int(batch)}_v{pattern_id}_{ordinal}'
                feed = {name: np.full(value.type.get_shape(), left if name == 'a' else right, np.int8)
                        for name, value in match.operands}
                expected = evaluate(match, feed)['out']
                argv = []
                for arg in match.kernel.args:
                    name = label + '_' + arg.name
                    argv.append(name)
                    buffers[name] = feed.get(arg.name, np.zeros(arg.shape, np.int32))
                    if arg.role == 'output':
                        outputs.add(name)
                        checks.append(runner.Check(name, expected, 0, label))
                calls.append((binding['symbol'], argv))
        for op in list(device.body.block.ops):
            if op.sym_name.data not in symbols:
                symbols.add(op.sym_name.data)
                op.detach()
                combined.body.block.add_op(op)
    receipt = compile_to_object(combined, out / 'build')
    test = runner.DeviceTest('mixed_integer', [Path(receipt['object'])], buffers, calls, checks, outputs)
    elf = runner.build_elf(test, out / 'build')
    final_elf_audit(elf)
    result = runner.run_spike(elf)
    (out / 'spike.log').write_text(result.log)
    (out / 'results.json').write_text(json.dumps(dict(
        elf=str(elf.resolve()), status=result.status, passed=len(result.passed),
        failed=result.failed), indent=2) + '\n')
    print(result.status, len(result.passed), result.failed)
    return result.ok


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    raise SystemExit(0 if run(parser.parse_args().out) else 1)
