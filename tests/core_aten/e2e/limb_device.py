"""Standalone F2/F4 component checks on Spike; exact splitting is test-only.

Uses the real model2mlir _int_mm fixture. All device results are compared as
full i32, then the Python test driver proves wide recombination with integers.
The driver does not provide a production host representation implementation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import ModuleOp

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'device'))
from test_limbs import FIXTURE, annotate, linear_source, split
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.core_aten.stream import simulate
import runner


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    combined, symbols = ModuleOp([]), set()
    calls, checks, buffers, outputs, records = [], [], {}, set(), []

    def component(label, text, feed, expected):
        module, manifest, matches = build(text)
        match, = [m for m in matches if 'limb_plan' in m.numeric]
        binding = next(b for b in manifest['bindings'] if b['numeric'].get('limb_plan'))
        assert np.array_equal(simulate(match.kernel.stream, feed)['out'], expected)
        args = []
        for arg in match.kernel.args:
            name = f'{label}_{arg.name}'
            buffers[name] = feed[arg.name] if arg.role == 'input' else np.zeros(arg.shape, np.int32)
            args.append(name)
            if arg.role == 'output':
                outputs.add(name)
                checks.append(runner.Check(name, expected.astype(np.int32), 0, label))
        calls.append((binding['symbol'], args))
        for op in list(module.body.blocks[0].ops):
            if op.sym_name.data not in symbols:
                symbols.add(op.sym_name.data)
                op.detach()
                combined.body.blocks[0].add_op(op)
        records.append(dict(label=label, numeric=binding['numeric']))

    rng = np.random.default_rng(481)
    native_checks = []
    for bits in (32, 64):
        left = rng.integers(-(1 << (bits-1)), (1 << (bits-1))-1, (3, 17), dtype=np.int64).astype(object)
        right = rng.integers(-(1 << (bits-1)), (1 << (bits-1))-1, (17, 5), dtype=np.int64).astype(object)
        left[0, 0], right[0, 0] = -(1 << (bits-1)), (1 << (bits-1))-1
        result = np.zeros((3, 5), object)
        for j, a in enumerate(split(left)):
            for k, b in enumerate(split(right)):
                raw = a.astype(np.int32) @ b.astype(np.int32)
                text = annotate(FIXTURE.read_text(), indices=(j, k))
                component(f'i{bits}_{j}_{k}', text, dict(a=a, b=b), raw)
                result += raw.astype(object)*256**(j+k)
        assert np.array_equal(result, left @ right)
        native_checks.append(f'int{bits}_exact_big_integer_recombination_and_modulo_wrap')

    # Dyadic float components and the programmatic four-product complex rule.
    # These deliberately need several limbs, rather than losslessly fitting i8.
    real = [rng.integers(-65536, 65536, s, dtype=np.int64).astype(object)
            for s in ((3, 17), (3, 17), (17, 5), (17, 5))]
    results = []
    for name, li, ri in (('rr', 0, 2), ('ii', 1, 3), ('ri', 0, 3), ('ir', 1, 2)):
        result = np.zeros((3, 5), object)
        for j, a in enumerate(split(real[li])):
            for k, b in enumerate(split(real[ri])):
                raw = a.astype(np.int32) @ b.astype(np.int32)
                text = annotate(FIXTURE.read_text(), indices=(j, k), exponents=(-16, -8), family='F4', component=name)
                component(f'complex_{name}_{j}_{k}', text, dict(a=a, b=b), raw)
                result += raw.astype(object)*256**(j+k)
        assert np.array_equal(result, real[li] @ real[ri])
        results.append(result)
    assert np.array_equal(results[0]-results[1], real[0]@real[2]-real[1]@real[3])
    assert np.array_equal(results[2]+results[3], real[0]@real[3]+real[1]@real[2])
    native_checks.append('four_real_imag_dyadic_products_exact_before_final_rounding')

    for reduction in (False, True):
        for c in (-254, -3, 0, 1, 128):
            text = annotate(linear_source(c, reduction), indices=(0,), exponents=(0,))
            x = np.resize(np.array([-128, 127, 0, -1, 1], np.int8), (3, 5) if reduction else (19,))
            expected = x.astype(np.int32)*c
            expected = expected.sum(1)+7 if reduction else expected+c
            component(f'linear_{int(reduction)}_{"neg" if c < 0 else "pos"}{abs(c)}', text, dict(x0=x), expected)

    # The same component interface from native finite half-format payloads.
    import torch
    generator = torch.Generator().manual_seed(151)
    for dtype_name in ('float16', 'bfloat16'):
        dtype = getattr(torch, dtype_name)
        tensors = [(torch.randint(-1500, 1501, shape, generator=generator).double()/512).to(dtype)
                   for shape in ((3, 17), (17, 5))]
        exact_inputs = [np.vectorize(int, otypes=[object])(t.double().numpy()*512) for t in tensors]
        result = np.zeros((3, 5), object)
        for j, a in enumerate(split(exact_inputs[0])):
            for k, b in enumerate(split(exact_inputs[1])):
                raw = a.astype(np.int32) @ b.astype(np.int32)
                component(f'{dtype_name}_{j}_{k}',
                          annotate(FIXTURE.read_text(), indices=(j, k), exponents=(-9, -9)), dict(a=a, b=b), raw)
                result += raw.astype(object)*256**(j+k)
        assert np.array_equal(result, exact_inputs[0]@exact_inputs[1])
        native_checks.append(dtype_name+'_exact_dyadic_recombination')
    a = np.resize(np.array([-128, 127, 0], np.int8), (2, 3, 17))
    b = np.resize(np.array([127, -128, 1], np.int8), (2, 17, 5))
    component('batched', annotate((FIXTURE.parent/'bmm.mlir').read_text()), dict(a=a, b=b), a.astype(np.int32)@b.astype(np.int32))

    obj = compile_to_object(combined, root / 'build')['object']
    test = runner.DeviceTest('limb_device', [Path(obj)], buffers, calls, checks, outputs)
    elf = runner.build_elf(test, root / 'build')
    from mlir_oot.golden_device_catalog import final_elf_audit
    final_elf_audit(elf)
    actual = runner.run_spike(elf)
    (root/'spike.log').write_text(actual.log)
    result = dict(elf=str(elf.resolve()), cases=records, native_checks=native_checks,
                  spike=dict(status=actual.status, passed=len(actual.passed), failed=actual.failed))
    (root/'results.json').write_text(json.dumps(result, indent=2))
    print(result['spike'])
    return actual.ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    raise SystemExit(0 if run(parser.parse_args().out) else 1)
