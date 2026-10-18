"""Real raw-dtype captures, byte oracle, provider wiring and optional Spike batch.

Experiment selections live here, never in production matching. Run with the
model2mlir venv and the device environment from e2e/README.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

DTYPES = ['float16', 'bfloat16', 'int32', 'int64', 'complex64', 'bool']
OPS = ['cat', 'clone', 'constant_pad_nd', 'reflection_pad1d', 'reflection_pad2d',
       'reflection_pad3d', 'replication_pad2d', 'replication_pad3d', 'repeat',
       'slice_scatter', 'select_scatter', 'full', 'full_like', 'fill_Scalar', '_to_copy']


def example(dtype, shape, seed=71):
    import torch
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    if dtype == 'bool':
        return torch.from_numpy(rng.integers(0, 2, shape).astype(bool))
    width = {'float16': 2, 'bfloat16': 2, 'int32': 4, 'int64': 8, 'complex64': 8}[dtype]
    bits = rng.integers(0, 256, n * width, dtype=np.uint8)
    # Exercise payloads and signed zero without computing on them.
    if dtype in ('float16', 'bfloat16'):
        patterns = [0, 0x8000, 1, 0x7fc1 if dtype == 'bfloat16' else 0x7e15]
        count = min(n, len(patterns))
        bits[:count * 2] = np.frombuffer(np.asarray(patterns[:count], dtype='<u2').tobytes(), dtype=np.uint8)
    if dtype == 'bfloat16':
        return torch.from_numpy(bits.view('<u2').reshape(shape)).view(torch.bfloat16)
    npdtype = {'float16': '<f2', 'int32': '<i4', 'int64': '<i8', 'complex64': '<c8'}[dtype]
    return torch.from_numpy(bits.view(npdtype).reshape(shape).copy())


def raw(tensor):
    import torch
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().view(np.int8).copy()


def case(name, dtype):
    import torch
    A = torch.ops.aten
    dt = getattr(torch, dtype)
    literal = (complex(-2, 0.75) if dtype == 'complex64' else True if dtype == 'bool'
               else -2 if dtype.startswith('int') else -0.75)
    x = example(dtype, (3, 5))
    table = {
        'cat': (lambda a, b: A.cat.default([a, b], 1), [x, example(dtype, (3, 2), 72)]),
        'clone': (lambda a: A.clone.default(a), [x]),
        'constant_pad_nd': (lambda a: A.constant_pad_nd.default(a, [-1, 2, 1, 0], literal), [x]),
        'repeat': (lambda a: A.repeat.default(a, [2, 2]), [x]),
        'slice_scatter': (lambda a, b: A.slice_scatter.default(a, b, 1, 1, 5, 2), [x, example(dtype, (3, 2), 72)]),
        'select_scatter': (lambda a, b: A.select_scatter.default(a, b, 1, 2), [x, example(dtype, (3,), 72)]),
        'full': (lambda a: A.full.default([3, 5], literal, dtype=dt), [x]),
        'full_like': (lambda a: A.full_like.default(a, literal), [x]),
        'fill_Scalar': (lambda a: A.fill.Scalar(a, literal), [x]),
        '_to_copy': (lambda a: A._to_copy.default(a, dtype=dt), [x]),
    }
    for dim, shape in [(1, (2, 3, 5)), (2, (1, 2, 3, 5)), (3, (1, 2, 3, 4, 5))]:
        for mode in ('reflection', 'replication'):
            if mode == 'replication' and dim == 1:
                continue
            op = getattr(A, f'{mode}_pad{dim}d').default
            table[f'{mode}_pad{dim}d'] = (lambda a, op=op, dim=dim: op(a, [1, 1] * dim), [example(dtype, shape)])
    fn, inputs = table[name]
    class Model(torch.nn.Module):
        def forward(self, *xs):
            return fn(*xs)
    return Model(), tuple(inputs)


def capture(name, dtype):
    import m2m
    model, inputs = case(name, dtype)
    expected = raw(model(*inputs))
    converted = m2m.convert(model.eval(), inputs, backend='fx_importer')
    if not converted.ok:
        raise ValueError('capture refused: ' + str(converted.diagnostics))
    return converted.mlir_text, inputs, expected


def wire(text, inputs, expected, prefix):
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.offload import rewrite, shim_source
    from mlir_oot.core_aten.stream import simulate, ACC_BIT
    from mlir_oot.frontend.parse import parse_module
    device, manifest, matches = build(text)
    # Exercise the same API and descriptors as Merlin, including dependent bindings.
    rewritten = rewrite(text, manifest)
    assert rewritten.count('func.call @ca_') == len(matches)
    assert shim_source(manifest)
    key = lambda src: ('arg', src['func_arg']) if 'func_arg' in src else ('op', src['op_ordinal'], src['result'])
    values = {('arg', i): raw(x) for i, x in enumerate(inputs)}
    names = {k: f'{prefix}_arg{k[1]}' for k in values}
    buffers = {names[k]: v for k, v in values.items()}
    calls, outputs = [], set()
    kernels = {k['symbol']: k for k in manifest['kernels']}
    for ordinal, (binding, match) in enumerate(zip(manifest['bindings'], matches)):
        assert match.numeric['numerics'] == 'bitwise_movement', binding
        assert not match.kernel.stream.readouts and not match.kernel.stream.packs
        assert not any(i.a('local') & ACC_BIT for i in match.kernel.stream.ins if i.kind == 'mvout')
        assert all(i.a('scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_ld')
        assert all(i.a('acc_scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_st')
        feed, argv = {}, []
        for arg in match.kernel.args:
            if arg.role == 'input':
                source = key(binding['operands'][arg.name])
                if source not in values:
                    raise ValueError('host producer remains for ' + str(source) + ': ' + str(manifest['declines']))
                feed[arg.name] = values[source]
                argv.append(names[source])
            else:
                buf = f'{prefix}_b{ordinal}_{arg.name}'
                buffers[buf] = np.zeros(arg.nbytes, dtype=np.int8)
                argv.append(buf)
                if arg.role == 'output':
                    outputs.add(buf)
                    out_source, = binding['result'].values()
                    names[key(out_source)] = buf
        actual = simulate(match.kernel.stream, feed)['y']
        values[key(out_source)] = np.ascontiguousarray(actual).reshape(-1).view(np.int8)
        calls.append((binding['symbol'], argv))
    module = parse_module(text)
    operations = list(module.walk())
    fn = next(o for o in operations if o.name == 'func.func')
    ret = next(o for o in fn.walk() if o.name == 'func.return')
    value, = ret.operands
    result_key = ('arg', list(fn.body.blocks[0].args).index(value)) if value in fn.body.blocks[0].args else ('op', operations.index(value.owner), list(value.owner.results).index(value))
    if result_key not in values:
        raise ValueError('returned result remains on host: ' + str(manifest['declines']))
    assert np.array_equal(values[result_key], expected), 'instruction model differs from PyTorch bytes'
    return device, manifest, buffers, calls, outputs, names[result_key]


def run(root, spike=False):
    from xdsl.dialects.builtin import ModuleOp
    root.mkdir(parents=True, exist_ok=True)
    combined, symbols = ModuleOp([]), set()
    buffers, calls, outputs, checks, records = {}, [], set(), [], []
    if spike:
        sys.path.insert(0, str(Path(__file__).parents[1] / 'device'))
        import runner
    for dtype in DTYPES:
        for name in OPS:
            label = name + '_' + dtype
            try:
                text, inputs, expected = capture(name, dtype)
            except (RuntimeError, ValueError, NotImplementedError) as exc:
                records.append(dict(op=name, dtype=dtype, status='source_refused', reason=str(exc)))
                continue
            folder = root / label
            folder.mkdir(exist_ok=True)
            (folder / 'model.mlir').write_text(text)
            try:
                module, manifest, bufs, cs, outs, result = wire(text, inputs, expected, label)
            except (ValueError, AssertionError) as exc:
                records.append(dict(op=name, dtype=dtype, status='backend_refused', reason=str(exc)))
                continue
            (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            for op in list(module.body.blocks[0].ops):
                symbol = op.sym_name.data
                if symbol not in symbols:
                    symbols.add(symbol)
                    op.detach()
                    combined.body.blocks[0].add_op(op)
            buffers.update(bufs); calls.extend(cs); outputs.update(outs)
            if spike:
                checks.append(runner.Check(result, expected, 0, label))
            records.append(dict(op=name, dtype=dtype, status='exact', bindings=len(cs)))
            print(label, 'exact', flush=True)
    result = dict(cases=records)
    if spike and calls:
        from mlir_oot.core_aten.emit import compile_to_object
        from mlir_oot.golden_device_catalog import final_elf_audit
        obj = compile_to_object(combined, root / 'build')['object']
        test = runner.DeviceTest('dtype_movement', [Path(obj)], buffers, calls, checks, outputs)
        elf = runner.build_elf(test, root / 'build')
        final_elf_audit(elf)
        actual = runner.run_spike(elf)
        result.update(elf=str(elf.resolve()), spike=dict(status=actual.status, passed=len(actual.passed),
            failed=actual.failed, tail=None if actual.ok else actual.log[-3000:]))
    (root / 'results.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--spike', action='store_true')
    args = ap.parse_args()
    result = run(args.out, args.spike)
    print(json.dumps(result.get('spike', {})))
    raise SystemExit(int(any(c['status'] == 'backend_refused' for c in result['cases']) or
                         (args.spike and result.get('spike', {}).get('status') != 'PASS')))
