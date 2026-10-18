"""Exact quotient theorem, kernel-bound certificates and emitted integer paths."""
from fractions import Fraction as F
import random

import pytest

from mlir_oot.core_aten.host_magic import prove, interval
from mlir_oot.core_aten.requant import Readout


@pytest.mark.parametrize('d', [1, 2, 3, 7, 50, 1 << 32, (1 << 63)-1])
def test_full_signed_i32_domain_and_every_small_tie(d):
    p = prove(-(1 << 31), (1 << 31)-1, d, 1, 255, 123, False)
    assert p['method'] == 'exact_reciprocal_multiply_shift'
    m, s = int(p['multiplier']), p['shift']
    assert (1 << 31)*(m*d-(1 << s)) < 1 << s
    rng = random.Random(d)
    values = [-(1 << 31), (1 << 31)-1, 0] + [rng.randrange(-(1 << 31), 1 << 31) for _ in range(2000)]
    if d % 2 == 0:
        values += [n for k in range(-1024, 1024) if -(1 << 31) <= (n := (2*k+1)*d//2) < 1 << 31]
    for n in values:
        q = abs(n)*m >> s
        r = abs(n)-q*d
        q += r > d-r or (r == d-r and q & 1)
        assert (-q if n < 0 else q) == round(F(n, d))


def test_exhaustive_and_fallback_certificates():
    p = prove(-10000, 10000, 50, -127, 127, -7, True)
    assert p['verification'] == 'exhaustive'
    assert p['tested_values'] == 20001
    for low, high, d in [(-(1 << 63), 1, 3), (-1, 1, 1 << 149), (-1, 1 << 127, 3)]:
        assert prove(low, high, d, -128, 127, 0, False)['method'] == 'comparison_search'


def test_bounds_include_each_term_constant_and_intermediate():
    r = Readout('y', [('primary', F(1)), ('extra', F(1))])
    meta = dict(accumulator_min=-10, accumulator_max=10,
                zero_point_corrections=[dict(tile='extra', accumulator_min=-100, accumulator_max=100)])
    lo, hi, src, safe = interval(r, [3, -2, 7], meta)
    assert (lo, hi, safe) == (-223, 237, True)
    assert all(b['bound_source'] == 'kernel_meta' for b in src)
    # Final cancellation cannot license an overflowing intermediate.
    shifted = dict(accumulator_min=100, accumulator_max=100,
                   zero_point_corrections=[dict(tile='extra', accumulator_min=-100, accumulator_max=-100)])
    lo, hi, _, safe = interval(r, [1, 1, (1 << 63)-2], shifted)
    assert lo == hi == (1 << 63)-2
    assert not safe


def test_catalog_records_exhaustive_bound_proof():
    from pathlib import Path
    from fixture_source import read
    from mlir_oot.core_aten.catalog import build
    _, manifest, _ = build(read(Path(__file__).parents[1]/'fixtures/core_aten/mean_dim/model.mlir'))
    p = manifest['kernels'][0]['meta']['host_requantization_proofs'][0]
    assert p['method'] == 'exact_reciprocal_multiply_shift'
    assert p['reachable_verification'] == 'exhaustive_accumulator_tuples'
    assert p['reachable_values_tested'] > 1
    assert p['accumulator_bounds'][0]['bound_source'] == 'kernel_meta'


def test_deduplication_authenticates_codegen_bounds():
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.catalog import _stream_digest
    k = ops.mean_dim((2, 3), [1], False, s_x=F(1, 3), s_out=F(1))
    original = _stream_digest(k)
    k.meta['accumulator_max'] += 1
    assert _stream_digest(k) != original


def test_near_i64_limits_require_the_high_product():
    a = (1 << 63)-2
    d = (1 << 62)+37
    p = prove(-a, a, d, -128, 127, 0, False)
    assert p['shift'] > 64
    m, s = int(p['multiplier']), p['shift']
    for n in [-a, -d-1, -d, -d+1, -1, 0, 1, d-1, d, d+1, a]:
        q = abs(n)*m >> s
        r = abs(n)-q*d
        q += r > d-r or (r == d-r and q & 1)
        assert (-q if n < 0 else q) == round(F(n, d))


def test_emitted_multiterm_and_wide_product_against_fraction(tmp_path):
    import ctypes
    import os
    from pathlib import Path
    import subprocess
    import numpy as np
    from mlir_oot.core_aten.stream import Stream, Arg
    from mlir_oot.core_aten.emit import build_module

    bindir = Path(os.environ.get('GEMMINI_LLVM_BIN', '/nonexistent'))
    if not (bindir/'mlir-translate').exists():
        pytest.skip('LLVM toolchain not configured')
    rng = np.random.default_rng(6728)
    x = np.r_[np.arange(-1024, 1025), rng.integers(-(1 << 31), 1 << 31, 2000),
              -(1 << 31), (1 << 31)-1].astype(np.int32)
    z = x[::-1].copy()
    for ordinal, (f, g, const) in enumerate([
        (F(17, 31), F(-5, 7), F(2, 3)),
        # Numerator approaches signed i64 limits, forcing shift >64.
        (F((1 << 31)-1, (1 << 62)+37), F(0), F(0)),
    ]):
        stream = Stream([Arg('x', 'i32', x.shape, 'input'),
                         Arg('z', 'i32', x.shape, 'input'),
                         Arg('y', 'i8', x.shape, 'output')])
        stream.readouts.append(Readout('y', [('x', f), ('z', g)], const,
                                       zero_point=123, quant_min=1, quant_max=255))
        meta = {}
        module, _ = build_module(stream, 'readout', meta=meta)
        proof, = meta['host_requantization_proofs']
        assert proof['method'] == 'exact_reciprocal_multiply_shift'
        if ordinal == 1:
            assert proof['shift'] > 64
        source = tmp_path/f'readout{ordinal}.mlir'
        source.write_text(str(module)+'\n')
        ir, obj, shared = (source.with_suffix(s) for s in ('.ll', '.o', '.so'))
        subprocess.run([str(bindir/'mlir-translate'), '--mlir-to-llvmir', str(source), '-o', str(ir)], check=True)
        subprocess.run([str(bindir/'llc'), '-mtriple=x86_64-unknown-linux-gnu', '-relocation-model=pic',
                        '-O2', '-filetype=obj', str(ir), '-o', str(obj)], check=True)
        subprocess.run(['cc', '-shared', str(obj), '-o', str(shared)], check=True)
        fn = ctypes.CDLL(str(shared)).readout
        fn.argtypes = [ctypes.c_void_p]*3
        out = np.zeros(x.shape, np.uint8)
        fn(x.ctypes.data, z.ctypes.data, out.ctypes.data)
        expected = np.array([min(255, max(1, round(int(a)*f+int(b)*g+const)+123))
                             for a, b in zip(x, z)], np.uint8)
        assert np.array_equal(out, expected)
