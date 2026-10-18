"""Real model2MLIR seeded MAC/readout bodies through the provider's region catalog."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import TensorType, i8, i32, DenseIntOrFPElementsAttr, ModuleOp
from xdsl.ir import Block, Region

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.stream import simulate
from exact_oracle import evaluate


def source(m=3,k=17,n=5,scale=.5, runtime=False, guarded=True):
    m2m = pytest.importorskip('m2m.ir.integer_readout')
    types = [TensorType(i8, (m,k)), TensorType(i8, (k,n))]
    if runtime:
        types.append(TensorType(i32, (n,)))
    block = Block(arg_types=types)
    if runtime:
        seed = block.args[2]
    else:
        values = [2*j-7 for j in range(n)]
        constant = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, (n,)), values))
        block.add_op(constant)
        seed = constant.result
    contraction = m2m.decompose_int_mm_seeded([block.args[0],block.args[1],seed], {}, 'seeded')
    block.add_ops(contraction.ops)
    if not guarded:
        for guard in list(block.ops):
            if guard.name == 'linalg.generic' and any(op.name == 'cf.assert' for op in guard.walk()):
                guard.results[0].replace_all_uses_with(guard.operands[0])
                guard.detach()
                guard.erase()
    readout = m2m.decompose_requantize([contraction.result], {'_fx_args': (None,scale,0,-128,127,False)}, 'readout')
    block.add_ops(readout.ops)
    block.add_op(func.ReturnOp(readout.result))
    function = func.FuncOp('forward', (types,[TensorType(i8,(m,n))]), Region(block))
    module = ModuleOp([function])
    module.verify()
    return str(module)


@pytest.mark.parametrize('m,k,n', [(1,1,1),(3,17,5),(17,33,19)])
def test_constant_seed_and_byte_readout_are_covered_and_exact(m,k,n):
    text = source(m,k,n)
    _, manifest, matches = build(text)
    regions = [match for match in matches if match.numeric.get('integer_readout')]
    assert len(regions) == 1, manifest['declines']
    rng = np.random.default_rng(41+k)
    inputs = {'a':rng.integers(-128,128,(m,k),dtype=np.int8), 'b':rng.integers(-128,128,(k,n),dtype=np.int8)}
    match, = regions
    assert np.array_equal(simulate(match.kernel.stream,inputs)['out'],evaluate(match,inputs)['out'])


def test_unbounded_runtime_seed_is_refused():
    _, manifest, matches = build(source(runtime=True,guarded=False))
    assert not any(m.numeric.get('integer_readout') for m in matches)
    assert any('runtime i32 seed' in d['reason'] for d in manifest['declines'])


def test_provider_proves_and_routes_the_actual_region(tmp_path):
    path = Path(__file__).parents[2] / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('provider_seeded_test',path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    text = source()
    policy = dict(numerical_semantics=dict(accumulator_dtype='i32',internal_arithmetic=dict(
        signed_operand_bits=8,mac_result_bits=20,mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))
    proof = provider.admit_numeric_schedule(text,package=path.parents[2],numeric_policy=policy)
    assert len(proof['kernels']) == 1 and proof['kernels'][0]['exact_arithmetic']
    assert proof['bindings'][0]['source_operation_ordinal'] >= 0
    from types import SimpleNamespace
    from merlin.targetgen.core_aten_device import selected_facts
    fixture=path.parents[1]/'tests/fixtures/core_aten_device/facts.json'
    (tmp_path/'model.mlir').write_text(text)
    with selected_facts('synthetic_core_aten',fixture) as facts:
        route=provider.routing(tmp_path,target='synthetic_core_aten',package=path.parents[2],facts=facts,
                               eligible=[(None,SimpleNamespace(dtypes=('i8','i8','i32')))])
    assert callable(route.region_offload) and route.catalog_builder is None
    assert callable(route.final_elf_audit)



def test_guarded_runtime_seed_is_admitted_without_erasing_the_guard():
    from mlir_oot.core_aten.offload import rewrite
    text = source(runtime=True)
    _, manifest, matches = build(text)
    match, = [m for m in matches if m.numeric.get('integer_readout')]
    assert match.kernel.meta['seed'] == 'guarded_runtime'
    assert 'cf.assert' in rewrite(text,manifest)


def test_guarded_runtime_seed_extremes_equal_rational_reference():
    text = source(m=2,k=17,n=3,runtime=True)
    _,manifest,matches = build(text)
    match, = [m for m in matches if m.numeric.get('integer_readout')]
    bound=17*16384
    for seed in (-(1<<31)+bound,(1<<31)-1-bound,0):
        for code in (-128,127):
            inputs=dict(a=np.full((2,17),code,np.int8), b=np.full((17,3),code,np.int8),bias=np.full(3,seed,np.int32))
            assert np.array_equal(simulate(match.kernel.stream,inputs)['out'],evaluate(match,inputs)['out'])


def test_weakened_seed_guard_refuses_overflow():
    text=source(runtime=True)
    # A guard covering the entire input type is insufficient for a seeded sum.
    text=text.replace('2147205119 : i64','2147483647 : i64')
    _,manifest,matches=build(text)
    assert not any(m.numeric.get('integer_readout') for m in matches)
    assert any('overflow bound' in d['reason'] for d in manifest['declines'])
