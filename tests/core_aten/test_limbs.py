"""F2/F4 device contracts, wide-integer reconstruction and refusal boundaries."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate


FIXTURE = Path(__file__).parents[1] / 'fixtures/core_aten/limbs/int_mm.mlir'
POLICY = dict(numerical_semantics=dict(accumulator_dtype='i32', internal_arithmetic=dict(
    signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
    full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))


def annotate(text, indices=(0, 0), exponents=(0, 0), family='F2', component='rr', mutate=None):
    module = parse_module(text)
    op = [o for o in module.walk() if o.name in ('linalg.generic', 'linalg.matmul')][-1]
    p = dict(version=1, family=family, radix=256, inputs=[
        dict(operand=i, index=k, count=max(9, k+1), exponent=e)
        for i, (k, e) in enumerate(zip(indices, exponents))])
    if family == 'F4':
        p['component'] = component
    if mutate:
        mutate(p)
    op.attributes['gemmini.limb'] = StringAttr(json.dumps(p))
    return str(module)


def linear_source(coefficient=-3, reduction=False, cast='arith.extsi', scalar_op='arith.muli'):
    # Captured-IR interface, independent of a frontend's current limb splitter.
    maps = ('affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>' if reduction else
            'affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>')
    source, output = ('3x5', '3') if reduction else ('19', '19')
    its = '"parallel", "reduction"' if reduction else '"parallel"'
    return f'''builtin.module {{
      func.func @forward(%x: tensor<{source}xi8>) -> tensor<{output}xi32> {{
        %seed = arith.constant 7 : i32
        %init = tensor.splat %seed : tensor<{output}xi32>
        %r = linalg.generic {{indexing_maps = [{maps}], iterator_types = [{its}]}}
          ins(%x : tensor<{source}xi8>) outs(%init : tensor<{output}xi32>) {{
          ^bb0(%a: i8, %acc: i32):
            %v = {cast} %a : i8 to i32
            %c = arith.constant {coefficient} : i32
            %w = {scalar_op} %v, %c : i32
            %z = arith.addi %w, {'%acc' if reduction else '%c'} : i32
            linalg.yield %z : i32
        }} -> tensor<{output}xi32>
        func.return %r : tensor<{output}xi32>
      }}
    }}'''


def ranked_linear_source(shape):
    text = linear_source()
    rank = len(shape)
    dims = ','.join(f'd{i}' for i in range(rank))
    tensor = 'x'.join(map(str,shape))+('x' if shape else '')
    return (text.replace('tensor<19x',f'tensor<{tensor}')
            .replace('affine_map<(d0)->(d0)>',f'affine_map<({dims})->({dims})>')
            .replace('iterator_types = ["parallel"]', 'iterator_types = ['+','.join(['"parallel"']*rank)+']'))


@pytest.mark.parametrize('shape', [(),(1,),(1,2,1,3,5)])
@pytest.mark.parametrize('count', [1,9,65])
def test_ranked_linear_components_have_no_native_width_or_rank_limit(shape,count):
    text = annotate(ranked_linear_source(shape),indices=(count-1,),exponents=(0,),
                    mutate=lambda p:p['inputs'][0].update(count=count))
    _, match = selected(text)
    x = np.resize(np.array([-128,127,-1,0,1],np.int8),shape)
    assert np.array_equal(simulate(match.kernel.stream,{'x0':x})['out'],x.astype(np.int32)*-3-3)
    assert match.numeric['binary_exponent'] == 8*(count-1)
    api,package = provider()
    assert api.admit_numeric_schedule(text,package=package,numeric_policy=POLICY)['status'] == 'proven'


def selected(text):
    _, manifest, matches = build(text)
    limbs = [m for m in matches if 'limb_plan' in m.numeric]
    assert len(limbs) == 1, manifest['declines']
    for binding in manifest['bindings']:
        if 'limb_plan' in binding['numeric']:
            assert binding['source_operation_ordinal'] == binding['root_ordinal']
    return manifest, limbs[0]


def provider():
    path = Path(__file__).parents[2] / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('provider_limbs_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, path.parents[2]


@pytest.mark.parametrize('code', [-128, 127])
@pytest.mark.parametrize('family', ['F2', 'F4'])
def test_captured_bilinear_tail_full_range_and_manifest(code, family):
    text = annotate(FIXTURE.read_text(), indices=(2, 3), exponents=(-16, 0), family=family)
    manifest, match = selected(text)
    a, b = np.full((3, 17), code, np.int8), np.full((17, 5), code, np.int8)
    assert np.array_equal(simulate(match.kernel.stream, dict(a=a, b=b))['out'], a.astype(np.int32) @ b.astype(np.int32))
    assert match.numeric['binary_exponent'] == 24
    assert match.numeric['bound_proof']['required_signed_mac_bits'] == 20
    assert match.numeric['readout'] == 'full_i32'
    assert not match.kernel.stream.readouts
    assert 'func.call @ca_' in rewrite(text, manifest)
    api, package = provider()
    proof = api.admit_numeric_schedule(text, package=package, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert proof['kernels'][0]['schedule']['family'] == family


@pytest.mark.parametrize('coefficient', [-254, -3, 0, 1, 128])
@pytest.mark.parametrize('reduction', [False, True])
def test_signed_linear_maps_and_reduction_seeds(coefficient, reduction):
    text = annotate(linear_source(coefficient, reduction), indices=(1,), exponents=(-8,))
    manifest, match = selected(text)
    x = np.resize(np.array([-128, -1, 0, 1, 127], np.int8), (3, 5) if reduction else (19,))
    expected = x.astype(np.int32)*coefficient
    expected = expected.sum(1) + 7 if reduction else expected + coefficient
    assert np.array_equal(simulate(match.kernel.stream, dict(x0=x))['out'], expected)
    assert match.numeric['binary_exponent'] == 0
    assert match.numeric['bound_proof']['required_signed_mac_bits'] == 15
    rewrite(text, manifest)
    api, package = provider()
    proof = api.admit_numeric_schedule(text, package=package, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof


@pytest.mark.parametrize('change,reason', [
    (lambda p: p.update(extra=1), 'unknown or missing limb plan keys'),
    (lambda p: p.update(radix=2), 'unsupported limb plan version/radix'),
    (lambda p: p['inputs'][0].update(index=-1), 'invalid limb index/count'),
    (lambda p: p['inputs'][0].update(exponent=True), 'limb input fields must be integers'),
    (lambda p: p['inputs'].pop(), 'limb plan does not cover'),
    (lambda p: p['inputs'].append(p['inputs'][0]), 'duplicate or invalid limb operand'),
])
def test_bad_plans_decline_without_fallback(change, reason):
    text = annotate(FIXTURE.read_text(), mutate=change)
    _, manifest, matches = build(text)
    assert not any('limb_plan' in m.numeric for m in matches)
    assert any(reason in d['reason'] for d in manifest['declines'])
    api, package = provider()
    assert api.admit_numeric_schedule(text, package=package, numeric_policy=POLICY)['status'] == 'declined'


@pytest.mark.parametrize('kwargs,reason', [
    (dict(coefficient=1 << 25), 'overflow bound'),
    (dict(cast='arith.extui'), 'unsupported limb data operation'),
    (dict(scalar_op='arith.divsi'), 'unsupported limb data operation'),
])
def test_unsafe_integer_data_path_declines(kwargs, reason):
    _, manifest, matches = build(annotate(linear_source(**kwargs), indices=(0,), exponents=(0,)))
    assert not any('limb_plan' in m.numeric for m in matches)
    assert any(reason in d['reason'] for d in manifest['declines'])


def split(value):
    """Test-only exact balanced radix256 representation; arbitrary Python integers."""
    value = np.asarray(value, object)
    out = []
    while np.any(value):
        digit = ((value+128) % 256 - 128).astype(np.int8)
        out.append(digit)
        value = (value-digit.astype(object)) // 256
    return out or [np.zeros(value.shape, np.int8)]


@pytest.mark.parametrize('bits', [32, 64])
def test_limb_pair_recombination_is_exact_with_integer_wrap(bits):
    rng = np.random.default_rng(32+bits)
    a = rng.integers(-(1 << (bits-1)), (1 << (bits-1))-1, (3, 17), dtype=np.int64).astype(object)
    b = rng.integers(-(1 << (bits-1)), (1 << (bits-1))-1, (17, 5), dtype=np.int64).astype(object)
    a[0, 0], b[0, 0] = -(1 << (bits-1)), (1 << (bits-1))-1
    _, match = selected(annotate(FIXTURE.read_text()))
    exact = np.zeros((3, 5), object)
    for j, left in enumerate(split(a)):
        for k, right in enumerate(split(b)):
            raw = simulate(match.kernel.stream, dict(a=left, b=right))['out']
            exact += raw.astype(object) * (256**(j+k))
    assert np.array_equal(exact, a @ b)
    wrap = lambda x: (x+(1 << (bits-1))) % (1 << bits) - (1 << (bits-1))
    assert np.array_equal(wrap(exact), wrap(a @ b))


def test_plan_is_authenticated_even_for_identical_instructions():
    m0, _ = selected(annotate(FIXTURE.read_text(), indices=(0, 0)))
    m1, _ = selected(annotate(FIXTURE.read_text(), indices=(1, 1)))
    assert m0['bindings'][-1]['symbol'] != m1['bindings'][-1]['symbol']


def test_selected_mac_width_must_cover_the_actual_primitive():
    api, package = provider()
    narrow = json.loads(json.dumps(POLICY))
    narrow['numerical_semantics']['internal_arithmetic'].update(mac_result_bits=19, mac_result_overflow='wrap_to_19_bits')
    proof = api.admit_numeric_schedule(annotate(FIXTURE.read_text()), package=package, numeric_policy=narrow)
    assert proof['status'] == 'declined' and 'selected exact MAC' in proof['reason']
    narrow = json.loads(json.dumps(POLICY))
    narrow['numerical_semantics']['internal_arithmetic']['signed_operand_bits'] = 4
    proof = api.admit_numeric_schedule(annotate(FIXTURE.read_text()), package=package, numeric_policy=narrow)
    assert proof['status'] == 'declined' and 'selected exact MAC' in proof['reason']


def test_real_batched_capture_full_i32():
    text = annotate((FIXTURE.parent/'bmm.mlir').read_text(), indices=(3, 7), exponents=(-8, -16))
    manifest, match = selected(text)
    a = np.resize(np.array([-128, 127, 0], np.int8), (2, 3, 17))
    b = np.resize(np.array([127, -128, 1, -1], np.int8), (2, 17, 5))
    assert np.array_equal(simulate(match.kernel.stream, dict(a=a, b=b))['out'], a.astype(np.int32) @ b.astype(np.int32))
    assert match.numeric['binary_exponent'] == 56
    assert match.kernel.meta['dimensions']['batch'] == 2
    rewrite(text, manifest)


def test_limb_route_uses_region_abi(tmp_path):
    from types import SimpleNamespace
    from merlin.targetgen.core_aten_device import selected_facts
    api, package = provider()
    (tmp_path/'model.mlir').write_text(annotate(FIXTURE.read_text()))
    fixture = package/'merlin-support/tests/fixtures/core_aten_device/facts.json'
    with selected_facts('synthetic_core_aten', fixture) as facts:
        route = api.routing(tmp_path, target='synthetic_core_aten', package=package, facts=facts,
                            eligible=[(None, SimpleNamespace(dtypes=('i8', 'i8', 'i32')))])
    assert callable(route.region_offload) and route.catalog_builder is None


def test_recombination_operations_remain_source_host_work():
    # Emulate one explicit host recombination term; no implicit native cast.
    text = FIXTURE.read_text().replace('-> tensor<3x5xi32> {', '-> tensor<3x5xi64> {', 1)
    text = text.replace('    func.return %4 : tensor<3x5xi32>', '''
      %empty = tensor.empty() : tensor<3x5xi64>
      %host = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%4 : tensor<3x5xi32>) outs(%empty : tensor<3x5xi64>) {
        ^bb1(%x: i32, %unused: i64):
          %wide = arith.extsi %x : i32 to i64
          %weight = arith.constant 65536 : i64
          %scaled = arith.muli %wide, %weight : i64
          linalg.yield %scaled : i64
      } -> tensor<3x5xi64>
      func.return %host : tensor<3x5xi64>''')
    module = parse_module(text)
    contraction = next(o for o in module.walk() if o.name == 'linalg.generic')
    contraction.attributes['gemmini.limb'] = StringAttr(json.dumps(dict(version=1, family='F2', radix=256,
        inputs=[dict(operand=i, index=1, count=4, exponent=0) for i in range(2)])))
    text = str(module)
    manifest, _ = selected(text)
    rewritten = rewrite(text, manifest)
    assert 'arith.extsi' in rewritten and 'arith.muli' in rewritten and '65536 : i64' in rewritten
    assert 'tensor<3x5xi64>' in rewritten


def test_signed_static_gather_map():
    text = linear_source(3).replace('tensor<19xi32>', 'tensor<7xi32>')
    text = text.replace('affine_map<(d0)->(d0)>,', 'affine_map<(d0)->(18 - d0*3)>,', 1)
    _, match = selected(annotate(text, indices=(0,), exponents=(0,)))
    x = np.arange(-9, 10, dtype=np.int8)
    assert np.array_equal(simulate(match.kernel.stream, dict(x0=x))['out'], x[[18, 15, 12, 9, 6, 3, 0]].astype(np.int32)*3+3)


@pytest.mark.parametrize('operator', ['arith.addi', 'arith.subi'])
def test_two_limb_inputs_with_same_exponent(operator):
    text = linear_source(1)
    text = text.replace('%x: tensor<19xi8>', '%x: tensor<19xi8>, %y: tensor<19xi8>', 1)
    text = text.replace('indexing_maps = [', 'indexing_maps = [affine_map<(d0)->(d0)>, ', 1)
    text = text.replace('ins(%x : tensor<19xi8>)', 'ins(%x, %y : tensor<19xi8>, tensor<19xi8>)')
    text = text.replace('%a: i8, %acc:', '%a: i8, %b: i8, %acc:')
    text = text.replace('%w = arith.muli %v, %c : i32', f'%v2 = arith.extsi %b : i8 to i32\n            %w = {operator} %v, %v2 : i32')
    text = annotate(text, indices=(1, 2), exponents=(0, -8))
    _, match = selected(text)
    x, y = np.arange(-9, 10, dtype=np.int8), np.full(19, -128, np.int8)
    expected = x.astype(np.int32)+(y.astype(np.int32) if operator == 'arith.addi' else -y.astype(np.int32))+1
    assert np.array_equal(simulate(match.kernel.stream, dict(x0=x, x1=y))['out'], expected)
    module = parse_module(text)
    root = next(o for o in module.walk() if 'gemmini.limb' in o.attributes)
    p = json.loads(root.attributes['gemmini.limb'].data)
    p['inputs'][1]['exponent'] = 0
    root.attributes['gemmini.limb'] = StringAttr(json.dumps(p))
    _, manifest, matches = build(str(module))
    assert not any('limb_plan' in m.numeric for m in matches)
    assert any('different binary exponents' in d['reason'] for d in manifest['declines'])


def test_static_index_math_never_wraps_or_uses_rounded_division():
    from mlir_oot.core_aten.linmap import Region, NotLinear, _affine
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i8
    from xdsl.ir.affine import AffineExpr as A, AffineMap
    reg = Region(lambda value: None)
    lhs, rhs = arith.ConstantOp.from_int_and_width(1, 64), arith.ConstantOp.from_int_and_width(3, 64)
    div = arith.DivSIOp(lhs, rhs)
    assert reg._scalar(div, [np.asarray((1 << 63)-1), np.asarray(3)], {}, [], np.asarray(True)) == ((1 << 63)-1)//3
    add = arith.AddiOp(lhs, rhs)
    with pytest.raises(NotLinear, match='nonoverflow proof'):
        reg._scalar(add, [np.asarray((1 << 63)-1), np.asarray(1)], {}, [], np.asarray(True))
    cast = arith.TruncIOp(lhs, i8)
    with pytest.raises(NotLinear, match='nonoverflow proof'):
        reg._scalar(cast, [np.asarray(257)], {}, [], np.asarray(True))
    with pytest.raises(NotLinear, match='signed 64-bit address range'):
        _affine(AffineMap(1, 0, (A.dimension(0)*(1 << 62),)), [np.arange(4)], (4,))


def test_reused_component_kernel_has_one_proof_and_multiple_bindings():
    module = parse_module(annotate(FIXTURE.read_text()))
    op = next(o for o in module.walk() if 'gemmini.limb' in o.attributes)
    from xdsl.dialects import func
    from xdsl.dialects.builtin import FunctionType
    duplicate = op.clone()
    op.parent_block().insert_op_after(duplicate, op)
    fn = next(o for o in module.walk() if o.name == 'func.func')
    ret = next(o for o in fn.walk() if o.name == 'func.return')
    ret.parent_block().insert_op_before(func.ReturnOp(op.results[0], duplicate.results[0]), ret)
    ret.parent_block().erase_op(ret)
    fn.properties['function_type'] = FunctionType.from_lists(
        [v.type for v in fn.body.block.args], [op.results[0].type]*2)
    api, package = provider()
    proof = api.admit_numeric_schedule(str(module), package=package, numeric_policy=POLICY)
    assert proof['status'] == 'proven', proof
    assert len(proof['kernels']) == 1
    assert len([b for b in proof['bindings'] if 'limb_plan' in b['numeric']]) == 2
