"""Provider integration on actual convert(profile=int8_accelerator) output."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate, FULL_BIT, ACC_BIT
from exact_oracle import evaluate


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16', 'int32', 'int64', 'complex64', 'bool'])
@pytest.mark.parametrize('name', ['cat', 'clone', 'constant_pad_nd', 'reflection_pad1d',
    'reflection_pad2d', 'reflection_pad3d', 'replication_pad2d', 'replication_pad3d',
    'repeat', 'slice_scatter', 'select_scatter', 'full', 'full_like', 'fill_Scalar', '_to_copy'])
def test_real_dtype_movement(name, dtype):
    pytest.importorskip('m2m')
    sys.path.insert(0, str(Path(__file__).parent / 'e2e'))
    from dtype_movement import capture, wire
    try:
        text, inputs, expected = capture(name, dtype)
    except RuntimeError as exc:
        if dtype == 'bool' and "not implemented for 'Bool'" in str(exc):
            pytest.skip(str(exc))
        raise
    _, manifest, _, _, _, _ = wire(text, inputs, expected, name + '_' + dtype)
    assert manifest['bindings']


@pytest.mark.parametrize('alpha', [-1.0, -0.375])
@pytest.mark.parametrize('bias', [False, True])
def test_real_signed_addmm_retains_exact_source_readout(alpha, bias):
    torch = pytest.importorskip('torch')
    m2m = pytest.importorskip('m2m')
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    from mlir_oot.core_aten.quant_schema import ratios, attributes

    class Addmm(torch.nn.Module):
        def forward(self, a, b, offset):
            qd = torch.ops.quantized_decomposed
            a = qd.dequantize_per_tensor(a, 0.5, 0, -128, 127, torch.int8)
            b = qd.dequantize_per_tensor(b, 0.25, 0, -128, 127, torch.int8)
            y = torch.ops.aten.addmm.default(offset, a, b, alpha=alpha, beta=1 if bias else 0)
            return qd.quantize_per_tensor(y, 0.125, 0, -128, 127, torch.int8)

    inputs = (torch.tensor([[-128, -3, 0, 1, 127], [3, 5, 7, 9, 11]], dtype=torch.int8),
              torch.arange(-7, 8, dtype=torch.int8).reshape(5, 3),
              torch.tensor([0.25, -0.5, 0.75]))
    graph = torch.export.export(Addmm(), inputs).module()
    integer, receipt = integerize_pt2e(graph, inputs)
    assert receipt['quantized_contractions_integerized'] == 1 and not receipt['refusals']
    allow_exported_model_train_eval(integer)
    converted = m2m.convert(integer, inputs, backend='fx_importer', fully_standard=True)
    assert converted.ok, converted.diagnostics
    _, manifest, matches = build(converted.mlir_text)
    selected, = [m for m in matches if m.kernel.meta.get('integer_readout')]
    assert all(int(n) < 0 for n in selected.kernel.meta['scale_numerator'])
    feed = dict(a=inputs[0].numpy(), b=inputs[1].numpy())
    raw = simulate(selected.kernel.stream, feed)['out']
    assert np.array_equal(raw, evaluate(selected, feed)['out'])
    rewritten = rewrite(converted.mlir_text, manifest)
    assert 'prov.quant.arithmetic = "exact"' in rewritten
    assert not selected.kernel.stream.readouts
    # Interpret the retained source readout independently using the raw result.
    from exact_oracle import SourceOracle
    from mlir_oot.frontend.parse import parse_module
    module = parse_module(converted.mlir_text)
    fn = next(o for o in module.walk() if o.name == 'func.func')
    oracle = SourceOracle(dict(zip(fn.body.blocks[0].args, [i.numpy().astype(object) for i in inputs])))
    ret = next(o for o in fn.walk() if o.name == 'func.return')
    expected = np.asarray(oracle.tensor(ret.operands[0]), dtype=np.int64).astype(np.int8)
    from fractions import Fraction
    acc = inputs[0].numpy().astype(object) @ inputs[1].numpy().astype(object)
    values = acc * Fraction(alpha)
    if bias:
        values += np.asarray([2, -4, 6], dtype=object)
    independent = np.array([min(127, max(-128, round(v))) for v in values.flat], dtype=np.int8).reshape(2, 3)
    assert np.array_equal(expected, independent)


@pytest.mark.parametrize('batch,m,k,n', [(1, 2, 17, 3), (3, 17, 33, 19)])
def test_real_bmm_proof_covers_producer_and_retains_source_readout(batch, m, k, n):
    torch = pytest.importorskip('torch')
    m2m = pytest.importorskip('m2m')
    from m2m.capture.torchao_pipeline import QuantizationConfig
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_spec_int8 import provider_coverage
    from mlir_oot.contraction_patterns import match_integer_gemm
    from mlir_oot.core_aten.lint import lint

    class Bmm(torch.nn.Module):
        def forward(self, a, b):
            return torch.ops.aten.bmm.default(a, b)

    converted = m2m.convert(Bmm().eval(), (torch.randn(batch, m, k), torch.randn(batch, k, n)),
        backend='fx_importer', quantization=QuantizationConfig(scheme='int8_static_act_int8_weight',
            extra_args={'profile': 'int8_accelerator'}))
    assert converted.ok, converted.diagnostics
    module = parse_mlir_text(converted.mlir_text)
    text = str(module)
    path = Path(__file__).parents[2] / 'merlin-support/core_aten/execution_provider.py'
    spec = importlib.util.spec_from_file_location('real_bmm_provider_test', path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    policy = dict(numerical_semantics=dict(accumulator_dtype='i32', internal_arithmetic=dict(
        signed_operand_bits=8, mac_result_bits=20, mac_result_overflow='wrap_to_20_bits',
        full_operation_overflow_policy='bounded_exact_requires_each_partial_sum')))
    proof = provider.admit_numeric_schedule(text, package=path.parents[2], numeric_policy=policy)
    contraction, = [op for op in module.walk() if match_integer_gemm(op, require_zero=False)]
    assert provider_coverage(module, proof, contraction) is None
    _, manifest, matches = build(text)
    match, = [m for m in matches if m.kernel.meta.get('integer_readout')]
    assert match.kernel.meta['dimensions'] == dict(batch=batch, m=m, k=k, n=n)
    rewritten = rewrite(text, manifest)
    assert 'prov.quant.arithmetic = "exact"' in rewritten
    rng = np.random.default_rng(282)
    feed = dict(a=rng.integers(-128, 128, (batch, m, k), dtype=np.int8),
                b=rng.integers(-128, 128, (batch, k, n), dtype=np.int8))
    lint(match.kernel.stream)
    assert np.array_equal(simulate(match.kernel.stream, feed)['out'], evaluate(match, feed)['out'])


READINESS = ['add_Scalar', 'sub_Scalar', '_to_copy_f64', 'full_like_f64', 'fill_Scalar',
    'repeat', 'avg_pool3d', '_adaptive_avg_pool3d', 'avg_pool2d_backward',
    '_adaptive_avg_pool2d_backward', 'col2im', 'reflection_pad1d', 'reflection_pad2d',
    'reflection_pad3d', 'replication_pad2d', 'replication_pad3d', 'convolution_backward',
    'mean_all_dims', 'mean_all_keepdim', 'mean_all']


@pytest.fixture(scope='module')
def readiness_cases():
    pytest.importorskip('m2m')
    torch = pytest.importorskip('torch')
    sys.path.insert(0, str(Path(__file__).parent / 'e2e'))
    from capture_rtl import rtl_set
    from capture import case, rnd, A, QD
    class FloatWidth(torch.nn.Module):
        # Match the lane's code copy/fill boundary: Q/DQ uses f32, while
        # the requested float-width conversion and returned dtype remain f64.
        def __init__(self, fill):
            super().__init__()
            self.fill = fill

        def forward(self, codes):
            x = QD.dequantize_per_tensor(codes, 0.03125, 0, -128, 127, torch.int8)
            wide = (A.full_like.default(x, 0.75, dtype=torch.float64) if self.fill else
                    A._to_copy.default(x, dtype=torch.float64))
            codes = QD.quantize_per_tensor(wide.float(), 0.03125, 0, -128, 127, torch.int8)
            return QD.dequantize_per_tensor(codes, 0.03125, 0, -128, 127, torch.int8).double()
    cases = rtl_set()
    cases += [
        ('_to_copy_f64', FloatWidth(False), [torch.arange(-7, 8, dtype=torch.int8).reshape(3, 5)]),
        ('full_like_f64', FloatWidth(True), [torch.arange(-7, 8, dtype=torch.int8).reshape(3, 5)]),
        case('mean_all_dims', lambda x: A.mean.dim(x, [0, 1]), [rnd(3, 5)]),
        case('mean_all_keepdim', lambda x: A.mean.dim(x, [0, 1], True), [rnd(3, 5)]),
        case('mean_all', lambda x: A.mean.default(x), [rnd(3, 5)]),
    ]
    return {name: (model, inputs) for name, model, inputs in cases if name in READINESS}


@pytest.mark.parametrize('name', READINESS)
def test_real_readiness_capture_is_exact_with_unit_device_scales(name, readiness_cases):
    import m2m
    from mlir_oot.frontend.parse import parse_module
    from exact_oracle import SourceOracle
    model, inputs = readiness_cases[name]
    converted = m2m.convert(model.eval(), tuple(inputs), backend='fx_importer')
    assert converted.ok, converted.diagnostics
    text = converted.mlir_text
    module = parse_module(text)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    fn = next(op for op in module.walk() if op.name == 'func.func')
    arguments = list(fn.body.blocks[0].args)
    values = {('arg', i): x.numpy() for i, x in enumerate(inputs)}
    def key(value):
        return ('arg', arguments.index(value)) if value in arguments else ('op', ordinals[value.owner], list(value.owner.results).index(value))
    def source_key(source):
        return ('arg', source['func_arg']) if 'func_arg' in source else ('op', source['op_ordinal'], source['result'])
    oracle = SourceOracle({arg: np.asarray(x.numpy(), dtype=object) for arg, x in zip(arguments, inputs)})
    offloaded_oracle = SourceOracle({arg: np.asarray(x.numpy(), dtype=object) for arg, x in zip(arguments, inputs)})
    def source_value(source):
        return arguments[source['func_arg']] if 'func_arg' in source else list(module.walk())[source['op_ordinal']].results[source['result']]
    _, manifest, matches = build(text)
    assert matches, manifest['declines']
    for binding, match in zip(manifest['bindings'], matches):
        feed = {name: values[source_key(src)] if source_key(src) in values else
                np.asarray(oracle.tensor(source_value(src)), dtype=np.int64).astype(np.int8)
                for name, src in binding['operands'].items()}
        actual = simulate(match.kernel.stream, feed)[match.result[0]]
        assert np.array_equal(actual, evaluate(match, feed)[match.result[0]])
        result_source, = binding['result'].values()
        values[source_key(result_source)] = actual
        offloaded_oracle.memo[source_value(result_source)] = actual.astype(object)
        assert all(i.a('scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_ld')
        assert all(i.a('acc_scale') == 1 for i in match.kernel.stream.ins if i.kind == 'config_st')
        assert all(i.a('local') & FULL_BIT for i in match.kernel.stream.ins if i.kind == 'mvout' and i.a('local') & ACC_BIT)
    ret = next(op for op in fn.walk() if op.name == 'func.return')
    for value in ret.operands:
        if name in ('_to_copy_f64', 'full_like_f64'):
            assert str(value.type.get_element_type()) == 'f64'
            assert np.array_equal(offloaded_oracle.tensor(value), oracle.tensor(value))
        else:
            assert key(value) in values, manifest['declines']
            expected = np.asarray(oracle.tensor(value), dtype=np.int64).astype(values[key(value)].dtype)
            assert np.array_equal(values[key(value)], expected)


@pytest.fixture(scope="module", params=[False, True], ids=["no_bias", "bias"])
def real_capture(request):
    torch = pytest.importorskip("torch")
    m2m = pytest.importorskip("m2m")
    from m2m.capture.torchao_pipeline import QuantizationConfig

    # Real conversion must use the same source revision as the vendored schema.
    lock = Path(__file__).parents[2] / "merlin-support/quant_schema.lock.json"
    revision = json.loads(lock.read_text())["revision"]
    checkout = Path(m2m.__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == revision, (
        "real model2MLIR capture must use the schema-pinned revision"
    )
    subprocess.run(
        ["git", "-C", str(checkout), "diff", "--exit-code", revision, "--", "m2m"],
        check=True,
        stdout=subprocess.PIPE,
    )

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(32, 16, bias=request.param)
            self.second = torch.nn.Linear(16, 8, bias=request.param)

        def forward(self, x):
            return self.second(torch.relu(self.first(x)))

    torch.manual_seed(0)
    result = m2m.convert(
        Net().eval(),
        (torch.randn(4, 32),),
        backend="fx_importer",
        quantization=QuantizationConfig(
            scheme="int8_static_act_int8_weight",
            extra_args={"profile": "int8_accelerator"},
        ),
    )
    assert result.ok
    assert result.integerization_receipt["quantized_contractions_integerized"] == 2
    return result.mlir_text


def test_real_capture_binds_both_contractions_and_retains_exact_host_readout(
    real_capture,
):
    _, manifest, matches = build(real_capture)
    selected = [m for m in matches if m.kernel.meta.get("integer_readout")]
    assert len(selected) == 2, manifest["declines"]
    rewritten = rewrite(real_capture, manifest)
    assert rewritten.count('prov.quant.arithmetic = "exact"') >= 2
    assert "prov.quant.scale_numerator" in rewritten
    rng = np.random.default_rng(4821)
    for match in selected:
        assert match.result[1].type.get_element_type().width.data == 32
        feed = {
            a.name: rng.integers(-128, 128, a.shape, dtype=np.int8)
            for a in match.kernel.args
            if a.role == "input"
        }
        assert np.array_equal(
            simulate(match.kernel.stream, feed)["out"], evaluate(match, feed)["out"]
        )
        assert (
            not match.kernel.stream.readouts
        )  # the actual source readout stays on host
        stores = [
            i
            for i in match.kernel.stream.ins
            if i.kind == "mvout" and i.a("local") & ACC_BIT
        ]
        assert stores and all(i.a("local") & FULL_BIT for i in stores)


def test_informational_scale_is_optional_and_unknown_keys_are_rejected(real_capture):
    import re

    text = re.sub(r",? prov.quant.scale = (?:\[[^\]]*\]|[^,}]+)", "", real_capture)
    _, manifest, matches = build(text)
    assert sum(bool(m.kernel.meta.get("integer_readout")) for m in matches) == 2, (
        manifest["declines"]
    )
    # A metadata misread must never fall back to narrowing or zero coverage.
    text = real_capture.replace(
        'prov.quant.arithmetic = "exact"',
        'prov.quant.arithmetic = "exact", prov.quant.typo = "bad"',
    )
    _, manifest, matches = build(text)
    assert not any(m.kernel.meta.get("integer_readout") for m in matches)
    assert any("unknown prov.quant" in d["reason"] for d in manifest["declines"])


def test_real_capture_provider_routes_and_records_two_producer_proofs(
    real_capture, tmp_path
):
    path = Path(__file__).parents[2] / "merlin-support/core_aten/execution_provider.py"
    spec = importlib.util.spec_from_file_location("real_provider_test", path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    policy = dict(
        numerical_semantics=dict(
            accumulator_dtype="i32",
            internal_arithmetic=dict(
                signed_operand_bits=8,
                mac_result_bits=20,
                mac_result_overflow="wrap_to_20_bits",
                full_operation_overflow_policy="bounded_exact_requires_each_partial_sum",
            ),
        )
    )
    proof = provider.admit_numeric_schedule(
        real_capture, package=path.parents[2], numeric_policy=policy
    )
    assert len(proof["kernels"]) == 2
    (tmp_path / "model.mlir").write_text(real_capture)
    from merlin.targetgen.core_aten_device import selected_facts

    fixture = path.parents[1] / "tests/fixtures/core_aten_device/facts.json"
    with selected_facts("synthetic_core_aten", fixture) as facts:
        route = provider.routing(
            tmp_path,
            target="synthetic_core_aten",
            package=path.parents[2],
            facts=facts,
            eligible=[(None, SimpleNamespace(dtypes=("i8", "i8", "i32")))],
        )
    assert callable(route.region_offload)


@pytest.mark.parametrize(
    "key,value", [("prov.quant.host_only", "true"), ("prov.quant.typo", "bad")]
)
def test_plain_catalog_cannot_bypass_quant_placement_policy(real_capture, key, value):
    from mlir_oot.golden_device_catalog import build_catalog
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.contraction_patterns import match_integer_gemm
    from xdsl.dialects.builtin import StringAttr

    module = parse_module(real_capture)
    _, baseline = build_catalog(real_capture)
    assert baseline["covered_contractions"] == 2
    for op in module.walk():
        if match_integer_gemm(op) is not None:
            op.attributes[key] = StringAttr(value)
    _, refused = build_catalog(str(module))
    assert refused["covered_contractions"] == 0
    assert not refused["coverage_complete"]
    assert len(refused["declines"]) == 2
