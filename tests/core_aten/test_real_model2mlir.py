"""Provider integration on actual convert(profile=int8_accelerator) output."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate, FULL_BIT, ACC_BIT
from exact_oracle import evaluate


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
