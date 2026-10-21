"""Exact rational ABI property tests, including all byte schemes and ties."""

from fractions import Fraction as F
from pathlib import Path
import numpy as np
import pytest
from mlir_oot.core_aten.requant import evaluate
from mlir_oot.core_aten.match import Decline
from mlir_oot.core_aten.exactness import prove_readout, prove_affine


@pytest.mark.parametrize(
    "low,high,zp",
    [
        (-128, 127, 0),
        (-127, 127, 0),
        (-128, 127, -7),
        (0, 255, 0),
        (0, 255, 123),
        (1, 255, 128),
    ],
)
def test_python_exact_requant_all_codes_random_accumulators_and_ties(low, high, zp):
    rng = np.random.default_rng(1743 + zp)
    values = np.unique(
        np.concatenate(
            [
                np.arange(-512, 513),
                rng.integers(-(1 << 31), (1 << 31) - 1, 2000),
                [-(1 << 31), (1 << 31) - 1],
            ]
        )
    ).astype(np.int32)
    scales = [
        F(1, 2),
        F(1, 3),
        F(3, 50),
        F(1, 1 << 149),
        F(1 << 127),
        F(float(np.float32(0.03))) / F(float(np.float32(0.01))),
    ]
    scales += [F(float(np.float32(rng.uniform(0.000001, 4)))) for _ in range(16)]
    for scale in scales:
        expected = np.array(
            [min(high, max(low, round(int(x) * scale) + zp)) for x in values],
            dtype=np.int64,
        )
        assert np.array_equal(
            evaluate(
                values,
                scale.numerator,
                scale.denominator,
                zero_point=zp,
                quant_min=low,
                quant_max=high,
            ).astype(np.int64),
            expected,
        )


def test_output_equivalence_is_insufficient_for_product_proof():
    # Saturation masks the product rounding at these inputs; the design rule
    # nevertheless rejects the rounded product.
    with pytest.raises(Decline, match="product"):
        prove_readout(F(3, 8), -10_000_000, 10_000_000)
    with pytest.raises(Decline, match="coefficients"):
        prove_affine([F(1, 3)], 0, [1], 0, F(float(np.float32(1 / 3))))
    assert prove_readout(F(3, 8), -4096, 4096)


def test_oracle_ignores_matcher_numeric_metadata():
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate as source_reference

    p = Path(__file__).parents[1] / "fixtures/core_aten/mean_dim"
    _, _, matches = build(__import__("fixture_source").read(p / "model.mlir"))
    (match,) = matches
    feed = {"x": np.load(p / "inputs.npz")["arg0"]}
    expected = source_reference(match, feed)["y"]
    match.numeric.clear()
    match.numeric.update(s_x=700, s_out=0, scale=123)
    assert np.array_equal(source_reference(match, feed)["y"], expected)
    match.kernel.stream.readouts[0].terms[0] = ("y__i32", F(0))
    assert not np.array_equal(simulate(match.kernel.stream, feed)["y"], expected)


def scheme_source(name, low, high, zp):
    """Vary Q/DQ parameters on a real captured graph, retaining its source ops."""
    from mlir_oot.frontend.parse import parse_module
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

    from fixture_source import read

    module = parse_module(
        read(Path(__file__).parents[1] / "fixtures/core_aten" / name / "model.mlir")
    )
    for op in module.walk():
        nm = getattr(getattr(op, "op_name", None), "data", op.name)
        if (
            not nm.startswith("quant_ext.")
            or str(op.operands[0].type.get_element_type()) == "i32"
        ):
            continue
        op.properties["quant_min"] = IntegerAttr(low, i64)
        op.properties["quant_max"] = IntegerAttr(high, i64)
        if "quantize" in nm and "dequantize" not in nm:
            op.properties["output_dtype"] = StringAttr("uint8" if low >= 0 else "int8")
        else:
            op.properties["input_dtype"] = StringAttr("uint8" if low >= 0 else "int8")
        splat = op.operands[2].owner
        splat.operands[0].owner.properties["value"] = IntegerAttr(zp, i64)
    return str(module)


@pytest.mark.parametrize(
    "name",
    [
        "mean_dim",
        "add_Tensor",
        "sub_Tensor",
        "mul_Scalar",
        "div_Scalar",
        "neg",
        "cat",
        "slice_scatter",
        "relu",
        "mm",
        "bmm",
    ],
)
@pytest.mark.parametrize(
    "low,high,zp", [(-128, 127, 0), (-127, 127, -7), (0, 255, 123), (1, 255, 128)]
)
def test_real_graph_schemes_equal_independent_source_oracle(name, low, high, zp):
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate as source_reference

    text = scheme_source(name, low, high, zp)
    _, manifest, matches = build(text)
    assert matches, manifest["declines"]
    rng = np.random.default_rng(942 + zp)
    for binding, match in zip(manifest["bindings"], matches):
        assert all("func_arg" in src for src in binding["operands"].values()), manifest
        feed = {
            a.name: rng.integers(low, high + 1, a.shape, dtype=np.int32).astype(np.int8)
            for a in match.kernel.args
            if a.role == "input"
        }
        expected = source_reference(match, feed)
        model = simulate(match.kernel.stream, feed)
        assert np.array_equal(model[match.result[0]], expected[match.result[0]])


def test_numpy_fp32_scale_preserves_its_exact_canonical_value():
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.stream import simulate

    si, so = np.float32(0.03), np.float32(0.01)
    x = np.zeros((3, 50), np.int8)
    x[:, :2] = [-62, 87]
    kernel = ops.mean_dim(x.shape, [1], False, s_x=si, s_out=so)
    assert np.array_equal(
        simulate(kernel.stream, {"x": x})["y"], np.full(3, 2, np.int8)
    )


@pytest.mark.parametrize("storage", ["int8", "uint8"])
def test_signedness_is_not_inferred_from_a_positive_narrow_range(storage):
    from mlir_oot.frontend.parse import parse_module
    from xdsl.dialects.builtin import StringAttr
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate as source_reference

    module = parse_module(scheme_source("neg", 0, 127, 0))
    for op in module.walk():
        nm = getattr(getattr(op, "op_name", None), "data", op.name)
        if not nm.startswith("quant_ext."):
            continue
        if "output_dtype" in op.properties:
            op.properties["output_dtype"] = StringAttr(storage)
        if "input_dtype" in op.properties:
            op.properties["input_dtype"] = StringAttr(storage)
    _, manifest, matches = build(str(module))
    assert matches, manifest["declines"]
    (match,) = matches
    arg = next(a for a in match.kernel.args if a.role == "input")
    x = np.resize(np.arange(-128, 128, dtype=np.int8), arg.shape)
    assert np.array_equal(
        simulate(match.kernel.stream, {"x": x})["y"],
        source_reference(match, {"x": x})["y"],
    )


def test_emitted_host_readout_matches_rationals_natively(tmp_path):
    from mlir_oot.core_aten.stream import Stream, Arg
    from mlir_oot.core_aten.requant import Readout
    from conftest import native_readout

    rng = np.random.default_rng(3021)
    values = np.r_[
        np.arange(-1024, 1025),
        rng.integers(-(1 << 31), (1 << 31) - 1, 2000),
        -(1 << 31),
        (1 << 31) - 1,
    ].astype(np.int32)
    for ordinal, (factor, constant) in enumerate(
        [
            (F(1), F(0)),
            (F(1, 2), F(0)),
            (F(-3, 8), F(1, 2)),
            (F(1, 3), F(-1, 5)),
            (F(1, 1 << 149), F(0)),
            (F(1 << 127), F(0)),
            (
                np.resize(np.array([F(1, 2), F(3, 8)], dtype=object), values.shape),
                np.resize(
                    np.array([F(0), F(1, 3), F(-1, 3)], dtype=object), values.shape
                ),
            ),
        ]
    ):
        stream = Stream(
            [
                Arg("x", "i32", values.shape, "input"),
                Arg("y", "i8", values.shape, "output"),
            ]
        )
        stream.readouts.append(
            Readout(
                "y",
                [("x", factor)],
                constant,
                zero_point=123,
                quant_min=1,
                quant_max=255,
            )
        )
        fn, _ = native_readout(stream, "readout", tmp_path)
        result = np.zeros(values.shape, np.uint8)
        fn(values.ctypes.data, result.ctypes.data)
        expected = np.array(
            [
                min(255, max(1, round(int(x) * F(f) + F(c)) + 123))
                for x, f, c in zip(
                    values,
                    np.broadcast_to(factor, values.shape),
                    np.broadcast_to(constant, values.shape),
                )
            ],
            np.uint8,
        )
        assert np.array_equal(result, expected)


@pytest.mark.parametrize("name", ["convolution", "convolution_backward"])
@pytest.mark.parametrize("low,high,zp", [(-127, 127, -7), (1, 255, 128)])
def test_convolution_zero_point_tiles_equal_source(name, low, high, zp):
    from mlir_oot.core_aten.catalog import build
    from mlir_oot.core_aten.stream import simulate
    from exact_oracle import evaluate as source_reference

    _, manifest, matches = build(scheme_source(name, low, high, zp))
    assert matches, manifest["declines"]
    rng = np.random.default_rng(881 + zp)
    for binding, match in zip(manifest["bindings"], matches):
        feed = {
            a.name: rng.integers(low, high + 1, a.shape, dtype=np.int32).astype(
                np.int32 if a.dtype == "i32" else np.int8
            )
            for a in match.kernel.args
            if a.role == "input"
        }
        expected = source_reference(match, feed)
        assert np.array_equal(
            simulate(match.kernel.stream, feed)[match.result[0]],
            expected[match.result[0]],
        )


def test_relu_does_not_hide_inexact_device_product():
    with pytest.raises(Decline, match="conversion is not exact"):
        prove_readout(F(1), -(1 << 25), 0, relu=True)
    with pytest.raises(Decline, match="product is not exact"):
        prove_readout(F(3, 8), -(1 << 24), 0, relu=True)


@pytest.mark.parametrize(
    "low,high,zp", [(-128, 127, 0), (-127, 127, -7), (0, 255, 123), (1, 255, 128)]
)
def test_merlin_exact_host_api_matches_target_reference(low, high, zp):
    api = pytest.importorskip("merlin.runtime.exact_readout")
    rng = np.random.default_rng(7291)
    values = np.concatenate(
        (
            np.arange(-256, 257, dtype=np.int32),
            rng.integers(-(1 << 31), 1 << 31, 1024, dtype=np.int32),
            np.array([-(1 << 31), (1 << 31) - 1], dtype=np.int32),
        )
    )
    for factor in (F(1, 2), F(-3, 8), F(17, 31), F(1, 1 << 149), F(1 << 127), F(0)):
        expected = evaluate(
            values,
            factor.numerator,
            factor.denominator,
            zero_point=zp,
            quant_min=low,
            quant_max=high,
        )
        actual = api.requantize_int32(
            values,
            scale_numerator=str(factor.numerator),
            scale_denominator=str(factor.denominator),
            zero_point=zp,
            quant_min=low,
            quant_max=high,
            rounding="half_to_even",
        )
        assert np.array_equal(actual, expected.astype(np.int64))
