"""Narrow independent semantics of a declared public resident-matmul primitive.

This reads the original interface and numerical stimulus. It neither lowers the
program nor substitutes a typed model-source attestation for a primitive.
"""

from __future__ import annotations

from math import prod

import numpy as np
from gemmini_packet_contractions import Contraction
from merlin.targetgen.contract.interface_emit import (
    emit_interface_mlir,
    parse_interface_mlir,
)


def bind_primitive(capsule, source, command_buffer=None):
    """Accept one complete rank-two raw signed resident contraction only."""
    if (
        capsule.get("kind") != "layer"
        or capsule.get("label") != "public"
        or capsule.get("operation", {}).get("op") != "matmul"
        or capsule.get("numeric_policy") != {"compare": "exact_int", "dtype": "i32"}
        or capsule.get("required_oracle_tiers") != ["L0", "L1", "L2", "L3"]
        or capsule.get("semantic", {}).get("must_accelerate") is not True
        or capsule.get("expected")
        != {
            "instruction_classes": [
                "LOAD_CMD",
                "PRELOAD_CMD",
                "COMPUTE_AND_FLIP_CMD",
                "STORE_CMD",
            ],
            "modes": {"relu": False, "acc_scale": False, "i8": False},
        }
    ):
        raise ValueError(
            "primitive diagnostic needs the original public exact-integer layer and unchanged tiers"
        )
    parsed = parse_interface_mlir(source)
    if emit_interface_mlir(parsed) != source:
        raise ValueError(
            "primitive source has no complete canonical interface round trip"
        )
    commands = parsed["commands"]
    if parsed["target"] != "gemmini" or [row["opcode"] for row in commands] != [
        "RES_PACK",
        "MATMUL_RESIDENT",
        "COMMIT",
        "EVICT",
    ]:
        raise ValueError(
            "primitive source must contain exactly pack, contraction, raw commit and eviction"
        )
    pack, mm, commit, evict = [row["operands"] for row in commands]
    lhs, rhs, output = mm["lhs"], pack["src"], commit["dst"]
    attrs = capsule["operation"].get("attributes", {})
    if (
        len({lhs, rhs, output, pack["dst"], mm["dst"]}) != 5
        or mm["rhs"] != pack["dst"]
        or commit["src"] != mm["dst"]
        or evict != {"handle": pack["dst"]}
        or commands[0].get("attributes", {}) != {"layout": "packed_rhs"}
        or commands[1].get("attributes", {})
        or commands[2].get("attributes", {}) != {"epilogue": [], "output_dtype": "i32"}
        or commands[3].get("attributes", {})
        or attrs
        != {
            "lhs": lhs,
            "weight": rhs,
            "out": output,
            "epilogue": [],
            "output_dtype": "i32",
        }
        or set(parsed["tensors"]) != {lhs, rhs}
    ):
        raise ValueError(
            "primitive source has different operand, lifetime or readout semantics"
        )
    a, b = parsed["tensors"][lhs], parsed["tensors"][rhs]
    if (
        a.get("dtype") != "i8"
        or b.get("dtype") != "i8"
        or a.get("role") != "input"
        or b.get("role") != "weight"
        or len(a["shape"]) != 2
        or len(b["shape"]) != 2
        or any(type(v) is not int or v <= 0 for v in [*a["shape"], *b["shape"]])
        or a["shape"][1] != b["shape"][0]
        or capsule.get("inputs")
        != [dict(name=name, **spec) for name, spec in parsed["tensors"].items()]
    ):
        raise ValueError(
            "primitive declaration and typed source disagree on complete signed operands"
        )
    m, k, n = a["shape"][0], a["shape"][1], b["shape"][1]
    expected = {
        **parsed["tensors"],
        output: {"shape": [m, n], "dtype": "i32", "role": "output"},
    }
    if command_buffer is None:
        command_buffer = dict(parsed, tensors=expected)
    if (
        command_buffer.get("target") != parsed["target"]
        or command_buffer.get("abi_version") != parsed["abi_version"]
    ):
        raise ValueError("primitive compiler changed target or command ABI")
    if set(command_buffer.get("tensors", {})) != set(expected):
        raise ValueError("primitive compiler changed complete input/output roster")
    for name, spec in expected.items():
        actual = command_buffer["tensors"][name]
        if any(actual.get(key) != value for key, value in spec.items()):
            raise ValueError("primitive compiler changed tensor geometry, type or role")
    normalized = lambda rows: [
        dict(row, attributes=row.get("attributes", {})) for row in rows
    ]
    if normalized(command_buffer.get("commands", [])) != normalized(commands):
        raise ValueError(
            "primitive compiler changed complete source commands or numerical settings"
        )
    if m * k * n > 16_777_216 or m * n > 2_000_000:
        raise ValueError("primitive proof exceeds its bounded work/output domain")
    return Contraction(0, lhs, rhs, output, m, k, n)


def check_integer_values(contraction, values, golden, rule):
    """Check exact current operands/reference and both distinct arithmetic scopes."""
    c = contraction
    if set(values) != {c.lhs, c.rhs} or set(golden) != {c.output}:
        raise ValueError("primitive complete operand/reference roster changed")
    arrays = {}
    for name, shape in ((c.lhs, (c.m, c.k)), (c.rhs, (c.k, c.n))):
        value = np.asarray(values[name])
        if (
            value.shape != shape
            or value.dtype.kind not in "iu"
            or value.size != prod(shape)
            or np.any(value < -128)
            or np.any(value > 127)
        ):
            raise ValueError(
                "primitive operand is not its complete original signed i8 tensor"
            )
        arrays[name] = value.astype(np.int64)
    a, b = arrays[c.lhs], arrays[c.rhs]
    bound = int(np.max(np.abs(a))) * int(np.max(np.abs(b)))
    if (
        rule.get("operand_bits") != 8
        or rule.get("accumulator_bits") != 32
        or type(rule.get("max_primitive_k")) is not int
        or not 0 < rule["max_primitive_k"] <= 256
        or type(rule.get("primitive_mac_bits")) is not int
        or not 16 <= rule["primitive_mac_bits"] <= 64
        or min(c.k, rule["max_primitive_k"]) * bound
        >= 1 << (rule["primitive_mac_bits"] - 1)
        or c.k * bound >= 1 << 31
    ):
        raise ValueError(
            "primitive operands do not prove selected panel and cross-panel integer prefix bounds"
        )
    result = a @ b
    expected = np.asarray(golden[c.output])
    if (
        expected.shape != (c.m, c.n)
        or expected.dtype.kind not in "iu"
        or not np.array_equal(result, expected)
    ):
        raise ValueError(
            "primitive complete independent raw i32 reference differs from exact source arithmetic"
        )
    return {
        "products": c.m * c.k * c.n,
        "output_values": c.m * c.n,
        "primitive_prefix_bound": min(c.k, rule["max_primitive_k"]) * bound,
        "accumulator_prefix_bound": c.k * bound,
        "nonconstant_inputs": all(np.unique(x).size > 1 for x in arrays.values()),
        "signed_inputs": all(np.any(x < 0) and np.any(x > 0) for x in arrays.values()),
    }
