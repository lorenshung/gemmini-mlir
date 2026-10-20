"""Static actual-emitter and independent state checks, never live qualification."""

import copy
import hashlib
import importlib
import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from merlin.compile.scheduling.block_schedule import (
    Preload,
    check_residency,
    execute,
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    oot = Path(os.environ.get("PANEL_OOT_ROOT", Path(__file__).resolve().parents[2]))
    baseline = oot / "phase1_generated/el4_fire_aligned_v26_original_29213759/package"
    assert (
        hashlib.sha256((baseline / "contraction.py").read_bytes()).hexdigest()
        == "b96d460ff0cdb19731f79104a16c3c2d0d7d6872a81c6826d208ab0d98369d59"
    )
    contraction = load("panel_original_contraction", baseline / "contraction.py")
    monkeypatch.setitem(__import__("sys").modules, "contraction", contraction)
    monkeypatch.syspath_prepend(str(oot / "merlin-support/backend"))
    panel = importlib.import_module("gemmini_v26_rhs_panel")
    builder = load("panel_original_builder", baseline / "builder.py")
    finite = importlib.import_module("gemmini_v26_staged_source")
    return panel, builder, finite


@pytest.fixture
def decoder(modules):
    from gemmini_selected_scheduling import SelectedSchedulingTarget

    if not os.environ.get("PANEL_FACTS") or not os.environ.get("PANEL_CONTRACT"):
        pytest.skip("explicit original facts/operand-contract metadata required")
    selected = SelectedSchedulingTarget.read(
        os.environ["PANEL_FACTS"], os.environ["PANEL_CONTRACT"]
    )
    observer = selected.operand_observer()
    constants = observer.isa_constants("gemmini")
    yield selected, observer, constants
    selected.revalidate()


def emitted(modules, shape, **kwargs):
    panel, builder, finite = modules
    record = finite._FiniteBuilder(builder.Builder(["a", "b", "c"]), 200000)
    ok = panel.try_emit(
        record,
        "%arg0",
        "%arg1",
        "%arg2",
        *shape,
        {"output_dtype": "i32", "epilogue": []},
        private_rows_owned=True,
        **kwargs,
    )
    assert ok
    return record


def stream(record, decoder):
    _selected, observer, constants = decoder
    addresses = {}
    for name, (pointer, index, dtype) in record.addresses.items():
        base = {"%arg0": 1 << 32, "%arg1": 2 << 32, "%arg2": 3 << 32}.get(
            pointer, 4 << 32
        )
        addresses[name] = base + index * (4 if dtype == "i32" else 1)
    result = []
    for funct, first, second, _start, _end in record.events:
        if funct is None:
            result.append({"class": "FENCE"})
            continue
        first = addresses.get(first, first)
        operands = {"raw": first, "kind": "const"}, {"raw": second, "kind": "const"}
        kind, fields = observer.decode_instruction(funct, *operands, constants)
        result.append(
            {"class": kind, "decoded": fields, "rs1": operands[0], "rs2": operands[1]}
        )
    return result


def check(commands, shape):
    from gemmini_packet_contractions import Contraction, check_packets

    m, k, n = shape

    def memory(address, width):
        for base, count, name, element in [
            (1 << 32, m * k, "a", 1),
            (2 << 32, k * n, "b", 1),
            (3 << 32, m * n, "c", 4),
        ]:
            if (
                width == element
                and base <= address < base + count * element
                and (address - base) % element == 0
            ):
                return name, (address - base) // element
        raise ValueError("unowned static source pointer")

    def ex(a, b):
        if (a, b) != (4 | (1 << 16) | (0x3F800000 << 32), 1 << 48):
            raise ValueError("changed original identity EX")

    def ld(command):
        f = command["decoded"]
        if f["state_id"] not in (0, 1, 2) or f["scale_bits"] != 0x3F800000:
            raise ValueError("changed identity stream")

    return check_packets(
        commands,
        contractions=[Contraction(0, "a", "b", "c", m, k, n)],
        dim=16,
        spad_rows=16384,
        acc_rows=1024,
        address_bits=32,
        data_bits=29,
        accumulator_flag=0x80000000,
        accumulate_flag=0x40000000,
        full_width_flag=0x20000000,
        memory_cell=memory,
        zero_cell=lambda a, w: w == 4 and 4 << 32 <= a < (4 << 32) + 64,
        execute_contract=ex,
        weight_shape=lambda x: (x >> 48, (x >> 32) & 65535),
        compute_shape=lambda x: (x >> 48, (x >> 32) & 65535),
        preload_shape=lambda x: (x >> 48, (x >> 32) & 65535),
        load_configuration=ld,
        load_streams={"MVIN": 0, "MVIN2": 1, "MVIN3": 2},
    )


@pytest.mark.parametrize(
    "shape",
    [
        (17, 16, 32),
        (64, 256, 64),
        (64, 464, 48),
        (81, 641, 63),
        (33, 33, 529),
        (32, 1041, 17),
    ],
)
def test_actual_original_builder_stream_full_products_tails_and_raw_ban(
    modules, decoder, shape
):
    record = emitted(modules, shape)
    result = check(stream(record, decoder), shape)
    assert result["products"] == np.prod(shape)
    assert result["raw_i32_values"] == shape[0] * shape[2]
    assert {f for f, *_ in record.events if f is not None} <= {0, 1, 2, 3, 4, 6, 14}
    assert not (
        {8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 25}
        & {f for f, *_ in record.events}
    )
    loads = [e for e in record.events if e[0] == 1]
    mt = (shape[0] + 15) // 16
    kt = (shape[1] + 15) // 16
    nt = (shape[2] + 15) // 16
    group = modules[0].choose_group(*shape).block.m
    assert len(loads) == nt * ((mt + group - 1) // group) * kt < mt * nt * kt
    llvm = record.builder.finish()
    assert 'asm sideeffect "fence", "~{memory}"' in llvm
    assert "kernel_complete:" in llvm and "zeroinitializer" in llvm


@pytest.mark.parametrize("shape", [(31, 47, 15), (33, 17, 16), (32, 65, 17)])
def test_shared_plan_independent_numeric_values_and_accumulator_lifetimes(
    modules, shape
):
    panel, _, _ = modules
    plan = panel.stripe_schedule(*shape)
    rng = np.random.default_rng(925)
    a = rng.integers(-5, 6, (shape[0], shape[1]), dtype=np.int32)
    b = rng.integers(-5, 6, (shape[1], shape[2]), dtype=np.int32)
    np.testing.assert_array_equal(execute(plan, a, b), a @ b)
    broken = list(plan.ops)
    at = next(
        i for i, x in enumerate(broken) if isinstance(x, Preload) and x.accumulate
    )
    broken[at] = replace(broken[at], accumulate=False)
    with pytest.raises(ValueError):
        check_residency(replace(plan, ops=tuple(broken)))


@pytest.mark.parametrize(
    "change", ["seed", "later_reset", "early_store", "clobber", "tail", "owner"]
)
def test_mutated_actual_stream_refuses_in_original_independent_packet_checker(
    modules, decoder, change
):
    shape = (33, 33, 529)
    commands = stream(emitted(modules, shape), decoder)
    if change == "seed":
        at = next(i for i, x in enumerate(commands) if x["class"] == "MVIN3")
        del commands[at]
    elif change == "later_reset":
        at = next(
            i for i, x in enumerate(commands) if x["class"] == "COMPUTE_PRELOADED"
        )
        seed = copy.deepcopy(next(x for x in commands if x["class"] == "MVIN3"))
        commands.insert(at + 1, seed)
    elif change == "early_store":
        at = next(i for i, x in enumerate(commands) if x["class"] == "MVOUT")
        store = commands.pop(at)
        commands.insert(
            next(i for i, x in enumerate(commands) if x["class"] == "PRELOAD"), store
        )
    elif change == "clobber":
        at = next(i for i, x in enumerate(commands) if x["class"] == "MVIN2")
        bad = commands[at]
        bad["decoded"]["addr"] = 0
        bad["rs2"]["raw"] = bad["rs2"]["raw"] & ~0xFFFFFFFF
    elif change == "tail":
        at = next(i for i, x in enumerate(commands) if x["class"] == "MVOUT")
        commands[at]["decoded"]["cols"] += 1
    else:
        at = next(i for i, x in enumerate(commands) if x["class"] == "MVIN2")
        commands[at]["rs1"]["raw"] = 1 << 32
    with pytest.raises(ValueError):
        check(commands, shape)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"private_rows_owned": False},
        {"transpose_rhs": True},
        {"resident": {}},
        {"attrs": {"output_dtype": "i8"}},
        {"attrs": {"bias": "b"}},
        {"attrs": {"acc_scale": 1.0}},
        {"attrs": {"epilogue": ["relu"]}},
        {"attrs": {"output_dtype": "i32", "acc_scale_by_column": [1.0]}},
    ],
)
def test_unsupported_semantics_decline_before_emitting_anything(modules, kwargs):
    panel, builder, _ = modules
    b = builder.Builder(["a", "b", "c"])
    old = (list(b.lines), list(b.globals))
    arguments = {"private_rows_owned": True, "attrs": {"output_dtype": "i32"}}
    arguments.update(kwargs)
    attrs = arguments.pop("attrs")
    assert (
        panel.try_emit(b, "%arg0", "%arg1", "%arg2", 64, 464, 48, attrs, **arguments)
        is False
    )
    assert (b.lines, b.globals) == old


def test_legacy_knee_resource_decline_and_no_unsupported_partialK(modules):
    panel, _, _ = modules
    assert panel.choose_group(17, 16, 32).block.m == 2
    assert panel.choose_group(16, 1024, 128) is None
    assert panel.choose_group(32, 16384, 32) is None
    c = panel.choose_group(81, 641, 63)
    assert c.block.m >= 2 and c.block.k == (641 + 15) // 16 and c.block.n == 1
    assert c.block.operand_bytes <= panel.GEOMETRY.operand_rows * 16
    assert c.block.accumulator_bytes <= panel.GEOMETRY.accumulator_rows * 64
    assert c.static_commands <= panel.STATIC_BUDGET


def test_selected_geometry_is_exact_pinned_facts_and_small_views_decline(modules):
    import json

    if not os.environ.get("PANEL_FACTS"):
        pytest.skip("explicit original facts metadata required")
    panel, _, _ = modules
    data = Path(os.environ["PANEL_FACTS"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == panel.FACTS_SHA256
    facts = json.loads(data)["facts"]
    stores = {row["name"]: row for row in facts["memories"]}
    assert (
        panel.GEOMETRY.operand_rows
        == stores["scratchpad"]["banks"] * stores["scratchpad"]["depth"]
    )
    assert (
        panel.GEOMETRY.accumulator_rows
        == stores["accumulator"]["banks"] * stores["accumulator"]["depth"]
    )
    assert (
        panel.GEOMETRY.block
        == stores["scratchpad"]["row_elems"]
        == stores["accumulator"]["lanes"]
    )
    assert (
        panel.choose_group(64, 464, 48, replace(panel.GEOMETRY, accumulator_rows=16))
        is None
    )
    assert (
        panel.choose_group(64, 464, 48, replace(panel.GEOMETRY, operand_rows=32))
        is None
    )
    assert (
        panel.choose_group(
            64, 464, 48, replace(panel.GEOMETRY, separate_accumulator_space=False)
        )
        is None
    )
