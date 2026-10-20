"""Static generator checks; no compiler service, assembler, simulator or hardware.

The concrete loop expander is a test instrument, not a selected target decoder
or a qualification owner. Final actual-ELF/source/effect checks remain mandatory.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("store_hoist_prepare", HERE / "prepare.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
BASELINE = Path(os.environ.get(
    "GEMMINI_V26_COMPILER_BASELINE",
    HERE.parents[1] / "examples/phase2/public_pass_inputs_v26/compiler",
))


class Expansion:
    """Interpret only generator scalar/layout loops; retain every emitted effect."""
    def __init__(self):
        self.events, self.globals, self.serial, self.addresses = [], [], 0, {}

    def name(self, prefix="v"):
        self.serial += 1
        return prefix + str(self.serial)

    def alloc(self, count, ty="i8", align=64):
        address = 0x100000 + self.serial * 0x10000
        self.serial += 1
        self.events.append(("alloc", count, ty, align, address))
        return address

    def gep(self, pointer, index, ty="i8"):
        if isinstance(pointer, str):
            pointer = self.addresses.setdefault(pointer, 0x800000 + len(self.addresses) * 0x10000)
        return pointer + index * {"i8": 1, "i32": 4}[ty]

    def cast(self, opcode, value, src, dst):
        assert opcode == "ptrtoint" and src == "ptr" and dst == "i64"
        return value

    def op(self, opcode, lhs, rhs, ty="i64"):
        return {"add": lambda: lhs + rhs, "mul": lambda: lhs * rhs,
                "shl": lambda: lhs << rhs, "or": lambda: lhs | rhs}[opcode]()

    def load(self, pointer, ty):
        self.events.append(("load", pointer, ty))
        return pointer % 127

    def store(self, value, pointer, ty):
        self.events.append(("store", value, pointer, ty))

    def cmp(self, predicate, lhs, rhs, ty="i64"):
        assert predicate == "eq"
        return lhs == rhs

    def branch_if(self, cond, callback, other=None):
        if cond:
            callback()
        elif other is not None:
            other()

    def rocc(self, funct, rs1, rs2):
        assert type(funct) is type(rs1) is type(rs2) is int
        self.events.append(("rocc", funct, rs1, rs2))

    def fence(self):
        self.events.append(("fence",))

    def loop(self, begin, end, step, callback):
        assert 0 <= len(range(begin, end, step)) <= 4096
        for value in range(begin, end, step):
            callback(value)


def module(source):
    namespace = {}
    # Only the pinned public generator and the exact candidate rewrite enter here.
    exec(compile(ast.parse(source), "contraction-static-input", "exec"), namespace)  # noqa: S102
    return namespace


def expansion(source, m, k, n, *, attrs=None, transpose=False, resident=False):
    namespace = module(source)
    b = Expansion()
    state = {"k": k, "n": n, "base": 256, "k_tiles": (k+15)//16,
             "n_tiles": (n+15)//16, "source": 0x400000} if resident else None
    namespace["matmul"](
        b, 0x200000, 0x400000, 0x600000, m, k, n,
        attrs or {"output_dtype": "i32"}, {}, {"bias": 0x900000},
        transpose_rhs=transpose, _resident=state,
    )
    return b.events


def is_store_config(event):
    return event[0] == "rocc" and event[1] == 0 and event[2] & 3 == 2


def observed_store_configs(events):
    current = None
    reads = []
    for event in events:
        if is_store_config(event):
            current = event[2:]
        elif event[:2] == ("rocc", 3):
            if current is None:
                raise ValueError("store lacks established configuration")
            reads.append((event, current))
    return reads


def sources():
    original = (BASELINE / "contraction.py").read_text()
    return original, P.rewrite(original)


@pytest.mark.parametrize("shape,transpose,resident,bias", [
    ((17,16,32), False, False, False),
    ((1,16,1), False, False, False),
    ((31,15,17), False, False, False),
    ((32,48,32), False, False, False),
    ((7,17,48), True, False, False),
    ((17,16,17), False, True, False),
    ((17,31,33), False, True, True),
    ((64,256,64), False, False, False),
])
def test_raw_tiles_keep_all_effects_and_each_store_configuration(shape, transpose, resident, bias):
    attrs = {"output_dtype": "i32"}
    if bias:
        attrs["bias"] = "bias"
    before, after = [expansion(source, *shape, attrs=attrs, transpose=transpose, resident=resident)
                     for source in sources()]
    assert observed_store_configs(before) == observed_store_configs(after)
    assert [event for event in before if not is_store_config(event)] == [
        event for event in after if not is_store_config(event)]
    assert sum(map(is_store_config, after)) == 1
    assert sum(map(is_store_config, before)) == ((shape[0]+15)//16)*((shape[2]+15)//16)
    assert {event[1] for event in after if event[0] == "rocc"} <= {0,1,2,3,4,6,14}


@pytest.mark.parametrize("attrs", [
    {"output_dtype": "i8"},
    {"output_dtype": "i8", "epilogue": ["relu"], "acc_scale": 0.5},
    {"output_dtype": "i8", "acc_scale_by_column": [0.5]*8 + [1.0]*9},
    {"output_dtype": "i8", "acc_scale_by_column": [0.5,1.0]*8 + [0.5]},
])
def test_narrow_or_per_column_readout_retains_exact_original_effects(attrs):
    assert expansion(sources()[0], 17, 16, 17, attrs=attrs) == expansion(
        sources()[1], 17, 16, 17, attrs=attrs)


def test_fresh_contraction_reestablishes_store_state_for_a_different_width():
    events = expansion(sources()[1], 17, 16, 32) + expansion(sources()[1], 17, 16, 17)
    assert [event[3] & 0xffffffff for event in events if is_store_config(event)] == [128,68]
    assert [state[1] & 0xffffffff for _,state in observed_store_configs(events)] == [128]*4+[68]*4


def test_induction_detects_omitted_or_wrong_initial_configuration():
    events = expansion(sources()[1], 17, 16, 32)
    without = [event for event in events if not is_store_config(event)]
    with pytest.raises(ValueError, match="lacks established"):
        observed_store_configs(without)
    changed = [(event[0],event[1],event[2],event[3]^4) if is_store_config(event) else event
               for event in events]
    assert observed_store_configs(changed) != observed_store_configs(events)


def test_complete_baseline_and_only_one_changed_source(tmp_path):
    report = P.prepare(BASELINE, tmp_path / "candidate")
    assert report["baseline_sha256"] == P.BASELINE_SHA256
    assert len(report["baseline_files"]) == len(report["candidate_files"]) == 16
    assert report["changed_files"] == ["contraction.py"]
    assert report["compiler_execution"] == "NOT_RUN"
    assert not report["retained_rhs_85_dependency"]


@pytest.mark.parametrize("attack", ["change", "missing", "extra", "link"])
def test_stale_incomplete_or_foreign_baseline_refuses_before_candidate(tmp_path, attack):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    for name in P.FILES:
        target = baseline / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((BASELINE / name).read_bytes())
    target = baseline / "contraction.py"
    if attack == "change":
        target.write_text(target.read_text() + "# changed\n")
    elif attack == "missing":
        target.unlink()
    elif attack == "extra":
        (baseline / "foreign.py").write_text("pass\n")
    else:
        target.unlink()
        target.symlink_to(BASELINE / "contraction.py")
    with pytest.raises(ValueError):
        P.prepare(baseline, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="absent"):
        P.prepare(BASELINE, tmp_path)


def test_rewrite_refuses_changed_generator_sites():
    with pytest.raises(ValueError, match="sites changed"):
        P.rewrite(sources()[0].replace(P.SETUP, ""))


@pytest.mark.parametrize("shape", [(0,16,32), (17,16,0), (17,0,32)])
def test_zero_trip_or_empty_reduction_retains_original_effects(shape):
    assert expansion(sources()[0], *shape) == expansion(sources()[1], *shape)


def test_overflowing_store_stride_never_hoists_configuration_into_setup():
    class SetupOnly(Expansion):
        def loop(self, begin, end, step, callback):
            # Inspect setup only; a billion-column loop is not expanded.
            self.events.append(("unexpanded-loop", begin, end, step))

    rows = []
    for source in sources():
        b = SetupOnly()
        module(source)["matmul"](b,0x200000,0x400000,0x600000,17,16,1<<30,
                                 {"output_dtype":"i32"},{},{})
        rows.append(b.events)
    assert rows[0] == rows[1]
    assert not any(map(is_store_config, rows[1]))
