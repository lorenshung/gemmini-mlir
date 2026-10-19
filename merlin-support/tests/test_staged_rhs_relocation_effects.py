"""Consume-aware actual operand comparisons; fixtures issue no target owner."""

import copy

import pytest
from gemmini_staged_rhs_relocation_effects import (
    RhsPlacementRealizationChecker,
    canonical_relocated_stream,
    original_source_capsule,
)
from gemmini_timing_resources import _pin

from merlin.compile.scheduling.staged_dense import StagedDenseProblem


def shape(addr, rows, cols):
    return addr | cols << 4 | rows << 8


def command(kind, first=0, second=0, **fields):
    return [kind, first, second], dict(class_name=kind, decoded=fields, rs1=dict(raw=first), rs2=dict(raw=second))


def example(*, candidate=False, load_after_issue=False):
    rows = [
        command("CONFIG_EX"),
        command("CONFIG_LD", stride=3),
        command("MVIN", ["lhs", 0], shape(0, 2, 2), addr=0, rows=2, cols=2, is_acc_addr=False),
        command("CONFIG_LD", stride=2),
        command("MVIN", ["rhs", 0], shape(2, 2, 2), addr=2, rows=2, cols=2, is_acc_addr=False),
        command("PRELOAD", shape(2, 2, 2)),
        command("COMPUTE_PRELOADED", shape(0, 2, 2)),
        (["FENCE", 0, 0], None),
        command("CONFIG_LD", stride=3),
        command("MVIN", ["lhs", 2], shape(0, 2, 1), addr=0, rows=2, cols=1, is_acc_addr=False),
        command("CONFIG_LD", stride=2),
    ]
    address = 4 if candidate else 2
    load = command("MVIN", ["rhs", 4], shape(address, 1, 2), addr=address, rows=1, cols=2, is_acc_addr=False)
    preload = command("PRELOAD", shape(address, 1, 2))
    rows.extend([preload, (["FENCE", 0, 0], None), load] if load_after_issue else [load, preload])
    return [*rows, command("COMPUTE_PRELOADED", shape(0, 2, 1)), (["FENCE", 0, 0], None)]


def normalize(stream, *, baseline):
    return canonical_relocated_stream(
        stream,
        lhs="lhs",
        rhs="rhs",
        problem=StagedDenseProblem(2, 3, 2),
        dim=2,
        data_bits=4,
        shape_fields={"rows": (8, 4), "cols": (4, 4)},
        baseline=baseline,
    )


@pytest.mark.parametrize("load_after_issue", [False, True])
def test_baseline_and_relocated_actual_operands_match_only_after_checked_cell_relocation(load_after_issue):
    original = example(load_after_issue=load_after_issue)
    before = copy.deepcopy(original)
    candidate = example(candidate=True, load_after_issue=load_after_issue)
    left, right = normalize(original, baseline=True), normalize(candidate, baseline=False)
    assert [event for event, _ in left] == [event for event, _ in right]
    assert original == before
    preload = [event for event, _ in left if event[0] == "PRELOAD"]
    assert preload[-1][1] & 15 == 4


@pytest.mark.parametrize(
    "damage", ["physical", "source", "stride", "descriptor", "missing_compute", "retained", "unknown_owner"]
)
def test_actual_cell_source_config_and_pending_state_mismatches_refuse(damage):
    rows = example(candidate=True)
    tail = next(i for i, (event, _) in enumerate(rows) if event[0] == "MVIN" and event[1] == ["rhs", 4])
    if damage == "physical":
        rows[tail][1]["decoded"]["addr"] = 5
    elif damage == "source":
        rows[tail][0][1][1] = 5
    elif damage == "stride":
        rows[tail - 1][1]["decoded"]["stride"] = 3
    elif damage == "descriptor":
        rows[tail + 1][0][1] = shape(2, 1, 2)
    elif damage == "missing_compute":
        rows = rows[:-2]
    elif damage == "retained":
        rows[-2][0][0] = "COMPUTE_ACCUMULATE"
    else:
        rows[tail][0][1][0] = "other"
    with pytest.raises(ValueError):
        normalize(rows, baseline=False)


def test_json_qualification_cannot_construct_original_relocation_checker():
    with pytest.raises(ValueError, match="original live selected"):
        RhsPlacementRealizationChecker(selected={"status": "qualified"}, arithmetic={"scratchpad_rows": 16384}, nm="/")


def test_explicit_original_publication_needs_pinned_identical_complete_typed_source(tmp_path):
    original, published = (tmp_path / name for name in ("diagnostic", "published"))
    for path in (original, published):
        path.mkdir()
        (path / "capsule.yaml").write_text("declaration")
        (path / "capsule.interface.mlir").write_text("complete typed source")
    native = dict(source=dict(capsule_declaration=_pin(original / "capsule.yaml")))
    assert original_source_capsule({}, native) == original
    arm = dict(source_capsule_declaration=_pin(published / "capsule.yaml"))
    assert original_source_capsule(arm, native) == published
    (published / "capsule.interface.mlir").write_text("different source with same output shape")
    with pytest.raises(ValueError, match="original typed source"):
        original_source_capsule(arm, native)
    (published / "capsule.interface.mlir").write_text("complete typed source")
    (published / "capsule.yaml").write_text("changed publication")
    with pytest.raises(ValueError):
        original_source_capsule(arm, native)
