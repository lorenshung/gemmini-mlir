"""Actual issued-event tile permutation controls; no target authority fixtures."""

import copy
import importlib
from pathlib import Path

import pytest


def command(kind, first=0, second=0, **fields):
    return (kind, first, second), {"class": kind, "decoded": fields}


def tile(m, n):
    return [
        command("CONFIG_LD", stride=0),
        command("MVIN", ("zero", 0), 1, is_acc_addr=True, addr=0, rows=2, cols=2),
        (("FENCE", 0, 0), None),
        command("CONFIG_LD", stride=2),
        command("MVIN", ("lhs", m * 2), 2, is_acc_addr=False, addr=0, rows=2, cols=2),
        (("FENCE", 0, 0), None),
        command("CONFIG_LD", stride=4),
        command("MVIN", ("rhs", n), 3, is_acc_addr=False, addr=2, rows=2, cols=2),
        command("PRELOAD", 1, 2),
        command("COMPUTE", 3, 4),
        (("FENCE", 0, 0), None),
        command("CONFIG_ST", stride=16),
        command("MVOUT", ("out", m * 16 + n * 4), 9, addr=0, rows=2, cols=2),
        (("FENCE", 0, 0), None),
    ]


def program(order):
    return [
        command("CONFIG_EX", 1, 2),
        *(op for m, n in order for op in tile(m, n)),
        (("FENCE", 0, 0), None),
    ]


@pytest.fixture
def check(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module(
        "gemmini_staged_tile_effects"
    ).check_output_tile_permutation


def test_complete_original_tiles_can_move_and_only_live_rhs_transfers_can_disappear(
    check,
):
    before = program(((0, 0), (0, 2), (2, 0), (2, 2)))
    after = program(((0, 0), (2, 0), (0, 2), (2, 2)))
    after = [op for i, op in enumerate(after) if i not in {1 + 14 + 7, 1 + 42 + 7}]
    report = check(before, after, rhs="rhs", operand_rows=4, dim=2)
    assert report["output_tiles"] == 4
    assert report["output_tile_order_changed"] is True
    assert report["actual_rhs_loads_removed"] == 2
    assert (
        check(before, before, rhs="rhs", operand_rows=4, dim=2)[
            "actual_rhs_loads_removed"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "damage",
    [
        "drop_tile",
        "duplicate_tile",
        "readout",
        "zero_stride",
        "drop_fence",
        "drain",
        "compute",
        "config",
        "clobber",
        "first_rhs",
    ],
)
def test_changed_work_lifetimes_settings_or_missing_cell_proof_refuses(check, damage):
    before = program(((0, 0), (0, 2), (2, 0), (2, 2)))
    after = copy.deepcopy(program(((0, 0), (2, 0), (0, 2), (2, 2))))
    if damage == "drop_tile":
        del after[15:29]
    elif damage == "duplicate_tile":
        after[15:29] = copy.deepcopy(after[1:15])
    elif damage == "readout":
        after[1 + 12] = command("MVOUT", ("out", 1), 9, addr=0, rows=2, cols=2)
    elif damage == "zero_stride":
        after[1][1]["decoded"]["stride"] = 1
    elif damage == "drop_fence":
        del after[14]
    elif damage == "drain":
        after.insert(10, command("CONFIG_EX", 1, 2))
    elif damage == "compute":
        after[1 + 9] = command("COMPUTE", 9, 4)
    elif damage == "config":
        after[1 + 11] = command("CONFIG_ST", 7, 8, stride=17)
    elif damage == "clobber":
        before[29 + 4] = command(
            "MVIN", ("lhs", 4), 4, is_acc_addr=False, addr=2, rows=2, cols=2
        )
        after[15 + 4] = copy.deepcopy(before[29 + 4])
        del after[15 + 7]
    else:
        del after[1 + 7]
    with pytest.raises(ValueError):
        check(before, after, rhs="rhs", operand_rows=4, dim=2)
