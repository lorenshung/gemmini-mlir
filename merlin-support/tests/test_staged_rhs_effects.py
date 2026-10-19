"""Actual-cell omission induction controls; immutability is a separate premise."""

import copy
import importlib
from pathlib import Path

import pytest


def command(kind, first=0, second=0, **fields):
    return (kind, first, second), {"class": kind, "decoded": fields}


def example():
    return [
        command("CONFIG_EX"),
        command("CONFIG_LD", stride=2),
        command("MVIN", ("lhs", 0), 1, is_acc_addr=False, addr=0, rows=2, cols=2),
        command("MVIN", ("rhs", 0), 2, is_acc_addr=False, addr=2, rows=2, cols=2),
        command("PRELOAD"),
        command("COMPUTE"),
        (("FENCE", 0, 0), None),
        command("MVIN", ("lhs", 4), 3, is_acc_addr=False, addr=0, rows=1, cols=2),
        command("MVIN", ("rhs", 0), 2, is_acc_addr=False, addr=2, rows=2, cols=2),
        command("PRELOAD"),
        command("COMPUTE"),
        (("FENCE", 0, 0), None),
    ]


@pytest.fixture
def check(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_staged_rhs_effects").check_rhs_omissions


def test_repeated_exact_rhs_cells_permit_only_the_redundant_transfer(check):
    before = example()
    after = before[:8] + before[9:]
    report = check(before, after, rhs="rhs", operand_rows=4, dim=2)
    assert report["actual_rhs_loads_removed"] == 1
    assert report["all_other_commands_fences_and_settings_unchanged"] is True
    assert report["rhs_immutability"].startswith("required")
    assert (
        check(before, before, rhs="rhs", operand_rows=4, dim=2)[
            "actual_rhs_loads_removed"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "damage",
    [
        "clobber",
        "offset",
        "stride",
        "reset",
        "first_load",
        "compute",
        "fence",
        "insert",
        "capacity",
        "tail",
    ],
)
def test_changed_owners_cells_dimensions_and_required_work_refuse(check, damage):
    before = example()
    if damage == "clobber":
        before[7][1]["decoded"]["addr"] = 2
    elif damage == "offset":
        before[8] = command(
            "MVIN", ("rhs", 1), 9, is_acc_addr=False, addr=2, rows=2, cols=2
        )
    elif damage == "stride":
        before.insert(8, command("CONFIG_LD", stride=3))
    elif damage == "reset":
        before.insert(8, command("CONFIG_EX"))
    elif damage == "capacity":
        before[3][1]["decoded"]["rows"] = 3
    elif damage == "tail":
        before[8][1]["decoded"]["cols"] = 3
    index = 9 if damage in {"stride", "reset"} else 8
    after = copy.deepcopy(before[:index] + before[index + 1 :])
    if damage == "first_load":
        after = before[:3] + before[4:]
    elif damage == "compute":
        after = before[:5] + before[6:]
    elif damage == "fence":
        after = before[:6] + before[7:]
    elif damage == "insert":
        after.append(command("COMPUTE"))
    with pytest.raises(ValueError):
        check(before, after, rhs="rhs", operand_rows=4, dim=2)
