"""Actual-footprint boundary controls; neutral event fixtures grant no device."""

import copy
import json

import pytest
from gemmini_staged_rhs_window import ClosedScratchEpochOwner, check_original_footprint
from gemmini_timing_resources import _pin


def command(kind, first=0, second=0, **fields):
    return ((kind, first, second), {"class": kind, "decoded": fields})


def example():
    return [
        (("FENCE", 0, 0), None),
        command("FLUSH"),
        command("CONFIG_EX"),
        command("CONFIG_LD"),
        command("MVIN", ("lhs", 0), 0, addr=0, is_acc_addr=False, rows=2, cols=2),
        command("MVIN", ("rhs", 0), 2, addr=2, is_acc_addr=False, rows=2, cols=2),
        command("PRELOAD", 2),
        command("COMPUTE_PRELOADED", 0),
        command("CONFIG_ST"),
        command("MVOUT", ("out", 0), 0),
        (("FENCE", 0, 0), None),
    ]


def check(stream, **kwargs):
    return check_original_footprint(stream, lhs="lhs", rhs="rhs", dim=2, data_bits=4, physical_rows=16, **kwargs)


def test_only_original_complete_footprint_supplies_a_conditional_epoch():
    before = example()
    result = check(before)
    assert before == example()
    assert result["original_operand_windows"] == [[0, 2], [2, 4]]
    assert result["maximum_private_physical_rows"] == 16
    assert result["hardware_execution_qualified"] is False and result["timing_authority"] == "NONE"
    assert result["scope"] == "one_complete_observed_hart0_functional_program"


@pytest.mark.parametrize(
    "damage",
    [
        "foreign_pointer",
        "foreign_window",
        "reconfigure",
        "reload_other_owner",
        "retained",
        "preload",
        "accumulator",
        "readout",
        "no_entry",
        "no_exit",
    ],
)
def test_foreign_or_carried_owners_cannot_turn_capacity_into_a_lease(damage):
    stream = copy.deepcopy(example())
    if damage == "foreign_pointer":
        stream[5] = command("MVIN", ("other", 0), 2, addr=2, is_acc_addr=False, rows=2, cols=2)
    elif damage == "foreign_window":
        stream[5][1]["decoded"]["addr"] = 4
    elif damage == "reconfigure":
        stream.insert(7, command("CONFIG_EX"))
    elif damage == "reload_other_owner":
        stream.insert(7, command("LOOP_WS"))
    elif damage == "retained":
        stream[7] = command("COMPUTE_ACCUMULATE")
    elif damage == "preload":
        stream[6] = command("PRELOAD", 4)
    elif damage == "accumulator":
        stream[5] = command("MVIN", ("zero", 0), 1, addr=1, is_acc_addr=True, rows=2, cols=2)
    elif damage == "readout":
        stream[9] = command("MVOUT", ("out", 0), 1)
    elif damage == "no_entry":
        stream = stream[1:]
    else:
        stream = stream[:-1]
    with pytest.raises(ValueError):
        check(stream)


def test_selected_address_geometry_and_live_source_owner_are_mandatory():
    with pytest.raises(ValueError, match="physical addressing"):
        check_original_footprint(example(), lhs="lhs", rhs="rhs", dim=2, data_bits=3, physical_rows=16)
    with pytest.raises(ValueError, match="original live current-source"):
        ClosedScratchEpochOwner({"status": "qualified", "all_rows_free": True}, capsule_dir="/")


def test_within_pair_source_proof_reuse_refuses_changed_capture_reference_leaves(tmp_path):
    owner = object.__new__(ClosedScratchEpochOwner)
    owner.revalidate = lambda: None
    owner._source_rows = {"original_capture": {"path": "original", "sha256": "selected"}}
    result = tmp_path / "native.json"
    result.write_text(json.dumps({"source": {"original_capture": {"path": "copied-success", "sha256": "selected"}}}))
    with pytest.raises(ValueError, match="exact same original source/input/reference leaves"):
        owner.check_source_arm({"native_result": _pin(result)})
