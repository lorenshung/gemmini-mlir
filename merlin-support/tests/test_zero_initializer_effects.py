"""Comparison primitives supply no saved-success or live-owner constructor."""

import copy

import pytest
from gemmini_zero_initializer_effects import (
    ZeroInitializerRealizationChecker,
    canonical_issue_stream,
)


def _cb(names):
    return {"kernel_abi": {"args": [{"tensor": name} for name in names]}}


def _stream(*, zero="old_task.zero", output="tensor_3", offset=0):
    return [
        (["CONFIG_LD", 1, 0], {"decoded": {"stride": 0}}),
        (["MVIN", [zero, offset], 99], {"decoded": {"is_acc_addr": True}}),
        (["FENCE", 0, 0], None),
        (["MVIN", ["tensor_0", 0], 16], {"decoded": {"is_acc_addr": False}}),
        (["MVIN", ["tensor_1", 0], 32], {"decoded": {"is_acc_addr": False}}),
        (["COMPUTE_PRELOADED", 1, 2], {"decoded": {}}),
        (["MVOUT", [output, 0], 33], {"decoded": {}}),
        (["FENCE", 0, 0], None),
    ]


def test_virtual_private_storage_and_proved_zero_symbol_renaming_preserve_logical_issued_owners():
    before, after = _stream(), _stream(zero="new_task.zero", output="tensor_2")
    saved = copy.deepcopy((before, after))
    old = canonical_issue_stream(before, _cb(["a", "b", "private_seed", "out"]))
    new = canonical_issue_stream(after, _cb(["a", "b", "out"]))
    assert [event for event, _ in old] == [event for event, _ in new]
    assert new[-2][0][1] == ["abi/out", 0]
    assert (before, after) == saved


@pytest.mark.parametrize("change", ["input_owner", "output_offset", "compute", "fence", "configuration"])
def test_actual_instruction_or_owner_changes_remain_distinguishable(change):
    original = _stream(output="tensor_2")
    candidate = copy.deepcopy(original)
    if change == "input_owner":
        candidate[3][0][1] = ["tensor_1", 0]
    elif change == "output_offset":
        candidate[-2][0][1][1] = 4
    elif change == "compute":
        candidate[5][0][1] = 19
    elif change == "fence":
        candidate.pop()
    else:
        candidate[0][0][1] = 9
    cb = _cb(["a", "b", "out"])
    old, new = (canonical_issue_stream(stream, cb) for stream in (original, candidate))
    assert [event for event, _ in old] != [event for event, _ in new]


@pytest.mark.parametrize("change", ["operand_static", "output_static", "nonzero_offset", "logical_alias"])
def test_unowned_static_or_aliased_storage_cannot_be_canonicalized(change):
    stream, cb = _stream(), _cb(["a", "b", "seed", "out"])
    if change == "operand_static":
        stream[3][0][1] = ["unowned", 0]
    elif change == "output_static":
        stream[-2][0][1] = ["unowned", 0]
    elif change == "nonzero_offset":
        stream[1][0][1][1] = 4
    else:
        cb["kernel_abi"]["args"][-1]["tensor"] = "a"
    with pytest.raises(ValueError):
        canonical_issue_stream(stream, cb)


def test_json_success_or_foreign_object_cannot_construct_realization_checker():
    with pytest.raises(ValueError, match="original live"):
        ZeroInitializerRealizationChecker(
            selected={"status": "pass"}, arithmetic={"source_rule": "signed_i8_i32"}, nm="/not-used"
        )
