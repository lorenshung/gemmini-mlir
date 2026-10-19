"""Independent final issue-sequence controls; no source request stream as oracle."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def checker():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/shared_load_configuration/qualify.py"
    )
    spec = importlib.util.spec_from_file_location(
        "load_configuration_final_check", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sequence():
    fence, setter, load = (
        ("FENCE", 0, 0),
        ("CONFIG_LD", 123, 17),
        ("MVIN", ("tensor_0", 0), 999),
    )
    return [fence, setter, load, fence, setter, load, fence]


def test_final_check_accepts_only_redundant_setter_omission(checker, sequence):
    assert checker._check_omissions(sequence, sequence[:4] + sequence[5:], 123) == 1


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 6])
def test_final_check_rejects_missing_fence_first_setter_or_accelerator_work(
    checker, sequence, index
):
    with pytest.raises(ValueError, match="actual target"):
        checker._check_omissions(
            sequence, sequence[:index] + sequence[index + 1 :], 123
        )


def test_final_check_rejects_changed_configuration_value(checker, sequence):
    changed = sequence.copy()
    changed[4] = ("CONFIG_LD", 123, 19)
    with pytest.raises(ValueError, match="actual target"):
        checker._check_omissions(sequence, changed, 123)


def test_final_check_does_not_assume_foreign_command_preserves_load_state(
    checker, sequence
):
    changed = sequence[:3] + [("LOOP_WS", 0, 0)] + sequence[3:]
    with pytest.raises(ValueError, match="actual target"):
        checker._check_omissions(changed, changed[:5] + changed[6:], 123)


def test_linked_pointer_normalization_requires_unique_real_symbol(checker):
    assert checker._memory_name(104, [(100, 32, "tensor_0")]) == ("tensor_0", 4, 32)
    with pytest.raises(ValueError, match="unique linked data symbol"):
        checker._memory_name(104, [(100, 32, "tensor_0"), (102, 8, "alias")])


@pytest.fixture
def transfers():
    return {
        "instructions": [
            {"class": "CONFIG_LD", "decoded": {"state_id": 0, "stride": 8}},
            {
                "class": "MVIN",
                "decoded": {"rows": 2, "cols": 4, "is_acc_addr": False},
                "rs1": {"raw": 100},
            },
            {"class": "CONFIG_ST", "decoded": {"out_stride_bytes": 16}},
            {
                "class": "MVOUT",
                "decoded": {"rows": 2, "cols": 4, "readout": "i32"},
                "rs1": {"raw": 200},
            },
        ]
    }


def test_final_logical_dma_windows_use_linked_capacity_and_actual_width(
    checker, transfers
):
    result = checker._check_transfer_windows(
        transfers,
        [(100, 32, "tensor_0"), (200, 32, "tensor_1")],
        {"tensors": {"A": {"role": "input"}, "Y": {"role": "output"}}},
    )
    assert result["logical_reads"] == result["logical_writes"] == 1
    assert result["physical_beat_overread_and_guards"] == "not_qualified"


@pytest.mark.parametrize(
    "change,match",
    [
        ("store_input", "nonoutput"),
        ("too_many_rows", "span exceeds"),
        ("unknown_width", "width remains unavailable"),
        ("other_state", "state zero"),
    ],
)
def test_final_dma_check_rejects_wrong_scope_and_unknowns(
    checker, transfers, change, match
):
    if change == "store_input":
        transfers["instructions"][-1]["rs1"]["raw"] = 100
    elif change == "too_many_rows":
        transfers["instructions"][-1]["decoded"]["rows"] = 3
    elif change == "unknown_width":
        transfers["instructions"][-1]["decoded"]["readout"] = None
    else:
        transfers["instructions"][0]["decoded"]["state_id"] = 1
    with pytest.raises(ValueError, match=match):
        checker._check_transfer_windows(
            transfers,
            [(100, 32, "tensor_0"), (200, 32, "tensor_1")],
            {"tensors": {"A": {"role": "input"}, "Y": {"role": "output"}}},
        )
