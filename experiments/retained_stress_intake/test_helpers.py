import copy

import pytest
from helpers import check_arm_context, compare_complete_integer_outputs


def test_nested_complete_outputs():
    assert (
        compare_complete_integer_outputs({"Y": [-3, 2, 0, 7]}, {"Y": [[-3, 2], [0, 7]]})
        == 4
    )


@pytest.mark.parametrize(
    "value",
    [
        [[-3, 2], [0, 8]],
        [-3, 2, 0],
        [-3, 2, 0, 7, 8],
        [-3.0, 2.0, 0.0, 7.0],
        [True, False, True, False],
    ],
)
def test_bad_complete_outputs_refuse(value):
    with pytest.raises(ValueError):
        compare_complete_integer_outputs({"Y": [-3, 2, 0, 7]}, {"Y": value})


def test_missing_output_refuses():
    with pytest.raises(ValueError):
        compare_complete_integer_outputs({"Y": [1]}, {"Z": [1]})


def test_changed_arm_keeps_own_compiler_identity():
    build = {"context": {"source_sha256": "b" * 64, "compiler_sha256": "a" * 64}}
    m = {"build": build, "source_sha256": "b" * 64}
    assert check_arm_context(m, copy.deepcopy(build)) == "a" * 64
    foreign = copy.deepcopy(build)
    foreign["context"]["compiler_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        check_arm_context(m, foreign)
    m["source_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        check_arm_context(m, build)


def test_reused_build_metadata_only_may_differ():
    build = {
        "context": {"source_sha256": "b" * 64, "compiler_sha256": "a" * 64},
        "build_reused": False,
    }
    m = {"build": copy.deepcopy(build), "source_sha256": "b" * 64}
    m["build"]["build_reused"] = True
    assert check_arm_context(m, build) == "a" * 64
    m["build"]["build_reused"] = 1
    with pytest.raises(ValueError):
        check_arm_context(m, build)
