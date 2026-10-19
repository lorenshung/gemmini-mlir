"""Historical stress operator checks; no grade, execution or timing authority."""

import numpy as np


def check_arm_context(measurement, retained_build):
    """A changed arm keeps its own recorded compiler; it cannot borrow another arm."""
    build = measurement["build"]
    left, right = dict(build), dict(retained_build)
    for value in (left, right):
        if "build_reused" in value and type(value.pop("build_reused")) is not bool:
            raise ValueError("original build-reuse metadata is malformed")
    if left != right:
        raise ValueError(
            "original measurement and its own retained build receipt differ"
        )
    context = build["context"]
    if context["source_sha256"] != measurement["source_sha256"]:
        raise ValueError("original arm source identity differs")
    digest = context["compiler_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or set(digest) - set("0123456789abcdef")
    ):
        raise ValueError("original compiler identity is malformed")
    return digest


def compare_complete_integer_outputs(expected, actual):
    if set(expected) != set(actual) or not expected:
        raise ValueError("complete output roster differs")
    count = 0
    for name in expected:
        left, right = np.asarray(expected[name]), np.asarray(actual[name])
        if left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
            raise ValueError("integer output comparison received noninteger values")
        if left.size != right.size:
            raise ValueError("complete output element counts differ")
        if not np.array_equal(left.reshape(-1), right.reshape(-1)):
            raise ValueError("complete independent integer reference mismatch")
        count += left.size
    return count
