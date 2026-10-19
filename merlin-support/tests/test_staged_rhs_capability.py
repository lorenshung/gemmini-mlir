"""Fixed coverage obligations are distinct from actual owner issuance."""

import copy

import pytest
from gemmini_staged_rhs_capability import (
    QualifiedRhsResidencyOwner,
    RhsResidencyCompilerPrototype,
    qualification_coverage,
)


def evidence():
    shapes = [(17, 17, 17), (32, 48, 32), (47, 48, 32), (32, 47, 32), (32, 48, 47), (32, 16, 32)]
    return [
        dict(
            status=(
                "checked_original_private_rhs_relocation_and_actual_issue"
                if k > 16
                else "checked_original_private_panel_fallback"
            ),
            original_source_execution=[dict(source_sha256=str(i), source_geometry=dict(m=m, k=k, n=n))],
            full_k_placement_applied=k > 16,
            actual_issue_effects=dict(actual_rhs_loads_removed=6 if k > 16 else 0),
        )
        for i, (m, k, n) in enumerate(shapes)
    ]


def test_independent_geometry_evidence_must_cover_each_positive_tail_and_fallback():
    checks = evidence()
    before = copy.deepcopy(checks)
    coverage = qualification_coverage(checks, 16)
    assert checks == before
    assert coverage["positive_tail_remainders"] == [0, 1, 15]
    assert coverage["independent_sources"] == 6 and coverage["single_panel_fallback"]
    assert coverage["hardware_execution_qualified"] is False


@pytest.mark.parametrize("damage", ["missing", "repeat", "m", "n", "k", "no_reuse", "no_fallback", "saved_success"])
def test_incomplete_or_undistinguishing_pairs_cannot_qualify(damage):
    checks = evidence()
    if damage == "missing":
        checks.pop()
    elif damage == "repeat":
        checks[-1]["original_source_execution"][0]["source_sha256"] = "0"
    elif damage in {"m", "n", "k"}:
        for check in checks:
            shape = check["original_source_execution"][0]["source_geometry"]
            if shape[damage] % 16 == 15:
                shape[damage] -= 15
    elif damage == "no_reuse":
        checks[0]["actual_issue_effects"]["actual_rhs_loads_removed"] = 0
    elif damage == "no_fallback":
        checks[-1]["full_k_placement_applied"] = True
    else:
        checks[0]["status"] = "success"
    with pytest.raises(ValueError):
        qualification_coverage(checks, 16)


def test_reports_and_source_rules_cannot_issue_compiler_or_qualified_owners():
    with pytest.raises(ValueError, match="original compiler"):
        RhsResidencyCompilerPrototype({"status": "qualified"}, {"physical_rows": 16384})
    with pytest.raises(ValueError, match="original owners"):
        QualifiedRhsResidencyOwner({"status": "qualified"}, tuple(evidence()))
