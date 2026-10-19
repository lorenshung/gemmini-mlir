"""Structural resource projection cannot invent physical or timing evidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_roofline_resources as R


def test_rectangular_explicit_arithmetic_differs_from_missing_multiplier_inventory():
    facts = {"facts": {"arrays": [{"rows": 3, "cols": 5, "mac_idiom": {"muls": 2}}]}}
    geometry = {"stores": [{"banks": 3, "depth": 101, "row_bytes": 20}]}
    proof = {
        "dimension": 3,
        "input_rows_per_cycle_maximum": 1,
        "minimum_reduced_input_rows": 2,
    }
    result = R.project(facts, geometry, proof)
    assert result["resources"]["arithmetic.sites"]["capacity_per_cycle_upper"] == 30
    assert result["resources"]["execute.input"]["capacity_per_cycle_upper"] == 1
    assert result["resources"]["dma.read"]["capacity_per_cycle_upper"] is None
    assert result["physical_traffic_bytes"] is None
    assert not result["convergence_eligible"]
    assert "bank_service" in result["unresolved"]
    facts["facts"]["arrays"].append({"rows": 7, "cols": 2})
    assert (
        R.project(facts, geometry, proof)["resources"]["arithmetic.sites"][
            "capacity_per_cycle_upper"
        ]
        is None
    )


def test_no_saved_document_constructor_and_original_owner_required():
    try:
        R.StructuralServiceResources({}, execute_controller=Path("missing"))
    except ValueError as error:
        assert "original selected" in str(error)
    else:
        raise AssertionError(
            "saved JSON cannot establish a live selected resource owner"
        )


def test_cutpoint_evaluation_does_not_claim_actual_invocation_and_rechecks_sources(
    tmp_path,
):
    import gemmini_execute_service as E
    import pytest
    from test_execute_service import fixture

    owner = object.__new__(R.StructuralServiceResources)
    owner._selected = type(
        "SyntheticSelection", (), {"revalidate": lambda self: None}
    )()
    source = tmp_path / "original.py"
    source.write_text("selected original synthetic source")
    owner._sources = {"original": R._pin(source)}
    owner._proof = {"proof": E.prove(fixture())}
    cutpoint = {
        "current_dataflow": 1,
        "d_garbage": 1,
        "a_should_be_fed_into_transposer": 0,
        "b_should_be_fed_into_transposer": 0,
        "d_should_be_fed_into_transposer": 0,
        "a_rows": 1,
        "b_rows": 7,
        "a_garbage": 0,
        "b_garbage": 1,
    }
    result = owner.input_rows_at_cutpoint(cutpoint)
    assert result["input_rows"] == 2
    assert not result["actual_invocation_bound"] and result["attainment"] is None
    assert owner.input_rows_at_cutpoint({**cutpoint, "b_garbage": 0})["input_rows"] == 7
    assert owner.input_rows_at_cutpoint({**cutpoint, "d_garbage": 0})["input_rows"] == 8
    with pytest.raises(ValueError, match="Boolean"):
        owner.input_rows_at_cutpoint({**cutpoint, "a_garbage": False})
    with pytest.raises(ValueError, match="row extent"):
        owner.input_rows_at_cutpoint({**cutpoint, "a_rows": 9})
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        owner.record()
