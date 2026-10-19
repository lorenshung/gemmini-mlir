"""New relocation-checker custody only; neutral fixtures grant no source/device owner."""

import copy
import hashlib

import gemmini_staged_rhs_relocation_effects as R
import pytest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def neutral_checker(tmp_path, monkeypatch):
    """Exercise the real constructor/revalidation, without any target execution."""
    selected_inputs = {"facts": {"sha256": "neutral-facts"}}

    class Selected:
        def report(self):
            return copy.deepcopy(
                {"selected_inputs": selected_inputs, "geometry": {"array_rows": 4}}
            )

    class Arithmetic:
        def record(self):
            return copy.deepcopy({"selected_inputs": selected_inputs})

    modules = tmp_path / "original-checker-sources"
    modules.mkdir()

    def source_path(name):
        path = modules / (name + ".py")
        if not path.exists():
            path.write_text('"""Neutral source-custody fixture."""\n')
        return path

    monkeypatch.setattr(R, "SelectedSchedulingTarget", Selected)
    monkeypatch.setattr(R, "IntegerArithmeticOwner", Arithmetic)
    monkeypatch.setattr(R, "module_source_path", source_path)
    tool = tmp_path / "original-nm"
    tool.write_text("original symbol tool bytes\n")
    return R.RhsPlacementRealizationChecker(
        selected=Selected(), arithmetic=Arithmetic(), nm=tool
    ), source_path


def test_original_symbol_tool_bytes_cannot_drift_between_checks(neutral_checker):
    owner, _ = neutral_checker
    original = owner.identity
    owner.nm.write_text("replaced symbol interpreter\n")
    with pytest.raises(ValueError):
        owner.revalidate()
    assert owner.identity == original


def test_symbol_tool_path_cannot_be_rebound_even_with_identical_bytes(
    neutral_checker, tmp_path
):
    owner, _ = neutral_checker
    replacement = tmp_path / "other-nm"
    replacement.write_bytes(owner.nm.read_bytes())
    owner.nm = replacement
    with pytest.raises(ValueError):
        owner.revalidate()


@pytest.mark.parametrize(
    "helper",
    [
        "gemmini_cleanup_final_guard",
        "gemmini_staged_dense_capability",
        "gemmini_timing_resources",
        "merlin.common.paths",
        "merlin.compile.scheduling.staged_dense",
    ],
)
def test_consumed_logical_and_artifact_helpers_remain_in_original_source_closure(
    neutral_checker, helper
):
    owner, source_path = neutral_checker
    helper_path = source_path(helper)
    helper_path.write_text('"""Changed checker dependency after selection."""\n')
    with pytest.raises(ValueError):
        owner.revalidate()


def test_existing_pinned_helper_drift_already_refuses(neutral_checker):
    owner, source_path = neutral_checker
    source_path("gemmini_staged_rhs_effects").write_text(
        '"""Changed existing dependency."""\n'
    )
    with pytest.raises(ValueError):
        owner.revalidate()


def test_record_and_identity_bind_exact_original_symbol_tool(neutral_checker, tmp_path):
    owner, _ = neutral_checker
    expected = {"path": str(owner.nm.resolve()), "sha256": digest(owner.nm)}
    assert owner.record()["symbol_tool"] == expected
    # A separately selected tool is legal, but is a different original checker.
    other = tmp_path / "different-symbol-tool"
    other.write_bytes(owner.nm.read_bytes())
    selected = R.RhsPlacementRealizationChecker(
        selected=owner.selected, arithmetic=owner.arithmetic, nm=other
    )
    assert selected.identity != owner.identity
