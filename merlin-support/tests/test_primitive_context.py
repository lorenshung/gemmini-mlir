"""Explicit seed context plumbing; these type fixtures issue no capability."""

from pathlib import Path

import pytest
from gemmini_primitive_context import PrimitiveCompilerContext, _selection
from gemmini_primitive_diagnostic import PrimitiveDiagnosticOwner
from gemmini_staged_rhs_capability import (
    QualifiedRhsResidencyOwner,
    RhsResidencyCompilerPrototype,
)
from gemmini_zero_initializer_capability import (
    QualifiedZeroInitializerOwner,
    ZeroInitializerCompilerPrototype,
)


def test_rhs_context_selects_only_the_exact_original_seed_binding_owner():
    # Uninitialized exact types exercise only the pure path/type selection.
    # Constructing a real context still requires original live revalidation.
    seed = object.__new__(ZeroInitializerCompilerPrototype)
    prototype = object.__new__(RhsResidencyCompilerPrototype)
    prototype.seed = seed
    qualification = object.__new__(QualifiedRhsResidencyOwner)
    qualification.prototype = prototype
    found, owner_type, binding_owner, _sources = _selection(
        qualification, "qualified_rhs_seed_only"
    )
    assert (
        found is seed
        and binding_owner is seed
        and owner_type is QualifiedRhsResidencyOwner
    )
    assert binding_owner is not qualification


@pytest.mark.parametrize(
    "damage", ["json", "foreign_owner", "foreign_prototype", "foreign_seed"]
)
def test_arbitrary_wrappers_cannot_select_the_original_rhs_seed(damage):
    prototype = object.__new__(RhsResidencyCompilerPrototype)
    prototype.seed = object.__new__(ZeroInitializerCompilerPrototype)
    qualification = object.__new__(QualifiedRhsResidencyOwner)
    qualification.prototype = prototype
    if damage == "json":
        qualification = {"status": "qualified", "prototype": prototype}
    elif damage == "foreign_owner":
        qualification = object.__new__(QualifiedZeroInitializerOwner)
    elif damage == "foreign_prototype":
        qualification.prototype = {"seed": prototype.seed}
    else:
        prototype.seed = {"status": "qualified"}
    with pytest.raises(ValueError, match="original"):
        _selection(qualification, "qualified_rhs_seed_only")


def test_saved_context_success_is_not_a_constructor():
    with pytest.raises(ValueError, match="original live"):
        PrimitiveCompilerContext({"status": "qualified"}, "qualified_rhs_seed_only")
    with pytest.raises(ValueError, match="original live"):
        PrimitiveDiagnosticOwner.from_rhs_qualification({"status": "qualified"})


def test_seed_context_explicitly_refuses_an_rhs_compiler_ledger(tmp_path, monkeypatch):
    # Bounded path refusal only; no revalidation, compiler or target is mocked
    # into successful authority by this fixture.
    context = object.__new__(PrimitiveCompilerContext)
    monkeypatch.setattr(context, "revalidate", lambda: None)
    (tmp_path / "staged-rhs-window-public.json").write_text("{}")
    with pytest.raises(ValueError, match="does not admit an RHS"):
        context.check_binding(Path(tmp_path))
