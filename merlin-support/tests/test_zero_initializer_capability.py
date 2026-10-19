"""Original preparation/final-grade boundaries; these fixtures grant no target."""

import ast
import inspect
import json

import pytest
import test_cleanup_final_guard as old_tests
from gemmini_zero_initializer_capability import (
    QualifiedZeroInitializerOwner,
    ZeroInitializerCompilerPrototype,
    _check_prepared_binding,
)
from gemmini_zero_initializer_final_guard import check_functional_guard


def test_json_or_missing_actual_target_proofs_cannot_construct_initializer_capability():
    with pytest.raises(ValueError, match="original live"):
        ZeroInitializerCompilerPrototype({"status": "qualified"})
    with pytest.raises(ValueError, match="original owners"):
        QualifiedZeroInitializerOwner({"status": "qualified"}, tuple({} for _ in range(7)))


@pytest.mark.parametrize(
    "policy", [{}, {"zero_initializer": True, "tile": {}}, {"zero_initializer": {"enabled": 1}, "tile": {}}]
)
def test_malformed_policy_refuses_before_copy_or_original_compilation(tmp_path, policy):
    prototype = object.__new__(ZeroInitializerCompilerPrototype)
    prototype.revalidate = lambda: None
    with pytest.raises(ValueError, match="explicit independent"):
        prototype.prepare_compiler(tmp_path / "source", tmp_path / "destination", policy=policy)
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize("linked_parent", [False, True])
def test_new_component_symlinks_never_enter_exact_compiler_binding(tmp_path, linked_parent):
    prototype = object.__new__(ZeroInitializerCompilerPrototype)
    prototype.revalidate = lambda: None
    prototype._binding = {}
    package = tmp_path / "package"
    package.mkdir()
    for name in ("mixed_codegen.py", "provenance.py", "zero_initializer_policy.py"):
        (package / name).write_text("# neutral source\n")
    private = tmp_path / "foreign"
    private.mkdir()
    (private / "__init__.py").write_text("")
    (private / "zero_initializer_groups.py").write_text("# outside ordinary component\n")
    if linked_parent:
        (package / "passes").symlink_to(private, target_is_directory=True)
    else:
        (package / "passes").mkdir()
        (package / "passes/__init__.py").write_text("")
        (package / "passes/zero_initializer_groups.py").symlink_to(private / "zero_initializer_groups.py")
    with pytest.raises(ValueError, match="exact regular"):
        prototype.check_compiler_binding(package)


@pytest.mark.parametrize("damage", [None, "library", "sources", "dense", "tile", "large", "symlink"])
def test_original_binding_leaf_check_preserves_all_contracts(tmp_path, damage):
    # This tests only the bounded byte comparison, never grants target support.
    package = tmp_path / "compiler"
    package.mkdir()
    expected = {
        name: {"original": name}
        for name in (
            "selected-shared-library.json",
            "shared-library-sources.json",
            "staged-dense-public.json",
            "staged-tile-public.json",
        )
    }
    for name, value in expected.items():
        (package / name).write_text(json.dumps(value))
    (package / "passes").mkdir()
    for name in (
        "mixed_codegen.py",
        "provenance.py",
        "zero_initializer_policy.py",
        "passes/__init__.py",
        "passes/zero_initializer_groups.py",
    ):
        (package / name).write_text("# original ordinary leaf\n")
    ledger = {
        "library": "selected-shared-library.json",
        "sources": "shared-library-sources.json",
        "dense": "staged-dense-public.json",
        "tile": "staged-tile-public.json",
    }.get(damage)
    if ledger:
        (package / ledger).write_text(json.dumps({"forged": ledger}))
    elif damage == "large":
        (package / "mixed_codegen.py").write_text("#" * ((1 << 20) + 1))
    elif damage == "symlink":
        original = package / "staged-tile-public.json"
        target = tmp_path / "foreign.json"
        original.rename(target)
        original.symlink_to(target)
    if damage is None:
        checked = _check_prepared_binding(package, expected)
        assert set(checked["source_ledgers"]) == set(expected)
        assert len(checked["candidate_owned_initializer_components"]) == 5
    else:
        with pytest.raises(ValueError, match="original|exact regular"):
            _check_prepared_binding(package, expected)


def test_new_guard_keeps_every_original_complete_grade_roster_and_tree_gate():
    # Only final native-pair dispatch differs. The full original grade consumer
    # body remains byte-independent of candidate transforms and diagnostics.
    import gemmini_cleanup_final_guard as old

    assert ast.dump(ast.parse(inspect.getsource(check_functional_guard))) == ast.dump(
        ast.parse(inspect.getsource(old.check_functional_guard))
    )


@pytest.fixture
def original_roster(tmp_path):
    return old_tests.original_roster.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "pending",
        "cert",
        "root",
        "duplicate",
        "missing",
        "foreign",
        "compiler",
        "receipt",
        "emission",
        "family",
        "integrity",
        "count",
        "unsafe_name",
    ],
)
def test_new_native_inverse_cannot_bypass_any_original_grade_gate(original_roster, monkeypatch, damage):
    import gemmini_zero_initializer_final_guard as new

    old_tests.test_incomplete_or_foreign_original_grade_never_substitutes_saved_success(
        new, original_roster, monkeypatch, damage
    )
