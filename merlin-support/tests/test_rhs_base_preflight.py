"""Early mechanical source checks issue no target capability or qualification."""

import copy

import gemmini_staged_rhs_registry as registry
import pytest
from gemmini_zero_initializer_capability import ZeroInitializerCompilerPrototype


@pytest.mark.parametrize("damage", ["mapping", "count", "relative", "sha", "extra"])
def test_malformed_optional_preflight_refuses_before_any_admission(damage):
    row = dict(path="/original/package", sha256="a" * 64)
    packages = [row]
    if damage == "mapping":
        packages = {"status": "success"}
    elif damage == "count":
        packages *= 9
    elif damage == "relative":
        row["path"] = "relative"
    elif damage == "sha":
        row["sha256"] = "saved-success"
    else:
        row["status"] = "qualified"
    with pytest.raises(ValueError, match="bounded explicit original"):
        registry._base_packages(dict(base_compiler_preflight_packages=packages))


def test_defaults_are_unchanged_and_json_cannot_supply_base_owner():
    assert registry._base_packages({}) == []
    with pytest.raises(ValueError, match="original live base"):
        registry._preflight_base_packages({"status": "qualified"}, [])


def test_each_selected_tree_is_reopened_before_and_after_original_bound_checker(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "compiler.py"
    source.write_text('print("neutral")\n')
    rows = [dict(path=str(package), sha256=registry._tree(package))]
    original = copy.deepcopy(rows)
    owner = object.__new__(ZeroInitializerCompilerPrototype)
    calls = []
    monkeypatch.setattr(
        ZeroInitializerCompilerPrototype, "check_compiler_binding", lambda self, path: calls.append(path)
    )
    registry._preflight_base_packages(owner, rows)
    assert rows == original and calls == [package]
    source.write_text("changed bytes")
    with pytest.raises(ValueError, match="differ from their original pin"):
        registry._preflight_base_packages(owner, rows)
    assert calls == [package]


def test_source_mutation_during_check_cannot_publish_preflight(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "compiler.py"
    source.write_text("neutral")
    rows = [dict(path=str(package), sha256=registry._tree(package))]
    owner = object.__new__(ZeroInitializerCompilerPrototype)
    monkeypatch.setattr(
        ZeroInitializerCompilerPrototype, "check_compiler_binding", lambda self, path: source.write_text("changed")
    )
    with pytest.raises(ValueError, match="changed during"):
        registry._preflight_base_packages(owner, rows)


def test_linked_tree_is_not_an_original_compiler_pin(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "compiler.py").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="unsupported file"):
        registry._preflight_base_packages(
            object.__new__(ZeroInitializerCompilerPrototype), [dict(path=str(package), sha256="a" * 64)]
        )
