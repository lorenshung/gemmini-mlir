"""A valid native program never upgrades a foreign/missing SDK source binding."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    module = importlib.import_module("gemmini_compiler_passes")
    owner = object.__new__(module.CompilerCleanupOwner)
    monkeypatch.setattr(owner, "revalidate", lambda: None)
    selected = {
        "contract_sha256": "exact-new-release",
        "root": "/selected",
        "contract": {"members": []},
    }
    sources = {"public.example": "exact-source"}
    owner.library = SimpleNamespace(
        record=lambda: selected, contract=SimpleNamespace(sha256="exact-new-release")
    )
    owner.cleanup = SimpleNamespace(
        revalidate=lambda: {"shared_library_sources": sources}
    )
    (tmp_path / "selected-shared-library.json").write_text(json.dumps(selected))
    (tmp_path / "shared-library-sources.json").write_text(json.dumps(sources))
    return owner, tmp_path


def test_exact_compiler_release_ledger_required(bound):
    owner, package = bound
    report = owner.check_compiler_binding(package)
    assert report["status"] == "verified_exact_compiler_library_binding"
    assert report["compiler_library_contract_sha256"] == "exact-new-release"


@pytest.mark.parametrize(
    "damage", ["release", "sources", "missing", "symlink", "oversize", "package_link"]
)
def test_foreign_or_missing_release_refuses_even_if_program_outputs_match(
    bound, damage
):
    owner, package = bound
    path = package / "selected-shared-library.json"
    if damage == "release":
        row = json.loads(path.read_bytes())
        row["contract_sha256"] = "old-release"
        path.write_text(json.dumps(row))
    elif damage == "sources":
        (package / "shared-library-sources.json").write_text("{}")
    elif damage == "missing":
        path.unlink()
    elif damage == "symlink":
        target = package / "foreign.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif damage == "oversize":
        path.write_bytes(b"x" * ((1 << 20) + 1))
    else:
        linked = package / "package-link"
        linked.symlink_to(package, target_is_directory=True)
        package = linked
    with pytest.raises(ValueError):
        owner.check_compiler_binding(package)
