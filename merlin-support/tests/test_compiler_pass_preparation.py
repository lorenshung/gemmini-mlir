"""Projection and policy boundary tests; synthetic fixtures grant no capability."""

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def hook(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    module = importlib.import_module("gemmini_compiler_pass_hook")
    library = importlib.import_module("merlin.compile.scheduling.library_sources")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "hook.py"))
    monkeypatch.setattr(
        library, "shared_library_sources", lambda: {"synthetic": "fixed"}
    )
    (tmp_path / "shared-library-sources.json").write_text(
        json.dumps({"synthetic": "fixed"})
    )
    leaf = tmp_path / "public.py"
    leaf.write_text("public reviewed source fixture\n")
    pin = {"path": leaf.name, "sha256": hashlib.sha256(leaf.read_bytes()).hexdigest()}
    row = {
        "runtime": pin,
        "selected_inputs": {"facts": pin, "observed_contract": pin},
        "sources": [pin],
    }
    projection = {
        "schema": "gemmini.cleanup_public_inputs.v1",
        "origin_contract_sha256": "0" * 64,
        "contract": row,
    }
    path = tmp_path / "cleanup-public-inputs.json"
    path.write_text(json.dumps(projection))
    policy = types.ModuleType("cleanup_policy")
    policy.choose_cleanup = lambda shape: shape[0] * shape[1] * shape[2] >= 32
    monkeypatch.setitem(sys.modules, "cleanup_policy", policy)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        module.LoadConfigurationContract,
        "read",
        lambda path: json.loads(path.read_bytes()),
    )

    def synthetic(statements, contract, *, enabled):
        calls.append((statements, contract, enabled))
        return statements, {"synthetic_only": True}

    monkeypatch.setattr(module, "optimize_statements", synthetic)
    return module, path, leaf, projection, policy, calls


@pytest.mark.parametrize("shape,enabled", [((1, 2, 3), False), ((2, 4, 4), True)])
def test_exact_public_projection_and_shape_policy(hook, shape, enabled):
    module, _path, leaf, _projection, _policy, calls = hook
    statements = ["fixture_statement();"]
    assert module.rewrite(statements, shape=shape) is statements
    assert calls[0][2] is enabled
    assert calls[0][1]["runtime"]["path"] == str(leaf)
    assert not tuple(leaf.parent.glob("cleanup-inputs-*"))


@pytest.mark.parametrize(
    "damage",
    [
        "absolute",
        "traversal",
        "symlink",
        "digest",
        "oversize",
        "roster",
        "library",
        "policy",
    ],
)
def test_unreviewed_or_missing_projection_never_reaches_rewrite(hook, damage):
    module, path, leaf, projection, policy, calls = hook
    if damage == "absolute":
        projection["contract"]["runtime"]["path"] = str(leaf)
    elif damage == "traversal":
        projection["contract"]["runtime"]["path"] = "../public.py"
    elif damage == "symlink":
        other = leaf.with_name("other.py")
        other.write_bytes(leaf.read_bytes())
        leaf.unlink()
        leaf.symlink_to(other)
    elif damage == "digest":
        leaf.write_text("changed bytes")
    elif damage == "oversize":
        leaf.write_bytes(b"x" * ((1 << 20) + 1))
        projection["contract"]["runtime"]["sha256"] = hashlib.sha256(
            leaf.read_bytes()
        ).hexdigest()
    elif damage == "roster":
        projection["contract"]["sources"] *= 65
    elif damage == "library":
        (leaf.parent / "shared-library-sources.json").write_text("{}")
    else:
        policy.choose_cleanup = lambda shape: 1
    path.write_text(json.dumps(projection))
    with pytest.raises(ValueError):
        module.rewrite(["fixture_statement();"], shape=(2, 4, 4))
    assert not calls


def test_saved_json_cannot_construct_original_admission(monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    module = importlib.import_module("gemmini_cleanup_admission")
    assert not hasattr(module.OriginalCleanupAdmission, "read")
    with pytest.raises(ValueError, match="original independent"):
        module.OriginalCleanupAdmission({}, {}, {}, {}, {}, _issuer=object())


def test_admission_child_cannot_wait_inside_parent_compute_lease(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    module = importlib.import_module("gemmini_cleanup_admission")
    from merlin.common import cpu_pool

    monkeypatch.setattr(cpu_pool, "current_thread_owns_cpu_lease", lambda: True)
    with pytest.raises(ValueError, match="release parent compute lease"):
        module.run_original_admission({}, work_root=tmp_path / "unused", library=None)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"enabled": 1, "min_macs": 0},
        {"enabled": True, "min_macs": -1},
        {"enabled": True, "min_macs": True},
        {"enabled": True, "min_macs": 0, "hidden": True},
    ],
)
def test_policy_cannot_silently_select_unknown_or_nonboolean_controls(
    tmp_path, monkeypatch, policy
):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    module = importlib.import_module("gemmini_compiler_passes")
    owner = object.__new__(module.CompilerCleanupOwner)
    monkeypatch.setattr(owner, "revalidate", lambda: None)
    with pytest.raises(ValueError, match="nonnegative shape threshold"):
        owner.prepare_compiler(tmp_path / "absent", tmp_path / "unused", policy=policy)
    assert not (tmp_path / "unused").exists()
