"""Synthetic transport tests issue no executable or timing qualification."""

import importlib
import json
from pathlib import Path

import pytest
from merlin.targetgen.compiler_library import (
    CompilerLibrarySelection,
    freeze_compiler_library,
)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    module = importlib.import_module("gemmini_live_load_cleanup")
    old = importlib.import_module("gemmini_load_cleanup_capability")
    cleanup = importlib.import_module("gemmini_quiescent_load_config")
    (tmp_path / "example.py").write_text("value = 1\n")
    contract = freeze_compiler_library(
        tmp_path,
        review_id="synthetic transport only",
        public_modules=("example",),
        sources=(("example.py", "example"),),
    )
    library = CompilerLibrarySelection(contract, tmp_path)
    sources = {member.module: member.sha256 for member in contract.members}
    monkeypatch.setattr(old, "shared_library_sources", lambda: sources)
    source = tmp_path / "source"
    source.write_text("synthetic source, no executable evidence")
    pin = old._pin(source)
    effect = tmp_path / "effect.json"
    effect.write_text(
        json.dumps(
            {
                "schema": cleanup.SCHEMA,
                "selected_inputs": {
                    name: pin
                    for name in ("facts", "observed_contract", "operand_sidecar")
                },
                "runtime": pin,
                "sources": [pin],
                "semantics": cleanup.SEMANTICS,
                "encoding": {"config_rs1": 123, "max_stride": 1023},
            }
        )
    )
    row = {
        "schema": old.SCHEMA,
        "scope": old.SCOPE,
        "admission": "conditional_final_pair_required",
        "hardware_execution_qualification": "not_established",
        "timing_authority": "NONE",
        "dense_v1_enabled": False,
        "owner_implementation": old._pin(old.__file__),
        "shared_library_sources": sources,
        "pinned_inputs": {"source": pin},
        "checks": {"synthetic_transport_only": True},
        "effect_contract": old._pin(effect),
        "checker": pin,
        "selected_inputs": {},
        "runtime_inputs": {},
        "completion_contract": pin,
        "qualified_pairs": [],
        "conditions": {},
        "compiler_library_selection": library.record(),
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(row))

    class OriginalSelected:
        def revalidate(self):
            pass

        def report(self):
            return {"selected_inputs": {}}

    owner = module._after_independent_checks(
        receipt=old.LoadCleanupOwner.read(receipt),
        selected=OriginalSelected(),
        library=library,
        checker=source,
    )
    return module, owner, library, source, receipt, row


def test_live_owner_has_no_json_constructor_or_hardware_admission(admitted):
    module, owner, library, _, _, _ = admitted
    assert not hasattr(module.LiveLoadCleanupOwner, "read")
    with pytest.raises(ValueError, match="original independent admission"):
        module.LiveLoadCleanupOwner(
            receipt=None,
            selected=None,
            library=library,
            checker=None,
            _checked=object(),
        )
    record = owner.record()
    assert record["hardware_execution_qualification"] == "not_established"
    assert record["timing_authority"] == "NONE"
    assert record["dense_v1_enabled"] is False
    assert record["serialized_record_authority"].startswith("NONE")
    assert owner.selected_binding()["compiler_library_selection"] == library.record()


@pytest.mark.parametrize("change", ["library", "checker", "receipt"])
def test_live_owner_reopens_exact_original_sources(admitted, change):
    _, owner, library, checker, receipt, _ = admitted
    if change == "library":
        (library.root / "example.py").write_text("value = 2\n")
    elif change == "checker":
        checker.write_text("changed checker")
    else:
        receipt.write_text("{}")
    with pytest.raises(ValueError):
        owner.revalidate()


def test_different_library_cannot_reconstruct_live_selection(admitted):
    module, owner, library, source, receipt, row = admitted
    row["compiler_library_selection"]["contract_sha256"] = "0" * 64
    receipt.write_text(json.dumps(row))
    old = importlib.import_module("gemmini_load_cleanup_capability")
    with pytest.raises(ValueError, match="different shared release"):
        module._after_independent_checks(
            receipt=old.LoadCleanupOwner.read(receipt),
            selected=owner._selected,
            library=library,
            checker=source,
        )
