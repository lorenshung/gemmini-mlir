"""Owner transport/refusal controls; synthetic receipts qualify no execution."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    owner_module = importlib.import_module("gemmini_load_cleanup_capability")
    cleanup = importlib.import_module("gemmini_quiescent_load_config")
    source = tmp_path / "public-synthetic-source"
    source.write_text("synthetic owner API stimulus; no target execution")
    pin = owner_module._pin(source)
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
    checker = (
        Path(__file__).resolve().parents[2]
        / "experiments/shared_load_configuration/qualify.py"
    )
    row = {
        "schema": owner_module.SCHEMA,
        "scope": owner_module.SCOPE,
        "admission": "conditional_final_pair_required",
        "hardware_execution_qualification": "not_established",
        "timing_authority": "NONE",
        "dense_v1_enabled": False,
        "owner_implementation": owner_module._pin(owner_module.__file__),
        "shared_library_sources": owner_module.shared_library_sources(),
        "pinned_inputs": {"source": pin, "checker": owner_module._pin(checker)},
        "checks": {"synthetic_api_only": True},
        "effect_contract": owner_module._pin(effect),
        "checker": owner_module._pin(checker),
        "selected_inputs": {},
        "runtime_inputs": {},
        "completion_contract": pin,
        "qualified_pairs": [],
        "conditions": {},
    }
    receipt = tmp_path / "owner.json"
    receipt.write_text(json.dumps(row))
    return owner_module, receipt, row, source


def test_owner_rewrites_bounded_source_and_requires_new_realization_check(setup):
    module, receipt, _, _ = setup
    owner = module.LoadCleanupOwner.read(receipt)
    before = [
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
        "gm_mvin(A,0,16,16);",
        "gm_fence();",
        "gm_ld_config(17,0,1.0f,0);",
    ]
    after, report = owner.rewrite(before, execution_domain="functional_spike")
    assert after == before[:4]
    assert report["requires_independent_final_pair_check"] is True
    assert owner.record()["dense_v1_enabled"] is False
    assert owner.record()["hardware_execution_qualification"] == "not_established"
    assert owner.identity == module.LoadCleanupOwner.read(receipt).identity


def test_original_actual_effect_checker_rejects_missing_fence_or_work(setup):
    module, receipt, _, _ = setup
    owner = module.LoadCleanupOwner.read(receipt)
    before = [
        ("FENCE", 0, 0),
        ("CONFIG_LD", 123, 17),
        ("MVIN", ("input", 0), 9),
        ("FENCE", 0, 0),
        ("CONFIG_LD", 123, 17),
    ]
    assert (
        owner.check_actual_pair(
            before, before[:4], execution_domain="functional_spike"
        )["actual_redundant_setters_removed"]
        == 1
    )
    with pytest.raises(ValueError, match="actual target"):
        owner.check_actual_pair(before, before[:3], execution_domain="functional_spike")


@pytest.mark.parametrize("domain", ["gsim", "native_event_timing", "hardware"])
def test_owner_cannot_enable_a_different_execution_or_timing_domain(setup, domain):
    module, receipt, _, _ = setup
    owner = module.LoadCleanupOwner.read(receipt)
    with pytest.raises(ValueError, match="functional Spike"):
        owner.rewrite([], execution_domain=domain)


@pytest.mark.parametrize(
    "change", ["source", "receipt", "library", "hardware", "check"]
)
def test_changed_selection_or_unevaluated_claim_cannot_keep_admission(setup, change):
    module, receipt, row, source = setup
    owner = module.LoadCleanupOwner.read(receipt)
    if change == "source":
        source.write_text("different source")
    elif change == "receipt":
        receipt.write_text("{}")
    else:
        if change == "library":
            row["shared_library_sources"] = {}
        elif change == "hardware":
            row["hardware_execution_qualification"] = "qualified"
        else:
            row["checks"]["synthetic_api_only"] = False
        receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        owner.revalidate() if change in {
            "source",
            "receipt",
        } else module.LoadCleanupOwner.read(receipt)
