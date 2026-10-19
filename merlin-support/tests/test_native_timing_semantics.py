"""Synthetic admission controls do not qualify a native model or hardware."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_native_timing_semantics")


@pytest.fixture
def selection(adapter):
    classes = {}
    for group, entries, queues in (("load", 5, 3), ("execute", 7, 4), ("store", 2, 3)):
        classes[group] = {
            "entries": {"value": entries},
            "command_queue": {"entries_total": queues},
            "downstream_buffers": []
            if group == "execute"
            else [{"instance": group + "/cmd_tracker", "entries": [2]}],
        }
    facts = {
        "facts": {
            "memories": [{"name": "scratchpad", "banks": 3, "depth": 19}],
            "timing_structure": {
                "command_classes": {"status": "derived", "classes": classes}
            },
        }
    }
    structure = {
        "accelerator_dispatch": {
            g: {"entries": r["entries"]["value"]} for g, r in classes.items()
        },
        "cpu_execution_resources": {"cpu.integer": "cpu.integer_alu"},
    }
    names = [
        "cpu.issue",
        "cpu.integer_alu",
        "cpu.muldiv",
        "cpu.control",
        "cpu.rocc_dispatch",
        "memory.shared",
        "accelerator.storage.accumulator.all",
    ]
    names += [
        f"accelerator.{stage}.{group}"
        for stage in ("issue", "controller")
        for group in classes
    ]
    names += [f"accelerator.storage.scratchpad.{i}" for i in range(3)]
    caches = {k: {"line_bytes": 32} for k in ("l1d", "l1i", "l2")}
    model = {
        "target": {
            "scheduling": {
                "version": 2,
                "require_fenced_phases": 1,
                "conservative_local_store_holds": 1,
                "scratchpad_bank_count": 3,
                "scratchpad_bank_rows": 19,
                "issue_latency_cycles": 1,
                "issue_occupancy_cycles": 1,
                "reservations": {g: r["entries"]["value"] for g, r in classes.items()},
                "controller_entries": {
                    g: r["command_queue"]["entries_total"] for g, r in classes.items()
                },
                "dma_commands": {"load": 2, "store": 2},
            }
        },
        "memory": {
            "caches": caches,
            "coherence_policy": "writeback_invalidate",
            "coherence_granule": 32,
            "paths": {
                "cpu_data": ["l1d", "l2"],
                "cpu_fetch": ["l1i", "l2"],
                "device": ["l2"],
            },
        },
        "parameters": {"cpu.issue_cycles": 1, "cpu.integer.latency_cycles": None},
        "resources": [
            {"id": i, "name": n, "capacity": 1, "queue_capacity": 0}
            for i, n in enumerate(names)
        ],
        "cpu_execution_resources": dict(
            structure["cpu_execution_resources"],
            **{
                "cpu.csr": "cpu.control",
                "cpu.fence": "cpu.control",
                "cpu.custom": "cpu.rocc_dispatch",
            },
        ),
        "domain": {"feature_engine_sha256": adapter.FEATURE_ENGINE},
    }
    return model, facts, structure, caches


def test_current_model_is_checked_without_filling_unknown_prices(adapter, selection):
    before = copy.deepcopy(selection[0])
    adapter.verify_model(*selection)
    assert (
        selection[0] == before
        and selection[0]["parameters"]["cpu.integer.latency_cycles"] is None
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("conservative_local_store_holds", 0),
        ("version", 3),
        ("scratchpad_bank_rows", 20),
        ("issue_latency_cycles", 0),
        ("version", True),
    ],
)
def test_policy_or_geometry_cannot_be_relabelled(adapter, selection, key, value):
    selection[0]["target"]["scheduling"][key] = value
    with pytest.raises(ValueError, match="policy/geometry"):
        adapter.verify_model(*selection)


@pytest.mark.parametrize(
    "change",
    ["resource", "boolean", "queue", "reservation", "dma", "cache", "cpu", "engine"],
)
def test_actual_capacity_assignment_and_source_mutations_refuse(
    adapter, selection, change
):
    m = selection[0]
    if change == "resource":
        m["resources"][-1]["capacity"] = 2
    elif change == "boolean":
        m["resources"][-1]["capacity"] = True
    elif change == "queue":
        m["target"]["scheduling"]["controller_entries"]["load"] += 1
    elif change == "reservation":
        m["target"]["scheduling"]["reservations"]["execute"] += 1
    elif change == "dma":
        m["target"]["scheduling"]["dma_commands"]["store"] += 1
    elif change == "cache":
        m["memory"]["coherence_policy"] = "invented"
    elif change == "cpu":
        m["cpu_execution_resources"]["cpu.atomic"] = "cpu.control"
    else:
        m["domain"]["feature_engine_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        adapter.verify_model(*selection)


def test_saved_json_or_duck_typed_owner_cannot_issue_capability(adapter):
    with pytest.raises(TypeError):
        adapter.NativeTimingSemanticsOwner({"qualified": True})
    with pytest.raises(TypeError, match="original live"):
        adapter.NativeTimingSemanticsOwner.bind(
            object(),
            source_root=None,
            spike_source=None,
            observer_build=None,
            provider_bundle=None,
            rebuild_dir=None,
        )
