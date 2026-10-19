"""Live source owner for the explicit conservative native reservation model.

This admits a checked modeling abstraction, not physical equivalence or timing
accuracy. Held timing validation and ordinary numerical admission stay separate.
No saved JSON success record constructs an owner.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

from gemmini_count_hardware import CountHardwareCorrespondenceOwner
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_staged_completion import derive as derive_completion
from gemmini_timing_resources import _pin, _revalidate_sources, extract
from merlin.targetgen.rtl import firrtl_struct as F
from merlin_timing import provider_bundle as B

SCHEMA = "gemmini.conservative_native_semantics.v1"
FEATURE_ENGINE = "e3c5f4f8499d68adcc209f929d349d91abb201254ef9d7cd2a433eb33755d166"
# A reviewed evaluator release, independent of the candidate compiler. A source
# change requires a new reviewed release; matching model names cannot enable it.
PROVIDER_SOURCES = {
    "provider.hpp": "1128c8e8678056da34d5a9a61648d70317275340559df3807e6c21658b638a2e",
    "staged.hpp": "5787bc9c0a07de83f3373a70bcd3eeebc5f834c04e0f6a624a298c8bfacd875c",
    "bridge.cpp": "fc4f69d6fbfc70790efd686ad4001d96712cb547f6a47cc3e3a54d50180e55e5",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _smems(body):
    rows = []
    pattern = re.compile(r"\bsmem\s+(\w+)\s*:\s*UInt<(\d+)>(?:\[(\d+)\])?\s+\[(\d+)\]")
    for line, text in body:
        match = pattern.search(F.split_locator(text)[0])
        if match:
            name, bits, lanes, depth = match.groups()
            rows.append(
                {
                    "name": name,
                    "bits": int(bits),
                    "lanes": int(lanes or 1),
                    "depth": int(depth),
                    "line": line,
                }
            )
    return rows


def cache_geometry(fir, dts, l2):
    """Corroborate declaration geometry; replacement/coherence stay approximate."""
    modules = F.read_module_bodies(
        Path(fir), {"ICache", "DCache", "DCacheDataArray", "Directory", "BankedStore"}
    )
    text = Path(dts).read_text()

    def literal(name):
        values = re.findall(r"\b" + re.escape(name) + r"\s*=\s*<\s*(\d+)\s*>\s*;", text)
        if len(values) != 1 or int(values[0]) <= 0:
            raise ValueError(
                "selected DTS cache property is absent or ambiguous: " + name
            )
        return int(values[0])

    result = {}
    witnesses = {}
    for label, prefix, tags, data in (
        ("l1i", "i-", "ICache", "ICache"),
        ("l1d", "d-", "DCache", "DCacheDataArray"),
    ):
        line, sets, size = (
            literal(prefix + "cache-" + field)
            for field in ("block-size", "sets", "size")
        )
        if size % (line * sets):
            raise ValueError("selected DTS cache geometry is inconsistent")
        ways = size // (line * sets)
        tag = [row for row in _smems(modules[tags]) if "tag_array" in row["name"]]
        arrays = [row for row in _smems(modules[data]) if "data_arrays" in row["name"]]
        if len(tag) != 1 or tag[0]["lanes"] != ways or tag[0]["depth"] != sets:
            raise ValueError("selected FIRRTL tag geometry differs from DTS")
        if sum(row["bits"] * row["lanes"] * row["depth"] for row in arrays) != size * 8:
            raise ValueError("selected FIRRTL data capacity differs from DTS")
        result[label] = {
            "line_bytes": line,
            "sets": sets,
            "ways": ways,
            "policy": "write_back_write_allocate",
        }
        witnesses[label] = {"tags": tag, "data": arrays}
    bank = json.loads(Path(l2).read_bytes())["banks"]
    if len(bank) != 1:
        raise ValueError("selected L2 needs a separately reviewed bank mapping")
    bank = bank[0]
    line_bits = bank["blockBytes"].bit_length() - 1
    if bank["blockBytes"] != 1 << line_bits or bank["setBits"] != list(
        range(line_bits, line_bits + len(bank["setBits"]))
    ):
        raise ValueError("selected L2 needs a separately reviewed set mapping")
    sets, line, ways = 1 << len(bank["setBits"]), bank["blockBytes"], bank["ways"]
    directory, data = _smems(modules["Directory"]), _smems(modules["BankedStore"])
    if (
        len(directory) != 1
        or directory[0]["lanes"] != ways
        or directory[0]["depth"] != sets
    ):
        raise ValueError("selected L2 directory differs from elaboration metadata")
    if (
        sum(row["bits"] * row["lanes"] * row["depth"] for row in data)
        != sets * ways * line * 8
    ):
        raise ValueError("selected L2 data capacity differs from elaboration metadata")
    result["l2"] = {
        "line_bytes": line,
        "sets": sets,
        "ways": ways,
        "policy": "write_back_write_allocate",
    }
    witnesses["l2"] = {"directory": directory, "data": data}
    return result, witnesses


def verify_model(model, facts, structure, caches):
    """Check actual current model inputs; no timing-price success is inferred."""
    policy = model["target"]["scheduling"]
    stores = {row["name"]: row for row in facts["facts"]["memories"]}
    scratchpad = stores["scratchpad"]
    required = {
        "version": 2,
        "require_fenced_phases": 1,
        "conservative_local_store_holds": 1,
        "scratchpad_bank_count": scratchpad["banks"],
        "scratchpad_bank_rows": scratchpad["depth"],
        "issue_latency_cycles": 1,
        "issue_occupancy_cycles": 1,
    }
    if any(
        type(policy.get(key)) is not int or policy[key] != value
        for key, value in required.items()
    ):
        raise ValueError(
            "native model does not select the checked conservative policy/geometry"
        )
    classes = facts["facts"]["timing_structure"]["command_classes"]
    if classes["status"] != "derived":
        raise ValueError("selected command capacities are unresolved")
    for group in ("load", "execute", "store"):
        row = classes["classes"][group]
        if (
            policy["reservations"][group] != row["entries"]["value"]
            or policy["reservations"][group]
            != structure["accelerator_dispatch"][group]["entries"]
        ):
            raise ValueError("native reservation capacity differs from selected RTL")
        if policy["controller_entries"][group] != row["command_queue"]["entries_total"]:
            raise ValueError(
                "native controller queue capacity differs from selected RTL"
            )
        if group != "execute":
            tracker = [
                x
                for x in row["downstream_buffers"]
                if x["instance"].endswith("/cmd_tracker")
            ]
            if len(tracker) != 1 or tracker[0]["entries"] != [
                policy["dma_commands"][group]
            ]:
                raise ValueError(
                    "native DMA command ownership differs from selected RTL"
                )
    if (
        model["memory"]["caches"] != caches
        or model["memory"]["coherence_policy"] != "writeback_invalidate"
        or model["memory"]["coherence_granule"] != caches["l2"]["line_bytes"]
    ):
        raise ValueError("native selected cache geometry/policy differs")
    if model["memory"]["paths"] != {
        "cpu_data": ["l1d", "l2"],
        "cpu_fetch": ["l1i", "l2"],
        "device": ["l2"],
    }:
        raise ValueError("native cache paths need a different reviewed selection")
    if model["parameters"].get("cpu.issue_cycles") != 1:
        raise ValueError("native scalar issue opportunity differs from selected source")
    resources = {r["name"]: r for r in model["resources"]}
    names = {
        "cpu.issue",
        "cpu.integer_alu",
        "cpu.muldiv",
        "cpu.control",
        "cpu.rocc_dispatch",
        "memory.shared",
        "accelerator.storage.accumulator.all",
    }
    names.update(
        f"accelerator.{stage}.{group}"
        for stage in ("issue", "controller")
        for group in ("load", "execute", "store")
    )
    names.update(
        f"accelerator.storage.scratchpad.{i}" for i in range(scratchpad["banks"])
    )
    if any(
        name not in resources
        or type(resources[name]["capacity"]) is not int
        or resources[name]["capacity"] != 1
        or type(resources[name]["queue_capacity"]) is not int
        or resources[name]["queue_capacity"] != 0
        for name in names
    ):
        raise ValueError(
            "native conservative resource mapping is incomplete or overcommitted"
        )
    assignments = dict(
        structure["cpu_execution_resources"],
        **{
            "cpu.csr": "cpu.control",
            "cpu.fence": "cpu.control",
            "cpu.custom": "cpu.rocc_dispatch",
        },
    )
    if model["cpu_execution_resources"] != assignments:
        raise ValueError("native CPU assignment is outside the reviewed scalar subset")
    if model["domain"]["feature_engine_sha256"] != FEATURE_ENGINE:
        raise ValueError("native generic observer is a different evaluator release")


class NativeTimingSemanticsOwner:
    """Original live owner; JSON receipts and passing statistics cannot issue it."""

    def __init__(self):
        raise TypeError(
            "construct through bind with the original live correspondence owner"
        )

    @classmethod
    def bind(
        cls,
        count_owner,
        *,
        source_root,
        spike_source,
        observer_build,
        provider_bundle,
        rebuild_dir,
        cxx="c++",
    ):
        if type(count_owner) is not CountHardwareCorrespondenceOwner:
            raise TypeError(
                "native semantics require the original live count correspondence owner"
            )
        count_owner.revalidate()
        model = json.loads(count_owner.model_path.read_bytes())
        selected = SelectedSchedulingTarget.read(
            count_owner.facts_path, count_owner.observed_contract_path
        )
        facts, _, _ = selected._documents()
        source_root, spike_source, observer_build, provider_bundle, rebuild_dir = map(
            Path,
            (source_root, spike_source, observer_build, provider_bundle, rebuild_dir),
        )
        fir = Path(facts["inputs"]["fir_path"])
        dts, l2 = fir.with_suffix(".dts"), fir.with_suffix(".l2.json")
        row = count_owner.correspondence.revalidate()
        deployment = json.loads(Path(row["pins"]["deployment"]["path"]).read_bytes())
        if (
            _pin(dts)["sha256"]
            != deployment["profile"]["hardware_association"]["host_dts_sha256"]
        ):
            raise ValueError(
                "selected DTS differs from original functional hardware association"
            )
        if deployment["profile"]["launch"]["harts"] != 1:
            raise ValueError("native release supports only the selected scalar hart")
        base = source_root / "generators/gemmini/src/main/scala/gemmini"
        structure = extract(
            selected,
            rocket_core=source_root
            / "generators/rocket-chip/src/main/scala/rocket/RocketCore.scala",
            reservation_station=base / "ReservationStation.scala",
            execute_controller=base / "ExecuteController.scala",
            spike_source=spike_source,
        )
        cpu = F.load_modules(fir, {"Rocket", "CSRFile"})
        instances = structure["cpu"]["instances"]
        csr = F.field_type(cpu["CSRFile"].port_type("io"), ("rw", "cmd"))
        rocc = F.field_type(cpu["Rocket"].port_type("io"), ("rocc", "cmd", "valid"))
        if (
            instances.get("csr") != "CSRFile"
            or sum(x == "CSRFile" for x in instances.values()) != 1
            or csr != ("uint", 3)
            or rocc != ("uint", 1)
        ):
            raise ValueError("selected scalar control/dispatch interfaces differ")
        completion = derive_completion(selected, source_root=source_root)
        caches, witnesses = cache_geometry(fir, dts, l2)
        verify_model(model, facts, structure, caches)
        manifest_path = observer_build / "timing-source-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest["sha256"] != FEATURE_ENGINE
            or _digest(manifest["members"]) != FEATURE_ENGINE
        ):
            raise ValueError(
                "native observer source manifest differs from reviewed release"
            )
        pins = {
            "implementation": _pin(__file__),
            "model": _pin(count_owner.model_path),
            "dts": _pin(dts),
            "l2": _pin(l2),
            "observer_manifest": _pin(manifest_path),
            "observer": _pin(observer_build / "spike"),
            "provider_manifest": _pin(provider_bundle / B.MANIFEST),
        }
        for name in (
            "gemmini_count_hardware",
            "gemmini_selected_scheduling",
            "gemmini_staged_completion",
            "gemmini_timing_resources",
            "merlin.targetgen.rtl.firrtl_struct",
            "merlin_timing.provider_bundle",
        ):
            pins["proof_helper:" + name] = _pin(importlib.import_module(name).__file__)
        for member in manifest["members"]:
            path = spike_source / member["path"]
            pin = _pin(path)
            if pin["sha256"] != member["sha256"]:
                raise ValueError("reviewed native generic observer source changed")
            pins["observer_source:" + member["path"]] = pin
        receipt = B.verify_bundle(provider_bundle)
        required_header = "timing/core/include/merlin_timing/reservations.hpp"
        if required_header not in receipt.get("runtime_headers", ()):
            raise ValueError(
                "native provider omits actual shared reservation-header custody"
            )
        roster = {member["path"]: member["sha256"] for member in receipt["members"]}
        for name, digest in PROVIDER_SOURCES.items():
            pin = _pin(spike_source / "examples/gemmini/timing" / name)
            if pin["sha256"] != digest or roster.get(name) != digest:
                raise ValueError(
                    "target provider is a different reviewed evaluator release"
                )
            pins["provider_source:" + name] = pin
        # Source-to-library association is executed here, never accepted from a
        # copied success receipt. Both candidate arms later use this same model.
        rebuild_dir.mkdir(parents=True, exist_ok=False)
        from merlin.common.cpu_pool import shared_compute_pool

        with shared_compute_pool():
            built = B.build(
                provider_bundle, spike_source, rebuild_dir / "provider.so", cxx
            )
        if (
            len(model["libraries"]) != 2
            or built["library_sha256"] != model["libraries"][1]
        ):
            raise ValueError(
                "actual source-rebuilt provider differs from selected model library"
            )
        pins["rebuilt_provider"] = _pin(built["library"])
        pins.update(
            {
                "generic_header:" + name: _pin(spike_source / name)
                for name in built["generic_header_sha256"]
            }
        )
        pins.update(
            {"rtl_source:" + name: pin for name, pin in completion["sources"].items()}
        )
        pins.update(
            {
                "structural_source:" + name: pin
                for name, pin in structure["sources"].items()
            }
        )
        owner = object.__new__(cls)
        owner._count_owner = count_owner
        owner._provider_bundle = provider_bundle.resolve()
        owner._pins = pins
        owner._record = {
            "schema": SCHEMA,
            "selected_binding": count_owner.selected_binding(),
            "pins": pins,
            "provider_build": built,
            "completion_proof": completion["proof"],
            "cache_geometry_witness": witnesses,
            "scope": "explicit_conservative_reservation_model_not_packet_cycle_equivalence",
            "limits": [
                "whole-command local store holds conservatively reduce overlap",
                "all accumulator aliases share one conservative conflict class",
                "aggregate cache LRU/coherence and future resource bookings are approximations",
                "prices require independent unchanged held validation",
                "unpriced/unmapped operations and unsupported mesh/tag overlap remain UNKNOWN",
                "large capacity/attention extrapolation requires independent labels",
            ],
            "hardware_execution_qualification": "not_established",
            "statistical_timing_qualification_issued": False,
        }
        owner.revalidate()
        return owner

    @property
    def identity(self):
        return SCHEMA + ":" + _digest(self._record)

    def revalidate(self):
        self._count_owner.revalidate()
        _revalidate_sources(self._pins)
        current = B.verify_bundle(self._provider_bundle)
        if (
            current["manifest_sha256"]
            != self._record["provider_build"]["manifest_sha256"]
        ):
            raise ValueError("native provider source bundle changed")
        if self._count_owner.selected_binding() != self._record["selected_binding"]:
            raise ValueError("native source owner differs from original count binding")

    def selected_binding(self):
        self.revalidate()
        return dict(self._record["selected_binding"])

    def record(self):
        self.revalidate()
        return json.loads(json.dumps(self._record))
