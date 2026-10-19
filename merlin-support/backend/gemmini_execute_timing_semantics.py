"""Source owner for an explicit operand-sensitive conservative timing variant.

The retained owner still checks legacy dependencies, completion and resources.
This additional owner checks only the source-derived input service substitution;
it issues no statistical timing or physical-equivalence qualification.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

from gemmini_count_hardware import CountHardwareCorrespondenceOwner
from gemmini_execute_service import CONTROL, derive
from gemmini_native_timing_semantics import PROVIDER_SOURCES, NativeTimingSemanticsOwner
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin_timing import provider_bundle as B

SCHEMA = "gemmini.execute_service_native_semantics.v1"
SOURCES = {
    "execute_service.hpp": "e0ac51d77a3b47b0a4b74c714f4c8af89f4a6c65135eac1ed56b0a68693e9e7b",
    "export_execute_service.py": "bbe78df0956f6f78f1a2764f2477863a29c705518d9d7c7bb21d93417ee05d72",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_variant(original, candidate, proof, *, fact_sha256):
    """Only the reviewed library and input-service layer may differ."""
    selected = candidate.get("target", {}).get("execution_service")
    if type(selected) is not dict or set(selected) != {
        "version",
        "dimension",
        "minimum_rows",
        "reduction_guard_mask",
        "cycles_per_row",
        "pipeline_cycles",
        "source_fact_sha256",
    }:
        raise ValueError("execute service needs an exact explicit field roster")
    mask = sum(
        1 << sum(row[key] << (len(CONTROL) - 1 - i) for i, key in enumerate(CONTROL))
        for row in proof["reduction_guard_cases"]
        if row["reduced"]
    )
    integers = {
        "version": 1,
        "dimension": proof["dimension"],
        "minimum_rows": proof["minimum_reduced_input_rows"],
        "reduction_guard_mask": mask,
    }
    if any(
        type(selected[key]) is not int or selected[key] != value
        for key, value in integers.items()
    ):
        raise ValueError(
            "execute service control or geometry differs from original source proof"
        )
    if (
        type(selected["cycles_per_row"]) not in (int, float)
        or selected["cycles_per_row"] != 1
        or selected["source_fact_sha256"] != fact_sha256
    ):
        raise ValueError("execute service rate/fact differs from the selected proof")
    pipeline = selected["pipeline_cycles"]
    if (
        type(pipeline) not in (int, float)
        or not math.isfinite(pipeline)
        or pipeline < 0
    ):
        raise ValueError("execute propagation needs a finite nonnegative fitted cost")
    if (
        len(candidate.get("libraries", ())) != 2
        or len(original.get("libraries", ())) != 2
    ):
        raise ValueError(
            "execute service needs the original functional and explicit timing library"
        )
    reduced = copy.deepcopy(candidate)
    del reduced["target"]["execution_service"]
    reduced["libraries"][1] = original["libraries"][1]
    if reduced != original:
        raise ValueError(
            "execute variant changed the original functional, resource, memory or price selection"
        )


class ExecuteServiceTimingOwner:
    """Composes two original live owners; copied success records cannot issue it."""

    def __init__(self):
        raise TypeError("construct through bind with original live source owners")

    @classmethod
    def bind(
        cls,
        count_owner,
        *,
        original_native_owner,
        source_root,
        spike_source,
        provider_bundle,
        source_fact,
        rebuild_dir,
        cxx="c++",
    ):
        if (
            type(count_owner) is not CountHardwareCorrespondenceOwner
            or type(original_native_owner) is not NativeTimingSemanticsOwner
        ):
            raise TypeError("execute semantics require the original live source owners")
        count_owner.revalidate()
        original_native_owner.revalidate()
        original_record = original_native_owner.record()
        original_binding = original_native_owner.selected_binding()
        binding = count_owner.selected_binding()
        if any(
            binding[key] != original_binding[key]
            for key in ("hardware", "software", "numerical_policy")
        ):
            raise ValueError(
                "execute variant changed selected hardware/software/count semantics"
            )
        paths = list(
            map(
                Path,
                (source_root, spike_source, provider_bundle, source_fact, rebuild_dir),
            )
        )
        source_root, spike_source, provider_bundle, source_fact, rebuild_dir = paths
        selected = SelectedSchedulingTarget.read(
            count_owner.facts_path, count_owner.observed_contract_path
        )
        fact = derive(
            selected,
            execute_controller=source_root
            / "generators/gemmini/src/main/scala/gemmini/ExecuteController.scala",
        )
        if json.loads(source_fact.read_bytes()) != fact:
            raise ValueError(
                "selected execute service fact is not freshly source-derived"
            )
        original = json.loads(
            Path(original_record["pins"]["model"]["path"]).read_bytes()
        )
        model = json.loads(count_owner.model_path.read_bytes())
        verify_variant(
            original, model, fact["proof"], fact_sha256=_pin(source_fact)["sha256"]
        )
        pins = {
            "implementation": _pin(__file__),
            "model": _pin(count_owner.model_path),
            "source_fact": _pin(source_fact),
            "provider_manifest": _pin(provider_bundle / B.MANIFEST),
        }
        root = spike_source / "examples/gemmini/timing"
        for name, digest in SOURCES.items():
            pin = _pin(root / name)
            if pin["sha256"] != digest:
                raise ValueError(
                    "execute service is a different reviewed source release"
                )
            pins["provider_source:" + name] = pin
        receipt = B.verify_bundle(provider_bundle)
        roster = {row["path"]: row["sha256"] for row in receipt["members"]}
        if set(roster) != {
            "provider.hpp",
            "staged.hpp",
            "bridge.cpp",
            "legacy_bridge.cpp",
            "README.md",
            B.BUILDER,
            "execute_service.hpp",
            "execute-service-export.json",
        }:
            raise ValueError("execute provider has a different closed source roster")
        for name, digest in PROVIDER_SOURCES.items():
            retained = "legacy_bridge.cpp" if name == "bridge.cpp" else name
            if roster[retained] != digest:
                raise ValueError(
                    "execute variant modified the retained staging substrate"
                )
        if roster["execute_service.hpp"] != SOURCES["execute_service.hpp"]:
            raise ValueError("execute service bundle differs from reviewed header")
        # Import only the exact reviewed exporter, then reconstruct the actual
        # compiled bridge. Manifest strings alone do not establish this binding.
        spec = importlib.util.spec_from_file_location(
            "reviewed_execute_export", root / "export_execute_service.py"
        )
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        expected = exporter.specialize_bridge(
            (provider_bundle / "legacy_bridge.cpp").read_bytes()
        )
        if (provider_bundle / "bridge.cpp").read_bytes() != expected:
            raise ValueError("execute bridge differs from the reviewed specialization")
        if set(receipt["runtime_headers"]) != set(
            original_record["provider_build"]["runtime_headers"]
        ):
            raise ValueError("execute variant changed generic runtime-header custody")
        for name in (
            "gemmini_execute_service",
            "gemmini_native_timing_semantics",
            "gemmini_count_hardware",
            "gemmini_selected_scheduling",
            "gemmini_timing_resources",
            "merlin.targetgen.rtl.firrtl_struct",
            "merlin_timing.provider_bundle",
        ):
            pins["proof_helper:" + name] = _pin(importlib.import_module(name).__file__)
        pins.update(
            {"service_source:" + name: pin for name, pin in fact["sources"].items()}
        )
        from merlin.common.cpu_pool import shared_compute_pool

        rebuild_dir.mkdir(parents=True, exist_ok=False)
        with shared_compute_pool():
            built = B.build(
                provider_bundle, spike_source, rebuild_dir / "provider.so", cxx
            )
        if built["library_sha256"] != model["libraries"][1]:
            raise ValueError(
                "source-rebuilt execute provider differs from selected library"
            )
        pins["rebuilt_provider"] = _pin(built["library"])
        pins.update(
            {
                "generic_header:" + name: _pin(spike_source / name)
                for name in built["generic_header_sha256"]
            }
        )
        owner = object.__new__(cls)
        owner._count_owner, owner._original_owner = count_owner, original_native_owner
        owner._provider_bundle, owner._pins = provider_bundle, pins
        owner._record = {
            "schema": SCHEMA,
            "selected_binding": binding,
            "pins": pins,
            "original_native_owner": original_native_owner.identity,
            "provider_build": built,
            "input_service_proof": fact["proof"],
            "scope": "source_checked_input_service_with_legacy_conservative_completion_and_resources",
            "limits": [
                "cutpoint service proof is not state reachability or physical cycle equivalence",
                "nonnegative propagation cost is fitted, not a derived physical floor",
                "whole-command holds and cache approximations remain unchanged",
                "statistical gates and independent numerical admission remain required",
                "capacity-scale and attention predictions require their own compatible labels",
            ],
            "statistical_timing_qualification_issued": False,
            "hardware_execution_qualification": "not_established",
        }
        owner.revalidate()
        return owner

    @property
    def identity(self):
        return SCHEMA + ":" + _digest(self._record)

    def revalidate(self):
        self._count_owner.revalidate()
        self._original_owner.revalidate()
        _revalidate_sources(self._pins)
        if (
            B.verify_bundle(self._provider_bundle)["manifest_sha256"]
            != self._record["provider_build"]["manifest_sha256"]
        ):
            raise ValueError("execute provider source bundle changed")
        if self._count_owner.selected_binding() != self._record["selected_binding"]:
            raise ValueError("execute owner differs from original count binding")

    def selected_binding(self):
        self.revalidate()
        return dict(self._record["selected_binding"])

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)
