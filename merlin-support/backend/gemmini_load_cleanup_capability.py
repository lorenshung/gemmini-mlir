"""Host-selected owner for qualified, bounded copied-emitter configuration cleanup.

The operator prepares its receipt after independent paired issue/output checks.
The owner never enables a dense backend, hardware timing, overlap or general
configuration removal. Each new realized pair still requires final checking.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from gemmini_quiescent_load_config import (
    LoadConfigurationContract,
    _sha,
    optimize_statements,
)
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.compile.scheduling.library_sources import shared_library_sources

SCHEMA = "gemmini.fenced_load_cleanup_capability.v1"
SCOPE = "functional_spike_fenced_load_source_effects"


@dataclass(frozen=True)
class LoadCleanupOwner:
    path: Path
    payload: bytes

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        payload = path.read_bytes()
        if len(payload) > 1 << 20:
            raise ValueError("load cleanup capability exceeds its explicit byte budget")
        owner = cls(path, payload)
        owner.revalidate()
        return owner

    @property
    def identity(self):
        return SCHEMA + ":" + _sha(self.payload)

    def revalidate(self):
        row = json.loads(self.payload)
        if (
            self.path.read_bytes() != self.payload
            or row.get("schema") != SCHEMA
            or row.get("scope") != SCOPE
            or row.get("admission") != "conditional_final_pair_required"
            or row.get("hardware_execution_qualification") != "not_established"
            or row.get("timing_authority") != "NONE"
            or row.get("dense_v1_enabled") is not False
            or row.get("shared_library_sources") != shared_library_sources()
            or row.get("owner_implementation") != _pin(__file__)
        ):
            raise ValueError(
                "load cleanup capability, library or implementation changed; requalify"
            )
        if not row.get("checks") or not all(
            value is True for value in row["checks"].values()
        ):
            raise ValueError(
                "load cleanup capability has unresolved independent checks"
            )
        _revalidate_sources(row["pinned_inputs"])
        contract = LoadConfigurationContract.read(row["effect_contract"]["path"])
        if contract.identity != row["effect_contract"]["sha256"]:
            raise ValueError("load cleanup effect selection changed")
        return row

    def selected_binding(self):
        row = self.revalidate()
        return {
            "selected_inputs": row["selected_inputs"],
            "effect_contract": row["effect_contract"],
            "runtime_inputs": row["runtime_inputs"],
            "completion_contract": row["completion_contract"],
            "shared_library_sources": row["shared_library_sources"],
            "qualification_receipt": _pin(self.path),
        }

    def record(self):
        row = self.revalidate()
        return {
            "identity": self.identity,
            "mechanism": "configuration_cleanup",
            "scope": SCOPE,
            "admission": row["admission"],
            "qualified_pairs": row["qualified_pairs"],
            "conditions": row["conditions"],
            "hardware_execution_qualification": "not_established",
            "timing_authority": "NONE",
            "dense_v1_enabled": False,
            "selected_binding": self.selected_binding(),
        }

    def rewrite(self, statements, *, enabled=True, execution_domain):
        if execution_domain != "functional_spike":
            raise ValueError(
                "load cleanup owner is qualified only for the selected functional Spike domain"
            )
        row = self.revalidate()
        contract = LoadConfigurationContract.read(row["effect_contract"]["path"])
        rewritten, report = optimize_statements(statements, contract, enabled=enabled)
        self.revalidate()
        return rewritten, {
            **report,
            "capability_identity": self.identity,
            "qualification_scope": SCOPE,
            "requires_independent_final_pair_check": True,
        }

    def check_actual_pair(self, before, after, *, execution_domain):
        if execution_domain != "functional_spike":
            raise ValueError("actual pair has a different selected execution domain")
        row = self.revalidate()
        checker_path = Path(row["checker"]["path"])
        spec = importlib.util.spec_from_file_location(
            "original_load_cleanup_checker", checker_path
        )
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        contract = LoadConfigurationContract.read(row["effect_contract"]["path"])
        removed = checker._check_omissions(
            before, after, contract.revalidate()["encoding"]["config_rs1"]
        )
        self.revalidate()
        return {
            "actual_redundant_setters_removed": removed,
            "other_commands_and_fences_unchanged": True,
            "numerical_and_memory_checks": "required_separately_for_this_realization",
            "timing_authority": "NONE",
        }
