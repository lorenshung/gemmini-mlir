"""Original source rederivation and functional-only staged capability admission."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from gemmini_staged_dense_capability import StagedDensePrototype, _digest
from gemmini_timing_resources import _pin, _revalidate_sources


class StagedCompletionOwner:
    """Rerun selected RTL completion proof; no JSON success constructor exists."""

    def __init__(self, selected, *, source_root):
        from gemmini_selected_scheduling import SelectedSchedulingTarget
        from gemmini_staged_completion import derive
        from merlin.common.paths import module_source_path

        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError("staged completion needs original exact target selection")
        self.selected = selected
        self._record = derive(selected, source_root=source_root)
        self._sources = {
            "owner": _pin(__file__),
            "derivation": _pin(module_source_path("gemmini_staged_completion")),
            **self._record["sources"],
        }
        self.identity = "gemmini.live_staged_completion.v1:" + _digest(
            [self._record, self._sources]
        )
        self.revalidate()

    def revalidate(self):
        self.selected.revalidate()
        _revalidate_sources(self._sources)
        if self._record["selected_inputs"] != self.selected.report()[
            "selected_inputs"
        ] or self.identity != "gemmini.live_staged_completion.v1:" + _digest(
            [self._record, self._sources]
        ):
            raise ValueError("original staged completion derivation changed")

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)


class QualifiedStagedDenseOwner:
    """Only actual original sealed-source realization checks issue this owner."""

    def __init__(self, prototype, pairs):
        if (
            type(prototype) is not StagedDensePrototype
            or type(pairs) is not tuple
            or len(pairs) < 5
        ):
            raise ValueError(
                "staged admission requires an original prototype and five independent boundary/control pairs"
            )
        self.prototype = prototype
        self._pairs = copy.deepcopy(pairs)
        self._checks = [
            prototype.check_pair_artifacts(**pair, execution_domain="functional_spike")
            for pair in pairs
        ]
        removals = [row["actual_rhs_loads_removed"] for row in self._checks]
        geometries = [
            row["original_source_execution"][0]["source_geometry"]
            for row in self._checks
        ]
        sources = [
            row["original_source_execution"][0]["source_sha256"] for row in self._checks
        ]
        dim = prototype.contract.dim
        positive_tails = {
            shape["m"] % dim
            for shape, removed in zip(geometries, removals, strict=True)
            if removed > 0
        }
        if (
            len(set(sources)) != len(sources)
            or not {0, 1, dim - 1} <= positive_tails
            or not any(
                removed == 0 and shape["k"] > dim and shape["n"] <= dim
                for shape, removed in zip(geometries, removals, strict=True)
            )
            or not any(
                removed == 0 and shape["n"] > dim and shape["k"] <= dim
                for shape, removed in zip(geometries, removals, strict=True)
            )
        ):
            raise ValueError(
                "staged admission needs independent M-boundary positives and independent K/N no-gain controls"
            )
        self._prototype_identity = prototype.identity
        self._source = _pin(__file__)
        self._identity = _digest(
            [self._source, self._prototype_identity, self._pairs, self._checks]
        )
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.qualified_staged_dense.v1:" + self._identity

    @property
    def cleanup(self):
        return self.prototype.cleanup

    @property
    def nm(self):
        return self.prototype.nm

    def revalidate(self):
        self.prototype.revalidate()
        if (
            _pin(__file__) != self._source
            or self.prototype.identity != self._prototype_identity
            or _digest(
                [self._source, self._prototype_identity, self._pairs, self._checks]
            )
            != self._identity
        ):
            raise ValueError(
                "original staged admission or independent evidence changed"
            )
        for pair in self._pairs:
            for arm in pair.values():
                for role in (
                    "native_result",
                    "command_buffer_artifact",
                    "lowered",
                    "issue_invocation",
                ):
                    pin = arm[role]
                    if _pin(pin["path"]) != {
                        name: pin[name] for name in ("path", "sha256")
                    }:
                        raise ValueError(
                            "original staged qualification artifact changed"
                        )
                from merlin.common import invocation_record
                from merlin.targetgen.native_model_execution import _pinned_native_file

                invocation_record.verify(Path(arm["issue_invocation"]["path"]))
                native = json.loads(Path(arm["native_result"]["path"]).read_bytes())
                for group in (native["source"], native["build_artifacts"]):
                    for role in group:
                        _pinned_native_file(group, role)
                for pin in native["host_build_sources"]:
                    if _pin(pin["path"]) != {
                        key: pin[key] for key in ("path", "sha256")
                    }:
                        raise ValueError(
                            "original staged native build implementation changed"
                        )

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "prototype": self.prototype.record(),
            "checks": self._checks,
            "qualified_pairs": self._pairs,
            "domain": "functional_spike_exact_source_and_inputs",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "dense_v1_enabled": False,
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        return self.prototype.prepare_compiler(source, destination, policy=policy)

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        return self.prototype.check_pair_artifacts(
            before, after, execution_domain=execution_domain
        )

    def check_functional_guard(
        self, baseline, candidate, baseline_result, candidate_result, *, grade_root
    ):
        self.revalidate()
        return self.prototype.check_functional_guard(
            baseline,
            candidate,
            baseline_result,
            candidate_result,
            grade_root=grade_root,
        )


def prepare_original_prototype(inputs, *, work_root):
    """Original host only: preparation is explicitly unqualified until checked."""
    from gemmini_arithmetic_scope import IntegerArithmeticOwner
    from gemmini_compiler_passes import restore_registry as cleanup_registry
    from merlin.common.cpu_pool import shared_compute_pool

    registry = cleanup_registry(inputs, work_root=Path(work_root) / "cleanup-admission")
    cleanup = registry._capabilities[0].owner
    with shared_compute_pool(timeout_s=120):
        arithmetic = IntegerArithmeticOwner(
            cleanup.cleanup.selected,
            **{
                name: Path(inputs[name])
                for name in (
                    "software_spec",
                    "source_root",
                    "receiver_source",
                    "runtime",
                )
            },
        )
        completion = StagedCompletionOwner(
            cleanup.cleanup.selected, source_root=Path(inputs["source_root"])
        )
        return StagedDensePrototype(cleanup, arithmetic, completion)


def restore_registry(inputs, *, work_root):
    """Explicit live reviewed registry; absent real source pairs stay disabled."""
    if not inputs.get("staged_qualification_pairs"):
        raise ValueError(
            "staged capability disabled: original sealed-source positive/control realization pairs are unavailable"
        )
    from merlin.llvmlower.reviewed_pass_capabilities import (
        PassCapabilityBinding,
        ReviewedPassCapabilities,
        ReviewedPassCapability,
    )

    prototype = prepare_original_prototype(inputs, work_root=work_root)
    owner = QualifiedStagedDenseOwner(
        prototype, tuple(inputs["staged_qualification_pairs"])
    )
    contract = prototype.contract
    binding = PassCapabilityBinding(
        contract.hardware,
        contract.software,
        contract.numerical_policy,
        contract.execution,
        prototype.library.contract.sha256,
    )
    return ReviewedPassCapabilities(
        (ReviewedPassCapability("staged_dense_raw_i32", binding, owner),)
    )
