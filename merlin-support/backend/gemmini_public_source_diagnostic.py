"""Original per-current-program diagnostics, without a scheduling qualification.

The source-derived prototype supplies arithmetic, completion and runtime facts.
It cannot authorize a compiler pass. Every newly emitted executable still needs
the original bounded trace, complete source/product/state and numerical checks.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from gemmini_source_execution_issuer import SourceExecutionIssuer
from gemmini_staged_dense_admission import prepare_original_prototype
from gemmini_staged_dense_capability import StagedDensePrototype, _digest
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)

MECHANISM = "source_execution_diagnostic"


class PublicSourceDiagnosticIssuer:
    def __init__(self, prototype):
        if type(prototype) is not StagedDensePrototype:
            raise ValueError(
                "public diagnostics require the original source-derived prototype"
            )
        with shared_compute_pool(timeout_s=120):
            prototype.revalidate()
        self._prototype = prototype
        self._prototype_identity = prototype.identity
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_public_source_diagnostic",
                "gemmini_source_execution_issuer",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
                "gemmini_staged_dense_capability",
                "gemmini_staged_dense_admission",
                "merlin.common.invocation_record",
                "merlin.common.cpu_pool",
                "merlin.common.paths",
                "merlin.targetgen.native_model_execution",
            )
        }
        self._trace = SourceExecutionIssuer.create_source_execution_owner
        self._trace_code = self._trace.__code__
        self._identity = _digest([self._prototype_identity, self._sources])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.public_current_source_diagnostic.v1:" + self._identity

    def revalidate(self):
        with shared_compute_pool(timeout_s=120):
            self._prototype.revalidate()
            self._check_implementation()

    def _check_implementation(self):
        _revalidate_sources(self._sources)
        if (
            type(self._prototype) is not StagedDensePrototype
            or self._prototype.identity != self._prototype_identity
            or SourceExecutionIssuer.create_source_execution_owner is not self._trace
            or self._trace.__code__ is not self._trace_code
            or _digest([self._prototype_identity, self._sources]) != self._identity
        ):
            raise ValueError(
                "original public diagnostic source/runtime/checker changed"
            )

    def record(self):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            return {
                "identity": self.identity,
                "prototype": self._prototype.record(),
                "scope": "fresh current-source trace and complete original source/output checks only",
                "scheduling_qualification": "NONE",
                "hardware_execution_qualified": False,
                "timing_authority": "NONE",
                "promotion_authority": "NONE",
            }

    def source_rule(self):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            return self._prototype.arithmetic.source_rule()

    def create_source_execution_owner(
        self, *, native_result, command_buffer, lowered, work_root, remaining_s
    ):
        # The unchanged algorithm performs full revalidation before executing
        # and after the actual fresh trace. No previous program verdict enters.
        if (
            type(remaining_s) not in (int, float)
            or not math.isfinite(remaining_s)
            or remaining_s <= 0
        ):
            raise ValueError(
                "public source diagnostics require a positive remaining wall budget"
            )
        deadline = time.monotonic() + remaining_s
        # This original algorithm launches a raw functional process inheriting
        # this leaf; it does not wait on a separately leased Python child.
        # Its nested same-thread pool scope reuses this exact owned grant.
        with shared_compute_pool(timeout_s=min(120, remaining_s)):
            self._check_implementation()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(
                    "public source diagnostic budget expired in CPU admission"
                )
            return self._trace(
                self,
                native_result=native_result,
                command_buffer=command_buffer,
                lowered=lowered,
                work_root=work_root,
                remaining_s=remaining,
            )


def from_original_prototype(prototype):
    """Issue only a diagnostic callback from actual source-derived live owners."""
    if type(prototype) is not StagedDensePrototype:
        raise ValueError(
            "public diagnostic registry cannot consume saved or foreign owners"
        )
    with shared_compute_pool(timeout_s=120):
        prototype.revalidate()
        contract = prototype.contract
        binding = PassCapabilityBinding(
            contract.hardware,
            contract.software,
            contract.numerical_policy,
            contract.execution,
            prototype.library.contract.sha256,
        )
        return ReviewedPassCapabilities(
            (
                ReviewedPassCapability(
                    MECHANISM, binding, PublicSourceDiagnosticIssuer(prototype)
                ),
            )
        )


def restore_registry(inputs, *, work_root):
    """Original host/verified worker: derive facts anew, never restore a verdict."""
    prototype = prepare_original_prototype(inputs, work_root=Path(work_root))
    return from_original_prototype(prototype)
