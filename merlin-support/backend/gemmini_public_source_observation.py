"""Explicit complete-commit diagnostics; no scheduling or timing authority.

The old public source factory remains V1. This separately selected owner uses
the original live prototype and trace/check algorithms, with V2 log flags.
"""

from __future__ import annotations

import math
import time
import weakref
from pathlib import Path

import gemmini_source_observation_runtime as _RUNTIME
from gemmini_public_source_diagnostic import (
    MECHANISM,
    PublicSourceDiagnosticIssuer,
)
from gemmini_source_observation_runtime import COMMIT_ONLY_PROTOCOL, source_trace_flags
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

_ISSUED = weakref.WeakKeyDictionary()


class PublicSourceObservationIssuer:
    """A live original public issuer with an immutable explicit log protocol."""

    def __init__(self, prototype):
        if type(self) is not PublicSourceObservationIssuer:
            raise TypeError("source observation issuer cannot be subclassed")
        if type(prototype) is not StagedDensePrototype:
            raise ValueError("source observation requires the original live prototype")
        original = PublicSourceDiagnosticIssuer(prototype)
        sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_public_source_observation",
                "gemmini_public_source_observation_factory",
                "gemmini_public_source_diagnostic",
                "gemmini_source_observation_runtime",
                "gemmini_source_execution_issuer",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
            )
        }
        sources.update(
            {
                "explicit_" + name: dict(pin)
                for name, pin in _RUNTIME.IMPLEMENTATION_PINS.items()
            }
        )
        callbacks = tuple(
            (
                owner,
                name,
                getattr(type(owner), name),
                getattr(type(owner), name).__code__,
            )
            for owner, names in ((original, ("revalidate", "source_rule", "record")),)
            for name in names
        )
        trace = _RUNTIME.create_source_execution_owner
        _ISSUED[self] = (
            original,
            prototype,
            original.identity,
            sources,
            callbacks,
            trace,
            trace.__code__,
        )
        try:
            self.revalidate()
        except BaseException:
            del _ISSUED[self]
            raise

    def _state(self):
        _RUNTIME.revalidate_export()
        for name, function, code in _ORIGINAL_API:
            method = getattr(self, name)
            if (
                getattr(method, "__self__", None) is not self
                or getattr(method, "__func__", None) is not function
                or function.__code__ is not code
            ):
                raise ValueError("original source observation implementation changed")
        state = _ISSUED.get(self)
        if type(self) is not PublicSourceObservationIssuer or state is None:
            raise ValueError("source observation owner was not originally issued")
        original, prototype, identity, sources, callbacks, trace, code = state
        _revalidate_sources(sources)
        if (
            type(original) is not PublicSourceDiagnosticIssuer
            or original._prototype is not prototype
            or type(prototype) is not StagedDensePrototype
            or original.identity != identity
            or _RUNTIME.create_source_execution_owner is not trace
            or trace.__code__ is not code
        ):
            raise ValueError("original source observation selection changed")
        for owner, name, function, code in callbacks:
            method = getattr(owner, name)
            if (
                getattr(method, "__self__", None) is not owner
                or getattr(method, "__func__", None) is not function
                or function.__code__ is not code
            ):
                raise ValueError("original source observation callback changed")
        return state

    @property
    def _prototype(self):
        return _ORIGINAL_STATE(self)[1]

    @property
    def _sources(self):
        # The unchanged trace algorithm records every original source leaf too.
        state = _ORIGINAL_STATE(self)
        return {**state[0]._sources, **state[3]}

    @property
    def identity(self):
        state = _ORIGINAL_STATE(self)
        return "gemmini.public_current_source_diagnostic.v2:" + _digest(
            [
                state[2],
                state[3],
                COMMIT_ONLY_PROTOCOL,
                source_trace_flags(COMMIT_ONLY_PROTOCOL),
            ]
        )

    def revalidate(self):
        with shared_compute_pool(timeout_s=120):
            state = _ORIGINAL_STATE(self)
            state[0].revalidate()
            _ORIGINAL_STATE(self)

    def record(self):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            result = _ORIGINAL_STATE(self)[0].record()
            result.update(
                identity=self.identity,
                source_observation={
                    "protocol": COMMIT_ONLY_PROTOCOL,
                    "flags": list(source_trace_flags(COMMIT_ONLY_PROTOCOL)),
                },
            )
            self.revalidate()
        return result

    def source_rule(self):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            result = _ORIGINAL_STATE(self)[0].source_rule()
            self.revalidate()
        return result

    def create_source_execution_owner(
        self, *, native_result, command_buffer, lowered, work_root, remaining_s
    ):
        if (
            type(remaining_s) not in (int, float)
            or not math.isfinite(remaining_s)
            or remaining_s <= 0
        ):
            raise ValueError(
                "source observation requires a positive remaining wall budget"
            )
        deadline = time.monotonic() + remaining_s
        with shared_compute_pool(timeout_s=min(120, remaining_s)):
            state = _ORIGINAL_STATE(self)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("source observation budget expired in CPU admission")
            result = state[5](
                self,
                native_result=native_result,
                command_buffer=command_buffer,
                lowered=lowered,
                work_root=work_root,
                remaining_s=remaining,
                observation_protocol=COMMIT_ONLY_PROTOCOL,
            )
            self.revalidate()
        return result


_ORIGINAL_STATE = PublicSourceObservationIssuer._state
_ORIGINAL_API = tuple(
    (
        name,
        getattr(PublicSourceObservationIssuer, name),
        getattr(PublicSourceObservationIssuer, name).__code__,
    )
    for name in (
        "_state",
        "revalidate",
        "record",
        "source_rule",
        "create_source_execution_owner",
    )
)


def from_original_prototype(prototype):
    if type(prototype) is not StagedDensePrototype:
        raise ValueError(
            "source observation cannot consume saved or foreign prototypes"
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
        result = ReviewedPassCapabilities(
            (
                ReviewedPassCapability(
                    MECHANISM, binding, PublicSourceObservationIssuer(prototype)
                ),
            )
        )
    return result


def restore_registry(inputs, *, work_root):
    if (
        type(inputs) is not dict
        or inputs.get("source_observation_protocol") != COMMIT_ONLY_PROTOCOL
    ):
        raise ValueError(
            "V2 source factory requires the explicit complete-commit protocol"
        )
    selected = dict(inputs)
    selected.pop("source_observation_protocol")
    prototype = prepare_original_prototype(selected, work_root=Path(work_root))
    return from_original_prototype(prototype)
