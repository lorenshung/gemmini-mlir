"""Explicit V2 log-mode namespace bridge, composed from original live owners.

Only the source observation identity changes. The original V1 structural
bridge still derives and checks the selected software/RTL/native correspondence.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import asdict

from gemmini_count_hardware import CountHardwareCorrespondenceOwner
from gemmini_public_source_observation import (
    _ORIGINAL_STATE,
    PublicSourceObservationIssuer,
)
from gemmini_source_native_correspondence import (
    SourceNativeCorrespondenceOwner,
)
from gemmini_staged_dense_capability import _digest
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)

_ISSUED = weakref.WeakKeyDictionary()


class SourceObservationNativeCorrespondenceOwner:
    def __init__(self):
        raise TypeError(
            "issue only from original live V2 source and count capabilities"
        )

    @classmethod
    def bind(cls, source_capability, count_capability):
        if cls is not SourceObservationNativeCorrespondenceOwner or any(
            type(cap) is not ReviewedPassCapability
            for cap in (source_capability, count_capability)
        ):
            raise TypeError("V2 correspondence requires original reviewed capabilities")
        registry = ReviewedPassCapabilities((source_capability, count_capability))
        with scoped_registry_metadata_verification(registry):
            source, count = source_capability.owner, count_capability.owner
            if (
                source_capability.mechanism != "source_execution_diagnostic"
                or count_capability.mechanism != "timing_model_correspondence"
                or type(source) is not PublicSourceObservationIssuer
                or type(count) is not CountHardwareCorrespondenceOwner
            ):
                raise TypeError(
                    "V2 correspondence requires exact original source/count owners"
                )
            # This is an actual original issuer retained by the live V2 owner,
            # not an unwrapped object supplied by a caller or a saved verdict.
            original = _ORIGINAL_STATE(source)[0]
            source_binding, count_binding = (
                source_capability.binding,
                count_capability.binding,
            )
            original_cap = ReviewedPassCapability(
                "source_execution_diagnostic", source_binding, original
            )
            bridge = SourceNativeCorrespondenceOwner.bind(
                original_cap, count_capability
            )
            sources = {
                name: _pin(module_source_path(name))
                for name in (
                    "gemmini_source_observation_native",
                    "gemmini_public_source_observation",
                    "gemmini_source_native_correspondence",
                )
            }
            state = (
                registry,
                source_capability,
                count_capability,
                source,
                count,
                original,
                bridge,
                source_binding,
                count_binding,
                source_capability.identity,
                count_capability.identity,
                sources,
            )
        value = object.__new__(cls)
        _ISSUED[value] = state
        try:
            value.revalidate()
        except BaseException:
            del _ISSUED[value]
            raise
        return value

    def _state(self):
        state = _ISSUED.get(self)
        if (
            type(self) is not SourceObservationNativeCorrespondenceOwner
            or state is None
        ):
            raise ValueError("V2 correspondence was not originally issued")
        for name, function, code in _ORIGINAL_API:
            method = getattr(self, name)
            if (
                getattr(method, "__self__", None) is not self
                or getattr(method, "__func__", None) is not function
                or function.__code__ is not code
            ):
                raise ValueError("original V2 correspondence callback changed")
        _revalidate_sources(state[11])
        (
            registry,
            source_cap,
            count_cap,
            source,
            count,
            original,
            bridge,
            sb,
            cb,
            si,
            ci,
            _,
        ) = state
        if (
            type(registry) is not ReviewedPassCapabilities
            or registry._capabilities != (source_cap, count_cap)
            or source_cap.owner is not source
            or count_cap.owner is not count
            or source_cap.binding != sb
            or count_cap.binding != cb
            or source_cap.identity != si
            or count_cap.identity != ci
            or _ORIGINAL_STATE(source)[0] is not original
            or type(bridge) is not SourceNativeCorrespondenceOwner
        ):
            raise ValueError("original V2 correspondence owner/binding changed")
        return state

    @contextmanager
    def _checked(self):
        state = _ISSUED.get(self)
        if (
            type(self) is not SourceObservationNativeCorrespondenceOwner
            or state is None
        ):
            raise ValueError("V2 correspondence was not originally issued")
        with scoped_registry_metadata_verification(state[0]):
            _ORIGINAL_STATE_OF_BRIDGE(self)
            state[6].revalidate()
            try:
                yield state
            finally:
                state[6].revalidate()
                _ORIGINAL_STATE_OF_BRIDGE(self)
        # Final callbacks may mutate state; no result escapes before this check.
        _ORIGINAL_STATE_OF_BRIDGE(self)

    @property
    def identity(self):
        state = _ORIGINAL_STATE_OF_BRIDGE(self)
        return "gemmini.source_native_correspondence.v2:" + _digest(
            [state[9], state[10], state[11], asdict(state[7]), asdict(state[8])]
        )

    def revalidate(self):
        with _ORIGINAL_CHECKED(self):
            pass

    def verify_compiler_binding(self, compiler_binding, *, native_binding):
        with _ORIGINAL_CHECKED(self) as state:
            state[6].verify_compiler_binding(
                compiler_binding, native_binding=native_binding
            )

    def selected_binding(self):
        with _ORIGINAL_CHECKED(self) as state:
            result = state[6].selected_binding()
        return result

    def record(self):
        with _ORIGINAL_CHECKED(self) as state:
            result = state[6].record()
            result.update(
                identity=self.identity,
                source_capability_identity=state[9],
                source_observation_protocol="gemmini.source_observation.v2",
            )
        return result


_ORIGINAL_STATE_OF_BRIDGE = SourceObservationNativeCorrespondenceOwner._state
_ORIGINAL_CHECKED = SourceObservationNativeCorrespondenceOwner._checked
_ORIGINAL_API = tuple(
    (
        name,
        getattr(SourceObservationNativeCorrespondenceOwner, name),
        getattr(SourceObservationNativeCorrespondenceOwner, name).__code__,
    )
    for name in (
        "_state",
        "_checked",
        "revalidate",
        "verify_compiler_binding",
        "selected_binding",
        "record",
    )
)
