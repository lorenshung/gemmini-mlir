"""Explicit structural bridge for the new generated-source diagnostic owner.

Native statistical and event-model gates remain independent. This joins exact
original live source/count owners, never a saved qualification or compiler pass.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType

from gemmini_count_hardware import CountHardwareCorrespondenceOwner
from gemmini_generated_source_diagnostic import _STATE, GeneratedSourceDiagnosticIssuer
from gemmini_staged_dense_capability import _digest
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)

MECHANISM = "compiler_source_native_correspondence"
_ISSUED = weakref.WeakKeyDictionary()


class GeneratedSourceNativeCorrespondenceOwner:
    def __init__(self):
        raise TypeError(
            "issue generated-source correspondence from original live capabilities only"
        )

    @classmethod
    def bind(cls, source_capability, count_capability):
        if cls is not GeneratedSourceNativeCorrespondenceOwner or any(
            type(c) is not ReviewedPassCapability
            for c in (source_capability, count_capability)
        ):
            raise TypeError(
                "generated-source correspondence needs original reviewed capabilities"
            )
        registry = ReviewedPassCapabilities((source_capability, count_capability))
        with scoped_registry_metadata_verification(registry):
            source, count = source_capability.owner, count_capability.owner
            if (
                source_capability.mechanism != "source_execution_diagnostic"
                or count_capability.mechanism != "timing_model_correspondence"
                or type(source) is not GeneratedSourceDiagnosticIssuer
                or type(count) is not CountHardwareCorrespondenceOwner
            ):
                raise TypeError(
                    "generated-source correspondence selected foreign owner types or mechanisms"
                )
            sources = {
                name: _pin(module_source_path(name))
                for name in (
                    "gemmini_generated_source_native",
                    "gemmini_generated_source_diagnostic",
                    "gemmini_count_hardware",
                    "merlin.llvmlower.reviewed_pass_capabilities",
                    "merlin.llvmlower.reviewed_metadata",
                )
            }
            callbacks = tuple(
                (
                    owner,
                    name,
                    getattr(type(owner), name),
                    getattr(type(owner), name).__code__,
                )
                for owner, names in (
                    (source, ("revalidate", "selected_binding")),
                    (count, ("revalidate", "selected_binding")),
                )
                for name in names
            )
            state = {
                "registry": registry,
                "source_cap": source_capability,
                "count_cap": count_capability,
                "source": source,
                "count": count,
                "source_binding": source_capability.binding,
                "native_binding": count_capability.binding,
                "source_id": source_capability.identity,
                "count_id": count_capability.identity,
                "sources": sources,
                "callbacks": callbacks,
                "source_state": _STATE(source),
            }
            state["identity"] = _digest(
                [
                    state["source_id"],
                    state["count_id"],
                    sources,
                    asdict(state["source_binding"]),
                    asdict(state["native_binding"]),
                ]
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
        if type(self) is not GeneratedSourceNativeCorrespondenceOwner or state is None:
            raise ValueError(
                "generated-source correspondence was not originally issued"
            )
        for owner, name, function, code in (
            *state["callbacks"],
            *((self, *row) for row in _API),
        ):
            method = getattr(owner, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not owner
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError(
                    "original generated-source correspondence callback changed"
                )
        _revalidate_sources(state["sources"])
        for cap, owner, binding, identity in (
            (
                state["source_cap"],
                state["source"],
                state["source_binding"],
                state["source_id"],
            ),
            (
                state["count_cap"],
                state["count"],
                state["native_binding"],
                state["count_id"],
            ),
        ):
            if (
                type(cap) is not ReviewedPassCapability
                or cap._owner is not owner
                or cap._binding != binding
                or cap._identity != identity
            ):
                raise ValueError(
                    "original generated-source capability changed at metadata boundary"
                )
        if (
            scoped_registry_metadata_verification is not _METADATA_FUNCTION
            or scoped_registry_metadata_verification.__code__ is not _METADATA_CODE
            or _CHECKED.__wrapped__ is not _CHECKED_BODY
            or _CHECKED_BODY.__code__ is not _CHECKED_CODE
        ):
            raise ValueError("original generated-source metadata operation changed")
        if (
            type(state["registry"]) is not ReviewedPassCapabilities
            or state["registry"]._capabilities
            != (state["source_cap"], state["count_cap"])
            or _digest(
                [
                    state["source_id"],
                    state["count_id"],
                    state["sources"],
                    asdict(state["source_binding"]),
                    asdict(state["native_binding"]),
                ]
            )
            != state["identity"]
        ):
            raise ValueError(
                "original generated-source correspondence registry/binding changed"
            )
        return state

    def _relation(self):
        state = _ORIGINAL_STATE(self)
        s, c = state["source_cap"], state["count_cap"]
        if (
            s.owner is not state["source"]
            or c.owner is not state["count"]
            or s.binding != state["source_binding"]
            or c.binding != state["native_binding"]
            or s.identity != state["source_id"]
            or c.identity != state["count_id"]
        ):
            raise ValueError("original generated-source/count capability changed")
        source = _STATE(state["source"])
        count = state["count"]
        if (
            source is not state["source_state"]
            or state["source"].selected_binding() != s.binding
        ):
            raise ValueError(
                "generated-source correspondence lost its original semantic owner"
            )
        native = count.correspondence.revalidate()
        proof = source["completion"].record()
        if (
            source["selected_inputs"] != native["selected_inputs"]
            or proof["selected_inputs"] != source["selected_inputs"]
            or s.binding.software
            != _digest([source["selected_inputs"], proof["proof"]])
            or c.binding != PassCapabilityBinding(**count.selected_binding())
            or c.binding.hardware != source["selected_inputs"]["facts"]["sha256"]
            or c.binding.software
            != source["selected_inputs"]["observed_contract"]["sha256"]
            or source["runtime"]["functional_extension"]
            != native["pins"]["functional_extension"]
        ):
            raise ValueError(
                "generated-source semantic facts/runtime differ from original native correspondence"
            )

    @contextmanager
    def _checked(self):
        state = _ORIGINAL_STATE(self)
        with scoped_registry_metadata_verification(state["registry"]):
            _RELATION(self)
            try:
                yield state
            finally:
                _RELATION(self)
        _ORIGINAL_STATE(self)

    @property
    def identity(self):
        return (
            "gemmini.generated_source_native_correspondence.v1:"
            + _ORIGINAL_STATE(self)["identity"]
        )

    def revalidate(self):
        with _CHECKED(self):
            pass

    def verify_compiler_binding(self, compiler_binding, *, native_binding):
        with _CHECKED(self) as state:
            if (
                type(compiler_binding) is not PassCapabilityBinding
                or type(native_binding) is not PassCapabilityBinding
                or compiler_binding != state["source_binding"]
                or native_binding != state["native_binding"]
            ):
                raise ValueError(
                    "actual generated compiler/native selection changed its full original binding"
                )

    def selected_binding(self):
        with _CHECKED(self) as state:
            result = asdict(state["native_binding"])
        return result

    def record(self):
        with _CHECKED(self) as state:
            result = {
                "identity": self.identity,
                "compiler_binding": asdict(state["source_binding"]),
                "native_binding": asdict(state["native_binding"]),
                "scope": "structural live generated-source/native namespace correspondence only",
                "software_digests_equated": False,
                "timing_qualification": False,
                "numerical_qualification": False,
                "promotion_authority": "NONE",
            }
        return result


_ORIGINAL_STATE = GeneratedSourceNativeCorrespondenceOwner._state
_RELATION = GeneratedSourceNativeCorrespondenceOwner._relation
_CHECKED = GeneratedSourceNativeCorrespondenceOwner._checked
_CHECKED_BODY = _CHECKED.__wrapped__
_CHECKED_CODE = _CHECKED_BODY.__code__
_METADATA_FUNCTION = scoped_registry_metadata_verification
_METADATA_CODE = scoped_registry_metadata_verification.__code__
_API = tuple(
    (
        name,
        getattr(GeneratedSourceNativeCorrespondenceOwner, name),
        getattr(GeneratedSourceNativeCorrespondenceOwner, name).__code__,
    )
    for name in (
        "_state",
        "_relation",
        "_checked",
        "revalidate",
        "verify_compiler_binding",
        "selected_binding",
        "record",
    )
)
