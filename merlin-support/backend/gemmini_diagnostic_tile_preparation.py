"""Original current-source preparation, never a scheduling qualification.

This prepares independent b847 descendants under the same library and fixed
source/effect contract. Every executable still needs the original source,
packet, complete-output and runtime checks. No timing or promotion is issued.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary

from gemmini_public_source_diagnostic import PublicSourceDiagnosticIssuer
from gemmini_staged_dense_capability import StagedDensePrototype, _digest
from gemmini_staged_tile_capability import StagedTilePrototype
from gemmini_timing_resources import _pin, _revalidate_sources

from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)

MECHANISM = "diagnostic_staged_tile_preparation"
_ISSUED = WeakKeyDictionary()


def _methods(owner, names):
    result = []
    for name in names:
        callback = getattr(owner, name, None)
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise ValueError("diagnostic preparation needs original bound methods")
        result.append((name, callback.__func__, callback.__func__.__code__))
    return tuple(result)


def _source_metadata(registry):
    """Fresh original source checks around this one readonly metadata snapshot."""
    with scoped_registry_metadata_verification(registry):
        capability = registry._capabilities[0]
        return capability.owner, capability.mechanism, capability.record()


class DiagnosticTilePreparationOwner:
    def __init__(self, source_capability, *, source_registry=None):
        with shared_compute_pool(timeout_s=120):
            self._initialize(source_capability, source_registry)

    def _initialize(self, source_capability, source_registry):
        if type(source_capability) is not ReviewedPassCapability:
            raise ValueError(
                "diagnostic preparation needs the actual original source capability"
            )
        if source_registry is None:
            source_registry = ReviewedPassCapabilities((source_capability,))
        if (
            type(source_registry) is not ReviewedPassCapabilities
            or len(source_registry._capabilities) != 1
            or source_registry._capabilities[0] is not source_capability
        ):
            raise ValueError(
                "diagnostic preparation needs its exact original source registry"
            )
        owner, mechanism, source_record = _source_metadata(source_registry)
        original_type = PublicSourceDiagnosticIssuer
        if type(owner) is not original_type:
            if type(owner).__module__ != "gemmini_public_source_observation":
                raise ValueError(
                    "diagnostic preparation needs an exact original source issuer/prototype"
                )
            from gemmini_public_source_observation import PublicSourceObservationIssuer

            original_type = PublicSourceObservationIssuer
        if (
            mechanism != "source_execution_diagnostic"
            or type(owner) is not original_type
            or type(owner._prototype) is not StagedDensePrototype
        ):
            raise ValueError(
                "diagnostic preparation needs an exact original source issuer/prototype"
            )
        self.source_capability = source_capability
        self._source_registry = source_registry
        self.tile = StagedTilePrototype(owner._prototype)
        self._binding = copy.deepcopy(source_record["binding"])
        self._source_record = source_record
        self._tile_record = self.tile.record()
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_diagnostic_tile_preparation",
                type(owner).__module__,
                "gemmini_staged_tile_capability",
                "gemmini_staged_dense_capability",
                "merlin.llvmlower.reviewed_pass_capabilities",
                "merlin.llvmlower.reviewed_metadata",
                "merlin.targetgen.compiler_pass_selection",
                "gemmini_timing_resources",
                "merlin.common.paths",
                "merlin.common.cpu_pool",
            )
        }
        self._identity = _digest(
            [self._binding, self._source_record, self._tile_record, self._sources]
        )
        _ISSUED[self] = (
            source_capability,
            self.tile,
            copy.deepcopy(
                (
                    self._binding,
                    self._source_record,
                    self._tile_record,
                    self._sources,
                    self._identity,
                )
            ),
            _methods(
                self,
                (
                    "revalidate",
                    "record",
                    "selected_binding",
                    "prepare_compiler",
                    "check_compiler_binding",
                ),
            ),
            _methods(
                self.tile,
                ("revalidate", "record", "prepare_compiler", "check_compiler_binding"),
            ),
            (_methods, _methods.__code__, _source_metadata, _source_metadata.__code__),
            source_registry,
        )
        try:
            self.revalidate()
        except BaseException:
            _ISSUED.pop(self, None)
            raise

    @property
    def identity(self):
        return "gemmini.diagnostic_staged_tile_preparation.v1:" + self._identity

    def revalidate(self):
        with shared_compute_pool(timeout_s=120):
            issued = _ISSUED.get(self)
            if (
                issued is None
                or (
                    _methods,
                    _methods.__code__,
                    _source_metadata,
                    _source_metadata.__code__,
                )
                != issued[5]
            ):
                raise ValueError(
                    "diagnostic preparation lost its original live issuance"
                )

            def check():
                if (
                    type(self) is not DiagnosticTilePreparationOwner
                    or self.source_capability is not issued[0]
                    or self.tile is not issued[1]
                    or self._source_registry is not issued[6]
                    or self._source_registry._capabilities != (self.source_capability,)
                    or (
                        self._binding,
                        self._source_record,
                        self._tile_record,
                        self._sources,
                        self._identity,
                    )
                    != issued[2]
                    or _methods(
                        self,
                        (
                            "revalidate",
                            "record",
                            "selected_binding",
                            "prepare_compiler",
                            "check_compiler_binding",
                        ),
                    )
                    != issued[3]
                    or _methods(
                        self.tile,
                        (
                            "revalidate",
                            "record",
                            "prepare_compiler",
                            "check_compiler_binding",
                        ),
                    )
                    != issued[4]
                ):
                    raise ValueError(
                        "diagnostic preparation lost its original owner/code/source binding"
                    )
                _revalidate_sources(self._sources)

            check()
            with scoped_registry_metadata_verification(self._source_registry):
                _, mechanism, source_record = _source_metadata(self._source_registry)
                if (
                    mechanism != "source_execution_diagnostic"
                    or source_record != self._source_record
                    or source_record["binding"] != self._binding
                    or self.tile.record() != self._tile_record
                    or self.tile.library.contract.sha256 != self._binding["library"]
                ):
                    raise ValueError(
                        "diagnostic preparation original source/library contract changed"
                    )
                check()
            check()

    def selected_binding(self):
        self.revalidate()
        return copy.deepcopy(self._binding)

    def record(self):
        self.revalidate()
        return dict(
            identity=self.identity,
            mechanism=MECHANISM,
            binding=copy.deepcopy(self._binding),
            sources=copy.deepcopy(self._sources),
            source=copy.deepcopy(self._source_record),
            scope="candidate preparation only; every current emitted realization must be independently checked",
            scheduling_qualification="NONE",
            hardware_execution_qualified=False,
            timing_authority="NONE",
            promotion_authority="NONE",
        )

    def prepare_compiler(self, source, destination, *, policy):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            result = self.tile.prepare_compiler(
                Path(source), Path(destination), policy=copy.deepcopy(policy)
            )
            self.revalidate()
            return result

    def check_compiler_binding(self, package):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            result = self.tile.check_compiler_binding(Path(package))
            self.revalidate()
            return result


def with_diagnostic_preparation(registry):
    if type(registry) is not ReviewedPassCapabilities:
        raise ValueError("diagnostic preparation needs the actual original registry")
    # This wrapper transports the narrow original source registry only. Broader
    # capabilities must be selected through their own independent protocol.
    with scoped_registry_metadata_verification(registry):
        sources = tuple(registry._capabilities)
        valid = (
            len(sources) == 1 and sources[0].mechanism == "source_execution_diagnostic"
        )
        binding = sources[0].binding if valid else None
    if not valid:
        raise ValueError(
            "diagnostic preparation needs one source owner and no existing preparation"
        )
    source = sources[0]
    owner = DiagnosticTilePreparationOwner(source, source_registry=registry)
    return ReviewedPassCapabilities(
        (
            *registry._capabilities,
            ReviewedPassCapability(MECHANISM, binding, owner),
        )
    )
