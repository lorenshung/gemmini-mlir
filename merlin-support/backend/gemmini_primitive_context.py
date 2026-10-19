"""Explicit original compiler contexts for the independent primitive diagnostic.

The RHS entry reuses only its exact original seed release binding. It grants no
RHS/zero transformation support and never reads a saved successful qualification.
"""

from __future__ import annotations

from types import MethodType

from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_initializer_capability import (
    QualifiedZeroInitializerOwner,
    ZeroInitializerCompilerPrototype,
)
from merlin.common.paths import module_source_path
from merlin.targetgen.contract.readback_policy import canonical_sha256

_ISSUER = object()


def _original_methods(owner, owner_type=QualifiedZeroInitializerOwner):
    if type(owner) is not owner_type:
        raise ValueError(
            "primitive diagnostic requires its original live compiler-owner type"
        )
    callbacks = {}
    for name in ("revalidate", "check_compiler_binding"):
        callback = getattr(owner, name, None)
        if (
            type(callback) is not MethodType
            or callback.__self__ is not owner
            or callback.__func__ is not getattr(owner_type, name)
        ):
            raise ValueError(
                "primitive diagnostic requires original bound compiler-owner callbacks"
            )
        callbacks[name] = callback.__func__.__code__
    return callbacks


def _selection(qualification, mode):
    if mode == "qualified_zero":
        if type(qualification) is not QualifiedZeroInitializerOwner:
            raise ValueError(
                "primitive diagnostic requires its original live qualified compiler owner"
            )
        seed = qualification.prototype
        owner_type, binding_owner = QualifiedZeroInitializerOwner, qualification
        source_modules = ("gemmini_zero_initializer_capability",)
    elif mode == "qualified_rhs_seed_only":
        from gemmini_staged_rhs_capability import (
            QualifiedRhsResidencyOwner,
            RhsResidencyCompilerPrototype,
        )

        if (
            type(qualification) is not QualifiedRhsResidencyOwner
            or type(qualification.prototype) is not RhsResidencyCompilerPrototype
        ):
            raise ValueError(
                "primitive RHS context needs its exact original live qualified RHS owner and prototype"
            )
        seed = qualification.prototype.seed
        owner_type, binding_owner = QualifiedRhsResidencyOwner, seed
        source_modules = (
            "gemmini_zero_initializer_capability",
            "gemmini_staged_rhs_capability",
        )
    else:
        raise ValueError("unknown original primitive compiler context")
    if type(seed) is not ZeroInitializerCompilerPrototype:
        raise ValueError(
            "primitive context has no exact original seed compiler prototype"
        )
    return seed, owner_type, binding_owner, source_modules


class PrimitiveCompilerContext:
    """A live checked origin; constructor is not a JSON restoration protocol."""

    def __init__(self, qualification, mode, *, _issuer=None):
        if _issuer is not _ISSUER:
            raise ValueError(
                "primitive context can only be selected from an original live compiler owner"
            )
        seed, owner_type, binding_owner, modules = _selection(qualification, mode)
        self.qualification, self.mode = qualification, mode
        self.seed, self._owner_type, self.binding_owner = (
            seed,
            owner_type,
            binding_owner,
        )
        self._binding_type = (
            QualifiedZeroInitializerOwner
            if mode == "qualified_zero"
            else ZeroInitializerCompilerPrototype
        )
        self._callbacks = (
            _original_methods(qualification, owner_type),
            _original_methods(binding_owner, self._binding_type),
        )
        qualification.revalidate()
        self._origin = qualification.identity
        self._seed = seed.identity
        self._sources = {
            name: _pin(module_source_path(name)) for name in (__name__, *modules)
        }
        self._identity = canonical_sha256(
            [mode, self._origin, self._seed, self._sources]
        )
        self.revalidate()

    @classmethod
    def from_zero(cls, qualification):
        if cls is not PrimitiveCompilerContext:
            raise ValueError(
                "primitive context subclasses cannot issue an original selection"
            )
        return cls(qualification, "qualified_zero", _issuer=_ISSUER)

    @classmethod
    def from_rhs(cls, qualification):
        if cls is not PrimitiveCompilerContext:
            raise ValueError(
                "primitive context subclasses cannot issue an original selection"
            )
        return cls(qualification, "qualified_rhs_seed_only", _issuer=_ISSUER)

    @property
    def identity(self):
        return "gemmini.primitive_compiler_context.v1:" + self._identity

    def revalidate(self):
        seed, owner_type, binding, _modules = _selection(self.qualification, self.mode)
        if (
            seed is not self.seed
            or owner_type is not self._owner_type
            or binding is not self.binding_owner
            or (
                _original_methods(self.qualification, owner_type),
                _original_methods(binding, self._binding_type),
            )
            != self._callbacks
        ):
            raise ValueError(
                "primitive original compiler context or bound callbacks changed"
            )
        self.qualification.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.qualification.identity != self._origin
            or seed.identity != self._seed
            or canonical_sha256([self.mode, self._origin, self._seed, self._sources])
            != self._identity
        ):
            raise ValueError(
                "primitive original compiler qualification/seed/source binding changed"
            )

    def check_binding(self, package):
        self.revalidate()
        if (package / "staged-rhs-window-public.json").exists():
            raise ValueError(
                "primitive seed-only diagnostic does not admit an RHS compiler/ledger"
            )
        checked = self.binding_owner.check_compiler_binding(package)
        self.revalidate()
        return checked

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "selection": self.mode,
            "original_qualification": self._origin,
            "original_seed": self._seed,
            "scope": "original selected base release binding only; current primitive checked independently",
            "transformation_support_issued": False,
            "timing_authority": "NONE",
        }
