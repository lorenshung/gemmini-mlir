"""Explicit current primitive proof with separately checked historical ancestry.

This opt-in never scopes metadata reuse over current native/source work. The
actual ancestor is fully revalidated before/after the proof; each current check
freshly reopens the original base compiler, selected sources/runtime and facts.
"""

from __future__ import annotations

import time
import weakref
from types import MethodType

from gemmini_primitive_context import PrimitiveCompilerContext
from gemmini_primitive_diagnostic import PrimitiveDiagnosticOwner
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.contract.readback_policy import canonical_sha256

_ISSUER = object()
_ACTIVE = weakref.WeakSet()
_SOURCE = _pin(__file__)
_ORIGINAL_CHECK = PrimitiveDiagnosticOwner.check_pair
_ORIGINAL_REVALIDATE = PrimitiveDiagnosticOwner.revalidate
_OWNER_CODES = (_ORIGINAL_CHECK.__code__, _ORIGINAL_REVALIDATE.__code__)
_CONTEXT_METHODS = {
    name: (
        getattr(PrimitiveCompilerContext, name),
        getattr(PrimitiveCompilerContext, name).__code__,
    )
    for name in (
        "revalidate_current_seed",
        "check_current_binding",
        "current_seed_record",
        "_check_current_structure",
    )
}


def _owner(owner):
    if _pin(__file__) != _SOURCE:
        raise ValueError("original current primitive proof source changed")
    callback = getattr(owner, "revalidate", None)
    if (
        type(owner) is not PrimitiveDiagnosticOwner
        or type(callback) is not MethodType
        or callback.__self__ is not owner
        or callback.__func__ is not _ORIGINAL_REVALIDATE
        or type(owner.context) is not PrimitiveCompilerContext
        or owner.context.mode != "qualified_rhs_seed_only"
        or (_ORIGINAL_CHECK.__code__, _ORIGINAL_REVALIDATE.__code__) != _OWNER_CODES
        or PrimitiveDiagnosticOwner.check_pair is not _ORIGINAL_CHECK
    ):
        raise ValueError(
            "current primitive proof requires its exact original live owner"
        )
    for name, (original, code) in _CONTEXT_METHODS.items():
        callback = getattr(owner.context, name)
        if (
            type(callback) is not MethodType
            or callback.__self__ is not owner.context
            or callback.__func__ is not original
            or original.__code__ is not code
        ):
            raise ValueError(
                "primitive current seed checker method/source was replaced"
            )


class _CurrentProofView:
    def __init__(self, owner, *, issuer=None):
        if type(self) is not _CurrentProofView or issuer is not _ISSUER:
            raise ValueError(
                "primitive current proof view cannot be restored from data"
            )
        _owner(owner)
        self.original, self.context = owner, owner.context
        self._original_identity = owner.identity
        self._identity = canonical_sha256([owner.identity, _SOURCE])
        self.selected, self.arithmetic = owner.selected, owner.arithmetic
        self.cleanup, self.nm = owner.cleanup, owner.nm
        _ACTIVE.add(self)
        try:
            self.revalidate()
        except BaseException:
            _ACTIVE.discard(self)
            raise

    @property
    def identity(self):
        return "gemmini.current_primitive_proof.v1:" + self._identity

    def revalidate(self):
        _owner(self.original)
        if (
            type(self) is not _CurrentProofView
            or self not in _ACTIVE
            or self.original.identity != self._original_identity
            or canonical_sha256([self.original.identity, _SOURCE]) != self._identity
            or self.original.context is not self.context
            or self.selected is not self.original.selected
            or self.arithmetic is not self.original.arithmetic
            or self.cleanup is not self.original.cleanup
            or self.nm != self.original.nm
        ):
            raise ValueError("primitive current proof selection changed or expired")
        with shared_compute_pool(timeout_s=120):
            _revalidate_sources(self.original._sources)
            self.context.revalidate_current_seed()
        if (
            self.selected is not self.context.seed.tile.selected
            or self.arithmetic is not self.context.seed.tile.arithmetic
            or self.cleanup is not self.context.seed.cleanup
            or self.nm != self.context.seed.nm
        ):
            raise ValueError("primitive current proof selected another base release")

    def check_package_binding(self, package):
        self.revalidate()
        with shared_compute_pool(timeout_s=120):
            result = self.context.check_current_binding(package)
        self.revalidate()
        return result

    def context_record(self):
        self.revalidate()
        with shared_compute_pool(timeout_s=120):
            return self.context.current_seed_record()


def select_from_rhs(qualification):
    """Bounded metadata construction only; full original custody at each edge."""
    from gemmini_staged_rhs_capability import scoped_original_rhs_metadata

    with (
        shared_compute_pool(timeout_s=180),
        scoped_original_rhs_metadata(qualification),
    ):
        return PrimitiveDiagnosticOwner.from_rhs_qualification(qualification)


def _historical(owner):
    queued = time.monotonic()
    with shared_compute_pool(timeout_s=180):
        admitted = time.monotonic()
        owner.revalidate()
    return {
        "queue_wall_seconds": admitted - queued,
        "active_wall_seconds": time.monotonic() - admitted,
    }


def check_current_pair(owner, **arguments):
    """Fresh current proof; ancestor custody is checked pre/post even on failure."""
    _owner(owner)
    from gemmini_staged_rhs_capability import _METADATA

    if _METADATA.get() is not None:
        raise ValueError(
            "current primitive proof must run outside historical metadata reuse"
        )
    ancestry_pre = _historical(owner)
    view, primary, result = None, None, None
    try:
        view = _CurrentProofView(owner, issuer=_ISSUER)
        result = _ORIGINAL_CHECK(view, **arguments)
        result["historical_ancestry"] = {
            "status": "original_live_owner_fully_revalidated_before_and_after_current_proof",
            "identity": owner.context._origin,
            "scope": "base-release ancestry only; no historical execution is a fresh current proof",
            "current_artifact_checks": "fresh original seed/source/runtime/binding plus current complete ELF/input/trace/products/outputs",
        }
        result["costs"]["historical_full_ancestry_pre"] = ancestry_pre
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            ancestry_post = _historical(owner)
            if result is not None:
                result["costs"]["historical_full_ancestry_post"] = ancestry_post
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                "Original primitive final ancestry custody failed: " + str(error)
            )
        finally:
            if view is not None:
                _ACTIVE.discard(view)
