"""Explicit selected load/zero effects for complete whole-program checking.

This owner reuses independent original RTL/runtime derivations. It grants no
source, host-computation, scheduling, timing or publication equivalence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import weakref
from types import MethodType

from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_load_streams import SelectedLoadStreams
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_fill import SelectedAccumulatorZeroFill
from merlin.common.paths import module_source_path

PROTOCOL = "gemmini.whole_program_memory_effects.v1"
_ISSUED = weakref.WeakKeyDictionary()


def require_protocol(value):
    if value is not None and (type(value) is not str or value != PROTOCOL):
        raise ValueError("unknown whole-program source memory-effects protocol")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class WholeProgramMemoryEffects:
    """Privately issued live bundle; a serialized record cannot replace it."""

    def __init__(self, selected, arithmetic):
        _check_helpers()
        if (
            type(self) is not WholeProgramMemoryEffects
            or type(selected) is not SelectedSchedulingTarget
            or type(arithmetic) is not IntegerArithmeticOwner
            or arithmetic._selected is not selected
        ):
            raise TypeError(
                "whole-program effects require original selected live owners"
            )
        selected.revalidate()
        arithmetic.revalidate()
        selected_inputs = selected.report()["selected_inputs"]
        if arithmetic.record()["selected_inputs"] != selected_inputs:
            raise ValueError(
                "whole-program memory and arithmetic selected different inputs"
            )
        loads = SelectedLoadStreams(selected)
        zeros = SelectedAccumulatorZeroFill(selected, arithmetic)
        sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_whole_program_memory_effects",
                "gemmini_load_streams",
                "gemmini_zero_fill",
                "gemmini_private_zero_storage",
            )
        }
        record = {
            "protocol": PROTOCOL,
            "selected_inputs": copy.deepcopy(selected_inputs),
            "arithmetic_identity": arithmetic.identity,
            "loads": loads.record(),
            "zero_fill": zeros.record(),
            "sources": sources,
        }
        state = {
            "selected": selected,
            "arithmetic": arithmetic,
            "loads": loads,
            "zeros": zeros,
            "record": record,
            "identity": _digest(record),
            "callbacks": tuple(
                (
                    owner,
                    name,
                    getattr(type(owner), name),
                    getattr(type(owner), name).__code__,
                )
                for owner, names in (
                    (selected, ("revalidate", "report")),
                    (arithmetic, ("revalidate", "record")),
                    (loads, ("revalidate", "record")),
                    (zeros, ("revalidate", "record", "accepts")),
                )
                for name in names
            ),
        }
        _ISSUED[self] = state
        try:
            self.revalidate()
        except BaseException:
            del _ISSUED[self]
            raise

    def _state(self):
        _check_helpers()
        state = _ISSUED.get(self)
        if type(self) is not WholeProgramMemoryEffects or state is None:
            raise ValueError("whole-program memory effects were not originally issued")
        for name, function, code in _ORIGINAL_API:
            method = getattr(self, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not self
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("whole-program memory-effects callback changed")
        for owner, name, function, code in state["callbacks"]:
            method = getattr(owner, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not owner
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("original selected memory-effects callback changed")
        if (
            type(state["selected"]) is not SelectedSchedulingTarget
            or type(state["arithmetic"]) is not IntegerArithmeticOwner
            or type(state["loads"]) is not SelectedLoadStreams
            or type(state["zeros"]) is not SelectedAccumulatorZeroFill
            or state["loads"].selected is not state["selected"]
            or state["zeros"].selected is not state["selected"]
            or state["zeros"].arithmetic is not state["arithmetic"]
            or state["arithmetic"]._selected is not state["selected"]
            or _digest(state["record"]) != state["identity"]
        ):
            raise ValueError("whole-program memory-effects ownership changed")
        return state

    @property
    def identity(self):
        _check_helpers()
        return PROTOCOL + ":" + _STATE(self)["identity"]

    def revalidate(self):
        _check_helpers()
        state = _STATE(self)
        _revalidate_sources(state["record"]["sources"])
        for owner in (
            state["selected"],
            state["arithmetic"],
            state["loads"],
            state["zeros"],
        ):
            owner.revalidate()
        if (
            state["selected"].report()["selected_inputs"]
            != state["record"]["selected_inputs"]
            or state["arithmetic"].record()["selected_inputs"]
            != state["record"]["selected_inputs"]
            or state["arithmetic"].identity != state["record"]["arithmetic_identity"]
            or state["loads"].record() != state["record"]["loads"]
            or state["zeros"].record() != state["record"]["zero_fill"]
        ):
            raise ValueError("selected source memory-effects derivation changed")
        _STATE(self)

    def packet_owners(self, selected):
        _check_helpers()
        self.revalidate()
        state = _STATE(self)
        if selected is not state["selected"]:
            raise ValueError("packet checking selected a foreign memory-effects target")
        return state["loads"], state["zeros"]

    def record(self):
        _check_helpers()
        self.revalidate()
        return copy.deepcopy(_STATE(self)["record"])


_STATE = WholeProgramMemoryEffects._state
_ORIGINAL_API = tuple(
    (
        name,
        getattr(WholeProgramMemoryEffects, name),
        getattr(WholeProgramMemoryEffects, name).__code__,
    )
    for name in ("__init__", "_state", "revalidate", "packet_owners", "record")
)


def _check_helpers():
    if any(
        globals()[name] is not function or function.__code__ is not code
        for name, function, code in _HELPER_API
    ):
        raise ValueError("original whole-program memory-effects helper changed")
    if any(
        globals()[name] is not cls
        or cls.__init__ is not function
        or function.__code__ is not code
        for name, cls, function, code in _CONSTRUCTORS
    ):
        raise ValueError("original selected memory-effects constructor changed")


_HELPER_API = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "_STATE",
        "_digest",
        "_pin",
        "_revalidate_sources",
        "require_protocol",
        "_check_helpers",
    )
)
_CONSTRUCTORS = tuple(
    (name, globals()[name], globals()[name].__init__, globals()[name].__init__.__code__)
    for name in ("SelectedLoadStreams", "SelectedAccumulatorZeroFill")
)
