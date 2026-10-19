"""Live structural bridge between original source and native timing namespaces.

The source composite software identity and raw observed-contract identity remain
DISTINCT. This joins their original owners; it grants no timing or numerical
qualification and never constructs an owner from a saved receipt.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import weakref
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from gemmini_count_hardware import CountHardwareCorrespondenceOwner
from gemmini_public_source_diagnostic import PublicSourceDiagnosticIssuer
from gemmini_staged_dense_admission import StagedCompletionOwner
from gemmini_staged_dense_capability import StagedDensePrototype
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapability,
)

MECHANISM = "compiler_source_native_correspondence"
_ISSUED = weakref.WeakKeyDictionary()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "namespace correspondence source must be original regular bytes"
        )
    return str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()


def _bound(owner, name, function):
    method = getattr(owner, name, None)
    if (
        type(method) is not MethodType
        or method.__self__ is not owner
        or method.__func__ is not function
    ):
        raise ValueError("namespace correspondence lost original bound callback")
    return method


def _check(
    source_binding,
    native_binding,
    source_inputs,
    count_inputs,
    completion,
    contract,
    library,
):
    """Pure comparison only; this function cannot issue a live owner."""
    expected = PassCapabilityBinding(
        contract.hardware,
        contract.software,
        contract.numerical_policy,
        contract.execution,
        library,
    )
    if (
        source_binding != expected
        or source_inputs != count_inputs
        or completion["selected_inputs"] != source_inputs
    ):
        raise ValueError("original source/native selected identities differ")
    if (
        source_binding.hardware != source_inputs["facts"]["sha256"]
        or native_binding.hardware != source_inputs["facts"]["sha256"]
        or native_binding.software != source_inputs["observed_contract"]["sha256"]
        or source_binding.software != _digest([source_inputs, completion["proof"]])
    ):
        raise ValueError(
            "composite software identity is not the original selected raw-contract proof"
        )


class SourceNativeCorrespondenceOwner:
    def __init__(self):
        raise TypeError("issue only from original live source and count capabilities")

    @classmethod
    def bind(cls, source_capability, count_capability):
        if (
            cls is not SourceNativeCorrespondenceOwner
            or type(source_capability) is not ReviewedPassCapability
            or type(count_capability) is not ReviewedPassCapability
        ):
            raise TypeError("namespace bridge needs original reviewed capability types")
        source_capability.revalidate()
        count_capability.revalidate()
        if (
            source_capability.mechanism != "source_execution_diagnostic"
            or count_capability.mechanism != "timing_model_correspondence"
        ):
            raise ValueError("namespace bridge selected another capability mechanism")
        source, count = source_capability.owner, count_capability.owner
        if (
            type(source) is not PublicSourceDiagnosticIssuer
            or type(count) is not CountHardwareCorrespondenceOwner
            or type(source._prototype) is not StagedDensePrototype
            or type(source._prototype.completion) is not StagedCompletionOwner
        ):
            raise TypeError("namespace bridge needs original source/count/proof owners")
        callbacks = tuple(
            (
                owner,
                name,
                getattr(type(owner), name),
                getattr(type(owner), name).__code__,
            )
            for owner, names in (
                (source, ("revalidate",)),
                (count, ("revalidate", "selected_binding")),
                (source._prototype, ("revalidate",)),
                (source._prototype.completion, ("record", "revalidate")),
            )
            for name in names
        )
        files = tuple(
            dict.fromkeys(
                _pin(inspect.getsourcefile(function)) for _, _, function, _ in callbacks
            )
        ) + (_pin(__file__),)
        value = object.__new__(cls)
        for name, function, code in _ORIGINAL_API:
            if getattr(cls, name) is not function or function.__code__ is not code:
                raise ValueError("original namespace correspondence callback changed")
        _ISSUED[value] = (
            source_capability,
            count_capability,
            source,
            count,
            source._prototype,
            source._prototype.completion,
            source_capability.binding,
            count_capability.binding,
            source_capability.identity,
            count_capability.identity,
            callbacks,
            files,
        )
        try:
            value.revalidate()
        except BaseException:
            del _ISSUED[value]
            raise
        return value

    @property
    def identity(self):
        state = _ISSUED.get(self)
        if type(self) is not SourceNativeCorrespondenceOwner or state is None:
            raise ValueError("namespace correspondence was not originally issued")
        return "gemmini.source_native_correspondence.v1:" + _digest(
            [state[8], state[9], asdict(state[6]), asdict(state[7]), state[11]]
        )

    def revalidate(self):
        state = _ISSUED.get(self)
        if type(self) is not SourceNativeCorrespondenceOwner or state is None:
            raise ValueError("namespace correspondence was not originally issued")
        (
            source_cap,
            count_cap,
            source,
            count,
            prototype,
            completion,
            source_binding,
            native_binding,
            source_id,
            count_id,
            callbacks,
            files,
        ) = state
        for name, function, code in _ORIGINAL_API:
            if function.__code__ is not code:
                raise ValueError("original namespace callback code changed")
            _bound(self, name, function)
        for owner, name, function, code in callbacks:
            if function.__code__ is not code:
                raise ValueError("original namespace callback code changed")
            _bound(owner, name, function)
        if tuple(_pin(p) for p, _ in files) != files:
            raise ValueError("namespace correspondence original source bytes changed")
        source_cap.revalidate()
        count_cap.revalidate()
        if (
            source_cap.owner is not source
            or count_cap.owner is not count
            or source._prototype is not prototype
            or prototype.completion is not completion
            or source_cap.binding != source_binding
            or count_cap.binding != native_binding
            or source_cap.identity != source_id
            or count_cap.identity != count_id
        ):
            raise ValueError(
                "namespace correspondence original owner or binding changed"
            )
        prototype.revalidate()
        completion.revalidate()
        selected = prototype.selected.report()["selected_inputs"]
        proof = completion.record()
        if completion.selected is not prototype.selected:
            raise ValueError("completion belongs to another original target selection")
        if PassCapabilityBinding(**count.selected_binding()) != native_binding:
            raise ValueError("native binding differs from original count owner")
        _check(
            source_binding,
            native_binding,
            selected,
            count.correspondence.revalidate()["selected_inputs"],
            proof,
            prototype.contract,
            prototype.library.contract.sha256,
        )

    def verify_compiler_binding(self, compiler_binding, *, native_binding):
        self.revalidate()
        state = _ISSUED[self]
        if (
            type(compiler_binding) is not PassCapabilityBinding
            or type(native_binding) is not PassCapabilityBinding
            or compiler_binding != state[6]
            or native_binding != state[7]
        ):
            raise ValueError(
                "actual compiler/native policy selected another complete binding"
            )
        self.revalidate()

    def selected_binding(self):
        self.revalidate()
        return asdict(_ISSUED[self][7])

    def record(self):
        self.revalidate()
        state = _ISSUED[self]
        return {
            "identity": self.identity,
            "source_capability_identity": state[8],
            "count_capability_identity": state[9],
            "compiler_binding": asdict(state[6]),
            "native_binding": asdict(state[7]),
            "scope": "structural original namespace correspondence only",
            "software_digests_equated": False,
            "timing_qualification": False,
            "numerical_qualification": False,
            "promotion_authority": "NONE",
        }


_ORIGINAL_API = tuple(
    (
        name,
        getattr(SourceNativeCorrespondenceOwner, name),
        getattr(SourceNativeCorrespondenceOwner, name).__code__,
    )
    for name in ("revalidate", "verify_compiler_binding", "selected_binding", "record")
)
