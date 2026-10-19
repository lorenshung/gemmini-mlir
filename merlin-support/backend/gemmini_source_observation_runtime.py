"""Explicit source export for V2 beside an immutable V1 target namespace.

No canonical module is rebound. The complete existing owner and exact issuer
method body come from the two selected implementation files beside this leaf.
All target/native helpers retain their original normal import resolution.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import gemmini_source_execution_issuer as _original_issuer
from gemmini_timing_resources import _pin, _revalidate_sources

_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_PINS = {
    name: _pin(_ROOT / (name + ".py"))
    for name in ("gemmini_source_execution", "gemmini_source_execution_issuer")
}

# A unique module object owns the class. It is never installed under a canonical
# name or substituted into another original module's globals.
_spec = importlib.util.spec_from_file_location(
    "gemmini_source_observation_runtime_explicit_owner_v2",
    IMPLEMENTATION_PINS["gemmini_source_execution"]["path"],
)
if _spec is None or _spec.loader is None:
    raise ValueError("explicit V2 source implementation is unavailable")
_owner_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_owner_module)
COMMIT_ONLY_PROTOCOL = _owner_module.COMMIT_ONLY_PROTOCOL
source_trace_flags = _owner_module.source_trace_flags


class SourceExecutionObservationOwner(_owner_module.SourceExecutionOwner):
    """Same independent checker, with its actual exported sources in custody."""

    def __init__(self, **kwargs):
        if (
            type(self) is not SourceExecutionObservationOwner
            or kwargs.get("observation_protocol") != COMMIT_ONLY_PROTOCOL
        ):
            raise ValueError("explicit observation owner requires exact V2 selection")
        _revalidate_sources(IMPLEMENTATION_PINS)
        super().__init__(**kwargs)
        self._pins.update(
            {"explicit_" + name: dict(pin) for name, pin in IMPLEMENTATION_PINS.items()}
        )
        self._pins["explicit_source_observation_export"] = _pin(__file__)
        self._identity = self._digest()
        self.revalidate()


_path = Path(IMPLEMENTATION_PINS["gemmini_source_execution_issuer"]["path"])
_tree = ast.parse(_path.read_bytes(), filename=str(_path))
_classes = [
    n
    for n in _tree.body
    if isinstance(n, ast.ClassDef) and n.name == "SourceExecutionIssuer"
]
_methods = (
    [
        n
        for n in _classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "create_source_execution_owner"
    ]
    if len(_classes) == 1
    else []
)
if len(_methods) != 1 or _methods[0].decorator_list:
    raise ValueError("explicit V2 source issuer has no unique original algorithm")
_globals = dict(vars(_original_issuer))
_globals.update(
    __name__=__name__,
    __file__=str(_path),
    FULL_LOG_PROTOCOL=_owner_module.FULL_LOG_PROTOCOL,
    source_trace_flags=source_trace_flags,
    SourceExecutionOwner=SourceExecutionObservationOwner,
)
_export = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        _methods[0],
    ],
    type_ignores=[],
)
ast.fix_missing_locations(_export)
# Both files are original host-selected, digest-pinned implementation sources;
# no candidate text, saved verdict or arbitrary AST enters this fixed export.
exec(compile(_export, str(_path), "exec"), _globals)  # noqa: S102
create_source_execution_owner = _globals["create_source_execution_owner"]
_revalidate_sources(IMPLEMENTATION_PINS)

_EXPORT_SELF_PIN = _pin(__file__)
_EXPORT_BINDINGS = tuple(
    (name, _globals[name], getattr(_globals[name], "__code__", None))
    for name in (
        "SourceExecutionOwner",
        "source_trace_flags",
        "_native_artifacts",
        "_limits",
    )
)
_EXPORT_CLASS_INIT = SourceExecutionObservationOwner.__init__
_EXPORT_CLASS_CODE = _EXPORT_CLASS_INIT.__code__
_EXPORT_TRACE = create_source_execution_owner
_EXPORT_TRACE_CODE = create_source_execution_owner.__code__


def revalidate_export():
    _revalidate_sources(IMPLEMENTATION_PINS)
    if (
        _pin(__file__) != _EXPORT_SELF_PIN
        or SourceExecutionObservationOwner.__init__ is not _EXPORT_CLASS_INIT
        or _EXPORT_CLASS_INIT.__code__ is not _EXPORT_CLASS_CODE
        or create_source_execution_owner is not _EXPORT_TRACE
        or _EXPORT_TRACE.__code__ is not _EXPORT_TRACE_CODE
    ):
        raise ValueError("original explicit source implementation export changed")
    for name, value, code in _EXPORT_BINDINGS:
        if _globals[name] is not value or getattr(value, "__code__", None) is not code:
            raise ValueError("original explicit source implementation binding changed")
