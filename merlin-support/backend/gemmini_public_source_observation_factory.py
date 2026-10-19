"""Separate private factory transport for explicit complete-commit diagnostics."""

from pathlib import Path

from gemmini_public_source_observation import (
    restore_registry as _canonical_restore_registry,
)
from gemmini_source_observation_runtime import IMPLEMENTATION_PINS
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return _canonical_restore_registry(inputs, work_root=work_root)


def original_factory(inputs, *, sources=()):
    canonical = (
        Path(module_source_path(name)).resolve()
        for name in (
            "gemmini_public_source_observation",
            "gemmini_source_observation_runtime",
        )
    )
    implementations = (Path(pin["path"]) for pin in IMPLEMENTATION_PINS.values())
    closure = tuple(dict.fromkeys((*canonical, *implementations, *map(Path, sources))))
    return OriginalPassFactory(restore_registry, inputs, sources=closure)
