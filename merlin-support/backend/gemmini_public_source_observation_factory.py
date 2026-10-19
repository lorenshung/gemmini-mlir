"""Separate private factory transport for explicit complete-commit diagnostics."""

from pathlib import Path

from gemmini_public_source_observation import (
    restore_registry as _canonical_restore_registry,
)
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return _canonical_restore_registry(inputs, work_root=work_root)


def original_factory(inputs, *, sources=()):
    canonical = Path(module_source_path("gemmini_public_source_observation"))
    closure = tuple(dict.fromkeys((canonical, *map(Path, sources))))
    return OriginalPassFactory(restore_registry, inputs, sources=closure)
