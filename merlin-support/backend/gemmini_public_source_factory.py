"""Source-pinned factory transport preserving the canonical issuer class.

Private factory restoration loads this wrapper in its own module namespace.
The actual issuer remains in its original canonical implementation module.
"""

from pathlib import Path

from gemmini_public_source_diagnostic import (
    restore_registry as _canonical_restore_registry,
)
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return _canonical_restore_registry(inputs, work_root=work_root)


def original_factory(inputs, *, sources=()):
    """Select the wrapper and pin its canonical callback's source closure."""
    canonical = Path(module_source_path("gemmini_public_source_diagnostic"))
    closure = tuple(dict.fromkeys((canonical, *map(Path, sources))))
    return OriginalPassFactory(restore_registry, inputs, sources=closure)
