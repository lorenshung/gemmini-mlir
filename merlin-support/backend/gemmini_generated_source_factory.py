"""Private factory transport preserving the original generated-source issuer."""

from pathlib import Path

from gemmini_generated_source_diagnostic import restore_registry as _original_restore
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return _original_restore(inputs, work_root=work_root)


def original_factory(inputs, *, sources=()):
    leaves = tuple(
        Path(module_source_path(name))
        for name in (
            "gemmini_generated_source_diagnostic",
            "gemmini_generated_source_trace",
        )
    )
    return OriginalPassFactory(
        restore_registry,
        inputs,
        sources=tuple(dict.fromkeys((*leaves, *map(Path, sources)))),
    )
