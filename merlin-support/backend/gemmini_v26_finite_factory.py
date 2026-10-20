"""Explicit generated-source plus finite preparation factory; default unchanged."""

from pathlib import Path

from gemmini_generated_source_diagnostic import restore_registry as _source_registry
from gemmini_v26_finite_preparation import with_finite_preparation
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return with_finite_preparation(_source_registry(inputs, work_root=work_root))


def original_factory(inputs, *, sources=()):
    modules = (
        "gemmini_v26_finite_preparation",
        "gemmini_v26_finite_hook",
        "gemmini_v26_finite_pair",
        "gemmini_v26_staged_source",
        "gemmini_v26_staged_actual_pair",
        "gemmini_v26_staged_pair_effects",
        "gemmini_generated_source_diagnostic",
        "gemmini_generated_source_factory",
        "gemmini_generated_source_trace",
    )
    return OriginalPassFactory(
        restore_registry,
        inputs,
        sources=tuple(
            dict.fromkeys(
                (
                    *[Path(module_source_path(name)) for name in modules],
                    *map(Path, sources),
                )
            )
        ),
    )
