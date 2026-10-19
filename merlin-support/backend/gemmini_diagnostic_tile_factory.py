"""Source-pinned diagnostic source/preparation factory; no saved owner loader."""

from pathlib import Path

from gemmini_diagnostic_tile_preparation import with_diagnostic_preparation

from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def _source_module(inputs):
    if "source_observation_protocol" not in inputs:
        import gemmini_public_source_diagnostic as source
    else:
        if inputs["source_observation_protocol"] != "gemmini.source_observation.v2":
            raise ValueError("unknown explicit diagnostic source observation protocol")
        import gemmini_public_source_observation as source
    return source


def restore_registry(inputs, *, work_root):
    # The canonical original issuer remains in its original module namespace.
    registry = _source_module(inputs).restore_registry(inputs, work_root=work_root)
    with shared_compute_pool(timeout_s=120):
        return with_diagnostic_preparation(registry)


def original_factory(inputs, *, sources=()):
    source = _source_module(inputs)
    closure = tuple(
        dict.fromkeys(
            (
                Path(source.__file__).resolve(),
                *(
                    Path(module_source_path(name))
                    for name in (
                        "gemmini_diagnostic_tile_preparation",
                        "gemmini_staged_dense_admission",
                        "gemmini_public_source_diagnostic",
                        "gemmini_source_execution",
                        "gemmini_source_execution_issuer",
                    )
                ),
                *map(Path, sources),
            )
        )
    )
    return OriginalPassFactory(restore_registry, inputs, sources=closure)
