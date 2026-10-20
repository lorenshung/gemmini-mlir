"""Private factory transport preserving the original generated-source issuer."""

from pathlib import Path

from gemmini_generated_source_diagnostic import restore_registry as _original_restore
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory


def restore_registry(inputs, *, work_root):
    return _original_restore(inputs, work_root=work_root)


def original_factory(inputs, *, sources=()):
    names = (
        "gemmini_generated_source_diagnostic",
        "gemmini_generated_source_trace",
        "merlin.targetgen.riscv_memory_effects",
    )
    if (
        inputs.get("source_binding_protocol")
        == "gemmini.pointer_source_contractions.v1"
    ):
        names += (
            "gemmini_direct_source_execution",
            "gemmini_source_packet_artifacts",
            "merlin.llvmlower.pointer_source_contractions",
        )
    protocol = inputs.get("source_memory_effects_protocol")
    if protocol is not None:
        from gemmini_whole_program_memory_effects import require_protocol

        require_protocol(protocol)
        if inputs.get("source_binding_protocol", "whole_program") != "whole_program":
            raise ValueError(
                "selected source memory effects require whole_program binding"
            )
        names += (
            "gemmini_whole_program_memory_effects",
            "gemmini_load_streams",
            "gemmini_zero_fill",
            "gemmini_private_zero_storage",
        )
    leaves = tuple(Path(module_source_path(name)) for name in names)
    return OriginalPassFactory(
        restore_registry,
        inputs,
        sources=tuple(dict.fromkeys((*leaves, *map(Path, sources)))),
    )
