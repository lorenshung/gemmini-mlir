"""Selected generated compiler protocol for original traversal qualification."""

from merlin.targetgen.compiler_emission_binding import check_compiler_emission


def check_original_emission(owner, arm):
    return check_compiler_emission(
        owner,
        arm,
        argument_templates=(
            "--convert-iface-to-gemmini",
            "--emit-command-buffer={output_json}",
            "--emit-target-artifact",
            "{input_mlir}",
        ),
        required_sources=(
            "compiler.py",
            "direct_codegen.py",
            "selected-shared-library.json",
            "shared-library-sources.json",
            "_selected_shared_sources.py",
            "gemmini_staged_tile_hook.py",
            "gemmini_staged_tile_source.py",
            "staged_dense_policy.py",
        ),
    )
