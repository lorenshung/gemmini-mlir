"""Actual V23 source/C/ABI grouping; no native or timing qualification."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from gemmini_zero_initializer_preparation import (
    prepare_compiler,
    rewrite_mixed_codegen,
    rewrite_provenance,
)

from merlin.common.cpu_pool import shared_compute_pool
from merlin.compile.scheduling import zero_initializer_groups
from merlin.targetgen.oot_starterkit.plan import validate_mixed_program_plan

PACKAGE = Path(__file__).resolve().parents[2] / "phase1_generated/el4_fire_aligned_v23_20261018/package"


def _source(m, k, n, *, returned_init=False):
    result = f"tensor<{m}x{n}xi32>"
    returns = f"({result}, {result})" if returned_init else result
    outputs = f"%r, %init : {result}, {result}" if returned_init else f"%r : {result}"
    return f"""module {{func.func @forward(%a:tensor<{m}x{k}xi8>, %b:tensor<{k}x{n}xi8>) -> {returns} {{
      %zero = arith.constant 0 : i32
      %init = "tensor.splat"(%zero) : (i32) -> {result}
      %r = linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
        affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
        iterator_types=["parallel","parallel","reduction"]}}
        ins(%a,%b:tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%init:{result}) {{
      ^bb0(%x:i8,%y:i8,%s:i32):
        %xx=arith.extsi %x:i8 to i32
        %yy=arith.extsi %y:i8 to i32
        %p=arith.muli %xx,%yy:i32
        %v=arith.addi %s,%p:i32
        linalg.yield %v:i32
      }} -> {result}
      return {outputs}
    }}}}"""


def _emit(tmp_path, text, *, enabled):
    package = tmp_path / ("candidate" if enabled else "baseline")
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if enabled:
        path = package / "mixed_codegen.py"
        path.write_text(rewrite_mixed_codegen(path.read_text()))
        (package / "passes").mkdir()
        (package / "passes/__init__.py").write_text("")
        (package / "zero_initializer_policy.py").write_text(
            "def choose_zero_initializer_groups(module):\n"
            "    from passes.zero_initializer_groups import find_zero_initializer_groups\n"
            "    return find_zero_initializer_groups(module)\n"
        )
        shutil.copyfile(
            zero_initializer_groups.__file__,
            package / "passes/zero_initializer_groups.py",
        )
        (package / "staged-tile-public.json").write_text(
            json.dumps(
                {
                    "schema": "gemmini.staged_tile_public.v1",
                    "contract": {
                        "readout": "raw_accumulator_identity",
                        "operand_bits": 8,
                        "accumulator_bits": 32,
                        "external_buffers_disjoint": True,
                        "fence_completion": "data_commands_preserve_pending_descriptor",
                    },
                }
            )
        )
    source = package / "source.mlir"
    source.write_text(text)
    script = """import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from mixed_codegen import emit_mixed
p=Path(sys.argv[1]); cb,c,functions=emit_mixed((p/'source.mlir').read_text())
(p/'result.json').write_text(json.dumps(dict(cb=cb,c=c,functions=functions)))
"""
    with shared_compute_pool(timeout_s=60):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(package)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((package / "result.json").read_bytes())


@pytest.mark.parametrize("shape", [(2, 3, 5), (17, 16, 32), (17, 17, 17)])
def test_actual_generated_program_groups_complete_source_and_omits_only_private_initializer(tmp_path, shape):
    text = _source(*shape)
    baseline, candidate = (_emit(tmp_path, text, enabled=flag) for flag in (False, True))
    assert validate_mixed_program_plan(text, baseline["cb"])["ok"] is True
    assert validate_mixed_program_plan(text, candidate["cb"])["ok"] is True
    old = baseline["cb"]["params"]["global_program_plan"]
    new = candidate["cb"]["params"]["global_program_plan"]
    assert old["source_sha256"] == new["source_sha256"] and old["source_op_count"] == new["source_op_count"] == 3
    assert [row["source_op_indices"] for row in new["tasks"]] == [[0, 1, 2]]
    assert [row["task_index"] for row in new["tasks"]] == [0]
    assert [row["tensor"] for row in candidate["cb"]["kernel_abi"]["args"]] == [
        "arg0",
        "arg1",
        "Y0",
    ]
    assert new["entry_bindings"] == old["entry_bindings"] and new["output_bindings"] == old["output_bindings"]
    assert candidate["cb"]["commands"] == baseline["cb"]["commands"]
    assert f"for(long i=0;i<{shape[0] * shape[2]}" in baseline["c"]
    assert f"for(long i=0;i<{shape[0] * shape[2]}" not in candidate["c"]


def test_returned_initializer_keeps_complete_original_program_and_output_roster(
    tmp_path,
):
    text = _source(3, 5, 7, returned_init=True)
    baseline, candidate = (_emit(tmp_path, text, enabled=flag) for flag in (False, True))
    assert candidate == baseline
    assert candidate["cb"]["kernel_abi"]["outputs"] == ["Y0", "Y1"]


@pytest.mark.parametrize(
    ("alter", "refusal"),
    [
        (
            lambda text: text.replace("arith.constant 0 : i32", "arith.constant 7 : i32"),
            "requires proved zero source initializer",
        ),
        (
            lambda text: text.replace(
                "linalg.generic {indexing_maps=", "linalg.generic {unreviewed_effect=1:i32,indexing_maps="
            ),
            None,
        ),
    ],
    ids=["nonzero-initializer", "unknown-semantic-attribute"],
)
def test_unproved_initializer_semantics_keep_original_materialization(tmp_path, alter, refusal):
    # This only checks that the proposal cannot change the original lowering.
    # It grants no numerical correctness to unsupported original source modes.
    text = alter(_source(3, 5, 7))
    if refusal is not None:
        for enabled in (False, True):
            with pytest.raises(AssertionError, match=refusal):
                _emit(tmp_path, text, enabled=enabled)
        return
    baseline, candidate = (_emit(tmp_path, text, enabled=flag) for flag in (False, True))
    assert candidate == baseline
    assert "for(long i=0;i<21" in candidate["c"]


def test_provenance_keeps_task_identity_distinct_from_source_group_identity():
    original = (PACKAGE / "provenance.py").read_text()
    changed = rewrite_provenance(original)
    compile(changed, "<owned-provenance-proposal>", "exec")
    assert "IntegerAttr(indices[0],64)" in changed
    assert "merlin.source_op_indices" in changed
    with pytest.raises(ValueError, match="exact original generated compiler boundary"):
        rewrite_provenance(changed)


def test_foreign_or_already_edited_generated_source_refuses():
    original = (PACKAGE / "mixed_codegen.py").read_text()
    with pytest.raises(ValueError, match="exact original generated compiler boundary"):
        rewrite_mixed_codegen(original.replace("    _, fn = read_module(text)", "    module, fn = read_module(text)"))
    with pytest.raises(ValueError, match="exact original generated compiler boundary"):
        rewrite_mixed_codegen(rewrite_mixed_codegen(original))


def test_preparation_preserves_original_and_same_selected_library_in_both_arms(tmp_path):
    from merlin.targetgen.compiler_pass_selection import _tree

    source = tmp_path / "source"
    shutil.copytree(PACKAGE, source)
    selections = {
        "selected-shared-library.json": {"fixture": "same original selection"},
        "shared-library-sources.json": {"fixture": "same exact source leaves"},
        "staged-tile-public.json": {"fixture": "no qualification from this test"},
    }
    for name, value in selections.items():
        (source / name).write_text(json.dumps(value))
    original = _tree(source)
    records = []
    for arm, enabled in (("baseline", False), ("candidate", True)):
        records.append(
            prepare_compiler(
                source,
                tmp_path / arm,
                enabled=enabled,
                generic_source=Path(zero_initializer_groups.__file__).resolve(),
            )
        )
        assert _tree(source) == original
        for name in selections:
            assert (tmp_path / arm / name).read_bytes() == (source / name).read_bytes()
        assert (tmp_path / arm / "passes/zero_initializer_groups.py").read_bytes() == Path(
            zero_initializer_groups.__file__
        ).read_bytes()
        assert records[-1]["hardware_execution_qualified"] is False
        assert records[-1]["timing_authority"] == "NONE"
    assert records[0]["selected_library"] == records[1]["selected_library"]
    assert records[0]["prepared"]["sha256"] != records[1]["prepared"]["sha256"]
    with pytest.raises(ValueError, match="explicit Boolean"):
        prepare_compiler(source, tmp_path / "bad", enabled=1, generic_source=Path(__file__))
    with pytest.raises(ValueError, match="fresh pass component"):
        prepare_compiler(
            tmp_path / "candidate",
            tmp_path / "twice",
            enabled=True,
            generic_source=Path(zero_initializer_groups.__file__).resolve(),
        )
