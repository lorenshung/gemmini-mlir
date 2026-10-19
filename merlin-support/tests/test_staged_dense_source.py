"""Actual immutable direct-emitter source projections; no target qualification."""

import importlib
import importlib.util
from pathlib import Path

import pytest
from merlin.compile.scheduling.staged_dense import StagedDenseContract


@pytest.fixture
def source(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "merlin-support/backend"))
    module = importlib.import_module("gemmini_staged_dense_source")
    path = (
        root
        / "phase1_generated/el4_fire_aligned_v23_20261018/package/direct_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("immutable_direct_emitter", path)
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    selected = StagedDenseContract(
        "exact-facts-fixture",
        "exact-effects-fixture",
        "unchanged-signed8-raw32",
        "functional-fixture-only",
        16,
        32,
        16,
        8,
        20,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        True,
    )
    return module, direct, selected


@pytest.mark.parametrize("m", [1, 16, 17, 31, 32, 33])
@pytest.mark.parametrize(
    "k,n", [(1, 1), (16, 16), (16, 3), (17, 16), (16, 17), (33, 31)]
)
def test_actual_current_C_program_projection_and_private_rhs_reuse(source, m, k, n):
    module, direct, selected = source
    declarations, statements = direct.emit_direct_matmul(
        "arg0", "arg1", "arg2", m, k, n
    )
    before = list(statements)
    after, report = module.optimize_statements(
        statements,
        declarations=declarations,
        lhs="arg0",
        rhs="arg1",
        output="arg2",
        shape=(m, k, n),
        contract=selected,
    )
    expected = (m - 1) // 16 if k <= 16 and n <= 16 else 0
    assert len(before) - len(after) == expected
    assert statements == before
    assert report["checks"][-1]["products"] == m * k * n
    # Every configuration, fence, compute and store retains exactly its order.
    assert [s for s in before if not s.startswith("gm_mvin(")] == [
        s for s in after if not s.startswith("gm_mvin(")
    ]
    baseline, _ = module.optimize_statements(
        statements,
        declarations=declarations,
        lhs="arg0",
        rhs="arg1",
        output="arg2",
        shape=(m, k, n),
        contract=selected,
        enabled=False,
    )
    assert baseline is statements


@pytest.mark.parametrize(
    "damage",
    ["loop", "mode", "scale", "bias", "alias", "ptr", "unknown", "zero", "stride"],
)
def test_unsupported_current_source_refuses_before_mutation(source, damage):
    module, direct, selected = source
    kwargs = {"bias": "other"} if damage == "bias" else {}
    declarations, statements = direct.emit_direct_matmul(
        "arg0", "arg1", "arg2", 17, 16, 16, **kwargs
    )
    if damage == "loop":
        statements.insert(2, "for(int i=0;i<2;i++) {")
    elif damage == "mode":
        statements[1] = "gm_ex_config(0,0);"
    elif damage == "scale":
        statements = [
            s.replace("gm_st_config(64,1.0f,0)", "gm_st_config(64,2.0f,0)")
            for s in statements
        ]
    elif damage == "ptr":
        statements = [
            s.replace("((size_t)(0)*16+(0))", "((size_t)(f())*16+(0))")
            for s in statements
        ]
    elif damage == "unknown":
        statements.insert(0, "foreign_writer();")
    elif damage == "zero":
        declarations[0] = declarations[0].replace("{0}", "{1}")
    elif damage == "stride":
        statements = [
            s.replace("gm_ld_config(16,0,1.0f,0)", "gm_ld_config(15,0,1.0f,0)")
            for s in statements
        ]
    with pytest.raises(ValueError):
        module.optimize_statements(
            statements,
            declarations=declarations,
            lhs="arg0",
            rhs="arg0" if damage == "alias" else "arg1",
            output="arg2",
            shape=(17, 16, 16),
            contract=selected,
        )
