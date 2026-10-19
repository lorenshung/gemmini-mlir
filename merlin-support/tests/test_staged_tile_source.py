"""Complete actual C tile permutation, pass order and strict renderer limits."""

import importlib
import importlib.util
from pathlib import Path

import pytest
from merlin.compile.scheduling.staged_dense import (
    OperandLoad,
    StagedDenseContract,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
)
from merlin.compile.scheduling.staged_dense_transforms import (
    order_output_tiles_n_before_m,
)


@pytest.fixture
def source(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "merlin-support/backend"))
    module = importlib.import_module("gemmini_staged_tile_source")
    path = (
        root
        / "phase1_generated/el4_fire_aligned_v23_20261018/package/direct_codegen.py"
    )
    spec = importlib.util.spec_from_file_location("original_tile_fixture", path)
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    contract = StagedDenseContract(
        "facts",
        "effects",
        "signed8raw32",
        "fixture-only",
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
        False,
    )
    return module, direct, contract


@pytest.mark.parametrize("m", [1, 16, 17, 31, 32, 33])
@pytest.mark.parametrize(
    "k,n", [(1, 1), (16, 16), (16, 17), (16, 31), (16, 32), (17, 31), (33, 31)]
)
def test_actual_original_C_tiles_reorder_and_keep_full_products_and_all_fences(
    source, m, k, n
):
    module, direct, contract = source
    declarations, statements = direct.emit_direct_matmul("lhs", "rhs", "out", m, k, n)
    before = list(statements)
    after, report = module.optimize_statements(
        statements,
        declarations=declarations,
        lhs="lhs",
        rhs="rhs",
        output="out",
        shape=(m, k, n),
        contract=contract,
    )
    assert statements == before
    assert report["checks"][-1]["products"] == m * k * n
    assert report["checks"][-1]["outputs"] == m * n
    assert after.count("gm_fence();") == before.count("gm_fence();")
    expected = ((m - 1) // 16) * ((n + 15) // 16) if k <= 16 else 0
    assert len(before) - len(after) == expected


def test_ordered_candidate_rewrites_change_actual_C_programs_under_identical_fixed_checks(
    source,
):
    module, direct, contract = source
    declarations, statements = direct.emit_direct_matmul(
        "lhs", "rhs", "out", 32, 16, 32
    )
    kwargs = {
        "declarations": declarations,
        "lhs": "lhs",
        "rhs": "rhs",
        "output": "out",
        "shape": (32, 16, 32),
        "contract": contract,
    }
    order = StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m)
    residency = StagedDenseRewrite("rhs_residency", eliminate_redundant_rhs_loads)
    left, left_report = module.optimize_statements(
        statements, **kwargs, rewrites=(order, residency)
    )
    right, right_report = module.optimize_statements(
        statements, **kwargs, rewrites=(residency, order)
    )
    assert left != right
    assert len(right) - len(left) == 2
    assert (
        left_report["checks"][-1]["products"]
        == right_report["checks"][-1]["products"]
        == 16384
    )
    assert [text for text in left if not text.startswith("gm_mvin(")] == [
        text for text in right if not text.startswith("gm_mvin(")
    ]


def test_logically_legal_new_transfers_still_need_a_new_independently_checked_renderer(
    source,
):
    module, direct, contract = source
    declarations, statements = direct.emit_direct_matmul(
        "lhs", "rhs", "out", 17, 16, 16
    )

    def add(operations, _problem, _contract):
        index = next(
            i
            for i, op in enumerate(operations)
            if type(op) is OperandLoad and op.role == "rhs"
        )
        return (*operations[:index], operations[index], *operations[index:])

    with pytest.raises(ValueError, match="physical tile renderer"):
        module.optimize_statements(
            statements,
            declarations=declarations,
            lhs="lhs",
            rhs="rhs",
            output="out",
            shape=(17, 16, 16),
            contract=contract,
            rewrites=(StagedDenseRewrite("new_transfer", add),),
        )
