"""Actual immutable-emitter C plus independent source/address controls."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from gemmini_staged_dense_source import project_statements
from gemmini_staged_rhs_placement import assigned_window, optimize_statements, render_statements

from merlin.compile.scheduling.staged_dense import (
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    order_output_tiles_n_before_m,
    rewrite_staged_dense,
)
from merlin.compile.scheduling.staged_rhs_residency import place_rhs_stripe


@pytest.fixture
def source():
    package = Path(__file__).resolve().parents[2] / "phase1_generated/el4_fire_aligned_v23_20261018/package"
    spec = importlib.util.spec_from_file_location("original_rhs_placement_fixture", package / "direct_codegen.py")
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    contract = StagedDenseContract(
        "facts",
        "effects",
        "i8raw32",
        "neutral-fixture-only",
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
    return direct, contract


def program(source, shape):
    direct, contract = source
    declarations, statements = direct.emit_direct_matmul("lhs", "rhs", "out", *shape)
    projection = dict(
        declarations=declarations,
        lhs="lhs",
        rhs="rhs",
        output="out",
        problem=StagedDenseProblem(*shape),
        contract=contract,
    )
    effects, positions = project_statements(statements, **projection, with_positions=True)
    return statements, effects, positions, projection


def rewrites():
    return (
        StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m),
        StagedDenseRewrite("full_k_rhs", place_rhs_stripe),
        StagedDenseRewrite("rhs_omissions", eliminate_redundant_rhs_loads),
    )


@pytest.mark.parametrize("shape", [(17, 17, 17), (31, 31, 31), (32, 48, 32), (33, 33, 47), (1, 33, 17)])
def test_actual_C_renders_joint_padded_stripe_and_only_proven_omissions(source, shape):
    statements, effects, _, projection = program(source, shape)
    contract, proof = assigned_window(
        effects, problem=projection["problem"], contract=projection["contract"], original_owned_rows=128
    )
    assert proof["full_k_stripe_fit"] and proof["hardware_execution_qualified"] is False
    after, report = optimize_statements(
        statements,
        declarations=projection["declarations"],
        lhs="lhs",
        rhs="rhs",
        output="out",
        shape=shape,
        contract=contract,
        rewrites=rewrites(),
    )
    m, k, n = shape
    assert report["checks"][-1]["products"] == m * k * n
    assert report["checks"][-1]["outputs"] == m * n
    expected = ((m - 1) // 16) * ((n + 15) // 16) * ((k + 15) // 16)
    assert len(statements) - len(after) == expected
    assert after.count("gm_fence();") == statements.count("gm_fence();")
    projected = project_statements(after, **{**projection, "contract": contract})
    assert all(op.local_row == 16 + op.source_row for op in projected if type(op) is OperandLoad and op.role == "rhs")


@pytest.mark.parametrize("owned,fit", [(63, False), (64, True), (65, True)])
def test_private_joint_capacity_knee_has_original_single_panel_fallback(source, owned, fit):
    statements, effects, _, projection = program(source, (17, 33, 17))
    contract, proof = assigned_window(
        effects, problem=projection["problem"], contract=projection["contract"], original_owned_rows=owned
    )
    assert proof["required_operand_rows"] == 64 and proof["full_k_stripe_fit"] is fit
    after, _ = optimize_statements(
        statements,
        declarations=projection["declarations"],
        lhs="lhs",
        rhs="rhs",
        output="out",
        shape=(17, 33, 17),
        contract=contract,
        rewrites=rewrites(),
    )
    assert len(after) < len(statements) if fit else len(after) == len(statements)


@pytest.mark.parametrize("damage", ["extra_load", "different_legal_placement", "noncanonical_owner", "pending"])
def test_independent_renderer_refuses_new_transfers_or_other_physical_placements(source, damage):
    statements, effects, positions, projection = program(source, (17, 17, 17))
    contract, _ = assigned_window(
        effects, problem=projection["problem"], contract=projection["contract"], original_owned_rows=64
    )
    after, _ = rewrite_staged_dense(effects, problem=projection["problem"], contract=contract, rewrites=rewrites())
    if damage == "extra_load":
        index = next(i for i, op in enumerate(after) if type(op) is OperandLoad and op.role == "rhs")
        after = (*after[:index], after[index], *after[index:])
    elif damage == "different_legal_placement":
        # This shifted stripe can remain arithmetically valid; it is outside
        # the independently admitted canonical physical renderer's scope.
        after = tuple(
            replace(op, local_row=op.local_row + 1)
            if type(op) is OperandLoad and op.role == "rhs"
            else replace(op, rhs_row=op.rhs_row + 1)
            if type(op) is DescriptorIssue
            else op
            for op in after
        )
    elif damage == "noncanonical_owner":
        damaged = tuple(
            replace(op, local_row=1) if type(op) is OperandLoad and op.role == "lhs" else op for op in effects
        )
        with pytest.raises(ValueError):
            assigned_window(
                damaged, problem=projection["problem"], contract=projection["contract"], original_owned_rows=64
            )
        return
    else:
        after = after[:-2]
    with pytest.raises(ValueError):
        render_statements(
            after,
            original_effects=effects,
            positions=positions,
            original_statements=statements,
            **{**projection, "contract": contract},
        )


def test_unknown_source_calls_and_nonidentity_readout_never_enter_projection(source):
    statements, _, _, projection = program(source, (17, 17, 17))
    for changed in (
        [*statements, "unreviewed_scratch_writer();"],
        [text.replace("gm_st_config(68,1.0f,0)", "gm_st_config(68,2.0f,0)") for text in statements],
    ):
        with pytest.raises(ValueError):
            project_statements(changed, **projection)
