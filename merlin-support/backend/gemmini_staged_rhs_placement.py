"""Independent finite source projection for a private full-K RHS stripe.

The selected original owner must establish whole-program scratch ownership and
the physical row bound separately. These functions check the current source
effects and renderer; a positive result grants no executable capability.
"""

from __future__ import annotations

from dataclasses import replace

from gemmini_quiescent_load_config import _arguments
from gemmini_staged_dense_source import project_statements
from gemmini_staged_tile_source import render_statements as render_tiles

from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
    rewrite_staged_dense,
)


def _verify(operations, *, problem, contract):
    # Use the existing public checked entry point. Copying this renderer must
    # not grant direct access to a private shared-library verifier module.
    return rewrite_staged_dense(operations, problem=problem, contract=contract)[1]["checks"][0]


def assigned_window(operations, *, problem, contract, original_owned_rows):
    """Prove a joint padded fit within an ORIGINAL assigned private epoch.

    ``original_owned_rows`` must come from independent complete-program/caller
    ownership, not from hardware capacity or a candidate's declaration. Final
    native/source/packet checks remain necessary for every emitted program.
    """
    if (
        type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or type(original_owned_rows) is not int
        or original_owned_rows < 2 * contract.dim
        or contract.operand_rows != 2 * contract.dim
        or contract.accumulator_rows != contract.dim
        or not contract.external_buffers_disjoint
        or not contract.rhs_immutable
        or contract.retained_weights
    ):
        raise ValueError("RHS placement lacks its original canonical private operand/accumulator epoch")
    original = _verify(operations, problem=problem, contract=contract)
    if not operations or type(operations[0]) is not CompletionFence or type(operations[-1]) is not CompletionFence:
        raise ValueError("RHS placement needs a complete original fenced program epoch")
    required = contract.dim + ((problem.k + contract.dim - 1) // contract.dim) * contract.dim
    # Validate all canonical ownership even on the no-fit fallback path.
    relocated_effects(operations, problem=problem, contract=contract)
    fits = required <= original_owned_rows
    assigned = replace(contract, operand_rows=required) if fits else contract
    return assigned, dict(
        status="bounded_private_source_window_assignment",
        required_operand_rows=required,
        assigned_operand_rows=assigned.operand_rows,
        original_owned_rows=original_owned_rows,
        full_k_stripe_fit=fits,
        original_source_effects=original,
        complete_program_ownership="required independently from original caller/source/actual issue",
        hardware_execution_qualified=False,
        timing_authority="NONE",
    )


def relocated_effects(operations, *, problem, contract):
    """Reconstruct the only allowed canonical row relocation independently.

    A descriptor reads its source cells when the following Compute consumes it.
    A load after descriptor issue or a completion fence may change those cells.
    This check deliberately does not import a candidate placement algorithm.
    """
    cells, pending, result = {}, None, list(operations)
    dim = contract.dim
    for index, operation in enumerate(operations):
        if type(operation) is OperandLoad:
            expected = 0 if operation.role == "lhs" else dim
            if (
                operation.role not in {"lhs", "rhs"}
                or operation.local_row != expected
                or operation.rows > dim
                or operation.cols > dim
            ):
                raise ValueError("RHS relocation found a foreign original private operand owner")
            if operation.role == "rhs":
                if operation.source_row % dim or operation.source_col % dim:
                    raise ValueError("RHS relocation needs complete aligned original panel origins")
                result[index] = replace(operation, local_row=dim + operation.source_row)
            for r in range(operation.rows):
                for c in range(operation.cols):
                    cells[operation.local_row + r, c] = (
                        operation.role,
                        operation.source_row + r,
                        operation.source_col + c,
                    )
        elif type(operation) is DescriptorIssue:
            if pending is not None or operation.rhs_row != dim or operation.accumulator_row != 0:
                raise ValueError("RHS relocation found foreign or overlapping pending descriptor ownership")
            pending = index, operation
        elif type(operation) is ComputeConsume:
            if pending is None or operation.lhs_row != 0 or operation.retain_weights:
                raise ValueError("RHS relocation needs original fresh-weight compute consumption")
            descriptor_index, descriptor = pending
            first = cells.get((dim, 0))
            if first is None or first[0] != "rhs" or first[1] % dim:
                raise ValueError("RHS relocation lacks the actual descriptor source cells at consumption")
            k0, n0 = first[1:]
            if any(
                cells.get((dim + r, c)) != ("rhs", k0 + r, n0 + c)
                for r in range(descriptor.k)
                for c in range(descriptor.n)
            ):
                raise ValueError("RHS relocation descriptor reads clobbered or mixed original source cells")
            result[descriptor_index] = replace(descriptor, rhs_row=dim + k0)
            pending = None
        elif type(operation) in (AccumulatorZero, AccumulatorReadout):
            row = operation.local_row if type(operation) is AccumulatorZero else operation.accumulator_row
            if row != 0:
                raise ValueError("RHS relocation changed the original sole accumulator lifetime")
        elif type(operation) is not CompletionFence:
            raise ValueError("RHS relocation cannot cross an unknown effect or scratch owner")
    if pending is not None:
        raise ValueError("RHS relocation leaves a pending descriptor outside its complete epoch")
    return tuple(result)


def render_statements(effects, *, original_effects, positions, original_statements, **projection):
    """Render only canonical row relocation, complete tiles and live omissions."""
    problem, contract = projection["problem"], projection["contract"]
    _verify(original_effects, problem=problem, contract=contract)
    _verify(effects, problem=problem, contract=contract)
    if not any(
        (type(op) is OperandLoad and op.role == "rhs" and op.local_row > contract.dim)
        or (type(op) is DescriptorIssue and op.rhs_row is not None and op.rhs_row > contract.dim)
        for op in effects
    ):
        return render_tiles(
            effects,
            original_effects=original_effects,
            positions=positions,
            original_statements=original_statements,
            **projection,
        )
    expected = relocated_effects(original_effects, problem=problem, contract=contract)
    statements = list(original_statements)
    for before, after, position in zip(original_effects, expected, positions, strict=True):
        if before == after:
            continue
        if type(before) is OperandLoad:
            args = _arguments(statements[position], "gm_mvin")
            if args is None or len(args) != 4 or args[1] != str(before.local_row):
                raise ValueError("RHS renderer lost the original physical DMA source boundary")
            statements[position] = f"gm_mvin({args[0]},{after.local_row},{args[2]},{args[3]});"
        elif type(before) is DescriptorIssue:
            args = _arguments(statements[position], "gm_cmd")
            if args is None or len(args) != 3 or args[0] != "6":
                raise ValueError("RHS renderer lost the original physical PRELOAD boundary")
            statements[position] = f"gm_cmd(6,gm_shape({after.rhs_row},{after.k},{after.n}),{args[2]});"
        else:
            raise ValueError("RHS renderer proposed a new instruction or non-RHS physical placement")
    if project_statements(statements, **projection) != expected:
        raise ValueError("RHS physical relocation does not reconstruct its complete source effects")
    return render_tiles(
        effects, original_effects=expected, positions=positions, original_statements=statements, **projection
    )


def optimize_statements(statements, *, declarations, lhs, rhs, output, shape, contract, rewrites=()):
    """Execute candidate-owned proposals under unchanged independent checks."""
    problem = StagedDenseProblem(*shape)
    projection = dict(declarations=declarations, lhs=lhs, rhs=rhs, output=output, problem=problem, contract=contract)
    original, positions = project_statements(statements, **projection, with_positions=True)
    effects, report = rewrite_staged_dense(original, problem=problem, contract=contract, rewrites=rewrites)
    rendered = render_statements(
        effects, original_effects=original, positions=positions, original_statements=statements, **projection
    )
    return rendered, report
