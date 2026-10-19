"""Narrow actual C-effect projection for the preserved finite direct emitter.

No source string or selected plan qualifies execution. The original owner must
check actual final packets, original source products and complete outputs.
Unknown calls, loops, bias, nonidentity readouts and external aliasing refuse.
"""

from __future__ import annotations

import re

from gemmini_quiescent_load_config import _arguments
from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    rewrite_staged_dense,
)


def _integer(text):
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise ValueError("staged source requires literal bounded unsigned geometry")
    return int(text)


def _shape(text):
    parts = _arguments(text + ";", "gm_shape")
    if parts is None or len(parts) != 3:
        raise ValueError("staged source has an unknown actual shape expression")
    return parts[0], _integer(parts[1]), _integer(parts[2])


def _address(text, prefix):
    if text == prefix:
        return 0
    match = re.fullmatch(r"\(" + re.escape(prefix) + r"\|([0-9]+)\)", text)
    if match is None:
        raise ValueError("staged source has unknown accumulator flags or physical row")
    return _integer(match[1])


def _pointer(text, base, dtype, stride):
    match = re.fullmatch(
        r"\(\("
        + dtype
        + r"\*\)\("
        + re.escape(base)
        + r"\)\)\+\(\(size_t\)\(([0-9]+)\)\*"
        + str(stride)
        + r"\+\(([0-9]+)\)\)",
        text,
    )
    if match is None:
        raise ValueError(
            "staged source pointer is not the complete current logical row/column expression"
        )
    return _integer(match[1]), _integer(match[2])


def _zero(declarations, dim):
    matches = [
        re.fullmatch(
            r"static const int32_t ([A-Za-z_][A-Za-z0-9_]*)\["
            + str(dim)
            + r"\] __attribute__\(\(aligned\(64\)\)\) = \{0\};",
            text.strip(),
        )
        for text in declarations
    ]
    if len(matches) != 1 or matches[0] is None:
        raise ValueError("staged source needs the original exact immutable zero row")
    return matches[0][1]


def project_statements(
    statements,
    *,
    declarations,
    lhs,
    rhs,
    output,
    problem,
    contract,
    with_positions=False,
):
    if (
        type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or contract.operand_bits != 8
        or contract.accumulator_bits != 32
    ):
        raise ValueError(
            "staged source requires current typed source and selected contracts"
        )
    if len({lhs, rhs, output}) != 3:
        raise ValueError(
            "staged source does not establish distinct external pointer ownership"
        )
    zero = _zero(declarations, contract.dim)
    effects, positions, load_stride, store_stride, execution = [], [], None, None, False

    def emit(effect):
        effects.append(effect)
        positions.append(position)

    for position, statement in enumerate(statements):
        if statement == "gm_fence();":
            emit(CompletionFence())
            continue
        if statement == "gm_ex_config(1,0);":
            if execution or any(type(op) is not CompletionFence for op in effects):
                raise ValueError(
                    "staged source cannot move or repeat an execution configuration"
                )
            execution = True
            continue
        args = _arguments(statement, "gm_ld_config")
        if args is not None:
            if len(args) != 4 or args[1:] != ["0", "1.0f", "0"]:
                raise ValueError(
                    "staged source load configuration is not identity state zero"
                )
            load_stride = _integer(args[0])
            continue
        args = _arguments(statement, "gm_st_config")
        if args is not None:
            if args != [str(problem.n * 4), "1.0f", "0"]:
                raise ValueError("staged source readout is not exact raw i32 identity")
            store_stride = problem.n * 4
            continue
        args = _arguments(statement, "gm_mvin")
        if args is not None:
            if len(args) != 4 or not execution:
                raise ValueError(
                    "staged source DMA lacks execution state or complete fields"
                )
            rows, cols = map(_integer, args[2:])
            if args[0] == zero and args[1] == "GM_ACC" and load_stride == 0:
                emit(AccumulatorZero(0, rows, cols))
            else:
                roles = []
                for role, base, stride in (
                    ("lhs", lhs, problem.k),
                    ("rhs", rhs, problem.n),
                ):
                    try:
                        row, col = _pointer(args[0], base, "const int8_t", stride)
                    except ValueError:
                        continue
                    if load_stride == stride:
                        roles.append(
                            OperandLoad(role, row, col, _integer(args[1]), rows, cols)
                        )
                if len(roles) != 1:
                    raise ValueError(
                        "staged source DMA has ambiguous input ownership or mismatched row stride"
                    )
                emit(roles[0])
            continue
        args = _arguments(statement, "gm_cmd")
        if args is not None:
            if len(args) != 3 or not execution:
                raise ValueError("staged source execution has incomplete fields")
            first, rows, cols = _shape(args[1])
            second, out_rows, out_cols = _shape(args[2])
            if args[0] == "6":
                if cols != out_cols:
                    raise ValueError(
                        "staged source descriptor input/output columns disagree"
                    )
                continuation = second.startswith(("GM_ACC_ADD", "(GM_ACC_ADD|"))
                emit(
                    DescriptorIssue(
                        None if first == "GM_GARBAGE" else _integer(first),
                        _address(second, "GM_ACC_ADD" if continuation else "GM_ACC"),
                        out_rows,
                        rows,
                        cols,
                        continuation,
                    )
                )
            elif (
                args[0] in {"4", "5"}
                and second == "GM_GARBAGE"
                and (out_rows, out_cols) == (contract.dim, contract.dim)
            ):
                emit(ComputeConsume(_integer(first), rows, cols, args[0] == "5"))
            else:
                raise ValueError(
                    "staged source compute mode or zero incoming partial sum is unsupported"
                )
            continue
        args = _arguments(statement, "gm_mvout")
        if args is not None:
            if len(args) != 4 or store_stride != problem.n * 4:
                raise ValueError(
                    "staged source output lacks exact current readout configuration"
                )
            row, col = _pointer(args[0], output, "int32_t", problem.n)
            emit(
                AccumulatorReadout(
                    _address(args[1], "GM_ACC_FULL"), row, col, *map(_integer, args[2:])
                )
            )
            continue
        raise ValueError(
            "staged source contains unknown control, arithmetic, calls or effects"
        )
    if not execution:
        raise ValueError("staged source lacks the selected identity WS execution mode")
    return (tuple(effects), tuple(positions)) if with_positions else tuple(effects)


def render_statements(
    effects,
    *,
    original_statements,
    original_effects,
    positions,
    declarations,
    lhs,
    rhs,
    output,
    problem,
    contract,
):
    """Preserve every configuration/fence and render only qualified RHS omissions.

    General new placements, instruction changes and added calls need another
    independently qualified physical renderer. A logical proposal alone cannot
    expand this current executable capability.
    """
    cursor, omitted = 0, set()
    for before, position in zip(original_effects, positions, strict=True):
        if cursor < len(effects) and effects[cursor] == before:
            cursor += 1
        elif type(before) is OperandLoad and before.role == "rhs":
            omitted.add(position)
        else:
            raise ValueError(
                "current staged source renderer permits only checked RHS transfer omissions"
            )
    if cursor != len(effects):
        raise ValueError(
            "candidate staged effects exceed the qualified physical renderer"
        )
    result = [text for i, text in enumerate(original_statements) if i not in omitted]
    projected = project_statements(
        result,
        declarations=declarations,
        lhs=lhs,
        rhs=rhs,
        output=output,
        problem=problem,
        contract=contract,
    )
    if projected != effects:
        raise ValueError(
            "actual staged C emission does not reconstruct its complete requested effects"
        )
    return result


def optimize_statements(
    statements,
    *,
    declarations,
    lhs,
    rhs,
    output,
    shape,
    contract,
    enabled=True,
    rewrites=None,
):
    if type(enabled) is not bool:
        raise ValueError("staged source enable must be a Boolean policy choice")
    problem = StagedDenseProblem(*shape)
    actual, positions = project_statements(
        statements,
        declarations=declarations,
        lhs=lhs,
        rhs=rhs,
        output=output,
        problem=problem,
        contract=contract,
        with_positions=True,
    )
    selected = (
        ()
        if not enabled
        else (
            (StagedDenseRewrite("rhs_residency", eliminate_redundant_rhs_loads),)
            if rewrites is None
            else rewrites
        )
    )
    candidate, report = rewrite_staged_dense(
        actual, problem=problem, contract=contract, rewrites=selected
    )
    if not enabled:
        return statements, report
    rendered = render_statements(
        candidate,
        original_statements=statements,
        original_effects=actual,
        positions=positions,
        declarations=declarations,
        lhs=lhs,
        rhs=rhs,
        output=output,
        problem=problem,
        contract=contract,
    )
    return rendered, report
