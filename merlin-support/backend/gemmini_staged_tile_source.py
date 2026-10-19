"""Render only complete original C tile permutations and checked RHS omissions.

The fixed source projection and final issued-program/source/output check remain
independent of candidate-owned traversal and rewrite ordering decisions.
"""

from gemmini_staged_dense_source import project_statements
from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    OperandLoad,
    StagedDenseProblem,
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    order_output_tiles_n_before_m,
    rewrite_staged_dense,
)


def _blocks(effects):
    cursor = 0
    while cursor < len(effects) and type(effects[cursor]) is CompletionFence:
        cursor += 1
    prefix, blocks = effects[:cursor], {}
    while cursor < len(effects) and type(effects[cursor]) is not CompletionFence:
        start = cursor
        if type(effects[cursor]) is not AccumulatorZero:
            raise ValueError(
                "physical tile renderer requires a complete original zeroed lifetime"
            )
        cursor += 1
        while cursor < len(effects) and type(effects[cursor]) is not AccumulatorReadout:
            if type(effects[cursor]) is AccumulatorZero:
                raise ValueError(
                    "physical tile renderer cannot split overlapping accumulator lifetimes"
                )
            cursor += 1
        if (
            cursor + 1 >= len(effects)
            or type(effects[cursor + 1]) is not CompletionFence
        ):
            raise ValueError(
                "physical tile renderer requires complete readout and fence"
            )
        readout = effects[cursor]
        key = (
            readout.output_row,
            readout.output_col,
            readout.rows,
            readout.cols,
            readout.accumulator_row,
        )
        cursor += 2
        if key in blocks:
            raise ValueError("physical tile renderer cannot duplicate output tiles")
        blocks[key] = (start, cursor)
    if any(type(op) is not CompletionFence for op in effects[cursor:]):
        raise ValueError("physical tile renderer cannot detach inter-tile effects")
    return prefix, blocks, effects[cursor:]


def render_statements(
    effects, *, original_effects, positions, original_statements, **projection
):
    prefix, blocks, suffix = _blocks(original_effects)
    after_prefix, after_blocks, after_suffix = _blocks(effects)
    if (
        not blocks
        or prefix != after_prefix
        or suffix != after_suffix
        or set(blocks) != set(after_blocks)
    ):
        raise ValueError(
            "physical tile renderer changed original complete output roster or boundary effects"
        )
    chunks, previous_end = {}, None
    for key, (start, end) in blocks.items():
        cstart, cend = positions[start] - 1, positions[end - 1] + 1
        if cstart < 0 or original_statements[cstart] != "gm_ld_config(0,0,1.0f,0);":
            raise ValueError(
                "physical tile renderer lost original zero configuration boundary"
            )
        if previous_end is not None and cstart != previous_end:
            raise ValueError(
                "physical tile renderer cannot detach configuration or other C effects"
            )
        chunks[key] = (cstart, cend)
        previous_end = cend
    first = next(iter(chunks.values()))[0]
    result = list(original_statements[:first])
    for key, (start, end) in after_blocks.items():
        before_start, before_end = blocks[key]
        proposal = effects[start:end]
        cursor, omitted = 0, set()
        for index in range(before_start, before_end):
            before = original_effects[index]
            if cursor < len(proposal) and proposal[cursor] == before:
                cursor += 1
            elif type(before) is OperandLoad and before.role == "rhs":
                omitted.add(positions[index])
            else:
                raise ValueError(
                    "physical tile renderer permits only original whole tiles and RHS omissions"
                )
        if cursor != len(proposal):
            raise ValueError(
                "physical tile renderer cannot synthesize new commands or placements"
            )
        cstart, cend = chunks[key]
        result.extend(
            text
            for index, text in enumerate(original_statements[cstart:cend], start=cstart)
            if index not in omitted
        )
    result.extend(original_statements[previous_end:])
    if project_statements(result, **projection) != effects:
        raise ValueError(
            "physical emitted C does not reconstruct the complete checked tile proposal"
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
        raise ValueError(
            "tile traversal enable must be an actual Boolean policy choice"
        )
    problem = StagedDenseProblem(*shape)
    projection = {
        "declarations": declarations,
        "lhs": lhs,
        "rhs": rhs,
        "output": output,
        "problem": problem,
        "contract": contract,
    }
    actual, positions = project_statements(
        statements, **projection, with_positions=True
    )
    selected = (
        ()
        if not enabled
        else (
            (
                StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m),
                StagedDenseRewrite("rhs_residency", eliminate_redundant_rhs_loads),
            )
            if rewrites is None
            else rewrites
        )
    )
    effects, report = rewrite_staged_dense(
        actual, problem=problem, contract=contract, rewrites=selected
    )
    if not enabled:
        return statements, report
    rendered = render_statements(
        effects,
        original_effects=actual,
        positions=positions,
        original_statements=statements,
        **projection,
    )
    return rendered, report
