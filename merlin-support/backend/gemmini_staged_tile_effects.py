"""Independent complete-tile permutation proof over actual issued operands.

The caller must separately prove original source/ELF/trace custody, exact zero
initialization, immutable source bytes and all complete numerical outputs.
Neither this comparison nor saved output issues authority.
"""

from gemmini_staged_rhs_effects import check_rhs_omissions


def _key(value):
    return (
        tuple(_key(item) for item in value)
        if isinstance(value, (list, tuple))
        else value
    )


def _partition(stream):
    starts = []
    for index, (event, command) in enumerate(stream):
        fields = {} if command is None else command["decoded"]
        if event[0] == "MVIN" and fields.get("is_acc_addr") is True:
            if (
                index == 0
                or stream[index - 1][0][0] != "CONFIG_LD"
                or stream[index - 1][1]["decoded"].get("stride") != 0
            ):
                raise ValueError(
                    "whole-tile issue comparison needs original zero-stride initialization"
                )
            starts.append(index - 1)
    if not starts:
        raise ValueError(
            "whole-tile issue comparison needs independently initialized output tiles"
        )
    blocks = {}
    cursor = starts[0]
    for start in starts:
        if start != cursor:
            raise ValueError(
                "whole-tile issue comparison cannot detach commands between output tiles"
            )
        cursor = start + 2
        while cursor < len(stream) and stream[cursor][0][0] != "MVOUT":
            if stream[cursor][0][0] in {"CONFIG_EX", "FLUSH"} or (
                stream[cursor][0][0] == "MVIN"
                and stream[cursor][1]["decoded"].get("is_acc_addr") is True
            ):
                raise ValueError(
                    "whole-tile issue comparison cannot cross state drains or accumulator lifetimes"
                )
            cursor += 1
        if cursor + 1 >= len(stream) or stream[cursor + 1][0][0] != "FENCE":
            raise ValueError(
                "whole-tile issue comparison needs complete readout and its original fence"
            )
        key = _key(stream[cursor][0])
        cursor += 2
        if key in blocks:
            raise ValueError(
                "whole-tile issue comparison found duplicate complete output tiles"
            )
        blocks[key] = stream[start:cursor]
    if any(event[0] != "FENCE" for event, _ in stream[cursor:]):
        raise ValueError(
            "whole-tile issue comparison cannot detach trailing accelerator effects"
        )
    return stream[: starts[0]], blocks, stream[cursor:]


def check_output_tile_permutation(baseline, candidate, *, rhs, operand_rows, dim):
    """Accept complete original tile permutations and independently live RHS omissions.

    Configurations and fences move only with their complete original tile.
    Their contents, multiplicity and final state are checked. This grants no
    asynchronous overlap, allocation, cross-operation lifetime or timing model.
    """
    before_prefix, before_blocks, before_suffix = _partition(baseline)
    after_prefix, after_blocks, after_suffix = _partition(candidate)
    if (
        before_prefix != after_prefix
        or before_suffix != after_suffix
        or set(before_blocks) != set(after_blocks)
    ):
        raise ValueError(
            "whole-tile candidate changed original prologue, completion or output roster"
        )
    permuted = [*before_prefix]
    for key in after_blocks:
        permuted.extend(before_blocks[key])
    permuted.extend(before_suffix)
    checked = check_rhs_omissions(
        permuted, candidate, rhs=rhs, operand_rows=operand_rows, dim=dim
    )
    return {
        "status": "checked_complete_output_tile_permutation",
        "output_tiles": len(before_blocks),
        "output_tile_order_changed": list(before_blocks) != list(after_blocks),
        "actual_rhs_loads_removed": checked["actual_rhs_loads_removed"],
        "all_original_tile_commands_and_fences_preserved_except_proven_rhs_reloads": True,
        "source_products_outputs_and_immutable_inputs": "required from original independent execution owner",
        "timing_authority": "NONE",
    }
