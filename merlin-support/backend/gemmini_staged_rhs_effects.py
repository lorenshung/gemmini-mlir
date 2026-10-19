"""Independent actual issued-cell proof under original RHS immutability.

This proves only the command transformation. The original caller must separately
establish source/trace custody, RHS immutable bytes and full numerical outputs.
Neither normalized event tuples nor this result issue a capability.
"""


def check_rhs_omissions(baseline, candidate, *, rhs, operand_rows, dim):
    cursor, stride, cells, removed = 0, None, {}, 0

    def requested(event, fields):
        if (
            stride is None
            or type(stride) is not int
            or stride < 0
            or any(type(fields.get(key)) is not int for key in ("addr", "rows", "cols"))
            or not 0 <= fields["addr"] < operand_rows
            or not 0 < fields["rows"] <= dim
            or not 0 < fields["cols"] <= dim
            or fields["addr"] + fields["rows"] > operand_rows
        ):
            raise ValueError(
                "actual staged DMA exceeds its assigned private window or load configuration"
            )
        return {
            (fields["addr"] + r, c): (event[1][0], event[1][1] + r * stride + c)
            for r in range(fields["rows"])
            for c in range(fields["cols"])
        }

    for event, command in baseline:
        kind = event[0]
        fields = {} if command is None else command["decoded"]
        matching = cursor < len(candidate) and candidate[cursor][0] == event
        if matching:
            cursor += 1
        elif (
            kind == "MVIN" and fields.get("is_acc_addr") is False and event[1][0] == rhs
        ):
            wanted = requested(event, fields)
            if not wanted or any(
                cells.get(cell) != value for cell, value in wanted.items()
            ):
                raise ValueError(
                    "actual omitted RHS transfer lacks live exact immutable source cells"
                )
            removed += 1
            continue
        else:
            raise ValueError(
                "staged final instructions changed beyond a proven immutable RHS reload"
            )
        if kind in {"FLUSH", "CONFIG_EX"}:
            cells.clear()
        elif kind == "CONFIG_LD":
            stride = fields["stride"]
        elif kind == "MVIN" and fields.get("is_acc_addr") is False:
            cells.update(requested(event, fields))
    if cursor != len(candidate):
        raise ValueError(
            "staged candidate inserted or changed final commands, fences or settings"
        )
    return {
        "actual_rhs_loads_removed": removed,
        "all_other_commands_fences_and_settings_unchanged": True,
        "rhs_immutability": "required from original independent source/write proof",
        "timing_authority": "NONE",
    }
