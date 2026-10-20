"""Data-only actual-command relation; live source authority is a separate caller.

Rows come from the original complete commit decoder, with linked-symbol pointer
normalization and exact command/fence transport bits. This helper cannot issue
a source owner, transform capability, numerical verdict or timing prediction.
"""

from __future__ import annotations


def _signature(row):
    event, command = row
    if (
        type(event) is not tuple
        or len(event) != 3
        or type(command) is not dict
        or command.get("class") != event[0]
        or type(command.get("transport")) is not int
        or not 0 <= command["transport"] < 1 << 32
        or type(command.get("decoded")) is not dict
    ):
        raise ValueError("actual pair has incomplete original decoded transport")
    if event[0] == "FENCE":
        if event != ("FENCE", 0, 0) or command["decoded"]:
            raise ValueError("actual pair has incomplete fence transport")
    else:
        if (
            any(
                type(value) is not int or not 0 <= value < 1 << 64
                for value in (
                    event[2],
                    command.get("rs1", {}).get("raw"),
                    command.get("rs2", {}).get("raw"),
                )
            )
            or command["rs2"]["raw"] != event[2]
            or type(event[1]) is int
            and command["rs1"]["raw"] != event[1]
            or type(command.get("funct")) is not int
        ):
            raise ValueError("actual pair has mismatched original command operands")
        if event[0] == "CONFIG_LD" and (
            type(command["decoded"].get("state_id")) is not int
            or type(command["decoded"].get("stride")) is not int
        ):
            raise ValueError("actual pair lacks original typed load-state fields")
    return event, command["transport"], command.get("funct"), command["decoded"]


def _partition(stream, ports):
    starts = []
    for index, (event, command) in enumerate(stream):
        _signature((event, command))
        if event[0] in ports and command["decoded"].get("is_acc_addr") is True:
            if index == 0 or stream[index - 1][0][0] != "FENCE":
                raise ValueError("actual V26 zero tile lost its original leading fence")
            starts.append(index - 1)
    if not starts:
        raise ValueError("actual V26 pair has no independently initialized tile")
    blocks, cursor = {}, starts[0]
    for start in starts:
        if cursor != start:
            raise ValueError("actual V26 pair detaches inter-tile commands or fences")
        cursor = start + 2
        while cursor < len(stream) and stream[cursor][0][0] != "MVOUT":
            event, command = stream[cursor]
            if (
                event[0] in {"CONFIG_EX", "FLUSH"}
                or event[0] in ports
                and command["decoded"].get("is_acc_addr") is True
            ):
                raise ValueError(
                    "actual V26 tile crosses a drain or accumulator lifetime"
                )
            cursor += 1
        if cursor + 1 >= len(stream) or stream[cursor + 1][0][0] != "FENCE":
            raise ValueError("actual V26 tile lost its complete readout/fence group")
        key = stream[cursor][0]
        if key in blocks:
            raise ValueError("actual V26 pair duplicates a complete output tile")
        cursor += 2
        blocks[key] = stream[start:cursor]
    if any(row[0][0] != "FENCE" for row in stream[cursor:]):
        raise ValueError("actual V26 pair has detached trailing device effects")
    return stream[: starts[0]], blocks, stream[cursor:]


def compare_decoded_pair(baseline, candidate, *, ports, rhs, operand_rows, dim):
    """Compare complete original tile permutations plus exact live RHS reloads.

    Source products, actual zero bytes, immutable ABI inputs and complete outputs
    MUST be independently proved on both streams by the original live owner.
    No JSON result from this pure helper establishes those premises.
    """
    if (
        type(baseline) is not tuple
        or type(candidate) is not tuple
        or not baseline
        or not candidate
        or max(len(baseline), len(candidate)) > 1_000_000
        or type(ports) is not dict
        or len(ports) != 3
        or any(
            type(k) is not str or type(v) is not int or v < 0 for k, v in ports.items()
        )
        or len(set(ports.values())) != 3
        or type(rhs) is not str
        or not rhs
        or any(type(v) is not int or v <= 0 for v in (operand_rows, dim))
        or dim > 256
    ):
        raise ValueError(
            "actual V26 pair needs explicit bounded original stream/storage bindings"
        )
    allowed = set(ports) | {
        "CONFIG_EX",
        "CONFIG_LD",
        "CONFIG_ST",
        "PRELOAD",
        "COMPUTE_PRELOADED",
        "MVOUT",
        "FLUSH",
        "FENCE",
    }
    if any(row[0][0] not in allowed for row in (*baseline, *candidate)):
        raise ValueError("actual V26 pair contains unsupported device effects")
    bp, bb, bs = _partition(baseline, ports)
    cp, cb, cs = _partition(candidate, ports)
    def signatures(rows):
        return [_signature(row) for row in rows]
    if (
        signatures(bp) != signatures(cp)
        or signatures(bs) != signatures(cs)
        or set(bb) != set(cb)
    ):
        raise ValueError(
            "actual V26 pair changed original prologue/completion/output roster"
        )
    ordered = (*bp, *(row for key in cb for row in bb[key]), *bs)
    cursor, states, cells, removed = 0, {}, {}, 0

    def requested(event, fields):
        state = ports[event[0]]
        pointer = event[1]
        if (
            state not in states
            or type(states[state]) is not int
            or states[state] < 0
            or type(pointer) is not tuple
            or len(pointer) != 2
            or type(pointer[0]) is not str
            or type(pointer[1]) is not int
            or pointer[1] < 0
            or any(type(fields.get(k)) is not int for k in ("addr", "rows", "cols"))
            or not 0 < fields["rows"] <= dim
            or not 0 < fields["cols"] <= dim
            or not 0 <= fields["addr"] < operand_rows
            or fields["addr"] + fields["rows"] > operand_rows
        ):
            raise ValueError(
                "actual V26 reload lacks its own configured bounded source/local cells"
            )
        return {
            (fields["addr"] + r, c): (pointer[0], pointer[1] + r * states[state] + c)
            for r in range(fields["rows"])
            for c in range(fields["cols"])
        }

    for row in ordered:
        event, command = row
        fields, kind = command["decoded"], event[0]
        if cursor < len(candidate) and _signature(row) == _signature(candidate[cursor]):
            cursor += 1
        elif (
            kind in ports
            and fields.get("is_acc_addr") is False
            and type(event[1]) is tuple
            and event[1][0] == rhs
        ):
            wanted = requested(event, fields)
            if any(cells.get(cell) != value for cell, value in wanted.items()):
                raise ValueError(
                    "actual omitted RHS reload lacks live exact immutable source cells"
                )
            removed += 1
            continue
        else:
            raise ValueError(
                "actual V26 commands/fences/settings changed beyond a proved RHS reload"
            )
        if kind in {"FLUSH", "CONFIG_EX"}:
            cells.clear()
        elif kind == "CONFIG_LD":
            state, stride = fields.get("state_id"), fields.get("stride")
            if (
                type(state) is not int
                or state not in ports.values()
                or type(stride) is not int
                or stride < 0
            ):
                raise ValueError(
                    "actual V26 pair lost independent load-state configuration"
                )
            states[state] = stride
        elif kind in ports:
            if type(fields.get("is_acc_addr")) is not bool:
                raise ValueError("actual V26 load has unknown storage ownership")
            if not fields["is_acc_addr"]:
                cells.update(requested(event, fields))
    if cursor != len(candidate):
        raise ValueError(
            "actual V26 candidate inserted commands or detached fence groups"
        )
    return {
        "schema": "gemmini.v26_actual_staged_pair_relation.v1",
        "status": "checked_decoded_relation_requires_original_source_owners",
        "output_tiles": len(bb),
        "output_tile_order_changed": list(bb) != list(cb),
        "actual_rhs_loads_removed": removed,
        "original_leading_reduction_readout_fences_and_configurations_preserved": True,
        "independent_load_states": dict(ports),
        "source_ownership_and_complete_outputs": "required separately on both actual programs",
        "target_capability_issued": False,
        "timing_authority": "NONE",
    }
