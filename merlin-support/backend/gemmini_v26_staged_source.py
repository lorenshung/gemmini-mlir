"""Explicit finite shared-pass connection to the original V26 emitter.

This is an unqualified emission adapter, disabled by default. It loads exact
original Builder/contraction source, preserves their emitted instruction lines,
and uses the selected shared verifier. Actual linked source/packet/output and
capability admission remain separate requirements; no saved report grants them.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path
from types import ModuleType

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
    order_output_tiles_n_before_m,
    rewrite_staged_dense,
)
from merlin.llvmlower.direct_source_contraction import bind_direct_source_contraction
from merlin.llvmlower.late_quant_rne import _tokens

_ORIGINAL = {
    "builder.py": "6ef98c486e55c84bf0b946f3174e74c66d11d2e70ee724c7e28c4937081894b0",
    "contraction.py": "b96d460ff0cdb19731f79104a16c3c2d0d7d6872a81c6826d208ab0d98369d59",
}


def _words(line):
    if type(line) is not str or len(line) > 16384:
        raise ValueError("finite V26 original line exceeds its lexical bound")
    tokens = [token.text for token in _tokens(line)]
    if not tokens or len(tokens) > 512:
        raise ValueError("finite V26 original line has invalid token extent")
    return tokens


def _original(package):
    package = Path(package)
    if not package.is_absolute() or package.resolve() != package:
        raise ValueError("V26 shared emission requires its original canonical package")
    modules, pins = {}, {}
    for name, expected in _ORIGINAL.items():
        path = package / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("V26 original emitter source cannot use symlinks")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("V26 original emitter source changed")
        module = ModuleType("original_v26_" + name[:-3])
        # Exact reviewed immutable original source, never caller-supplied code.
        exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
        modules[name] = module
        pins[name] = {"path": str(path), "sha256": expected}
    return modules, pins


def _single_command(buffer, binding):
    if (
        set(buffer)
        != {"abi_version", "target", "tensors", "commands", "compiler_pointer_abi"}
        or buffer["abi_version"] != "0.1"
        or buffer["target"] != "gemmini"
        or type(buffer["commands"]) is not list
        or len(buffer["commands"]) != 4
    ):
        raise ValueError(
            "V26 staged emission supports only its complete single-dot command buffer"
        )
    pack, dot, _commit, _evict = buffer["commands"]
    try:
        handle, accumulator = pack["operands"]["dst"], dot["operands"]["dst"]
        if any(type(value) is not str or not value for value in (handle, accumulator)):
            raise ValueError("invalid private command identities")
        if (
            len(
                {handle, accumulator, binding["lhs"], binding["rhs"], binding["output"]}
            )
            != 5
        ):
            raise ValueError("private command identities alias original pointers")
        expected = [
            {
                "opcode": "RES_PACK",
                "operands": {"src": binding["rhs"], "dst": handle},
                "attributes": {"layout": "packed_rhs"},
            },
            {
                "opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": binding["lhs"], "rhs": handle, "dst": accumulator},
            },
            {
                "opcode": "COMMIT",
                "operands": {"src": accumulator, "dst": binding["output"]},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": handle}},
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "V26 staged emission has incomplete original command roles"
        ) from error
    if buffer["commands"] != expected:
        raise ValueError(
            "V26 staged emission refuses extra arithmetic, layouts or commands"
        )


class _FiniteBuilder:
    """Record actual original lines; unroll only bounded constant original loops."""

    def __init__(self, builder, limit):
        self.builder, self.limit = builder, limit
        self.events, self.pointers, self.addresses = [], {}, {}
        self.iterations = 0

    def __getattr__(self, name):
        return getattr(self.builder, name)

    def loop(self, start, end, step, callback):
        if (
            any(type(v) is not int for v in (start, end, step))
            or start < 0
            or end < start
            or step <= 0
        ):
            raise ValueError(
                "finite V26 emission refuses dynamic or invalid loop bounds"
            )
        self.iterations += len(range(start, end, step))
        if self.iterations > self.limit:
            raise ValueError(
                "finite V26 emission exceeds its original-loop expansion budget"
            )
        for value in range(start, end, step):
            callback(value)

    def gep(self, pointer, index, ty="i8"):
        if type(index) is not int or pointer not in {"%arg0", "%arg1", "%arg2"}:
            raise ValueError("finite V26 emission refuses unresolved source pointers")
        result = self.builder.gep(pointer, index, ty)
        self.pointers[result] = pointer, index, ty
        return result

    def cast(self, opcode, value, source, destination):
        if (opcode, source, destination) != ("ptrtoint", "ptr", "i64"):
            raise ValueError("finite V26 emission has an unknown pointer conversion")
        result = self.builder.cast(opcode, value, source, destination)
        self.addresses[result] = self.pointers.get(value, (value, 0, "i32"))
        return result

    def rocc(self, funct, rs1, rs2):
        start = len(self.builder.lines)
        self.builder.rocc(funct, rs1, rs2)
        self.events.append((funct, rs1, rs2, start, len(self.builder.lines)))
        if len(self.events) > self.limit * 16:
            raise ValueError("finite V26 emission exceeds its instruction budget")

    def fence(self):
        start = len(self.builder.lines)
        self.builder.fence()
        self.events.append((None, None, None, start, len(self.builder.lines)))


def _project(recorder, problem, dim):
    """Keep physical fence groups separate from canonical logical boundaries."""
    states, blocks, current, effects, positions = {}, [], None, [], []
    zero = recorder.builder.globals
    tokens = _words(zero[0]) if len(zero) == 1 else []
    if (
        len(tokens) != 13
        or not tokens[0].startswith("@scratch_g")
        or not tokens[0][10:].isascii()
        or not tokens[0][10:].isdigit()
        or tokens[1:]
        != [
            "=",
            "internal",
            "global",
            "[",
            "16",
            "x",
            "i32",
            "]",
            "zeroinitializer",
            ",",
            "align",
            "64",
        ]
    ):
        raise ValueError(
            "finite V26 emission lacks its exact original private zero row"
        )
    zero_name = tokens[0]
    execution, pending = False, None

    def append(effect, event):
        if current is None:
            raise ValueError(
                "finite V26 data effect lies outside a complete output tile"
            )
        effects.append(effect)
        positions.append(event[3])

    def shape(value):
        if type(value) is not int or value < 0 or value >= 1 << 64:
            raise ValueError("finite V26 emission has unresolved packed dimensions")
        return value & 0xFFFFFFFF, value >> 48, (value >> 32) & 0xFFFF

    for ordinal, event in enumerate(recorder.events):
        funct, rs1, rs2, _start, end = event
        if funct is None:
            if current is not None:
                append(CompletionFence(), event)
                if current.get("readout") is not None:
                    current.update(
                        end=end, effects=tuple(effects), positions=tuple(positions)
                    )
                    blocks.append(current)
                    current, effects, positions = None, [], []
            continue
        if funct == 0:
            if type(rs1) is not int or type(rs2) is not int:
                raise ValueError("finite V26 configuration operands are not literal")
            subtype = rs1 & 3
            if subtype == 0:
                if (
                    execution
                    or states
                    or blocks
                    or current
                    or (rs1, rs2) != (4 | (1 << 16) | (0x3F800000 << 32), 1 << 48)
                ):
                    raise ValueError(
                        "finite V26 execution mode or configuration placement changed"
                    )
                execution = True
            elif subtype == 1:
                state = (rs1 >> 3) & 3
                expected = (
                    (0x3F800000 << 32) | (dim << 16) | (1 << 8) | (state << 3) | 1
                )
                if (
                    not execution
                    or state not in {0, 1, 2}
                    or state in states
                    or current
                    or blocks
                    or rs1 != expected
                    or rs2 != (problem.k, problem.n, 0)[state]
                ):
                    raise ValueError(
                        "finite V26 identity load states or independent strides changed"
                    )
                states[state] = rs2
            elif (
                subtype != 2
                or current is None
                or (rs1, rs2) != (2, (0x3F800000 << 32) | (problem.n * 4))
            ):
                raise ValueError(
                    "finite V26 raw accumulator readout configuration changed"
                )
            continue
        if not execution or set(states) != {0, 1, 2}:
            raise ValueError(
                "finite V26 data effects lack original complete configuration"
            )
        if funct in {1, 2, 14}:
            address, rows, cols = shape(rs2)
            pointer, index, dtype = recorder.addresses.get(rs1, (None, None, None))
            if funct == 14:
                if (
                    current is not None
                    or (pointer, index, dtype, address)
                    != (zero_name, 0, "i32", 0x80000000)
                    or ordinal == 0
                    or recorder.events[ordinal - 1][0] is not None
                ):
                    raise ValueError(
                        "finite V26 accumulator zero or its original leading fence changed"
                    )
                current = {"start": recorder.events[ordinal - 1][3]}
                append(AccumulatorZero(0, rows, cols), event)
            else:
                role, stride, base, local = (
                    ("lhs", problem.k, "%arg0", 0)
                    if funct == 2
                    else ("rhs", problem.n, "%arg1", dim)
                )
                if (
                    pointer != base
                    or dtype != "i8"
                    or type(index) is not int
                    or index < 0
                    or address != local
                ):
                    raise ValueError(
                        "finite V26 transfer changed exact source or assigned storage"
                    )
                row, col = divmod(index, stride)
                append(OperandLoad(role, row, col, address, rows, cols), event)
        elif funct == 6:
            rhs, k, n = shape(rs1)
            accumulator, m, out_n = shape(rs2)
            if rhs != dim or accumulator != 0xC0000000 or out_n != n:
                raise ValueError(
                    "finite V26 descriptor ownership or continuation changed"
                )
            if pending is not None:
                raise ValueError("finite V26 descriptor overwrites pending ownership")
            pending = m, k, n
            append(DescriptorIssue(rhs, 0, m, k, n, True), event)
        elif funct == 4:
            lhs, m, k = shape(rs1)
            garbage, out_m, out_n = shape(rs2)
            if (
                lhs != 0
                or garbage != 0xFFFFFFFF
                or out_m != m
                or pending != (m, k, out_n)
            ):
                raise ValueError("finite V26 compute mode or source extent changed")
            append(ComputeConsume(lhs, m, k), event)
            pending = None
        elif funct == 3:
            accumulator, rows, cols = shape(rs2)
            pointer, index, dtype = recorder.addresses.get(rs1, (None, None, None))
            if (
                current is None
                or accumulator != 0xA0000000
                or pointer != "%arg2"
                or dtype != "i32"
                or type(index) is not int
                or index < 0
            ):
                raise ValueError("finite V26 complete raw readout changed")
            row, col = divmod(index, problem.n)
            effect = AccumulatorReadout(0, row, col, rows, cols)
            append(effect, event)
            current["readout"] = (row, col)
        else:
            raise ValueError(
                "finite V26 emission contains unsupported original effects"
            )
    if current is not None or pending is not None or not blocks:
        raise ValueError("finite V26 output tile is incomplete")
    return blocks


def _closed_chunk(lines, prologue_definitions):
    definitions = set(prologue_definitions)
    for line in lines:
        tokens = _words(line)
        definition = (
            tokens[0]
            if len(tokens) > 2 and tokens[0].startswith("%") and tokens[1] == "="
            else None
        )
        opcode = tokens[2] if definition is not None else tokens[0]
        if (
            opcode in {"phi", "br", "switch", "indirectbr", "invoke", "callbr"}
            or tokens[-1] == ":"
        ):
            raise ValueError("finite V26 tile contains unproved control flow")
        uses = {token for token in tokens if token.startswith("%")}
        if any(
            not token[1:]
            or not token[1:].isascii()
            or not all(c.isalnum() or c == "_" for c in token[1:])
            for token in uses
        ):
            raise ValueError("finite V26 tile has unsupported original SSA spelling")
        if definition is not None:
            if definition in definitions:
                raise ValueError("finite V26 tile duplicates a definition")
            uses.discard(definition)
        if uses - definitions - {"%arg0", "%arg1", "%arg2"}:
            raise ValueError("finite V26 tile depends on another tile's SSA values")
        if definition is not None:
            definitions.add(definition)


def emit_v26_staged_dense(
    *,
    package,
    source,
    proof,
    command_buffer,
    contract,
    enabled=False,
    rewrite_policy="n_before_m_rhs",
    max_products=4 << 20,
    max_tiles=4096,
):
    """Return ``(LLVM, diagnostic)`` only for explicit bounded source opt-in.

    Default returns None so the caller retains its original backend unchanged.
    The caller must independently bind the same library/source/facts/owners,
    and qualify actual paired programs before enabling a capability.
    """
    if type(enabled) is not bool:
        raise ValueError("V26 staged source opt-in must be an explicit Boolean")
    if not enabled:
        return None
    if type(rewrite_policy) is not str or rewrite_policy not in {
        "verify_only",
        "n_before_m_rhs",
    }:
        raise ValueError("unknown finite V26 shared rewrite policy")
    if (
        type(contract) is not StagedDenseContract
        or any(type(v) is not int or v <= 0 for v in (max_products, max_tiles))
        or max_products > 4 << 20
        or max_tiles > 4096
    ):
        raise ValueError("V26 staged emission needs exact bounded selected contracts")
    if (
        contract.dim,
        contract.operand_rows,
        contract.accumulator_rows,
        contract.operand_bits,
        contract.primitive_bits,
        contract.accumulator_bits,
        contract.external_buffers_disjoint,
        contract.rhs_immutable,
        contract.retained_weights,
    ) != (16, 32, 16, 8, 20, 32, True, True, False):
        raise ValueError(
            "V26 staged emission lacks exact assigned storage/arithmetic/ownership premises"
        )
    source = Path(source)
    source_before = source.read_bytes()
    binding = bind_direct_source_contraction(source, command_buffer, proof)
    _single_command(command_buffer, binding)
    problem = StagedDenseProblem(*(binding[name] for name in ("m", "k", "n")))
    tiles = (
        ((problem.m + 15) // 16) * ((problem.k + 15) // 16) * ((problem.n + 15) // 16)
    )
    if problem.m * problem.k * problem.n > max_products or tiles > max_tiles:
        raise ValueError("finite V26 source exceeds its independent expansion budget")
    before_inputs = copy.deepcopy((command_buffer, proof, contract))
    modules, pins = _original(package)
    original = modules["builder.py"].Builder(
        command_buffer["compiler_pointer_abi"]["arguments"]
    )
    recorder = _FiniteBuilder(original, max_tiles * 4)
    modules["contraction.py"].matmul(
        recorder,
        "%arg0",
        "%arg1",
        "%arg2",
        problem.m,
        problem.k,
        problem.n,
        {"output_dtype": "i32", "epilogue": []},
        {},
        {},
    )
    recorder.fence()  # Original backend.emit's final completion, not a new program fence.
    blocks = _project(recorder, problem, contract.dim)
    effects = tuple(effect for block in blocks for effect in block["effects"])
    rewrites = (
        ()
        if rewrite_policy == "verify_only"
        else (
            StagedDenseRewrite("n_before_m", order_output_tiles_n_before_m),
            StagedDenseRewrite("rhs_reloads", eliminate_redundant_rhs_loads),
        )
    )
    proposed, report = rewrite_staged_dense(
        effects,
        problem=problem,
        contract=contract,
        rewrites=rewrites,
        max_commands=max_tiles * 32,
        max_products=max_products,
    )
    by_key = {block["readout"]: block for block in blocks}
    if len(by_key) != len(blocks):
        raise ValueError("finite V26 source has duplicate complete output tiles")
    new_blocks, segment, removed = [], [], []
    for effect in proposed:
        segment.append(effect)
        if (
            type(effect) is CompletionFence
            and len(segment) > 1
            and type(segment[-2]) is AccumulatorReadout
        ):
            readout = segment[-2]
            block = by_key.pop((readout.output_row, readout.output_col))
            cursor, omissions = 0, set()
            for before, position in zip(
                block["effects"], block["positions"], strict=True
            ):
                if cursor < len(segment) and segment[cursor] == before:
                    cursor += 1
                elif type(before) is OperandLoad and before.role == "rhs":
                    omissions.add(position)
                else:
                    raise ValueError(
                        "V26 physical mapping changes more than complete tiles and RHS reloads"
                    )
            if cursor != len(segment):
                raise ValueError("V26 physical mapping inserts unknown source effects")
            chunk = [
                line
                for index, line in enumerate(original.lines)
                if block["start"] <= index < block["end"] and index not in omissions
            ]
            _closed_chunk(chunk, ())
            new_blocks.extend(chunk)
            removed.extend(original.lines[index] for index in sorted(omissions))
            segment = []
    if segment or by_key:
        raise ValueError("V26 physical mapping omits complete original tiles")
    original_lines = list(original.lines)
    original.lines = [
        *original_lines[: blocks[0]["start"]],
        *new_blocks,
        *original_lines[blocks[-1]["end"] :],
    ]
    if Counter(original_lines) - Counter(original.lines) != Counter(removed) or Counter(
        original.lines
    ) - Counter(original_lines):
        raise ValueError("V26 physical mapping changed original emitted lines")
    _, after_pins = _original(package)
    if (
        after_pins != pins
        or before_inputs != (command_buffer, proof, contract)
        or source.read_bytes() != source_before
    ):
        raise ValueError("V26 source/library premises changed during shared emission")
    return original.finish(), {
        "schema": "gemmini.v26_shared_staged_emission.v1",
        "status": "unqualified_emission_requires_actual_pair_checks",
        "source_binding": binding,
        "original_sources": pins,
        "proposal": report,
        "physical_rhs_calls_removed": len(removed),
        "all_other_original_finite_body_lines_preserved": True,
        "finite_original_loop_expansion": True,
        "original_runtime_loop_control_preserved": False,
        "paired_control": "use enabled=True,rewrite_policy='verify_only' for the same finite expansion; default None retains original runtime loops",
        "physical_fence_count": sum(
            'asm sideeffect "fence"' in line for line in original.lines
        ),
        "fence_mapping": "each original leading/reduction/readout fence stays in its complete original tile; prologue/suffix unchanged",
        "ssa_scope": "every moved tile uses only its own preceding definitions and the three original ABI arguments",
        "enabled_by_default": False,
        "target_capability_issued": False,
        "hardware_execution_qualified": False,
        "timing_authority": "NONE",
    }
