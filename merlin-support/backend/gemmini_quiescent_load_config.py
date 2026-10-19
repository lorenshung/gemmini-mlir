"""Narrow source-relative load-state cleanup for the copied compiler.

Only an unrolled C statement stream and quiescent state-zero identity-scale
loads are admitted. Shared configuration induction performs the rewrite; this
OOT layer owns its selected helper/effect contract. Final target execution and
hardware timing qualification remain independent.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from merlin.compile.scheduling.configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    eliminate_redundant_configuration,
)
from merlin.sched.ir import Kernel, call, evaluate

SCHEMA = "gemmini.quiescent_load_configuration.v1"
SEMANTICS = {
    "load_state": 0,
    "scale": "f32_identity",
    "shrink": 0,
    "ownership": "one_hart_no_external_configuration_writer",
    "removal_boundary": "preceding_original_completion_fence",
    "load_effect": "reads_configuration_without_writing_it",
    "unknown_effect": "barrier",
    "loops": "retain_original",
    "hardware_qualification": "not_established",
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha(path.read_bytes())}


def _runtime_helper(path, name):
    module = ast.parse(Path(path).read_text())
    constants = [
        ast.literal_eval(node.value)
        for node in module.body
        if type(node) is ast.Assign
        and len(node.targets) == 1
        and type(node.targets[0]) is ast.Name
        and node.targets[0].id == "RUNTIME"
    ]
    if len(constants) != 1 or type(constants[0]) is not str:
        raise ValueError("preserved runtime needs its original literal C source")
    source = constants[0]
    marker = "static inline void " + name + "("
    if source.count(marker) != 1:
        raise ValueError("preserved runtime helper is absent or ambiguous")
    start = source.index("{", source.index(marker)) + 1
    depth, cursor = 1, start
    while cursor < len(source) and depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    if depth:
        raise ValueError("preserved runtime helper has no complete body")
    body = source[start : cursor - 1]
    while "/*" in body:
        before, _, comment = body.partition("/*")
        _, end, after = comment.partition("*/")
        if not end:
            raise ValueError("preserved helper has an incomplete comment")
        body = before + after
    return "".join(body.split())


@dataclass(frozen=True)
class LoadConfigurationContract:
    path: Path
    payload: bytes

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        payload = path.read_bytes()
        if len(payload) > 1 << 20:
            raise ValueError("load configuration contract exceeds its byte budget")
        contract = cls(path, payload)
        contract.revalidate()
        return contract

    @property
    def identity(self):
        return _sha(self.payload)

    def revalidate(self):
        row = json.loads(self.payload)
        if (
            set(row)
            != {
                "schema",
                "selected_inputs",
                "runtime",
                "sources",
                "semantics",
                "encoding",
            }
            or row["schema"] != SCHEMA
            or row["semantics"] != SEMANTICS
        ):
            raise ValueError("unsupported load configuration effect contract")
        if self.path.read_bytes() != self.payload:
            raise ValueError("selected load configuration contract changed")
        pins = [*row["selected_inputs"].values(), row["runtime"], *row["sources"]]
        if not 4 <= len(pins) <= 64:
            raise ValueError(
                "load configuration needs a bounded explicit source closure"
            )
        for pin in pins:
            if set(pin) != {"path", "sha256"} or _pin(pin["path"]) != pin:
                raise ValueError("selected load configuration source changed")
        encoding = row["encoding"]
        if set(encoding) != {"config_rs1", "max_stride"} or any(
            type(v) is not int or v < 0 for v in encoding.values()
        ):
            raise ValueError(
                "load configuration encoding needs selected nonnegative bounds"
            )
        return row


def prepare_contract(selected, *, runtime, rocket_core, output):
    """Freeze a Phase 2 contract separately from the original observed contract.

    These source-relative effect premises are reviewed by this explicit adapter;
    neither their presence nor a functional simulator check qualifies hardware.
    The runtime must match its preserved package manifest exactly.
    """
    selected.revalidate()
    observer = selected.operand_observer()
    mapping = observer.load_states
    if mapping is None or mapping.port("MVIN") != 0 or 0 not in mapping.states:
        raise ValueError("selected source does not establish state-zero load selection")
    runtime = Path(runtime).resolve(strict=True)
    manifest = json.loads((runtime.parent.parent / "package-files.json").read_text())
    if _sha(runtime.read_bytes()) != manifest["files"][runtime.name]["sha256"]:
        raise ValueError("runtime is not the preserved compiler helper")
    facts, _, operands = selected._documents()
    bundles = next(
        row["bundles"]
        for row in facts["facts"]["interfaces"]
        if row["name"] == "register_bundle_layouts"
    )
    fields = bundles["ConfigMvinRs1"]["fields"]
    observed = {
        row["name"]: row
        for row in operands["layouts"]["fields"]
        if row["bundle"] == "ConfigMvinRs1" and row["status"] == "derived"
    }
    for name in ("scale", "stride", "state_id", "shrink"):
        if name not in observed or fields[name]["offset"] != observed[name]["offset"]:
            raise ValueError("load configuration operand layout remains unresolved")
    block = selected.report()["geometry"]["array_cols"]
    # Selected helper always uses identity f32 scale, the derived row block,
    # state zero, no shrinking and the selected configuration subtype.
    scale = int.from_bytes(__import__("struct").pack("<f", 1.0), "little")
    isa = observer.isa_constants("gemmini")
    subtype = next(
        code for code, name in isa["CONFIG_SUBTYPE"].items() if name == "CONFIG_LD"
    )
    rs1 = (
        (scale << fields["scale"]["offset"])
        | (block << fields["stride"]["offset"])
        | subtype
    )
    config_function = next(
        code for code, name in isa["FUNCT_CLASS"].items() if name == "CONFIG"
    )
    load_function = next(
        code for code, name in isa["FUNCT_CLASS"].items() if name == "MVIN"
    )
    expected = (
        f"uint64_trs1=((uint64_t)gm_float_bits(scale)<<{fields['scale']['offset']})|({block}ull<<{fields['stride']['offset']})"
        f"|((uint64_t)(id&{(1 << observed['state_id']['width']) - 1})<<{fields['state_id']['offset']})"
        f"|((uint64_t)(shrink&{(1 << observed['shrink']['width']) - 1})<<{fields['shrink']['offset']})|{subtype};"
        f"gm_cmd({config_function},rs1,(uint64_t)(uint32_t)stride_bytes);"
    )
    if _runtime_helper(runtime, "gm_ld_config") != expected:
        raise ValueError("preserved load helper differs from selected operand encoding")
    if (
        _runtime_helper(runtime, "gm_mvin")
        != f"gm_cmd({load_function},ptr,gm_shape(addr,rows,cols));"
    ):
        raise ValueError(
            "preserved transfer helper differs from selected state-zero instruction"
        )
    if _runtime_helper(runtime, "gm_fence") != '__asm__volatile("fence":::"memory");':
        raise ValueError(
            "preserved completion helper differs from the selected fence ABI"
        )
    rocket_core = Path(rocket_core).resolve(strict=True)
    core = rocket_core.read_text()
    if "id_rocc_busy && (id_ctrl.fence || id_csr_rocc_write)" not in core:
        raise ValueError(
            "selected Rocket source does not state the expected fence/RoCC busy dependency"
        )
    record = {
        "schema": SCHEMA,
        "selected_inputs": selected.report()["selected_inputs"],
        "runtime": _pin(runtime),
        "sources": [*operands["source_pins"], _pin(rocket_core)],
        "semantics": SEMANTICS,
        "encoding": {
            "config_rs1": rs1,
            "max_stride": (1 << min(32, mapping.stride_width)) - 1,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    selected.revalidate()
    return LoadConfigurationContract.read(output)


def _arguments(statement, name):
    prefix = name + "("
    text = statement.strip()
    if not text.startswith(prefix) or not text.endswith(");"):
        return None
    body = text[len(prefix) : -2]
    depth, start, parts = 0, 0, []
    for index, char in enumerate(body):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
        if depth < 0 or char in "\"';":
            return None
    if depth:
        return None
    parts.append(body[start:].strip())
    return parts


def _pure_expression(text):
    # Only the preserved emitter's scalar casts and integer pointer arithmetic.
    # Nested calls, loads, updates and other C constructs are not inferred pure.
    for spelling in (
        "const int8_t*",
        "const int32_t*",
        "int8_t*",
        "int32_t*",
        "size_t",
    ):
        text = text.replace("(" + spelling + ")", "")
    try:
        expression = ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return all(
        type(node)
        in (
            ast.Expression,
            ast.Name,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.UAdd,
            ast.USub,
        )
        or type(node) is ast.Constant
        and type(node.value) is int
        for node in ast.walk(expression)
    )


def optimize_statements(statements, contract, *, enabled=True):
    """Remove only proved redundant setters; keep every other C statement exact."""
    if (
        type(contract) is not LoadConfigurationContract
        or type(statements) is not list
        or any(type(s) is not str for s in statements)
    ):
        raise ValueError(
            "load cleanup needs an explicit selected contract and statement list"
        )
    row = contract.revalidate()
    if len(statements) > 4096 or any("{" in s or "}" in s for s in statements):
        return list(statements), {
            "status": "refused",
            "reason": "bounded_unrolled_straight_line_required",
            "removed": 0,
        }
    calls, quiescent = [], False
    for index, statement in enumerate(statements):
        if statement.strip() == "gm_fence();":
            calls.append(call("completion_fence", statement=index))
            quiescent = True
            continue
        args = _arguments(statement, "gm_ld_config")
        if (
            args is not None
            and len(args) == 4
            and args[0].isdecimal()
            and int(args[0]) <= row["encoding"]["max_stride"]
            and args[1:] == ["0", "1.0f", "0"]
            and quiescent
        ):
            calls.append(
                call(
                    "load_configuration",
                    statement=index,
                    rs1=row["encoding"]["config_rs1"],
                    rs2=int(args[0]),
                )
            )
            # A repeated setter is safe only at an independently retained fence.
            quiescent = False
            continue
        load = _arguments(statement, "gm_mvin")
        if (
            load is not None
            and len(load) == 4
            and all(_pure_expression(arg) for arg in load)
        ):
            calls.append(call("operand_load", statement=index))
        else:
            calls.append(call("unknown_effect", statement=index))
        quiescent = False

    def project(instruction):
        values = {key: evaluate(value, {}) for key, value in instruction.args}
        if instruction.instr == "load_configuration":
            return ConfigurationEffects(
                writes=(
                    ("load_state0.rs1", values["rs1"]),
                    ("load_state0.rs2", values["rs2"]),
                ),
                ordered=True,
                pure_write=True,
            )
        if instruction.instr in ("operand_load", "completion_fence"):
            return ConfigurationEffects(ordered=True)
        return None

    adapter = ConfigurationAdapter(
        project, contract.identity, "source_relative_only:" + contract.identity
    )
    original = Kernel("selected_load_configuration", (), tuple(calls))
    result = eliminate_redundant_configuration(original, adapter, enabled=enabled)
    retained = [
        evaluate(dict(instruction.args)["statement"], {})
        for instruction in result.kernel.body
    ]
    rewritten = [statements[index] for index in retained]
    contract.revalidate()
    return rewritten, {
        "schema": "gemmini.quiescent_load_configuration_rewrite.v1",
        "status": "checked_source_relative",
        "contract_sha256": contract.identity,
        "removed": len(statements) - len(rewritten),
        "source_sha256": _sha(json.dumps(statements).encode()),
        "output_sha256": _sha(json.dumps(rewritten).encode()),
        "configuration_check": result.report,
        "target_execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def rewrite_copied_compiler_statements(statements, *, enabled=True):
    """Explicit copied-package hook; no ambient curated backend or target lookup."""
    contract = LoadConfigurationContract.read(
        Path(__file__).with_name("load-config-contract.json")
    )
    rewritten, report = optimize_statements(statements, contract, enabled=enabled)
    print(
        "LOAD_CONFIGURATION " + json.dumps(report, sort_keys=True),
        file=__import__("sys").stderr,
    )
    return rewritten
