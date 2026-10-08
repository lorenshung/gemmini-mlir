"""Fact-grounded structural screens, not compiler or numerical answers.

The selected facts supply the observed decoder domain and opcode. A partial
decoder observation is never an exhaustive legality proof. Ordering, operand
geometry and numerical correctness remain unproved by this screen.
"""

from merlin.common.facts_view import interface
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.rtl_checks import Check, CheckReport, RtlChecksUnavailable


def project_facts(record):
    body = record.get("facts", record)
    decoder = interface(body, "funct_decode_table")
    if not isinstance(decoder, dict) or decoder.get("custom_opcode") is None:
        raise RtlChecksUnavailable("selected facts have no observed custom opcode")
    return {
        "custom_opcode": int(decoder["custom_opcode"]),
        "legal_funct": list(decoder.get("legal_funct") or []),
        "complete_isa": decoder.get("complete_isa") is True,
        "source": decoder.get("method"),
        "evidence": decoder.get("evidence"),
    }


def load_default_facts(target):
    return project_facts(load_facts(target))


def _integer(value):
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (ValueError, TypeError):
        return None


def _observation(trace, facts):
    instructions = trace.get("instructions")
    if not isinstance(instructions, list) or any(not isinstance(i, dict) for i in instructions):
        raise RtlChecksUnavailable("trace instructions must be a list of records")
    commands = [i for i in instructions if i.get("class") != "FENCE"]
    unknown = sum(i.get("class") in (None, "UNKNOWN") or _integer(i.get("funct")) is None for i in commands)
    outside = sum(_integer(i.get("funct")) not in facts["legal_funct"] for i in commands)
    return {
        "commands": len(commands),
        "unknown": unknown,
        "opcode": _integer((trace.get("abi") or {}).get("custom_opcode")),
        "outside_observed_domain": outside,
    }


def screen(trace, capsule=None, rtl_facts=None, *, target, command_buffer=None):
    facts = load_default_facts(target) if rtl_facts is None else rtl_facts
    if "facts" in facts:
        facts = project_facts(facts)
    if facts.get("custom_opcode") is None:
        raise RtlChecksUnavailable("screen requires projected opcode evidence")
    seen = _observation(trace, facts)
    report = CheckReport((capsule or {}).get("name"), trace.get("source"), facts)
    for name, ok, message, expected, got in (
        ("nonempty_device_trace", seen["commands"] > 0, "device command presence", ">0", seen["commands"]),
        ("trace_decode", seen["unknown"] == 0, "transport decoded every command", 0, seen["unknown"]),
        (
            "custom_opcode",
            seen["opcode"] == facts["custom_opcode"],
            "selected observed custom opcode",
            facts["custom_opcode"],
            seen["opcode"],
        ),
    ):
        report.checks.append(Check(name, "T0", "error", "pass" if ok else "fail", message, expected, got))
    exhaustive = facts.get("complete_isa") is True and bool(facts.get("legal_funct"))
    report.checks.append(
        Check(
            "isa_legality",
            "T0",
            "error",
            ("pass" if seen["outside_observed_domain"] == 0 else "fail") if exhaustive else "skipped",
            "exhaustive decoder membership" if exhaustive else "decoder observation is not declared complete",
            0,
            seen["outside_observed_domain"],
            evidence={"source": facts.get("source"), "evidence": facts.get("evidence")},
        )
    )
    for name in ("protocol_ordering", "movement_geometry", "numerical_correctness"):
        report.checks.append(Check(name, "T0", "info", "skipped", "not established by this structural screen"))
    return report


def compile_trace_checks(facts_rec, capsule, prefix="TRACE"):
    facts = project_facts(facts_rec)
    lines = [
        "// Structural transport checks; no numerical or protocol-order proof",
        f"// {prefix}-DAG: DEVICE_TRACE_PRESENT yes{{{{$}}}}",
        f"// {prefix}-DAG: UNKNOWN_COMMAND_COUNT 0{{{{$}}}}",
        f"// {prefix}-DAG: CUSTOM_OPCODE {facts['custom_opcode']}{{{{$}}}}",
    ]
    if facts["complete_isa"] and facts["legal_funct"]:
        lines.append(f"// {prefix}-DAG: ILLEGAL_FUNCT_COUNT 0{{{{$}}}}")
    return "\n".join(lines) + "\n"


def render_trace(trace, facts_rec):
    facts = project_facts(facts_rec)
    seen = _observation(trace, facts)
    lines = [
        f"DEVICE_TRACE_PRESENT {'yes' if seen['commands'] else 'no'}",
        f"UNKNOWN_COMMAND_COUNT {seen['unknown']}",
        f"CUSTOM_OPCODE {seen['opcode']}",
    ]
    if facts["complete_isa"] and facts["legal_funct"]:
        lines.append(f"ILLEGAL_FUNCT_COUNT {seen['outside_observed_domain']}")
    else:
        lines.append("ISA_LEGALITY unknown: decoder observation is not complete")
    return "\n".join(lines) + "\n"
