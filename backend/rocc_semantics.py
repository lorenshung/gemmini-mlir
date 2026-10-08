"""Host RoCC names and operand fields derived from selected facts/contracts.

This interprets emitted transport, not a compiler lowering or a numerical oracle.
Unresolved widths remain unknown; slot reservations are not hardware widths.
"""

from merlin.common.facts_view import interface
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.target_registry import resolve

from . import rtl_checks

# Protocol relationships owned by this target. Numeric fields come from facts.
_BUNDLES = {
    "CONFIG_EX": ("ConfigExRs1", "ConfigExRs2"),
    "CONFIG_LD": ("ConfigMvinRs1", None),
    "CONFIG_ST": ("ConfigMvoutRs1", "ConfigMvoutRs2"),
    "MVIN": (None, "MvinRs2"),
    "MVOUT": (None, "MvoutRs2"),
    "PRELOAD": ("PreloadRs", "PreloadRs"),
}


def isa_constants(target):
    record = load_facts(target)
    body = record.get("facts", record)
    decoder = interface(body, "funct_decode_table") or {}
    if decoder.get("custom_opcode") is None or decoder.get("funct3") is None:
        raise ValueError("selected facts lack RoCC opcode/register-usage evidence")
    encoding = resolve(target).load_contract().get("encoding") or {}
    classes = {int(code): name for code, name in (decoder.get("names") or {}).items()}
    classes.update({int(code): name for code, name in (encoding.get("semantic_class") or {}).items()})
    layouts = (interface(body, "register_bundle_layouts") or {}).get("bundles") or {}
    return {
        "CUSTOM_OPCODE": int(decoder["custom_opcode"]),
        "FUNCT3": int(decoder["funct3"]),
        "FUNCT_CLASS": classes,
        "CONFIG_SUBTYPE": {int(code): name for code, name in (encoding.get("config_subtype") or {}).items()},
        "BUNDLES": layouts,
    }


def _fields(bundle, operand, isa):
    declarations = (isa["BUNDLES"].get(bundle) or {}).get("fields") or {}
    raw = operand.get("raw")
    values = {}
    for name, field in declarations.items():
        width, offset = field.get("width"), field.get("offset")
        known = isinstance(raw, int) and isinstance(width, int) and width > 0 and isinstance(offset, int)
        values[name] = (raw >> offset) & ((1 << width) - 1) if known else None
    return values


def _config_selector(rs1, isa):
    return _fields("ConfigExRs1", {"raw": rs1}, isa).get("cmd_type")


def instruction_funct(name, rs1, isa):
    for code, subtype in isa["CONFIG_SUBTYPE"].items():
        if name == subtype:
            if _config_selector(rs1, isa) != code:
                raise ValueError("CONFIG operand does not encode the requested declared subtype")
            name = "CONFIG"
            break
    matches = [code for code, klass in isa["FUNCT_CLASS"].items() if klass == name]
    if len(matches) != 1:
        raise ValueError(f"instruction {name!r} has no unique selected funct declaration")
    return matches[0]


def decode_instruction(funct, rs1, rs2, isa):
    klass = isa["FUNCT_CLASS"].get(funct, "UNKNOWN")
    if klass == "CONFIG":
        klass = isa["CONFIG_SUBTYPE"].get(_config_selector(rs1.get("raw"), isa), "UNKNOWN")
    first, second = _BUNDLES.get(klass, (None, None))
    return klass, {
        "rs1_fields": _fields(first, rs1, isa),
        "rs2_fields": _fields(second, rs2, isa),
        "operand_semantics": "only explicitly resolved packed field widths",
    }
