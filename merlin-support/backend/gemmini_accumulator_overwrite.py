"""Explicit first-product overwrite effect from selected RTL and receiver sources.

This conditional functional rule supplies no timing, initialization of reads,
physical overlap, hardware execution qualification, or compiler authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import weakref
from pathlib import Path
from types import MethodType

from gemmini_arithmetic_scope import IntegerArithmeticOwner, _driver, _expand
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl import firrtl_struct as F

PROTOCOL = "gemmini.accumulator_first_product_overwrite.v2"
_ISSUED = weakref.WeakKeyDictionary()
# This review supports the exact original deployed C++ implementation only.
_REVIEWED_FIR_SHA256 = (
    "b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0"
)
_REVIEWED_RECEIVER_SHA256 = (
    "c468a86deb68cb2e12e3fb1177821075b0eb87e27a37ff835280ee7caf61b13a"
)


def require_protocol(value):
    if value is not None and (type(value) is not str or value != PROTOCOL):
        raise ValueError("unknown accumulator overwrite effects protocol")


def _same(module, destination, expected):
    row = _driver(module, destination)
    if row.guards or row.rhs != F.parse_expr(expected):
        raise ValueError(
            "selected accumulator overwrite routing differs: " + destination
        )


def _project(module, expression, suffix=(), active=()):
    """Project original aggregate wires/muxes without guessing field aliases."""
    if len(active) > 128 or (expression, suffix) in active:
        raise ValueError("overwrite aggregate cone is cyclic or oversized")
    nested = (*active, (expression, suffix))
    if type(expression) is F.Ref:
        path = F.Ref(expression.root, expression.segments + suffix, expression.dynamic)
        rows = [r for r in module.connects if r.lhs == path]
        if rows:
            if len(rows) != 1 or rows[0].guards:
                raise ValueError("overwrite aggregate has ambiguous field drivers")
            return _project(module, rows[0].rhs, active=nested)
        if expression.segments:
            return path.text()
        if expression.root in module.nodes:
            return _project(module, module.nodes[expression.root][0], suffix, nested)
        rows = [r for r in module.connects if r.lhs == expression]
        if rows:
            if len(rows) != 1 or rows[0].guards:
                raise ValueError("overwrite aggregate has ambiguous drivers")
            return _project(module, rows[0].rhs, suffix, nested)
        return path.text()
    if type(expression) is F.Prim and expression.op == "mux":
        cond, yes, no = expression.args
        yes, no = (_project(module, arm, suffix, nested) for arm in (yes, no))
        return yes if yes == no else ("mux", (_expand(module, cond), yes, no))
    if suffix:
        raise ValueError("unsupported accumulator aggregate projection")
    return _expand(module, expression)


def prove_memory_overwrite(module, dim):
    """Prove the selected two-port write bypass and complete byte-mask routing."""
    _same(module, "pipelined_writes[0].bits", "io.write.bits")
    _same(module, "pipelined_writes[1]", "pipelined_writes[0]")
    # No later subfield assignment may change the captured write packet.
    permitted = {"pipelined_writes[0].bits", "pipelined_writes[1]"}
    for row in module.connects:
        name = row.lhs.text()
        if (
            name.startswith("pipelined_writes[")
            and not name.endswith(".valid")
            and name not in permitted
        ):
            raise ValueError("selected accumulator write packet is overridden")
    _same(module, "mem.io.waddr", "pipelined_writes[1].bits.addr")
    _same(module, "mem.io.wen", "pipelined_writes[1].valid")
    if _expand(module, F.Ref("_T")) != (
        "mux",
        (
            "pipelined_writes[1].bits.acc",
            "io.adder.sum",
            "pipelined_writes[1].bits.data",
        ),
    ):
        raise ValueError(
            "selected nonaccumulating write is not the incoming-data bypass"
        )
    if _expand(module, _driver(module, "io.adder.valid").rhs) != (
        "and",
        ("pipelined_writes[0].valid", "pipelined_writes[0].bits.acc"),
    ):
        raise ValueError("selected overwrite changed the conditional accumulator adder")
    for column in range(dim):
        _same(module, f"mem.io.wdata[{column}][0]", f"_T[{column}][0]")
    for byte in range(4 * dim):
        _same(module, f"mem.io.mask[{byte}]", f"pipelined_writes[1].bits.mask[{byte}]")
    return {"columns": dim, "byte_masks": 4 * dim}


def prove_execute_overwrite(execute, scratch, *, banks, dim, accumulate_offset):
    """Check the selected output flag, local routing and all tail byte masks."""
    field = _expand(execute, _driver(execute, "c_address_rs2.accumulate").rhs)
    if field != ("bits", ("_c_address_rs2_WIRE", accumulate_offset, accumulate_offset)):
        raise ValueError(
            "selected PRELOAD overwrite flag differs from its operand field"
        )
    _same(
        execute,
        "mesh_cntl_signals_q.io.enq.bits.c_addr.accumulate",
        "c_address_rs2.accumulate",
    )
    if (
        _project(execute, F.Ref("w_address"), ("accumulate",))
        != "mesh.io.resp.bits.tag.addr.accumulate"
    ):
        raise ValueError("selected output-address adjustment changes overwrite flag")
    for bank in range(banks):
        port = f"io.acc.write[{bank}]"
        _same(execute, port + ".bits.acc", "w_address.accumulate")
        _same(execute, port + ".bits.addr", "w_row")
        for byte in range(4 * dim):
            _same(execute, port + f".bits.mask[{byte}]", f"w_mask_{byte // 4}")
        for column in range(dim):
            if _expand(execute, F.Ref(f"w_mask_{column}")) != (
                "lt",
                (column, "mesh.io.resp.bits.tag.cols"),
            ):
                raise ValueError("selected accumulator column mask changes tail writes")
        for field in ("acc", "addr"):
            target = f"bank_ios_1[{bank}].write.bits.{field}"
            tree = _expand(scratch, _driver(scratch, target).rhs)
            if (
                type(tree) is not tuple
                or tree[:1] != ("mux",)
                or tree[1][:2]
                != (
                    port + ".valid",
                    port + ".bits." + field,
                )
            ):
                raise ValueError(
                    "selected scratchpad changes execute write flag/address"
                )
            _same(scratch, f"acc_mems_{bank}.io.write.bits.{field}", target)
        for field, count in (("data", dim), ("mask", 4 * dim)):
            for lane in range(count):
                tail = f".bits.{field}[{lane}]" + ("[0]" if field == "data" else "")
                target = f"bank_ios_1[{bank}].write" + tail
                drivers = [r for r in scratch.connects if r.lhs == F.parse_expr(target)]
                positive = F.Guard(F.parse_expr(port + ".valid"), True)
                negative = F.Guard(F.parse_expr(port + ".valid"), False)
                rows = [r for r in drivers if r.guards == (positive,)]
                if (
                    len(rows) != 1
                    or rows[0].rhs != F.parse_expr(port + tail)
                    or any(
                        negative not in r.guards for r in drivers if r is not rows[0]
                    )
                ):
                    raise ValueError(
                        "selected scratchpad changes accepted execute data/mask"
                    )
                _same(scratch, f"acc_mems_{bank}.io.write" + tail, target)


def _tokens(text):
    """Bounded C++ lexical review: comments/strings cannot supply proof tokens."""
    if type(text) is not str or len(text.encode()) > 1 << 20:
        raise ValueError("receiver lexical review exceeds bound")
    result, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end + 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated receiver comment")
            i = end + 2
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < len(text) and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            if i >= len(text):
                raise ValueError("unterminated receiver string")
            i += 1
            result.append("<quoted>")
        elif text[i].isalnum() or text[i] == "_":
            end = i + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            result.append(text[i:end])
            i = end
        else:
            result.append(text[i])
            i += 1
    return result


def _function_tokens(text, name):
    tokens = _tokens(text)
    header = _tokens("void gemmini_t::" + name + "(")
    starts = [i for i in range(len(tokens)) if tokens[i : i + len(header)] == header]
    if len(starts) != 1:
        raise ValueError("receiver has no unique original compute implementation")
    start, depth = starts[0] + len(header), 1
    cursor = start
    while cursor < len(tokens) and depth:
        depth += (tokens[cursor] == "(") - (tokens[cursor] == ")")
        cursor += 1
    if depth or cursor >= len(tokens) or tokens[cursor] != "{":
        raise ValueError("receiver compute has no complete original body")
    start, depth = cursor, 1
    cursor += 1
    while cursor < len(tokens) and depth:
        depth += (tokens[cursor] == "{") - (tokens[cursor] == "}")
        cursor += 1
    if depth:
        raise ValueError("receiver compute body is incomplete")
    return tokens[start:cursor]


def prove_receiver_overwrite(text):
    tokens = _function_tokens(text, "compute")
    preload = _function_tokens(text, "preload")
    for fragment in (
        "gemmini_state.output_sp_addr = static_cast<uint32_t>(c_addr & 0xFFFFFFFF);",
        "gemmini_state.output_cols = (c_addr >> addr_len) & 0xFFFF;",
        "gemmini_state.output_rows = (c_addr >> (addr_len + 16)) & 0xFFFF;",
    ):
        part = _tokens(fragment)
        if sum(preload[i : i + len(part)] == part for i in range(len(preload))) != 1:
            raise ValueError("selected receiver PRELOAD output descriptor differs")
    expected = (
        "bool const acc = (gemmini_state.output_sp_addr >> 31) & 0x1;",
        "bool const acc_accum = (gemmini_state.output_sp_addr >> 30) & 0x1;",
        "auto const base_sp_addr = gemmini_state.output_sp_addr & 0x1FFFFFFF;",
        "for (size_t i = 0; i < gemmini_state.output_rows; ++i) { for (size_t j = 0; j < gemmini_state.output_cols; ++j) {",
        "acc_t value = gemmini_state.mode == gemmini_state_t::OS ? gemmini_state.pe_state.at(i).at(j) : results.at(i).at(j);",
        "if (acc_accum) { gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) += value; } else { gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) = value; }",
    )
    for fragment in expected:
        part = _tokens(fragment)
        if sum(tokens[i : i + len(part)] == part for i in range(len(tokens))) != 1:
            raise ValueError(
                "selected receiver overwrite/extent/address semantics differ"
            )


class SelectedAccumulatorOverwrite:
    """Exact privately issued selected-source effect; records cannot restore it."""

    def __init__(self, selected, arithmetic):
        if (
            type(self) is not SelectedAccumulatorOverwrite
            or type(selected) is not SelectedSchedulingTarget
            or type(arithmetic) is not IntegerArithmeticOwner
            or arithmetic._selected is not selected
        ):
            raise TypeError("overwrite needs original selected arithmetic owners")
        selected.revalidate()
        arithmetic.revalidate()
        record = arithmetic.record()
        if record["selected_inputs"] != selected.report()["selected_inputs"]:
            raise ValueError(
                "overwrite selected inputs differ from original arithmetic"
            )
        fir = Path(record["sources"]["firrtl"]["path"])
        receiver = Path(record["sources"]["gemmini.cc"]["path"])
        sources = {
            "firrtl": _pin(fir),
            "receiver": _pin(receiver),
            **{
                name: _pin(module_source_path(name))
                for name in (
                    "gemmini_accumulator_overwrite",
                    "gemmini_arithmetic_scope",
                    "gemmini_selected_scheduling",
                    "merlin.targetgen.rtl.firrtl_struct",
                )
            },
        }
        if sources["firrtl"]["sha256"] != _REVIEWED_FIR_SHA256:
            raise ValueError(
                "accumulator overwrite needs its exact reviewed selected FIRRTL"
            )
        if sources["receiver"]["sha256"] != _REVIEWED_RECEIVER_SHA256:
            raise ValueError(
                "accumulator overwrite needs its exact reviewed deployed receiver"
            )
        if (
            sources["firrtl"] != record["sources"]["firrtl"]
            or sources["receiver"] != record["sources"]["gemmini.cc"]
        ):
            raise ValueError("overwrite changed original deployed receiver/FIRRTL")
        modules = F.load_modules(fir, {"ExecuteController", "Scratchpad"})
        geometry = selected.report()["geometry"]
        accumulator = next(
            row for row in geometry["stores"] if row["name"] == "accumulator"
        )
        banks, dim = accumulator["banks"], geometry["array_cols"]
        names = {
            modules["Scratchpad"].decls[f"acc_mems_{bank}"][1] for bank in range(banks)
        }
        modules.update(F.load_modules(fir, names))
        memory = [prove_memory_overwrite(modules[name], dim) for name in sorted(names)]
        local = next(
            row
            for row in selected._documents()[2]["layouts"]["fields"]
            if row["bundle"] == "PreloadRs" and row["name"] == "local_addr"
        )
        flag = next(row for row in local["subfields"] if row["field"] == "accumulate")
        prove_execute_overwrite(
            modules["ExecuteController"],
            modules["Scratchpad"],
            banks=banks,
            dim=dim,
            accumulate_offset=flag["offset"],
        )
        prove_receiver_overwrite(receiver.read_text())
        self.selected, self.arithmetic = selected, arithmetic
        evidence = {
            "protocol": PROTOCOL,
            "selected_inputs": record["selected_inputs"],
            "arithmetic_identity": arithmetic.identity,
            "sources": sources,
            "bank_modules": sorted(names),
            "memory": memory,
            "accumulate_offset": flag["offset"],
        }
        _ISSUED[self] = {
            "selected": selected,
            "arithmetic": arithmetic,
            "evidence": copy.deepcopy(evidence),
            "identity": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest(),
            "callbacks": tuple(
                (
                    owner,
                    name,
                    getattr(type(owner), name),
                    getattr(type(owner), name).__code__,
                )
                for owner, names in (
                    (selected, ("report", "revalidate")),
                    (arithmetic, ("record", "revalidate")),
                )
                for name in names
            ),
        }
        try:
            self.revalidate()
        except BaseException:
            del _ISSUED[self]
            raise

    def revalidate(self):
        state = _ISSUED.get(self)
        if (
            type(self) is not SelectedAccumulatorOverwrite
            or state is None
            or self.selected is not state["selected"]
            or self.arithmetic is not state["arithmetic"]
            or self.arithmetic._selected is not self.selected
            or state["evidence"]["sources"]["firrtl"]["sha256"] != _REVIEWED_FIR_SHA256
            or state["evidence"]["sources"]["receiver"]["sha256"]
            != _REVIEWED_RECEIVER_SHA256
        ):
            raise ValueError(
                "original accumulator overwrite owner changed or was not issued"
            )
        for owner, name, function, code in (
            *((self, *r) for r in _API),
            *state["callbacks"],
        ):
            method = getattr(owner, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not owner
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("original accumulator overwrite callback changed")
        if any(
            globals()[name] is not function or function.__code__ is not code
            for name, function, code in _PROOFS
        ):
            raise ValueError("original accumulator overwrite proof changed")
        if (
            type(self).identity.fget is not _IDENTITY_GETTER
            or _IDENTITY_GETTER.__code__ is not _IDENTITY_CODE
        ):
            raise ValueError("original accumulator overwrite identity callback changed")
        if (
            hashlib.sha256(
                json.dumps(state["evidence"], sort_keys=True).encode()
            ).hexdigest()
            != state["identity"]
        ):
            raise ValueError("original accumulator overwrite evidence changed")
        self.selected.revalidate()
        self.arithmetic.revalidate()
        _revalidate_sources(state["evidence"]["sources"])
        if (
            self.arithmetic.identity != state["evidence"]["arithmetic_identity"]
            or self.selected.report()["selected_inputs"]
            != state["evidence"]["selected_inputs"]
        ):
            raise ValueError(
                "original accumulator overwrite source/runtime binding changed"
            )

    @property
    def identity(self):
        self.revalidate()
        return PROTOCOL + ":" + _ISSUED[self]["identity"]

    def record(self):
        self.revalidate()
        return {
            **copy.deepcopy(_ISSUED[self]["evidence"]),
            "scope": "conditional accepted nonaccumulating product replaces masked accumulator cells; source prefix/ownership/readout checked separately",
            "limits": [
                "conditional functional cutpoints; no pipeline reachability, overlap or completion-cycle proof",
                "only first exact zero-seeded source reduction prefix may replace destination; accumulating reads remain initialized",
            ],
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }


_IDENTITY_GETTER = SelectedAccumulatorOverwrite.identity.fget
_IDENTITY_CODE = _IDENTITY_GETTER.__code__
_API = tuple(
    (
        name,
        getattr(SelectedAccumulatorOverwrite, name),
        getattr(SelectedAccumulatorOverwrite, name).__code__,
    )
    for name in ("__init__", "revalidate", "record")
)
_PROOFS = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "_same",
        "_project",
        "prove_memory_overwrite",
        "prove_execute_overwrite",
        "_tokens",
        "_function_tokens",
        "prove_receiver_overwrite",
        "_driver",
        "_expand",
    )
)
