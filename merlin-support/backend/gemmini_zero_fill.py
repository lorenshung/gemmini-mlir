"""Selected functional accumulator zero-fill; no implicit DRAM-zero assumption.

This is a source/effect diagnostic. It grants no hardware timing, completion
or general initialized-storage ownership.
"""

import copy
import weakref
from pathlib import Path
from types import MethodType

from gemmini_arithmetic_scope import IntegerArithmeticOwner, _driver, _expand
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl.firrtl_bit_transport import observe_bit_transport

_ISSUED = weakref.WeakKeyDictionary()


def prove_zero_route(load, scratch):
    """Prove the original command predicate and dedicated zero-writer routing."""
    if _expand(load, _driver(load, "io.dma.req.bits.all_zeros").rhs) != (
        "eq",
        ("cmd_q.io.deq.bits.cmd.rs1", 0),
    ):
        raise ValueError(
            "selected zero fill is not the original exact zero-address predicate"
        )
    if _expand(scratch, _driver(scratch, "zero_writer.io.req.valid").rhs) != (
        "and",
        ("io.dma.read.req.valid", "io.dma.read.req.bits.all_zeros"),
    ):
        raise ValueError("selected zero fill has a different zero-writer request gate")
    for field in (
        "laddr.data",
        "laddr.is_acc_addr",
        "laddr.accumulate",
        "cols",
        "block_stride",
    ):
        if _driver(scratch, "zero_writer.io.req.bits." + field).rhs != F.parse_expr(
            "io.dma.read.req.bits." + field
        ):
            raise ValueError("selected zero fill changes requested storage or extent")
    for field in ("laddr.data", "laddr.is_acc_addr", "laddr.accumulate"):
        if _driver(
            scratch, "zero_writer_pixel_repeater.io.req.bits." + field
        ).rhs != F.parse_expr("zero_writer.io.resp.bits." + field):
            raise ValueError("selected zero fill changes repeated storage ownership")
    if _driver(
        scratch, "zero_writer_pixel_repeater.io.req.bits.pixel_repeats"
    ).rhs != F.parse_expr("zero_writer.io.resp.bits.tag.pixel_repeats"):
        raise ValueError("selected zero fill changes repetition ownership")


def prove_receiver_zero(text):
    """Review exact selected runtime implementation, not compiler-provided effects."""
    begin, end = "void gemmini_t::mvin(", "void gemmini_t::mvout("
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("selected receiver has no unique original mvin implementation")
    body = text[text.index(begin) : text.index(end)]
    required = (
        "bool is_zeros = dram_addr == 0;",
        "bool const accumulator = (sp_addr >> 31) & 0x1;",
        "bool const accumulate = (sp_addr >> 30) & 0x1;",
        "if (accumulator) {",
        "if (is_zeros) {\n              value = 0;\n            } else if (!load_shrunk)",
        "if (accumulate) {\n              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) += value;\n            } else {\n              gemmini_state.accumulator.at(spad_row - pixel).at(spad_col + pixel*cols) = value;",
    )
    if any(fragment not in body for fragment in required):
        raise ValueError(
            "selected receiver zero-address/overwrite semantics require a new review"
        )


def _positive_atom(tree, atom):
    if tree == atom:
        return True
    return (
        type(tree) is tuple
        and len(tree) == 2
        and tree[0] == "and"
        and any(_positive_atom(child, atom) for child in tree[1])
    )


def prove_accumulator_zero_data(scratch, modules, banks, dim):
    """Check every selected accumulator data lane on its zero-writer branch.

    This observes conditional data only. No accepted-cycle or reachability
    premise is supplied by this cutpoint check.
    """
    checked = 0
    for bank in range(banks):
        for column in range(dim):
            sink = f"acc_mems_{bank}.io.write.bits.data[{column}][0]"
            wire = _driver(scratch, sink).rhs
            if type(wire) is not F.Ref:
                raise ValueError(
                    "selected zero fill lacks original accumulator data routing"
                )
            rows = [
                row
                for row in scratch.connects
                if row.lhs == wire
                and any(
                    guard.positive
                    and _positive_atom(
                        _expand(scratch, guard.cond),
                        "zero_writer_pixel_repeater.io.resp.valid",
                    )
                    and _positive_atom(
                        _expand(scratch, guard.cond),
                        "zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr",
                    )
                    for guard in row.guards
                )
            ]
            if len(rows) != 1:
                raise ValueError(
                    "selected zero fill has no unique positive accumulator branch"
                )
            values = observe_bit_transport(
                scratch,
                rows[0].rhs,
                sources=("io.dma.read.req.valid",),
                modules=modules,
            )
            if not values or any(
                any(bit != 0 for bit in value.bits) for value in values
            ):
                raise ValueError(
                    "selected accumulator zero-writer data is not entirely zero"
                )
            checked += 1
    return checked


class SelectedAccumulatorZeroFill:
    """Live source derivation tied to the original functional arithmetic owner."""

    def __init__(self, selected, arithmetic):
        if (
            type(self) is not SelectedAccumulatorZeroFill
            or type(selected) is not SelectedSchedulingTarget
            or type(arithmetic) is not IntegerArithmeticOwner
            or arithmetic._selected is not selected
        ):
            raise TypeError("zero fill needs exact original selected arithmetic owners")
        selected.revalidate()
        arithmetic.revalidate()
        record = arithmetic.record()
        fir = Path(record["sources"]["firrtl"]["path"])
        receiver = Path(record["sources"]["gemmini.cc"]["path"])
        self.sources = {
            "firrtl": _pin(fir),
            "receiver": _pin(receiver),
            "implementation": _pin(__file__),
            "arithmetic": _pin(module_source_path("gemmini_arithmetic_scope")),
            "parser": _pin(F.__file__),
            "bit_transport": _pin(module_source_path(observe_bit_transport.__module__)),
        }
        if (
            self.sources["firrtl"] != record["sources"]["firrtl"]
            or self.sources["receiver"] != record["sources"]["gemmini.cc"]
        ):
            raise ValueError("zero fill changed original deployed receiver or FIRRTL")
        modules = F.load_modules(fir, {"LoadController", "Scratchpad"})
        prove_zero_route(modules["LoadController"], modules["Scratchpad"])
        scratch = modules["Scratchpad"]
        # The actual sink may carry signed lanes; use original bit transport,
        # preserving their exact integer types and every data alternative.
        children = {row[1] for row in scratch.decls.values() if row[0] == "inst"}
        modules.update(F.load_modules(fir, children))
        geometry = selected.report()["geometry"]
        accumulator = next(
            row for row in geometry["stores"] if row["name"] == "accumulator"
        )
        self._data_lanes = prove_accumulator_zero_data(
            scratch, modules, accumulator["banks"], geometry["array_cols"]
        )
        prove_receiver_zero(receiver.read_text())
        self.selected, self.arithmetic = selected, arithmetic
        self._selected, self._arithmetic = selected, arithmetic
        self._arithmetic_identity = arithmetic.identity
        _ISSUED[self] = {
            "selected": selected,
            "arithmetic": arithmetic,
            "identity": arithmetic.identity,
            "sources": copy.deepcopy(self.sources),
            "data_lanes": self._data_lanes,
        }
        try:
            self.revalidate()
        except BaseException:
            del _ISSUED[self]
            raise

    def revalidate(self):
        state = _ISSUED.get(self)
        if state is None or (
            type(self) is not SelectedAccumulatorZeroFill
            or self.selected is not state["selected"]
            or self._selected is not state["selected"]
            or self.arithmetic is not state["arithmetic"]
            or self._arithmetic is not state["arithmetic"]
            or self._arithmetic_identity != state["identity"]
            or self._data_lanes != state["data_lanes"]
            or self.sources != state["sources"]
        ):
            raise ValueError("selected original zero-fill owner changed")
        for owner, name, function, code in (
            *((self, *row) for row in _API),
            *((self.arithmetic, *row) for row in _ARITHMETIC_API),
            (
                self.selected,
                "revalidate",
                _SELECTED_REVALIDATE,
                _SELECTED_REVALIDATE_CODE,
            ),
        ):
            method = getattr(owner, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not owner
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("selected original zero-fill callback changed")
        if any(
            globals()[name] is not function or function.__code__ is not code
            for name, function, code in _PROOF_API
        ):
            raise ValueError("selected original zero-fill proof function changed")
        self.selected.revalidate()
        self.arithmetic.revalidate()
        _revalidate_sources(self.sources)
        if self.arithmetic.identity != self._arithmetic_identity:
            raise ValueError("selected zero-fill arithmetic/runtime owner changed")

    def accepts(self, command):
        # Caller separately enforces exact stream configuration, dimensions,
        # local flags and complete storage/product effects before consuming it.
        raw = command["rs1"]["raw"]
        return (
            type(raw) is int and raw == 0 and command["decoded"]["is_acc_addr"] is True
        )

    def record(self):
        self.revalidate()
        return {
            "schema": "gemmini.selected_accumulator_zero_fill.v1",
            "sources": {name: dict(pin) for name, pin in self.sources.items()},
            "scope": "zero-address accumulator overwrite under independently checked identity load state and full rows/columns",
            "conditional_accumulator_zero_data_lanes": self._data_lanes,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }


_API = tuple(
    (
        name,
        getattr(SelectedAccumulatorZeroFill, name),
        getattr(SelectedAccumulatorZeroFill, name).__code__,
    )
    for name in ("revalidate", "accepts", "record")
)
_ARITHMETIC_API = tuple(
    (
        name,
        getattr(IntegerArithmeticOwner, name),
        getattr(IntegerArithmeticOwner, name).__code__,
    )
    for name in ("revalidate", "record")
)
_SELECTED_REVALIDATE = SelectedSchedulingTarget.revalidate
_SELECTED_REVALIDATE_CODE = _SELECTED_REVALIDATE.__code__
_PROOF_API = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "prove_zero_route",
        "prove_receiver_zero",
        "prove_accumulator_zero_data",
        "_positive_atom",
        "observe_bit_transport",
    )
)
