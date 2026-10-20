"""Selected per-load-stream state and identity pixel normalization.

This versioned source contract supplies functional packet ownership only, not
timing, overlap or physical transaction authority.
"""

from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F


def _one(module, destination):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(destination)]
    if len(rows) != 1 or type(rows[0].rhs) is not F.Ref:
        raise ValueError("selected pixel normalization has no unique original node")
    return rows[0].rhs


def _contains(module, expression, expected, active=()):
    if expression == expected:
        return True
    if len(active) >= 128 or expression in active:
        raise ValueError("pixel source normalization cone is cyclic or oversized")
    if (
        type(expression) is F.Ref
        and not expression.segments
        and expression.root in module.nodes
    ):
        return _contains(
            module, module.nodes[expression.root][0], expected, (*active, expression)
        )
    return type(expression) is F.Prim and any(
        _contains(module, arg, expected, (*active, expression))
        for arg in expression.args
    )


def _identity_pixels(load_cone, pixel_cone, field, normalized, capture):
    """Check both admitted encodings at original unsigned source cutpoints."""
    for raw in (0, 1):
        value = load_cone.evaluate(
            {"cmd_q.io.deq.bits.cmd.rs1": raw << field["offset"]}, [normalized]
        )[normalized]
        remaining = pixel_cone.evaluate(
            {"io.req.bits.pixel_repeats": value}, [capture]
        )[capture]
        if value != 1 or remaining != 0:
            raise ValueError(
                "selected raw zero/one pixel configuration is not identity repetition"
            )


class SelectedLoadStreams:
    """Re-derived original selector, banked strides and zero-repeat cutpoints."""

    def __init__(self, selected):
        if type(selected) is not SelectedSchedulingTarget:
            raise TypeError("load streams require original selected target")
        selected.revalidate()
        observer = selected.operand_observer()
        mapping = observer.load_states
        if mapping is None or mapping.config_class != "CONFIG_LD":
            raise ValueError(
                "selected load stream/state/config correspondence is unavailable"
            )
        facts, _, operands = selected._documents()
        fir = Path(facts["inputs"]["fir_path"])
        if mapping.source_sha256 != _pin(fir)["sha256"]:
            raise ValueError("load streams differ from selected original FIRRTL")
        modules = F.load_modules(fir, {"LoadController", "Scratchpad"})
        scratchpad = modules["Scratchpad"]
        repeater = scratchpad.decls.get("mvin_scale_pixel_repeater")
        if repeater is None or repeater[0] != "inst":
            raise ValueError("selected ordinary load repeater is unavailable")
        pixel = F.load_modules(fir, {repeater[1]})[repeater[1]]
        normalized = _one(modules["LoadController"], "pixel_repeats[state_id]")
        captures = [
            row.rhs
            for row in pixel.connects
            if row.lhs == F.parse_expr("req.bits.pixel_repeats")
            and type(row.rhs) is F.Ref
            and not row.rhs.segments
            and _contains(pixel, row.rhs, F.parse_expr("io.req.bits.pixel_repeats"))
        ]
        if len(captures) != 1:
            raise ValueError("selected identity pixel capture is absent or ambiguous")
        children = {
            row[1]
            for row in modules["LoadController"].decls.values()
            if row[0] == "inst"
        }
        modules.update(F.load_modules(fir, children))
        load_cone, pixel_cone = (
            C.ScalarCutpoints(modules["LoadController"], modules=modules),
            C.ScalarCutpoints(pixel),
        )
        field = next(
            row
            for row in operands["layouts"]["fields"]
            if row["bundle"] == "ConfigMvinRs1"
            and row["name"] == "pixel_repeats"
            and row["status"] == "derived"
        )
        _identity_pixels(
            load_cone, pixel_cone, field, normalized.root, captures[0].root
        )
        self.selected, self.mapping = selected, mapping
        self.ports = dict(mapping.ports)
        self.states = tuple(mapping.states)
        self.sources = {
            "firrtl": _pin(fir),
            "implementation": _pin(__file__),
            "state_derivation": _pin(
                module_source_path("merlin.targetgen.rtl.firrtl_load_states")
            ),
            "scalar_cone": _pin(C.__file__),
            "parser": _pin(F.__file__),
        }
        self._ports, self._states = dict(self.ports), self.states
        self._selected, self._mapping = selected, mapping
        self.revalidate()

    def revalidate(self):
        if self.selected is not self._selected or self.mapping is not self._mapping:
            raise ValueError("selected original load stream owner changed")
        self.selected.revalidate()
        _revalidate_sources(self.sources)
        if (
            self.ports != self._ports
            or self.states != self._states
            or dict(self.mapping.ports) != self.ports
            or tuple(self.mapping.states) != self.states
            or self.mapping.config_class != "CONFIG_LD"
            or self.mapping.source_sha256 != self.sources["firrtl"]["sha256"]
        ):
            raise ValueError("selected original load stream mapping changed")

    def record(self):
        self.revalidate()
        return {
            "schema": "gemmini.selected_load_streams.v1",
            "ports": dict(self.ports),
            "states": list(self.states),
            "identity_pixel_encodings": [0, 1],
            "sources": {name: dict(pin) for name, pin in self.sources.items()},
            "scope": "functional selected load state and identity repetition only",
            "timing_authority": False,
        }
