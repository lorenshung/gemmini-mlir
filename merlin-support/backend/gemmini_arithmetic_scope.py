"""Additive selected-source integer arithmetic scope, never a saved grant.

The 20-bit mesh MAC output and 32-bit accumulator have different scopes.
This owner rederives its rule from the selected elaboration and exact receiver
deployment. A compiler must separately prove its actual WS panel and readout
implementation; this source rule supplies no executable or timing admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _field, _pin, _revalidate_sources, _value
from merlin.common.paths import module_source_path
from merlin.targetgen.rtl import firrtl_struct as F

SCHEMA = "gemmini.selected_integer_arithmetic_scope.v1"


def _expand(module, expression, active=()):
    if len(active) > 128 or expression in active:
        raise ValueError("selected arithmetic expression is cyclic or oversized")
    if (
        type(expression) is F.Ref
        and not expression.segments
        and expression.root in module.nodes
    ):
        return _expand(module, module.nodes[expression.root][0], (*active, expression))
    if type(expression) is F.Prim:
        return expression.op, tuple(
            _expand(module, arg, (*active, expression)) for arg in expression.args
        )
    if type(expression) is F.Ref:
        return expression.text()
    if type(expression) in (F.IntArg, F.Lit):
        return expression.value
    raise ValueError("unknown selected arithmetic primitive")


def _driver(module, destination):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(destination)]
    if len(rows) != 1:
        raise ValueError("selected arithmetic destination is absent or ambiguous")
    return rows[0]


def _ws_driver(module, destination, propagate):
    environment = {
        F.parse_expr("io.in_control.dataflow"): 1,
        F.parse_expr("io.in_control.propagate"): propagate,
        F.parse_expr("io.in_valid"): 1,
    }
    rows = [
        row
        for row in module.connects
        if row.lhs == F.parse_expr(destination)
        and all(
            bool(_value(module, guard.cond, environment)) == guard.positive
            for guard in row.guards
        )
    ]
    if len(rows) != 1:
        raise ValueError("selected WS arithmetic driver is absent or ambiguous")
    return rows[0]


def prove_local_arithmetic(mac, pe, accumulator):
    """Check actual arithmetic ASTs and the active WS effect, not module labels."""
    widths = tuple(_field(mac, field) for field in ("in_a", "in_b", "in_c", "out_d"))
    if any(value[0] != "sint" for value in widths):
        raise ValueError("selected integer MAC has a nonsigned interface")
    a, b, incoming, result = (value[1] for value in widths)
    actual = _expand(mac, _driver(mac, "io.out_d").rhs)
    expected = (
        "asSInt",
        (("tail", (("add", (("mul", ("io.in_a", "io.in_b")), "io.in_c")), 1)),),
    )
    if actual != expected or a != b or incoming < result or result < 2 * a:
        raise ValueError("selected MAC is not the exact signed multiply/add truncation")
    for propagate in (0, 1):
        if _ws_driver(pe, "mac_unit.io.in_c", propagate).rhs != F.parse_expr(
            "io.in_b"
        ) or _ws_driver(pe, "io.out_b", propagate).rhs != F.parse_expr(
            "mac_unit.io.out_d"
        ):
            raise ValueError("selected WS path is not a spatial partial sum")
    acc_widths = tuple(_field(accumulator, field) for field in ("op1", "op2", "sum"))
    if len(set(acc_widths)) != 1 or acc_widths[0][0] != "sint":
        raise ValueError("selected accumulator addition widths disagree")
    register = _driver(accumulator, "io.sum").rhs
    update = _driver(accumulator, register.text())
    if any(
        type(g.cond) is not F.Lit or g.cond.value != int(g.positive)
        for g in update.guards
    ):
        raise ValueError(
            "selected accumulator arithmetic is not the unconditional registered sum"
        )
    if _expand(accumulator, update.rhs) != (
        "asSInt",
        (("tail", (("add", ("io.op1", "io.op2")), 1)),),
    ):
        raise ValueError("selected accumulator is not the exact width-bounded sum")
    return {
        "operand_bits": a,
        "primitive_mac_bits": result,
        "accumulator_bits": acc_widths[0][1],
    }


def _mesh_sources(module, reference, active=()):
    """Follow actual value wires/registers; preserve both guarded alternatives.

    A hold edge contributes no new value. No arithmetic except bit transport is
    admitted between physical MACs. Aggregate aliases retain their full index.
    """
    if len(active) > 128:
        raise ValueError("selected mesh value transport is oversized")
    if type(reference) is F.Prim:
        if reference.op not in {"asUInt", "asSInt", "pad", "cat", "bits"}:
            raise ValueError("selected mesh adds an unaccounted arithmetic operation")
        return set().union(
            *(_mesh_sources(module, arg, active) for arg in reference.args)
        )
    if type(reference) in (F.IntArg, F.Lit):
        return set()
    if type(reference) is not F.Ref:
        raise ValueError("selected mesh contains an unsupported value expression")
    if reference in active:
        return set()
    text = reference.text()
    if re.fullmatch(r"mesh_[0-9]+_[0-9]+\.io\.out_b\[0\]", text) or text.startswith(
        "io.in_b["
    ):
        return {text}
    if not reference.segments and reference.root in module.nodes:
        return _mesh_sources(
            module, module.nodes[reference.root][0], (*active, reference)
        )
    drivers = []
    for row in module.connects:
        prefix = row.lhs.text()
        if text == prefix:
            drivers.append(row.rhs)
        elif text.startswith((prefix + "[", prefix + ".")):
            if type(row.rhs) is not F.Ref:
                raise ValueError(
                    "selected mesh aggregate transport is not an exact alias"
                )
            drivers.append(F.parse_expr(row.rhs.text() + text[len(prefix) :]))
    if not drivers:
        raise ValueError("selected mesh value lacks an original producer: " + text)
    return set().union(
        *(_mesh_sources(module, rhs, (*active, reference)) for rhs in drivers)
    )


def prove_mesh_scope(mesh, rows, cols):
    instances = {name: row[1] for name, row in mesh.decls.items() if row[0] == "inst"}
    expected = {f"mesh_{r}_{c}" for r in range(rows) for c in range(cols)}
    if set(instances) != expected:
        raise ValueError("selected mesh has unexpected physical MAC tile membership")
    witnesses = []
    for row in range(rows):
        for column in range(cols):
            destination = f"mesh_{row}_{column}.io.in_b[0]"
            sources = _mesh_sources(mesh, F.parse_expr(destination))
            allowed = (
                {f"mesh_{row - 1}_{column}.io.out_b[0]"}
                if row
                else {f"io.in_b[{column}][0]"}
            )
            if sources != allowed:
                raise ValueError(
                    "selected WS partial sum does not traverse one spatial MAC per row"
                )
            witnesses.append({"destination": destination, "sources": sorted(sources)})
    return witnesses


class IntegerArithmeticOwner:
    """Original live source derivation; JSON output cannot construct this owner."""

    def __init__(
        self, selected, *, software_spec, source_root, receiver_source, runtime
    ):
        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError(
                "integer arithmetic requires the exact original target selection"
            )
        self._selected = selected
        selected.revalidate()
        facts, observed, _ = selected._documents()
        fir = Path(facts["inputs"]["fir_path"])
        if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
            raise ValueError("selected arithmetic FIRRTL changed")
        names = {
            "Mesh",
            "AccPipe",
            "AccumulatorMem",
            "ExecuteController",
            "AccumulatorScale",
        }
        modules = F.load_modules(fir, names)
        if set(modules) != names:
            raise ValueError("selected arithmetic modules are unavailable")
        tile_names = {
            row[1] for row in modules["Mesh"].decls.values() if row[0] == "inst"
        }
        tiles = F.load_modules(fir, tile_names)
        pe_names = set()
        for tile in tiles.values():
            instances = {
                name: row[1] for name, row in tile.decls.items() if row[0] == "inst"
            }
            if set(instances) != {"tile_0_0"}:
                raise ValueError(
                    "selected arithmetic requires one actual PE per spatial tile"
                )
            for destination, source in (
                ("tile_0_0.io.in_b", "io.in_b[0]"),
                ("io.out_b[0]", "tile_0_0.io.out_b"),
            ):
                if _driver(tile, destination).rhs != F.parse_expr(source):
                    raise ValueError("selected tile changes the spatial partial sum")
            pe_names.update(instances.values())
        pes = F.load_modules(fir, pe_names)
        mac_names = {pe.decls["mac_unit"][1] for pe in pes.values()}
        macs = F.load_modules(fir, mac_names)
        rules = [
            prove_local_arithmetic(
                macs[pe.decls["mac_unit"][1]], pe, modules["AccPipe"]
            )
            for pe in pes.values()
        ]
        if not rules or any(rule != rules[0] for rule in rules):
            raise ValueError("selected physical PE arithmetic is not uniform")
        rule = rules[0]
        geometry = selected.report()["geometry"]
        witnesses = prove_mesh_scope(
            modules["Mesh"], geometry["array_rows"], geometry["array_cols"]
        )
        rule["max_primitive_k"] = geometry["array_rows"]
        rule["accumulation"] = "exact_i32_before_readout"
        if rule["operand_bits"] != 8 or rule["accumulator_bits"] != 32:
            raise ValueError(
                "original signed_i8_i32 source contract does not match selected arithmetic"
            )
        epilogue = observed["output_epilogue"]
        if (
            epilogue["accumulator_dtype"] != "i32"
            or epilogue["full_width_readout"]["dtype"] != "i32"
        ):
            raise ValueError(
                "selected raw accumulator readout differs from the source contract"
            )
        accumulator = modules["AccumulatorMem"]
        for path in (
            ("write", "bits", "data", 0, 0),
            ("read", "resp", "bits", "data", 0, 0),
            ("adder", "sum", 0, 0),
        ):
            if _field(accumulator, *path) != ("sint", rule["accumulator_bits"]):
                raise ValueError(
                    "selected full accumulator storage/readout width changed"
                )
        source_root, receiver_source, runtime = map(
            Path, (source_root, receiver_source, runtime)
        )
        deployment = json.loads((runtime / "deployment.json").read_bytes())
        critical = deployment["sources"]["sources"]["libgemmini"]["critical_files"]
        receiver = {}
        for name in ("gemmini.cc", "gemmini.h", "gemmini_params.h"):
            pin = _pin(receiver_source / name)
            if pin["sha256"] != critical[name]:
                raise ValueError(
                    "receiver source differs from the selected functional deployment"
                )
            receiver[name] = pin
        params = (receiver_source / "gemmini_params.h").read_text()
        receiver_text = (receiver_source / "gemmini.cc").read_text()
        if (
            "typedef int8_t elem_t;" not in params
            or "typedef int32_t acc_t;" not in params
            or f"#define DIM {rule['max_primitive_k']}\n" not in params
            or "results.at(i).at(j) += a * gemmini_state.pe_state.at(k).at(j);"
            not in receiver_text
            or "gemmini_state.accumulator.at(base_sp_addr + gemmini_state.c_stride * i).at(j) += value;"
            not in receiver_text
        ):
            raise ValueError(
                "selected receiver arithmetic requires a new explicit review"
            )
        pins = {
            "firrtl": _pin(fir),
            "software_spec": _pin(software_spec),
            "runtime_deployment": _pin(runtime / "deployment.json"),
            "implementation": _pin(__file__),
            "firrtl_parser": _pin(module_source_path(F.__name__)),
            **receiver,
        }
        for name in (
            "PE",
            "Arithmetic",
            "Mesh",
            "Tile",
            "AccumulatorMem",
            "ExecuteController",
            "AccumulatorScale",
        ):
            pins[name] = _pin(
                source_root
                / "generators/gemmini/src/main/scala/gemmini"
                / (name + ".scala")
            )
        for name, item in deployment["artifacts"].items():
            pin = _pin(runtime / name)
            if pin["sha256"] != item["sha256"]:
                raise ValueError("selected functional runtime artifact changed")
            pins["runtime_" + name] = pin
        self._record = {
            "schema": SCHEMA,
            "selected_inputs": selected.report()["selected_inputs"],
            "sources": pins,
            "arithmetic_rule": rule,
            "mesh_transport_witnesses": witnesses,
            "physical_pes_checked": len(pes),
            "scope": {
                "dataflow": "weight_stationary",
                "primitive": "one spatial mesh column, zero incoming partial sum",
                "full_reduction": "multiple primitive panels added in the distinct signed_i32 accumulator",
                "source_requires": [
                    "every local panel prefix fits signed primitive_mac_bits",
                    "every cross-panel source prefix fits signed accumulator_bits",
                ],
                "compiler_requires": [
                    "actual WS configuration",
                    "actual panel K within max_primitive_k",
                    "zero incoming B/D partial sum",
                    "actual private storage and exact continuation bits",
                    "selected raw_i32 readout",
                ],
                "narrow_scalar_readout": "original source policy remains independent; hardware half-even is not integer half-up",
            },
            "hardware_execution_qualified": False,
            "compiler_implementation_qualified": False,
            "timing_authority": "NONE",
        }
        self._identity = hashlib.sha256(
            json.dumps(self._record, sort_keys=True).encode()
        ).hexdigest()
        self.revalidate()

    @property
    def identity(self):
        self.revalidate()
        return SCHEMA + ":" + self._identity

    def revalidate(self):
        self._selected.revalidate()
        _revalidate_sources(self._record["sources"])
        if (
            hashlib.sha256(
                json.dumps(self._record, sort_keys=True).encode()
            ).hexdigest()
            != self._identity
        ):
            raise ValueError("selected arithmetic rule or premises changed")

    def source_rule(self):
        self.revalidate()
        return dict(self._record["arithmetic_rule"])

    def record(self):
        self.revalidate()
        return json.loads(json.dumps(self._record))
