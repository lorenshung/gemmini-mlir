"""Source-backed structural CPU resources and target queue ownership audit.

No target latency is guessed. The CPU issue interval is the selected scalar
pipeline's one-cycle service opportunity; readiness, replays and stalls remain
separate. Accelerator queue sizes are never substituted for engine lanes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from merlin.targetgen.rtl import firrtl_struct as F


def _pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _field(module, *path):
    result = F.field_type(module.port_type("io"), path)
    if result is None:
        raise ValueError("selected structural interface field is unavailable")
    return result


def _value(module, expression, environment, active=()):
    """Evaluate only the bounded selected configuration-routing Boolean cone."""
    if len(active) > 128 or expression in active:
        raise ValueError("selected routing expression is cyclic or oversized")
    if expression in environment:
        return environment[expression]
    nested = (*active, expression)
    if type(expression) in (F.Lit, F.IntArg):
        return expression.value
    if type(expression) is F.Ref:
        if not expression.segments and expression.root in module.nodes:
            return _value(module, module.nodes[expression.root][0], environment, nested)
        drivers = [row for row in module.connects if row.lhs == expression]
        if len(drivers) != 1:
            raise ValueError(
                "selected routing reference has missing or ambiguous drivers"
            )
        for guard in drivers[0].guards:
            condition = _value(module, guard.cond, environment, nested)
            if condition not in (0, 1) or bool(condition) != guard.positive:
                raise ValueError("selected routing driver is not active in the query")
        return _value(module, drivers[0].rhs, environment, nested)
    if type(expression) is not F.Prim:
        raise ValueError("selected routing expression is unsupported")
    values = [
        _value(module, argument, environment, nested) for argument in expression.args
    ]
    if expression.op == "eq":
        return int(values[0] == values[1])
    if expression.op == "neq":
        return int(values[0] != values[1])
    if expression.op == "and":
        return values[0] & values[1]
    if expression.op == "or":
        return values[0] | values[1]
    if expression.op == "mux":
        return values[1] if values[0] else values[2]
    if expression.op == "bits":
        return (values[0] >> values[2]) & ((1 << (values[1] - values[2] + 1)) - 1)
    if expression.op in {"asUInt", "asBool", "pad"}:
        return values[0]
    raise ValueError(
        "selected routing cone needs unsupported primitive: " + expression.op
    )


def extract(
    selected, *, rocket_core, reservation_station, execute_controller, spike_source
):
    """Read exact selected FIRRTL; Scala and consumer sources are audit citations."""
    selected.revalidate()
    facts, contract, operands = selected._documents()
    fir = Path(facts["inputs"]["fir_path"])
    fir_pin = _pin(fir)
    if fir_pin["sha256"] != facts["inputs"]["fir_sha256"]:
        raise ValueError("selected timing FIRRTL source changed")
    names = {"Rocket", "IBuf", "ReservationStation", "ExecuteController"}
    modules = F.load_modules(fir, names)
    if set(modules) != names:
        raise ValueError("selected timing modules are absent")
    rocket, ibuf, station = (
        modules[name] for name in ("Rocket", "IBuf", "ReservationStation")
    )
    decode = _field(ibuf, "inst")
    retire = _field(rocket, "trace", "insns")
    if decode[0] != "vec" or retire[0] != "vec" or decode[2] != 1 or retire[2] != 1:
        raise ValueError(
            "selected scalar issue/retire width is not independently established as one"
        )
    instances = {name: row[1] for name, row in rocket.decls.items() if row[0] == "inst"}
    if instances.get("alu") != "RocketALU" or instances.get("div") != "MulDiv":
        raise ValueError("selected scalar execution instances differ")
    if sum(name == "RocketALU" for name in instances.values()) != 1:
        raise ValueError(
            "selected integer execution resource has ambiguous replication"
        )
    multiplier_instances = [
        name for name, module in instances.items() if module == "PipelinedMultiplier"
    ]
    if multiplier_instances:
        raise ValueError("selected multiplier requires its separate pipeline mapping")
    port = _field(rocket, "rocc", "cmd")
    if port[0] != "bundle":
        raise ValueError("selected RoCC command issue port is not scalar")
    resources = [
        {
            "name": "cpu.issue",
            "capacity": 1,
            "queue_capacity": 0,
            "meaning": "one scalar pipeline admission opportunity per cycle; no post-admission waiting queue",
        },
        {
            "name": "cpu.integer_alu",
            "capacity": 1,
            "queue_capacity": 0,
            "meaning": "selected single RocketALU request path; operation occupancy/latency unpriced",
        },
        {
            "name": "cpu.muldiv",
            "capacity": 1,
            "queue_capacity": 0,
            "meaning": "selected shared nonpipelined MulDiv request interface; variable operation duration unpriced",
        },
    ]
    assignments = {
        "cpu." + kind: "cpu.integer_alu"
        for kind in (
            "integer",
            "bitmanip",
            "branch",
            "load",
            "store",
            "fp_load",
            "fp_store",
        )
    }
    assignments.update({"cpu.multiply": "cpu.muldiv", "cpu.divide": "cpu.muldiv"})
    issue = _field(station, "issue")
    if issue[0] != "bundle":
        raise ValueError("selected command issue classes are unavailable")
    entries = {}
    for short, label in (("ld", "load"), ("ex", "execute"), ("st", "store")):
        typ = F.parse_type(station.decls["entries_" + short][1])
        if (
            typ[0] != "vec"
            or typ[2] <= 0
            or _field(station, "issue", short, "valid") != ("uint", 1)
        ):
            raise ValueError("selected dispatch queue or issue port differs")
        entries[label] = {
            "entries": typ[2],
            "issue_width": 1,
            "entry_ownership": "allocation_until_completion_or_complete_on_issue",
        }
    table = next(
        row
        for row in facts["facts"]["interfaces"]
        if row["name"] == "funct_decode_table"
    )
    config_function = next(
        int(code) for code, label in table["names"].items() if label == "CONFIG_CMD"
    )
    selector = next(
        row
        for row in operands["layouts"]["fields"]
        if row["bundle"] == "ConfigMvoutRs1" and row["name"] == "cmd_type"
    )
    routing = []
    for subtype in range(1 << selector["width"]):
        environment = {
            F.parse_expr("io.alloc.bits.cmd.inst.funct"): config_function,
            F.parse_expr("io.alloc.bits.cmd.rs1"): subtype << selector["offset"],
            F.parse_expr("io.alloc.valid"): 1,
        }
        chosen = [
            label
            for name, label in (
                ("is_load", "load"),
                ("is_ex", "execute"),
                ("is_store", "store"),
            )
            if _value(station, F.Ref(name), environment)
        ]
        if len(chosen) != 1:
            raise ValueError(
                "configuration does not select exactly one observed issue class"
            )
        completion = _value(
            station, F.parse_expr("new_entry.complete_on_issue"), environment
        )
        routing.append(
            {
                "subtype": subtype,
                "dispatch_class": chosen[0],
                "complete_on_issue": bool(completion),
            }
        )
    sources = {
        role: _pin(path)
        for role, path in (
            ("selected_firrtl", fir),
            ("rocket_scala", rocket_core),
            ("reservation_station_scala", reservation_station),
            ("execute_controller_scala", execute_controller),
            ("native_cpu_observer", Path(spike_source) / "riscv/annotated_timing.cc"),
            (
                "native_event_engine",
                Path(spike_source) / "timing/core/include/merlin_timing/engine.hpp",
            ),
            (
                "target_provider",
                Path(spike_source) / "examples/gemmini/timing/provider.hpp",
            ),
        )
    }
    # These selected semantic source bytes already have an independent sidecar pin.
    old_execute = next(
        row
        for row in operands["source_pins"]
        if Path(row["path"]).name == "ExecuteController.scala"
    )
    if sources["execute_controller_scala"] != old_execute:
        raise ValueError(
            "execute controller source differs from selected operand provenance"
        )
    selected.revalidate()
    _revalidate_sources(sources)
    return {
        "schema": "gemmini.structural_timing_resources.v1",
        "selected_inputs": selected.report()["selected_inputs"],
        "hardware": {
            "config": contract["rtl_source"]["config"],
            "firrtl_sha256": fir_pin["sha256"],
        },
        "sources": sources,
        "cpu": {
            "per_hart": True,
            "decode_width": decode[2],
            "retire_width": retire[2],
            "instances": instances,
            "issue_service_interval_cycles": 1,
        },
        "native_safe_resources": resources,
        "cpu_execution_resources": assignments,
        "cpu_unmapped": {
            "floating_arithmetic": "format-specific FPU pipelines and shared issue/scoreboard effects need separate binding",
            "csr_system_fence_custom": "side effects, wait/drain and RoCC admission cannot be represented by a guessed shared execution unit",
            "atomic": "cache atomic service and address generation require multiple resource/effect phases",
        },
        "accelerator_dispatch": entries,
        "configuration_routing": routing,
        "native_provider_gaps": [
            "configure currently sends every subtype through one accelerator.config resource; source routes it through load/execute/store classes",
            "load/store configuration completes on issue, without a DMA command-tracker allocation; it cannot inherit load/store tracker occupancy",
            "execute configuration waits for no matmul_in_progress and no pending completed ROB ids before completing",
            "native queue_capacity counts waiting reservations only; reservation-station entry counts include issued outstanding commands",
            "one execute issue port is not one outstanding compute; preload/compute pairing and output-row pipeline require separate issue/service/completion phases",
            "provider preload currently writes descriptor state without reading execute configuration; actual hardware preload consumes execute mode and may await a paired compute",
        ],
        "accelerator_resources_applied": [],
        "timing_prices_inferred": {"cpu.issue_cycles": 1},
        "remaining_prices": "all CPU execution/branch/memory and target operation latency/occupancy remain explicit unknowns",
        "timing_qualification": "not_established",
        "prediction_authority": "NONE",
    }


def _revalidate_sources(sources):
    for pin in sources.values():
        if _pin(pin["path"]) != pin:
            raise ValueError("timing source changed during extraction or augmentation")


def augment_model(model, evidence, *, harts):
    """Add proved single-hart bindings; preserve every unknown execution price."""
    if harts != 1 or evidence.get("schema") != "gemmini.structural_timing_resources.v1":
        raise ValueError(
            "structural native augmentation currently supports one explicitly selected hart"
        )
    result = copy.deepcopy(model)
    resources = result.setdefault("resources", [])
    by_name = {row["name"]: row for row in resources}
    next_id = max((row["id"] for row in resources), default=-1) + 1
    for row in evidence["native_safe_resources"]:
        expected = {key: row[key] for key in ("name", "capacity", "queue_capacity")}
        if row["name"] in by_name:
            actual = {key: by_name[row["name"]][key] for key in expected}
            if actual != expected:
                raise ValueError(
                    "existing native resource conflicts with selected structural binding"
                )
        else:
            resources.append(dict(expected, id=next_id))
            next_id += 1
    parameters = result.setdefault("parameters", {})
    if parameters.get("cpu.issue_cycles", 1) != 1:
        raise ValueError(
            "existing CPU issue price is not the structural service interval"
        )
    parameters["cpu.issue_cycles"] = 1
    assignments = result.setdefault("cpu_execution_resources", {})
    for name, resource in evidence["cpu_execution_resources"].items():
        if name in assignments and assignments[name] != resource:
            raise ValueError(
                "existing CPU execution binding conflicts with selected hardware"
            )
        assignments[name] = resource
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "facts",
        "observed-contract",
        "rocket-core",
        "reservation-station",
        "execute-controller",
        "spike-source",
        "input-model",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--harts", required=True, type=int)
    args = parser.parse_args()
    from merlin.common.cpu_pool import shared_compute_pool

    selected = SelectedSchedulingTarget.read(args.facts, args.observed_contract)
    with shared_compute_pool():
        evidence = extract(
            selected,
            rocket_core=args.rocket_core,
            reservation_station=args.reservation_station,
            execute_controller=args.execute_controller,
            spike_source=args.spike_source,
        )
    original = args.input_model.read_bytes()
    model = augment_model(json.loads(original), evidence, harts=args.harts)
    selected.revalidate()
    _revalidate_sources(evidence["sources"])
    if args.input_model.read_bytes() != original:
        raise ValueError("original native model changed during augmentation")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "model.json").write_text(
        json.dumps(model, sort_keys=True, indent=2) + "\n"
    )
    evidence["original_model_sha256"] = hashlib.sha256(original).hexdigest()
    evidence["output_model_sha256"] = _pin(args.output / "model.json")["sha256"]
    evidence["implementation"] = _pin(__file__)
    (args.output / "resources.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
