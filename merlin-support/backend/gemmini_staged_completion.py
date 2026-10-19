"""Selected RTL completion predicates and a versioned native event contract.

This is effect/protocol evidence, not a timing implementation or execution
qualification. In particular a CPU fence need not retire a solitary PRELOAD.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources, _value
from merlin.targetgen.rtl import firrtl_struct as F

SCHEMA = "gemmini.staged_completion_contract.v1"


def _leaves(module, expression, active=()):
    if len(active) > 128 or expression in active:
        raise ValueError("completion cone is cyclic or oversized")
    nested = (*active, expression)
    if type(expression) is F.Prim:
        return set().union(*(_leaves(module, x, nested) for x in expression.args))
    if type(expression) is not F.Ref:
        return set()
    if not expression.segments and expression.root in module.nodes:
        return _leaves(module, module.nodes[expression.root][0], nested)
    drivers = [row for row in module.connects if row.lhs == expression]
    if not drivers:
        return {expression}
    if len(drivers) != 1 or drivers[0].guards:
        raise ValueError("completion cone has ambiguous or conditional drivers")
    return _leaves(module, drivers[0].rhs, nested)


def _guard(module, row, environment):
    values = [_value(module, g.cond, environment) for g in row.guards]
    if any(value not in (0, 1) for value in values):
        raise ValueError("completion guard is not Boolean")
    return all(bool(value) == g.positive for g, value in zip(row.guards, values))


def _witness(row):
    return {
        "firrtl_line": row.line,
        "source_locator": row.locator,
        "destination": row.lhs.text(),
        "guards": [g.text() for g in row.guards],
    }


def prove(modules, *, config_function, preload_function):
    """Evaluate selected control cones under explicit, recorded state cutpoints."""
    station, execute, top = (
        modules[name] for name in ("ReservationStation", "ExecuteController", "Gemmini")
    )
    busy = []
    for count, solitary in itertools.product((0, 1, 2), (0, 1)):
        env = {
            F.Ref("empty"): int(count == 0),
            F.Ref("utilization"): count,
            F.Ref("solitary_preload"): solitary,
        }
        observed = _value(station, F.parse_expr("io.busy"), env)
        if observed != int(count != 0 and not (count == 1 and solitary)):
            raise ValueError("selected reservation busy exception differs")
        busy.append(
            {
                "valid_entries": count,
                "solitary_preload": bool(solitary),
                "busy": bool(observed),
            }
        )
    top_leaves = _leaves(top, F.parse_expr("io.busy"))
    rs_busy = F.parse_expr("reservation_station.io.busy")
    if rs_busy not in top_leaves or any(
        ref.root == "ex_controller" for ref in top_leaves
    ):
        raise ValueError("selected global busy ownership differs")
    for busy_value in (0, 1):
        env = {leaf: int(leaf == rs_busy) * busy_value for leaf in top_leaves}
        if _value(top, F.parse_expr("io.busy"), env) != busy_value:
            raise ValueError("selected global busy does not preserve the RS exception")
    starts = [
        row
        for row in execute.connects
        if row.lhs == F.Ref("performing_single_preload")
        and type(row.rhs) is F.Lit
        and row.rhs.value == 1
    ]
    if len(starts) != 1:
        raise ValueError("selected preload-start guard is ambiguous")
    preload = []
    for head, following, hazard in itertools.product((0, 1), repeat=3):
        env = {
            F.Ref("control_state"): 0,
            F.Ref("matmul_in_progress"): 0,
            F.Ref("raw_hazard_pre"): hazard,
            F.parse_expr("cmd_q.io.deq.valid[0]"): head,
            F.parse_expr("cmd_q.io.deq.valid[1]"): following,
            F.parse_expr("cmd_q.io.deq.bits[0].cmd.inst.funct"): preload_function,
            F.parse_expr("pending_completed_rob_ids[0].valid"): 0,
            F.parse_expr("pending_completed_rob_ids[1].valid"): 0,
        }
        observed = _guard(execute, starts[0], env)
        if observed != bool(head and following and not hazard):
            raise ValueError(
                "selected preload requires a different following-command or hazard predicate"
            )
        preload.append(
            {
                "head_valid": bool(head),
                "following_valid": bool(following),
                "raw_hazard": bool(hazard),
                "start": observed,
            }
        )
    # The EX configuration pop is the one selected queue-pop write guarded by
    # the same drained matmul/pending-completion predicate for every test state.
    pops = [
        row
        for row in execute.connects
        if row.lhs == F.parse_expr("cmd_q.io.deq.pop")
        and type(row.rhs) is F.Lit
        and row.rhs.value == 1
        and row.guards
    ]
    matching, config_rows = [], []
    for row in pops:
        outcomes = []
        try:
            for head, matmul, pending in itertools.product((0, 1), repeat=3):
                env = {
                    F.Ref("control_state"): 0,
                    F.Ref("matmul_in_progress"): matmul,
                    F.parse_expr("cmd_q.io.deq.valid[0]"): head,
                    F.parse_expr(
                        "cmd_q.io.deq.bits[0].cmd.inst.funct"
                    ): config_function,
                    F.parse_expr("pending_completed_rob_ids[0].valid"): pending,
                    F.parse_expr("pending_completed_rob_ids[1].valid"): 0,
                }
                outcomes.append(_guard(execute, row, env))
        except ValueError:
            continue
        expected = [
            bool(head and not matmul and not pending)
            for head, matmul, pending in itertools.product((0, 1), repeat=3)
        ]
        if outcomes == expected:
            matching.append(row)
            config_rows = [
                {
                    "head_valid": bool(head),
                    "matmul_in_progress": bool(matmul),
                    "pending_completion": bool(pending),
                    "accepted": value,
                }
                for (head, matmul, pending), value in zip(
                    itertools.product((0, 1), repeat=3), outcomes
                )
            ]
    if len(matching) != 1:
        raise ValueError(
            "selected execute configuration completion guard is unavailable or ambiguous"
        )
    return {
        "reservation_busy": busy,
        "global_busy_leaf_cutpoints": sorted(x.text() for x in top_leaves),
        "preload_start": preload,
        "execute_config_accept": config_rows,
        "witnesses": {
            "preload_start": _witness(starts[0]),
            "execute_config_complete": _witness(matching[0]),
        },
    }


def event_contract():
    """Protocol requirements for a future independent native implementation."""
    return {
        "stages": [
            "reservation_allocate",
            "controller_issue",
            "controller_consume",
            "effect_complete",
            "reservation_release",
        ],
        "reservation_ownership": "allocation_until_release_including_issued_commands",
        "load_store_configuration": {
            "reservation_release": "controller_issue",
            "configuration_visible": "controller_consume_in_fifo_order",
            "dma_tracker_allocation": False,
        },
        "execute_configuration": {
            "reservation_release": "effect_complete",
            "consume_requires": [
                "no_matmul_in_progress",
                "no_pending_completed_rob_ids",
            ],
        },
        "preload": {
            "reservation_release": "actual_tag_or_pending_rob_completion",
            "requires_following_command_to_start": True,
            "reads_tokens": ["configuration.execute", "rhs_local_storage"],
            "writes_tokens": ["preload.descriptor"],
            "cpu_fence_implies_retirement": False,
        },
        "compute": {
            "reads_tokens": [
                "configuration.execute",
                "preload.descriptor",
                "lhs_local_storage",
                "accumulator_if_continuing",
            ],
            "writes_tokens": ["accumulator"],
            "retirement": "selected_mesh_tag_or_pending_rob_completion",
        },
        "cpu_fence": {
            "wait_predicate": "selected_global_rocc_busy_is_false_and_original_cpu_fence_rules",
            "must_preserve_tokens": [
                "configuration.*",
                "pending_preload.descriptor",
                "pending_preload.reservation",
            ],
            "must_not_substitute": "unconditional_global_event_engine_drain",
        },
        "prices": "UNKNOWN",
        "native_implementation_qualified": False,
        "examples": [
            {
                "sequence": ["load_allocate", "load_issue", "cpu_fence"],
                "before_load_complete": "busy_wait",
                "after_load_complete_and_other_paths_idle": "fence_may_return",
            },
            {
                "sequence": [
                    "preload_allocate",
                    "preload_issue",
                    "cpu_fence",
                    "following_compute",
                ],
                "only_preload_pending_other_paths_idle": "fence_may_return_without_preload_retirement",
                "following_compute": "consumes_retained_configuration_and_preload_descriptor; completes_with_original_target_protocol",
            },
        ],
    }


def derive(selected, *, source_root):
    selected.revalidate()
    facts, _, operands = selected._documents()
    fir = Path(facts["inputs"]["fir_path"])
    if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
        raise ValueError("selected FIRRTL changed")
    table = next(
        row
        for row in facts["facts"]["interfaces"]
        if row["name"] == "funct_decode_table"
    )["names"]
    function = lambda label: next(
        int(key) for key, name in table.items() if name == label
    )
    modules = F.load_modules(
        fir, {"ReservationStation", "ExecuteController", "Gemmini"}
    )
    proof = prove(
        modules,
        config_function=function("CONFIG_CMD"),
        preload_function=function("PRELOAD_CMD"),
    )
    base = Path(source_root)
    sources = {"selected_firrtl": _pin(fir)}
    for name in (
        "ReservationStation",
        "ExecuteController",
        "LoadController",
        "StoreController",
        "Controller",
        "DMACommandTracker",
    ):
        sources[name] = _pin(
            base / "generators/gemmini/src/main/scala/gemmini" / (name + ".scala")
        )
    sources["RocketCore"] = _pin(
        base / "generators/rocket-chip/src/main/scala/rocket/RocketCore.scala"
    )
    for pin in operands["source_pins"]:
        if Path(pin["path"]).stem in sources and sources[Path(pin["path"]).stem] != pin:
            raise ValueError("selected controller differs from operand sidecar source")
    selected.revalidate()
    _revalidate_sources(sources)
    return {
        "schema": SCHEMA,
        "selected_inputs": selected.report()["selected_inputs"],
        "sources": sources,
        "proof": proof,
        "event_requirements": event_contract(),
        "prediction_authority": "NONE",
        "dense_v1_admission": {
            "enabled": False,
            "reason": "standalone_preload_synchronous_completion_not_established",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("facts", "observed-contract", "source-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    from merlin.common.cpu_pool import shared_compute_pool

    selected = SelectedSchedulingTarget.read(args.facts, args.observed_contract)
    with shared_compute_pool():
        report = derive(selected, source_root=args.source_root)
    report["implementation"] = _pin(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
