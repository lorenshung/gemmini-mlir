"""Read selected FIRRTL input-row service, without guessing mesh latency.

This checks a bounded combinational cone and its row counter at explicit
cutpoints. It grants neither reachable-state, completion nor timing qualification.
"""

from __future__ import annotations

import itertools

from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_struct as F

SCHEMA = "gemmini.execute_input_service.v1"
CONTROL = (
    "current_dataflow",
    "d_garbage",
    "a_should_be_fed_into_transposer",
    "b_should_be_fed_into_transposer",
    "d_should_be_fed_into_transposer",
)


def _width(module, expression, active=()):
    if len(active) > 128 or expression in active:
        raise ValueError("execute service width is cyclic or oversized")
    nested = (*active, expression)
    if type(expression) is F.Lit and expression.kind == "UInt":
        return expression.width or max(1, expression.value.bit_length())
    if type(expression) is F.Ref:
        if not expression.segments and expression.root in module.nodes:
            return _width(module, module.nodes[expression.root][0], nested)
        if not expression.segments and expression.root in module.decls:
            kind = F.parse_type(module.decls[expression.root][1])
            if kind[0] == "uint" and kind[1] is not None:
                return kind[1]
        drivers = [row for row in module.connects if row.lhs == expression]
        if drivers:
            return max(_width(module, row.rhs, nested) for row in drivers)
    if type(expression) is F.Prim:
        if expression.op in {"eq", "neq", "gt", "lt", "geq", "leq"}:
            return 1
        if expression.op in {"add", "sub"}:
            return max(_width(module, arg, nested) for arg in expression.args) + 1
        if expression.op == "tail":
            return _width(module, expression.args[0], nested) - expression.args[1].value
        if expression.op == "bits":
            return expression.args[1].value - expression.args[2].value + 1
        if expression.op == "mux":
            return max(_width(module, arg, nested) for arg in expression.args[1:])
        if expression.op in {"and", "or", "asUInt", "asBool"}:
            return max(_width(module, arg, nested) for arg in expression.args)
        if expression.op == "pad":
            return max(
                _width(module, expression.args[0], nested), expression.args[1].value
            )
    raise ValueError("execute service needs an explicit unsigned width")


def _active(module, row, environment, active=()):
    values = [_eval(module, guard.cond, environment, active) for guard in row.guards]
    if any(value not in (0, 1) for value in values):
        raise ValueError("execute service guard is not Boolean")
    return all(
        bool(value) == guard.positive for guard, value in zip(row.guards, values)
    )


def _eval(module, expression, environment, active=()):
    if expression in environment:
        return environment[expression]
    if len(active) > 128 or expression in active:
        raise ValueError("execute service cone is cyclic or oversized")
    nested = (*active, expression)
    if type(expression) in (F.Lit, F.IntArg):
        return expression.value
    if type(expression) is F.Ref:
        if not expression.segments and expression.root in module.nodes:
            return _eval(module, module.nodes[expression.root][0], environment, nested)
        drivers = [row for row in module.connects if row.lhs == expression]
        selected = [row for row in drivers if _active(module, row, environment, nested)]
        if not selected:
            raise ValueError(
                "execute service cone has an unbound state/input: " + expression.text()
            )
        # FIRRTL connects have source order priority. The public proof separately
        # constrains the output's default and guarded driver roster.
        return _eval(module, selected[-1].rhs, environment, nested)
    if type(expression) is not F.Prim:
        raise ValueError("unsupported execute service expression")
    values = [_eval(module, arg, environment, nested) for arg in expression.args]
    operation = expression.op
    if operation in {"asUInt", "asBool", "pad"}:
        return values[0]
    if operation == "eq":
        return int(values[0] == values[1])
    if operation == "neq":
        return int(values[0] != values[1])
    if operation == "gt":
        return int(values[0] > values[1])
    if operation == "lt":
        return int(values[0] < values[1])
    if operation == "geq":
        return int(values[0] >= values[1])
    if operation == "leq":
        return int(values[0] <= values[1])
    if operation == "and":
        return values[0] & values[1]
    if operation == "or":
        return values[0] | values[1]
    if operation == "mux":
        return values[1] if values[0] else values[2]
    if operation == "bits":
        return (values[0] >> values[2]) & ((1 << (values[1] - values[2] + 1)) - 1)
    if operation == "add":
        return values[0] + values[1]
    if operation == "sub":
        return values[0] - values[1]
    if operation == "tail":
        width = _width(module, expression.args[0]) - values[1]
        if not 1 <= width <= 64:
            raise ValueError("execute service tail has no bounded unsigned width")
        return values[0] & ((1 << width) - 1)
    raise ValueError("unsupported execute service primitive: " + operation)


def _witness(row):
    return {
        "line": row.line,
        "locator": row.locator,
        "guards": [guard.text() for guard in row.guards],
    }


def prove(module):
    drivers = [row for row in module.connects if row.lhs == F.Ref("total_rows")]
    if (
        len(drivers) != 2
        or drivers[0].guards
        or not drivers[1].guards
        or type(drivers[0].rhs) is not F.Lit
    ):
        raise ValueError(
            "execute service needs one constant default and one guarded driver"
        )
    dimension = drivers[0].rhs.value
    if not 1 <= dimension <= 64 or F.parse_type(
        module.decls["current_dataflow"][1]
    ) != ("uint", 1):
        raise ValueError(
            "execute input extent/dataflow exceeds the bounded selected proof"
        )
    guards = []
    floors = set()
    shape_checks = 0
    for values in itertools.product((0, 1), repeat=len(CONTROL)):
        control = dict(zip(CONTROL, values))
        base = {F.Ref(name): value for name, value in control.items()}
        base.update(
            {
                F.Ref("a_rows"): 0,
                F.Ref("b_rows"): 0,
                F.Ref("a_garbage"): 0,
                F.Ref("b_garbage"): 0,
            }
        )
        floor = _eval(module, F.Ref("total_rows"), base)
        if not 1 <= floor <= dimension:
            raise ValueError(
                "execute service has no positive primitive-bounded row floor"
            )
        reduced = floor != dimension
        if reduced:
            floors.add(floor)
        guards.append(dict(control, reduced=reduced))
        for a, b, a_garbage, b_garbage in itertools.product(
            range(dimension + 1), range(dimension + 1), (0, 1), (0, 1)
        ):
            env = dict(base)
            env.update(
                {
                    F.Ref("a_rows"): a,
                    F.Ref("b_rows"): b,
                    F.Ref("a_garbage"): a_garbage,
                    F.Ref("b_garbage"): b_garbage,
                }
            )
            actual = _eval(module, F.Ref("total_rows"), env)
            expected = (
                max(1 if a_garbage else a, 1 if b_garbage else b, floor)
                if reduced
                else dimension
            )
            if actual != expected:
                raise ValueError(
                    "execute service differs from the selected maximum-row rule"
                )
            shape_checks += 1
    if len(floors) != 1:
        raise ValueError("execute service has no unique reduced-row floor")
    counter = [row for row in module.connects if row.lhs == F.Ref("a_fire_counter")]
    if len(counter) != 2 or any(not row.guards for row in counter):
        raise ValueError("execute input counter has a different update contract")
    counter_checks = 0
    for total in range(1, dimension + 1):
        for old in range(total):
            for firing, fire, ready in itertools.product((0, 1), repeat=3):
                env = {
                    F.Ref("total_rows"): total,
                    F.Ref("a_fire_counter"): old,
                    F.Ref("firing"): firing,
                    F.Ref("a_fire"): fire,
                    F.Ref("cntl_ready"): ready,
                    F.parse_expr("mesh_cntl_signals_q.io.enq.ready"): ready,
                }
                selected = [row for row in counter if _active(module, row, env)]
                actual = _eval(module, selected[-1].rhs, env) if selected else old
                expected = (
                    0 if not firing else (old + 1) % total if fire and ready else old
                )
                if actual != expected:
                    raise ValueError(
                        "execute row counter is not single-step modulo selected total_rows"
                    )
                counter_checks += 1
    return {
        "dimension": dimension,
        "minimum_reduced_input_rows": next(iter(floors)),
        "reduction_guard_cases": guards,
        "input_rows_per_cycle_maximum": 1,
        "shape_cutpoint_checks": shape_checks,
        "counter_cutpoint_checks": counter_checks,
        "total_rows_witnesses": [_witness(row) for row in drivers],
        "counter_witnesses": [_witness(row) for row in counter],
        "scope": "combinational and row-counter cutpoints only; no state reachability, final completion or mesh latency proof",
    }


def derive(selected, *, execute_controller):
    selected.revalidate()
    facts, _, _ = selected._documents()
    fir = facts["inputs"]["fir_path"]
    fir_pin = _pin(fir)
    if fir_pin["sha256"] != facts["inputs"]["fir_sha256"]:
        raise ValueError("selected execute-service FIRRTL changed")
    module = F.load_modules(fir, {"ExecuteController"})["ExecuteController"]
    proof = prove(module)
    arrays = facts["facts"]["arrays"]
    if len(arrays) != 1 or arrays[0]["rows"] != proof["dimension"]:
        raise ValueError("selected stream extent differs from array geometry")
    return {
        "schema": SCHEMA,
        "proof": proof,
        "sources": {
            "firrtl": fir_pin,
            "scala_citation": _pin(execute_controller),
            "implementation": _pin(__file__),
        },
        "qualification": "NOT_ESTABLISHED",
        "limits": [
            "Input-row service alone is not total instruction latency.",
            "PRELOAD feed and COMPUTE/result completion remain distinct.",
            "Bank stalls, transposer behavior, output service and mesh propagation need separate evidence.",
            "A shortened input stream does not change the existing nominal padded-MAC counter definition.",
        ],
    }
