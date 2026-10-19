"""Live selected structural service facts, without invocation or overlap claims.

These source-rederived rate ceilings can feed generic lower-bound arithmetic.
No JSON receipt establishes actual packet/control-state bindings. Unobserved
physical traffic, ports, complete readout and host work remain explicit UNKNOWN.
"""

from __future__ import annotations

import hashlib
import json

import gemmini_execute_service as E
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin
from merlin.perf.capsule_roofline import compute_peak_from_facts
from merlin.perf.movement_peak import movement_peak


def project(facts, geometry, execute_proof):
    """Pure projection of already selected structural proofs; issues no authority."""
    peak, basis = compute_peak_from_facts(facts)
    ports = movement_peak(facts)
    resources = {
        "arithmetic.sites": {
            "unit": "MACs",
            "capacity_per_cycle_upper": peak,
            "provenance": basis,
        },
        "execute.input": {
            "unit": "input_rows",
            "capacity_per_cycle_upper": execute_proof["input_rows_per_cycle_maximum"],
            "provenance": "selected total_rows cone and single-step input counter proof",
        },
    }
    for direction in ("read", "write"):
        resources["dma." + direction] = {
            "unit": "bytes",
            "capacity_per_cycle_upper": ports.get(direction + "_bytes_per_cycle"),
            "provenance": "selected memory-client data widths and request-channel inventory; no fitted round trip",
        }
    return {
        "schema": "gemmini.structural_service_resources.v1",
        "resources": resources,
        "execute_input": execute_proof,
        "stores": geometry["stores"],
        "movement_ports": ports,
        "unresolved": {
            "actual_invocations": "issued packets are not yet bound to reachable execute control/counter states",
            "physical_transactions": "request sizes and beat width do not establish actual physical packet coverage",
            "bank_service": "bank counts/dimensions do not establish read/write ports, arbitration or overlap",
            "readout": "payload width alone does not price accumulator scaling, activation or completion",
            "host_packing": "no complete-call compulsory service derivation for packing, layout, issue or host work",
            "completion": "input service does not establish output/mesh propagation or final retirement",
        },
        "physical_traffic_bytes": None,
        "measured_cycles": None,
        "attainment": None,
        "convergence_eligible": False,
        "qualification": "structural source facts only; no executable invocation or timing qualification",
    }


class StructuralServiceResources:
    """Original live selected source owner, not constructible from saved receipts."""

    def __init__(self, selected, *, execute_controller):
        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError("require the original selected target source owner")
        self._selected = selected
        self._proof = E.derive(selected, execute_controller=execute_controller)
        self._sources = {
            "implementation": _pin(__file__),
            "service_derivation": _pin(E.__file__),
            **{
                "execute_" + key: value for key, value in self._proof["sources"].items()
            },
            "generic_arithmetic": _pin(compute_peak_from_facts.__code__.co_filename),
            "generic_movement": _pin(movement_peak.__code__.co_filename),
        }
        facts, _, _ = selected._documents()
        self._record = project(
            facts, selected.report()["geometry"], self._proof["proof"]
        )
        self._record["sources"] = self._sources
        self._record["selected_inputs"] = selected.report()["selected_inputs"]
        self.identity = hashlib.sha256(
            json.dumps(self._record, sort_keys=True).encode()
        ).hexdigest()

    def revalidate(self):
        self._selected.revalidate()
        if any(_pin(row["path"]) != row for row in self._sources.values()):
            raise ValueError("original structural service source changed")

    def record(self):
        self.revalidate()
        return json.loads(json.dumps(self._record))

    def input_rows_at_cutpoint(self, cutpoint):
        """Evaluate a proved cone; this does not prove a runtime invocation exists."""
        self.revalidate()
        proof = self._proof["proof"]
        required = {*E.CONTROL, "a_rows", "b_rows", "a_garbage", "b_garbage"}
        if not isinstance(cutpoint, dict) or set(cutpoint) != required:
            raise ValueError("complete explicit execute cutpoint required")
        for name in (*E.CONTROL, "a_garbage", "b_garbage"):
            if type(cutpoint[name]) is not int or cutpoint[name] not in (0, 1):
                raise ValueError("execute cutpoint controls must be Boolean bits")
        for name in ("a_rows", "b_rows"):
            if (
                type(cutpoint[name]) is not int
                or not 0 <= cutpoint[name] <= proof["dimension"]
            ):
                raise ValueError(
                    "execute cutpoint row extent exceeds selected source proof"
                )
        match = [
            row
            for row in proof["reduction_guard_cases"]
            if all(cutpoint[name] == row[name] for name in E.CONTROL)
        ]
        if len(match) != 1:
            raise ValueError("execute cutpoint has no unique selected source case")
        rows = proof["dimension"]
        if match[0]["reduced"]:
            rows = max(
                1 if cutpoint["a_garbage"] else cutpoint["a_rows"],
                1 if cutpoint["b_garbage"] else cutpoint["b_rows"],
                proof["minimum_reduced_input_rows"],
            )
        return {
            "input_rows": rows,
            "service_floor_cycles": rows / proof["input_rows_per_cycle_maximum"],
            "scope": "explicit combinational/counter cutpoint only",
            "actual_invocation_bound": False,
            "measured_cycles": None,
            "attainment": None,
        }
