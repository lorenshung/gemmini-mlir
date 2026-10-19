"""Actual operand to selected execute cutpoint observations, not reachability.

The packet and configuration are independently decoded. The controller/queue
state is an explicit conditional premise: an architectural fence trace alone
does not prove that a PRELOAD retired or that these RTL states were reached.
"""

from __future__ import annotations

from pathlib import Path

import gemmini_execute_service as E
import gemmini_execute_state as X
from gemmini_roofline_resources import StructuralServiceResources
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F


class SelectedExecuteCutpoints:
    def __init__(self, selected, resources):
        if (
            type(selected) is not SelectedSchedulingTarget
            or type(resources) is not StructuralServiceResources
        ):
            raise ValueError(
                "execute cuts require original live selected source owners"
            )
        selected.revalidate()
        resources.revalidate()
        if (
            resources.record()["selected_inputs"]
            != selected.report()["selected_inputs"]
        ):
            raise ValueError(
                "execute cuts and service proof selected different sources"
            )
        self.selected, self.resources = selected, resources
        self.ex = X.ExecuteStateContract(selected)
        self.decoder = selected.operand_observer()
        self.isa = self.decoder.isa_constants("gemmini")
        facts, _, _ = selected._documents()
        fir = Path(facts["inputs"]["fir_path"])
        module = F.load_modules(fir, {"ExecuteController"})["ExecuteController"]
        children = {row[1] for row in module.decls.values() if row[0] == "inst"}
        self.observer = C.ScalarCutpoints(
            module, modules=F.load_modules(fir, children), max_depth=256
        )
        self.sources = {
            name: _pin(path)
            for name, path in {
                "firrtl": fir,
                "implementation": __file__,
                "scalar_observer": C.__file__,
                "parser": F.__file__,
                "execute_fields": X.__file__,
                "row_service": E.__file__,
            }.items()
        }

    def revalidate(self):
        self.selected.revalidate()
        self.resources.revalidate()
        self.ex.revalidate()
        if any(_pin(row["path"]) != row for row in self.sources.values()):
            raise ValueError("original selected execute cutpoint source changed")

    def at_compute(self, command, *, configuration):
        self.revalidate()
        for row in (command, configuration):
            kind, fields = self.decoder.decode_instruction(
                row["funct"], row["rs1"], row["rs2"], self.isa
            )
            if kind != row["class"] or fields != row["decoded"]:
                raise ValueError(
                    "packet metadata differs from original selected operands"
                )
        if (
            not command["class"].startswith("COMPUTE_")
            or configuration["class"] != "CONFIG_EX"
        ):
            raise ValueError("execute cuts require actual compute and EX configuration")
        config = self.ex.identity_ws(
            configuration["rs1"]["raw"], configuration["rs2"]["raw"]
        )
        cuts = {
            "current_dataflow": config["dataflow"],
            "a_transpose": config["a_transpose"],
            "bd_transpose": config["b_transpose"],
            "control_state": 0,
            "perform_single_mul": 0,
            "perform_single_preload": 0,
            "perform_mul_pre": 0,
            "matmul_in_progress": 0,
            "pending_completed_rob_ids[0].valid": 0,
            "pending_completed_rob_ids[1].valid": 0,
            "cmd_q.io.deq.valid[0]": 1,
            "cmd_q.io.deq.valid[1]": 0,
            "cmd_q.io.deq.valid[2]": 0,
            "cmd_q.io.deq.bits[0].cmd.inst.funct": command["funct"],
            "cmd_q.io.deq.bits[0].cmd.rs1": command["rs1"]["raw"],
            "cmd_q.io.deq.bits[0].cmd.rs2": command["rs2"]["raw"],
        }
        outputs = [
            "performing_single_mul",
            "start_inputting_a",
            "start_inputting_b",
            "start_inputting_d",
            "a_rows",
            "b_rows",
            "a_garbage",
            "b_garbage",
            "d_garbage",
            "a_should_be_fed_into_transposer",
            "b_should_be_fed_into_transposer",
            "d_should_be_fed_into_transposer",
        ]
        values = self.observer.evaluate(cuts, outputs)
        if values["performing_single_mul"] != 1 or values["start_inputting_d"] != 0:
            raise ValueError(
                "actual source cone does not select conditional single-MUL"
            )
        control = {
            name: values[name]
            for name in (*E.CONTROL, "a_rows", "b_rows", "a_garbage", "b_garbage")
            if name != "current_dataflow"
        }
        control["current_dataflow"] = config["dataflow"]
        return {
            "schema": "gemmini.selected_execute_cutpoint.v1",
            "sources": self.sources,
            "explicit_cuts": cuts,
            "derived_values": values,
            "conditional_input_service": self.resources.input_rows_at_cutpoint(control),
            "actual_invocation_bound": False,
            "unresolved": "reachable queue/controller state and configuration visibility require sequential source/RTL proof; packet order and FENCE alone do not establish it",
            "completion_latency": None,
            "attainment": None,
            "convergence_eligible": False,
        }
