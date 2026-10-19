"""Reopen original source execution and derive bounded resource effects.

This joins a complete sealed source, actual linked inputs/output checks and the
original commit trace before inspecting request effects. It grants no hardware
execution, event timing, overlap or complete-call roofline qualification.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml
from gemmini_dma_service import DmaServiceContract
from gemmini_execute_cutpoints import SelectedExecuteCutpoints
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_source_execution import SourceExecutionOwner
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.perf.capsule_roofline import compose_service_floor
from merlin.targetgen.native_model_execution import _pinned_native_file


class SourceResourceEffects:
    """Original adjacent OOT owners, never serialized receipts or success labels."""

    def __init__(self, *, selected, realization, dma, execute):
        if (
            type(selected) is not SelectedSchedulingTarget
            or type(realization) is not SourceExecutionOwner
            or type(dma) is not DmaServiceContract
            or type(execute) is not SelectedExecuteCutpoints
        ):
            raise ValueError(
                "source resources require original live checked source/effect owners"
            )
        if (
            realization._selected is not selected
            or dma.selected is not selected
            or execute.selected is not selected
        ):
            raise ValueError(
                "source resource owners did not select the same original target"
            )
        self.selected, self.realization, self.dma, self.execute = (
            selected,
            realization,
            dma,
            execute,
        )
        self.framer = (
            Path(__file__).resolve().parents[2]
            / "experiments/shared_load_configuration/qualify.py"
        )
        self.pins = {
            "implementation": _pin(__file__),
            "framer": _pin(self.framer),
            "generic_service_arithmetic": _pin(
                compose_service_floor.__code__.co_filename
            ),
            "source_proof": _pin(
                module_source_path(
                    "merlin_experiments.phase0.connected_integer_qualification"
                )
            ),
        }

    def revalidate(self):
        for owner in (self.selected, self.realization, self.dma, self.execute):
            owner.revalidate()
        if any(_pin(row["path"]) != row for row in self.pins.values()):
            raise ValueError("original source resource derivation changed")

    def derive(self, capsule_dir):
        from merlin_experiments.phase0.connected_integer_qualification import (
            source_proof,
        )

        self.revalidate()
        capsule_dir = Path(capsule_dir).resolve(strict=True)
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_bytes())
        proof = source_proof(declaration, capsule_dir)
        source = capsule_dir / "capsule.interface.mlir"
        checked = self.realization.verify_source_execution(source, proof, capsule_dir)
        # Adjacent original OOT owner retains these paths; no caller-provided
        # receipt, trace or executable can replace its original selection.
        paths = self.realization._paths
        result = json.loads(paths["result"].read_bytes())
        elf = _pinned_native_file(result, "elf")
        invocation = invocation_record.verify(paths["trace_invocation"])
        log = Path(invocation["stderr"]["path"])
        if log.stat().st_size > 64 << 20:
            raise ValueError("original packet trace exceeds 64MiB resource admission")
        spec = importlib.util.spec_from_file_location(
            "original_resource_packet_framing", self.framer
        )
        framer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(framer)
        observed, events = framer._events(
            log.read_text(), self.dma.isa, framer._symbols(elf, paths["nm"])
        )
        instructions = iter(observed["instructions"])
        stream = [
            {"class": "FENCE"} if event[0] == "FENCE" else next(instructions)
            for event in events
        ]
        if next(instructions, None) is not None:
            raise ValueError("actual resource instruction/fence framing is incomplete")
        load = store = ex = None
        dma, execute = [], []
        for index, command in enumerate(stream):
            kind = command["class"]
            if kind == "FLUSH":
                load = store = ex = None
            elif kind == "CONFIG_LD":
                load = command
            elif kind == "CONFIG_ST":
                store = command
            elif kind == "CONFIG_EX":
                ex = command
            elif kind in {"MVIN", "MVOUT"}:
                requests = (
                    self.dma.load_row_requests(command, configuration=load)
                    if kind == "MVIN"
                    else self.dma.store_row_requests(command, configuration=store)
                )
                dma.append(
                    {
                        "command_index": command["index"],
                        "class": kind,
                        "requests": requests,
                    }
                )
            elif kind.startswith("COMPUTE_"):
                execute.append(
                    {
                        "command_index": command["index"],
                        "immediate_following_fence": index + 1 < len(stream)
                        and stream[index + 1]["class"] == "FENCE",
                        "conditional_cutpoint": self.execute.at_compute(
                            command, configuration=ex
                        ),
                    }
                )
        effects = checked["actual_packet_effects"]
        if len(execute) != effects["computes"]:
            raise ValueError("resource mapping omitted an actually checked compute")
        totals = {}
        for direction, kind in (("read", "MVIN"), ("write", "MVOUT")):
            rows = [
                request
                for command in dma
                if command["class"] == kind
                for request in command["requests"]
            ]
            packets = [packet for row in rows for packet in row["packets"]]
            totals[direction] = {
                "row_requests": len(rows),
                "packets": len(packets),
                "payload_bytes": sum(row["payload_bytes"] for row in rows),
                "no_retry_client_data_bytes": sum(
                    packet["packet_bytes"] for packet in packets
                ),
                "data_beats": sum(packet["data_beats"] for packet in packets),
            }
        raw_bytes = effects["raw_i32_values"] * 4
        if totals["write"]["payload_bytes"] != raw_bytes:
            raise ValueError(
                "actual plain full-width stores do not cover exact checked output bytes"
            )
        resources = self.execute.resources.record()["resources"]
        required_resources = {
            name: resources[name] for name in ("arithmetic.sites", "dma.write")
        }
        if (
            resources["dma.write"]["capacity_per_cycle_upper"]
            != self.dma.selector.beats["write"]
        ):
            raise ValueError(
                "selected source writer beat and facts movement ceiling disagree"
            )
        compulsory = [
            {
                "id": "source-products",
                "demand": {"arithmetic.sites": effects["products"]},
                "compulsory": True,
                "provenance": "sealed source contractions, exact required accelerator product coverage; independent of padded requests",
            },
            {
                "id": "required-raw-readout",
                "demand": {"dma.write": raw_bytes},
                "compulsory": True,
                "provenance": "exact unchanged raw_i32 source output contract, all distinct required outputs checked in actual MVOUTs",
            },
        ]
        unresolved = {
            "boundary": "no selected counter-window/service-snapshot binding for this functional commit trace",
            "execute": "explicit single-MUL controller cutpoints are not a reachable-state induction",
            "physical_traffic": "source-derived client request envelopes do not observe translation retries, caches or DRAM traffic",
            "banks_readout": "bank service, scaling pipeline and full retirement are not derived",
            "host_packing": "in-window layouts, packing, dispatch, host arithmetic and carried state service unknown",
            "overlap": "no overlap or complete-service chain proof; resource terms combine with max only",
        }
        evidence = {
            "schema": "merlin.phase2.service_effect_evidence.v1",
            "scope": "workload",
            "source_sha256": proof["interface_sha256"],
            "elf_sha256": _pin(elf)["sha256"],
            "device_sha256": _pin(self.selected.facts_path)["sha256"],
            "scope_sha256": None,
            "resources": required_resources,
            "events": compulsory,
            "completion_chains": [],
            "unresolved": unresolved,
        }
        self.revalidate()
        return {
            "schema": "gemmini.selected_source_resource_effects.v1",
            "sources": self.pins,
            "selected_inputs": self.selected.report()["selected_inputs"],
            "bindings": {
                "source": _pin(source),
                "elf": _pin(elf),
                "trace_invocation": _pin(paths["trace_invocation"]),
                "trace_stderr": _pin(log),
                "result": _pin(paths["result"]),
                "capsule": _pin(capsule_dir / "capsule.yaml"),
            },
            "fresh_execution_correspondence": checked,
            "dma_contract": self.dma.record(),
            "dma_commands": dma,
            "dma_totals": totals,
            "conditional_execute": execute,
            "workload_service_inventory": evidence,
            "partial_workload_floor": compose_service_floor(
                required_resources, compulsory, unresolved=unresolved
            ),
            "port_service_interpretation": "minimum useful arithmetic/readout service under unchanged required accelerator offload/raw_i32 contract; not a runtime prediction",
            "source_packet_interpretation": "conditional no-retry client envelopes at explicit controller/request cuts; not observed hardware or DRAM transactions",
            "fixed_floor_reuse": "only while original source and required accelerator/raw_i32 contract remain identical; candidate request waste is excluded",
            "measured_cycles": None,
            "attainment": None,
            "hardware_execution_qualified": False,
            "convergence_eligible": False,
            "timing_qualified": False,
        }
