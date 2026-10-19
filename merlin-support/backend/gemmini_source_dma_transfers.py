"""Join original checked source/ELF/trace effects to v2 client DMA service.

The actual source owner is re-run; saved resource JSON is never a qualification
issuer. Channel bounds describe the emitted schedule, separately from the fixed
source workload floor and any independently selected counter-window timing.
"""

from gemmini_dma_resources import DmaTransferContract
from gemmini_source_resources import SourceResourceEffects
from gemmini_timing_resources import _pin
from merlin.perf.channel_service import compose_transfer_service


class SourceDmaTransferEffects:
    def __init__(self, *, source_effects, transfer_contract):
        if (
            type(source_effects) is not SourceResourceEffects
            or type(transfer_contract) is not DmaTransferContract
        ):
            raise ValueError(
                "source DMA service needs original live source/resource owners"
            )
        if source_effects.selected is not transfer_contract.selected:
            raise ValueError(
                "source DMA service owners selected different original targets"
            )
        self.source_effects, self.transfer_contract = source_effects, transfer_contract
        self.sources = {
            "implementation": _pin(__file__),
            "channel_consumer": _pin(compose_transfer_service.__code__.co_filename),
        }

    def revalidate(self):
        self.source_effects.revalidate()
        self.transfer_contract.revalidate()
        if any(_pin(pin["path"]) != pin for pin in self.sources.values()):
            raise ValueError("original source DMA join changed")

    def derive(self, capsule_dir):
        self.revalidate()
        original = self.source_effects.derive(capsule_dir)
        transfers = []
        for command in original["dma_commands"]:
            transfers.append(
                {
                    "id": f"actual-command-{command['command_index']}",
                    "kind": {"MVIN": "read", "MVOUT": "write"}[command["class"]],
                    "packets": [
                        {
                            key: packet[key]
                            for key in ("payload_bytes", "packet_bytes", "data_beats")
                        }
                        for row in command["requests"]
                        for packet in row["packets"]
                    ],
                    "provenance": "current complete-source/linked-input/product/output owner plus original actual instruction trace and selected row/address packet cones",
                }
            )
        contract = self.transfer_contract.channel_contract()
        service = compose_transfer_service(contract, transfers)
        self.revalidate()
        return {
            "schema": "gemmini.selected_source_dma_transfers.v2",
            "sources": self.sources,
            "bindings": original["bindings"],
            "fresh_execution_correspondence": original[
                "fresh_execution_correspondence"
            ],
            "selected_inputs": original["selected_inputs"],
            "transfer_contract": self.transfer_contract.record(),
            "transfers": transfers,
            "emitted_schedule_service": service,
            "fixed_workload_service": original["workload_service_inventory"],
            "fixed_workload_partial_floor": original["partial_workload_floor"],
            "counter_window": None,
            "measured_cycles": None,
            "attainment": None,
            "convergence_eligible": False,
            "interpretation": "necessary per-client schedule service; no physical traffic, executable latency, reachable overlap, complete call or ROI attribution claim",
        }
