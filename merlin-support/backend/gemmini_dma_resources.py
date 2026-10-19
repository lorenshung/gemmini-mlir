"""Selected client-channel service and finite DMA ownership, version two.

This composes the original address-aware packet projection with independently
checked handshake/ownership cones. It does not replace a timing model, observe
physical transactions, or establish interconnect/DRAM bandwidth or overlap.
"""

from __future__ import annotations

from pathlib import Path

from gemmini_dma_ownership import DmaOwnershipCones
from gemmini_dma_ownership import __file__ as _OWNERSHIP_SOURCE
from gemmini_dma_service import DmaServiceContract
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_struct as F


class DmaTransferContract:
    """Original selected-source v2 owner; JSON records carry no live authority."""

    def __init__(self, selected):
        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError(
                "DMA transfer service requires original selected source owner"
            )
        self.packet = DmaServiceContract(selected)
        self.selected = selected
        self.sources = {
            **self.packet.sources,
            "transfer_implementation": _pin(__file__),
            "ownership_cones": _pin(_OWNERSHIP_SOURCE),
        }
        modules = F.load_modules(
            Path(self.sources["firrtl"]["path"]),
            {
                "StreamReaderCore",
                "StreamWriter",
                "StreamReader",
                "XactTracker",
                "BeatMerger",
            },
        )
        self.cones = DmaOwnershipCones(
            modules["StreamReaderCore"],
            modules["StreamWriter"],
            modules["XactTracker"],
            modules["StreamReader"],
            modules,
            new_write_state=self.packet.selector.new_write_state,
        )
        self.checks = self.cones.check()
        response_cases = 0
        for size in self.packet.selector.sizes["read"]:
            beat = self.packet.selector.beats["read"]
            for counter in range(size // beat):
                for flags in range(4):
                    self.cones.reader_response_step(
                        packet_bytes=size,
                        beat_bytes=beat,
                        counter=counter,
                        ready=flags & 1,
                        valid=(flags >> 1) & 1,
                    )
                    response_cases += 1
        self.checks["read_response_counter_cases"] = response_cases

    def revalidate(self):
        self.packet.revalidate()
        if any(_pin(pin["path"]) != pin for pin in self.sources.values()):
            raise ValueError("original DMA transfer service source changed")

    def channel_contract(self):
        self.revalidate()
        source = f"selected FIRRTL sha256:{self.sources['firrtl']['sha256']}"
        resources, kinds = {}, {}
        for kind, module in (("read", "StreamReaderCore"), ("write", "StreamWriter")):
            for phase, channel in (("request", "a"), ("response", "d")):
                name = f"dma.{kind}.{channel}"
                resources[name] = {
                    "unit": "channel_handshake",
                    "capacity_per_cycle_upper": 1,
                    "provenance": f"{source}; {module}.auto.out.{channel} single ready/valid port",
                }
            kinds[kind] = {
                "data_beat_bytes": self.packet.selector.beats[kind],
                "request": {
                    "resource": f"dma.{kind}.a",
                    "units_per_packet" if kind == "read" else "units_per_data_beat": 1,
                },
                "response": {
                    "resource": f"dma.{kind}.d",
                    "units_per_data_beat" if kind == "read" else "units_per_packet": 1,
                },
            }
        return {
            "schema": "merlin.perf.channel_service_contract.v1",
            "resources": resources,
            "transfer_kinds": kinds,
            "unresolved": {
                "downstream_sharing": "client ports do not prove independent interconnect/cache/DRAM service",
                "translation_retries": "no-retry packet projection excludes replay/translation stalls",
                "storage_completion": "reader source-ID release precedes merged local row consumption; store acknowledgement is not a global fence",
                "bank_service": "no per-bank arbitration/port throughput or physical address trace is derived",
                "complete_call": "host/packing/dispatch/compute/readout dependencies are not complete",
            },
            "provenance": source
            + "; legal Get/Put response protocol, no denied/corrupt responses",
        }

    def transfer(self, identity, kind, command, *, configuration):
        self.revalidate()
        if type(identity) is not str or not identity or kind not in {"read", "write"}:
            raise ValueError("transfer requires explicit identity and read/write kind")
        rows = (
            self.packet.load_row_requests
            if kind == "read"
            else self.packet.store_row_requests
        )(command, configuration=configuration)
        return {
            "id": identity,
            "kind": kind,
            "packets": [
                {
                    key: packet[key]
                    for key in ("payload_bytes", "packet_bytes", "data_beats")
                }
                for row in rows
                for packet in row["packets"]
            ],
            "provenance": f"selected command operands + original row/address packet cones; source sha256:{self.sources['transfer_implementation']['sha256']}",
        }

    def record(self):
        contract = self.channel_contract()
        return {
            "schema": "gemmini.selected_dma_transfer_service.v2",
            "sources": self.sources,
            "selected": self.selected.report(),
            "channel_contract": contract,
            "checks": self.checks,
            "ownership": {
                "reader_source_ids": self.cones.read_slots,
                "writer_source_ids": self.cones.write_slots,
                "reader_allocate": "atomic reserve.fire and untranslated request.fire",
                "reader_release": "accepted final D beat enters BeatMerger, not final local write completion",
                "writer_allocate": "first untranslated A beat, not downstream A completion",
                "writer_release": "accepted D response source; io.busy also includes non-idle writer state",
            },
            "measurement": None,
            "timing_model_changed": False,
            "physical_overlap_qualified": False,
            "convergence_eligible": False,
        }
