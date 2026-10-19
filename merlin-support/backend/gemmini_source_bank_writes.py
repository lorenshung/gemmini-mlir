"""Executed-command to source-derived plain scratchpad write demand, version one.

This never calls a compiler/simulator or loads success from a saved receipt.
Accepted-row demand is derived from selected effects, not a physical bank trace.
"""

import importlib.util
import json
from pathlib import Path

from gemmini_bank_service import ScratchpadBankService
from gemmini_mvin_bank_path import CheckedBankService, PlainMvinBankPath, _bound
from gemmini_source_resources import SourceResourceEffects
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.perf.capsule_roofline import compose_service_floor
from merlin.targetgen.native_model_execution import _pinned_native_file

_SOURCE_DERIVE = SourceResourceEffects.derive
_SOURCE_REVALIDATE = SourceResourceEffects.revalidate
_PATH_ROWS = PlainMvinBankPath.command_rows
_PATH_REVALIDATE = PlainMvinBankPath.revalidate


def account_stream(stream, path):
    """Preserve the complete stream; unsupported effects do not become zero."""
    if not isinstance(stream, (list, tuple)) or len(stream) > 100000:
        raise ValueError(
            "bank-write accounting needs the complete bounded actual stream"
        )
    commands, events, configuration = [], [], None
    for ordinal, command in enumerate(stream):
        kind = command["class"]
        row = {
            "stream_ordinal": ordinal,
            "command_index": command.get("index"),
            "class": kind,
            "writes": [],
        }
        if kind == "FLUSH":
            configuration = None
        if kind == "CONFIG_LD":
            configuration = command
        if kind == "MVIN":
            try:
                writes = path.command_rows(command, configuration)
                if (
                    not isinstance(writes, list)
                    or not writes
                    or any(
                        not isinstance(write, dict)
                        or not isinstance(write.get("resource"), str)
                        or not write["resource"]
                        or not isinstance(write.get("byte_mask"), list)
                        or not any(write["byte_mask"])
                        or any(
                            type(bit) is not int or bit not in (0, 1)
                            for bit in write["byte_mask"]
                        )
                        or write.get("physical_event_observed") is not False
                        for write in writes
                    )
                ):
                    raise ValueError(
                        "actual MVIN has incomplete local write/mask effects"
                    )
            except (ValueError, TypeError, KeyError) as error:
                row.update(status="UNKNOWN", reason=str(error))
            else:
                row.update(status="source_derived_plain_spad_writes", writes=writes)
                for index, write in enumerate(writes):
                    events.append(
                        {
                            "id": f"actual-{ordinal}-local-write-{index}",
                            "demand": {write["resource"]: 1},
                            "compulsory": False,
                            "provenance": "original executed MVIN + checked plain row/mask/identity path; conditional accepted bank write, not observed hardware event",
                        }
                    )
        elif kind in {"CONFIG_LD", "CONFIG_ST", "CONFIG_EX", "FENCE", "FLUSH"}:
            row.update(
                status="no_local_row_write_from_this_control",
                reason="control effects retained; no row-write service assigned",
            )
        else:
            row.update(
                status="UNKNOWN",
                reason="accumulator, execute/readout, spatial or other bank effects have no admitted local event mapping",
            )
        commands.append(row)
    return commands, events


class SourceBankWriteEffects:
    def __init__(self, *, source_effects, bank_service):
        if (
            type(source_effects) is not SourceResourceEffects
            or type(bank_service) is not ScratchpadBankService
        ):
            raise ValueError(
                "bank writes require original live complete-source and bank owners"
            )
        if source_effects.selected is not bank_service.selected:
            raise ValueError("bank writes selected a different original target")
        self.source = source_effects
        self._original_source = source_effects
        self._original_owners = (
            source_effects.realization,
            source_effects.selected,
            source_effects.dma,
        )
        self.banks = CheckedBankService(bank_service, source_effects.selected)
        self.path = PlainMvinBankPath(source_effects.dma, self.banks)
        self._issued_path, self._issued_banks = self.path, self.banks
        self.sources = {
            "implementation": _pin(__file__),
            "path": _pin(PlainMvinBankPath.__init__.__code__.co_filename),
            "generic_floor": _pin(compose_service_floor.__code__.co_filename),
        }

    def revalidate(self):
        if self.source is not self._original_source or any(
            a is not b
            for a, b in zip(
                self._original_owners,
                (self.source.realization, self.source.selected, self.source.dma),
                strict=True,
            )
        ):
            raise ValueError("source-bank original owner selection changed")
        _bound(self.source, "revalidate", _SOURCE_REVALIDATE)()
        if (
            self.path is not self._issued_path
            or self.banks is not self._issued_banks
            or self.path.banks is not self.banks
        ):
            raise ValueError("source-bank path owner selection changed")
        _bound(self.path, "command_rows", _PATH_ROWS)
        _bound(self.path, "revalidate", _PATH_REVALIDATE)()
        if self.source.selected is not self.banks.owner.selected or any(
            _pin(pin["path"]) != pin for pin in self.sources.values()
        ):
            raise ValueError("original source-bank accounting binding changed")

    def derive(self, capsule_dir):
        self.revalidate()
        original = _bound(self.source, "derive", _SOURCE_DERIVE)(capsule_dir)
        paths = self.source.realization._paths
        invocation = invocation_record.verify(paths["trace_invocation"])
        log = Path(invocation["stderr"]["path"])
        if log.stat().st_size > 64 << 20:
            raise ValueError("original bank-write trace exceeds bounded intake")
        result = json.loads(paths["result"].read_bytes())
        elf = _pinned_native_file(result, "elf")
        spec = importlib.util.spec_from_file_location(
            "original_bank_write_framer", self.source.framer
        )
        framer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(framer)
        observed, events = framer._events(
            log.read_text(), self.source.dma.isa, framer._symbols(elf, paths["nm"])
        )
        instructions = iter(observed["instructions"])
        stream = [
            {"class": "FENCE"} if event[0] == "FENCE" else next(instructions)
            for event in events
        ]
        if next(instructions, None) is not None:
            raise ValueError("actual bank-write stream omitted original instructions")
        if [
            (c["index"], c["class"]) for c in stream if c["class"] in {"MVIN", "MVOUT"}
        ] != [(c["command_index"], c["class"]) for c in original["dma_commands"]]:
            raise ValueError(
                "bank-write stream differs from original full source resource stream"
            )
        commands, bank_events = account_stream(stream, self.path)
        bank_record = self.banks.owner.record()
        unresolved = {
            **bank_record["unresolved"],
            "event_scope": "only source-derived plain identity byte-MVIN writes mapped; all other command effects retained UNKNOWN",
            "counter_window": "functional issue trace has no selected cycle-counter bank-service join",
            "physical_events": "logical local row effects are not observed SRAM/DRAM transactions or stall timestamps",
        }
        floor = compose_service_floor(
            bank_record["resources"],
            bank_events,
            scope="emitted_schedule",
            unresolved=unresolved,
        )
        invocation_record.verify(paths["trace_invocation"])
        self.revalidate()
        return {
            "schema": "gemmini.selected_source_bank_writes.v1",
            "sources": self.sources,
            "bindings": original["bindings"],
            "selected_inputs": original["selected_inputs"],
            "fresh_execution_correspondence": original[
                "fresh_execution_correspondence"
            ],
            "bank_contract": bank_record,
            "commands": commands,
            "events": bank_events,
            "partial_schedule_floor": floor,
            "path_sources": self.path.sources,
            "counter_window": None,
            "measured_cycles": None,
            "attainment": None,
            "physical_events_observed": False,
            "convergence_eligible": False,
            "qualification_issued": False,
        }
