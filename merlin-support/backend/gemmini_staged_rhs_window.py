"""Original complete functional-program scratch epoch for RHS placement.

This issues a source-bound functional allocation premise, not ownership of an
entire real device. Capacity alone and copied qualification JSON cannot create
it. Every new actual baseline must repeat complete source/native/packet checks.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from gemmini_source_execution import SourceExecutionOwner
from gemmini_source_packet_artifacts import _cpu_writes
from gemmini_staged_dense_capability import _digest
from gemmini_staged_rhs_placement import assigned_window
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin_experiments.phase0.connected_integer_qualification import source_proof
from merlin_experiments.phase2.contracts import exact_tree_record

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.llvmlower.source_contraction_bindings import bind_rank_two_contractions


def check_original_footprint(stream, *, lhs, rhs, dim, data_bits, physical_rows):
    """All actual commands in the complete single-hart program need one owner.

    The original caller must already prove all source products, immutable input
    bytes, zero/continuation/readout and complete outputs for this exact stream.
    This function does not issue an execution or allocator capability itself.
    """
    if (
        lhs == rhs
        or any(type(v) is not int or v <= 0 for v in (dim, data_bits, physical_rows))
        or physical_rows > 1 << data_bits
        or physical_rows < 2 * dim
    ):
        raise ValueError("scratch epoch lacks selected physical addressing or distinct source owners")
    mask = (1 << data_bits) - 1
    counts = {}
    data = False
    for event, command in stream:
        kind = event[0]
        counts[kind] = counts.get(kind, 0) + 1
        fields = {} if command is None else command["decoded"]
        if kind in {"FLUSH", "CONFIG_EX"}:
            if data or counts[kind] > 1:
                raise ValueError("scratch epoch crosses another device ownership or configuration epoch")
        elif kind == "MVIN":
            data = True
            address = fields["addr"] & mask
            if fields.get("is_acc_addr") is True:
                if address != 0:
                    raise ValueError("scratch epoch changes its original sole accumulator stripe")
            elif (event[1][0], address) not in {(lhs, 0), (rhs, dim)} or fields["rows"] > dim or fields["cols"] > dim:
                raise ValueError("scratch epoch contains a foreign or differently placed live operand owner")
        elif kind == "PRELOAD":
            if event[1] & mask != dim:
                raise ValueError("scratch epoch has an unowned original PRELOAD window")
        elif kind == "COMPUTE_PRELOADED":
            if event[1] & mask != 0:
                raise ValueError("scratch epoch changes its original LHS window")
        elif kind == "MVOUT":
            if event[2] & mask != 0:
                raise ValueError("scratch epoch changes its original sole readout stripe")
        elif kind not in {"FENCE", "CONFIG_LD", "CONFIG_ST"}:
            raise ValueError("scratch epoch contains a foreign command or retained device owner")
    if (
        not stream
        or stream[0][0][0] != "FENCE"
        or stream[-1][0][0] != "FENCE"
        or counts.get("CONFIG_EX") != 1
        or not counts.get("PRELOAD")
        or not counts.get("MVOUT")
    ):
        raise ValueError("scratch epoch is not a complete original initialized/fenced contraction program")
    return dict(
        scope="one_complete_observed_hart0_functional_program",
        original_operand_windows=[[0, dim], [dim, 2 * dim]],
        original_accumulator_window=[0, dim],
        maximum_private_physical_rows=physical_rows,
        address_data_bits=data_bits,
        command_counts=counts,
        no_other_device_owner="all complete issued commands checked; current source proof required",
        hardware_execution_qualified=False,
        timing_authority="NONE",
    )


class ClosedScratchEpochOwner:
    """Derive a finite allocation epoch from original current execution evidence."""

    def __init__(self, source_owner, *, capsule_dir):
        if type(source_owner) is not SourceExecutionOwner:
            raise ValueError("scratch epoch needs the original live current-source execution owner")
        self.source_owner = source_owner
        self._owner_identity = source_owner.identity
        self.cap = Path(capsule_dir)
        self._capsule = exact_tree_record(self.cap)
        source = self.cap / "capsule.interface.mlir"
        proof = source_proof(yaml.safe_load((self.cap / "capsule.yaml").read_bytes()), self.cap)
        self._checked = source_owner.verify_source_execution(source, proof, self.cap)
        self._rule = source_owner.source_rule()
        import json

        cb = json.loads(source_owner._paths["command_buffer"].read_bytes())
        contractions = bind_rank_two_contractions(source, cb, proof)
        if (
            len(contractions) != 1
            or contractions[0]["rhs"] not in cb["params"]["global_program_plan"]["entry_bindings"]
        ):
            raise ValueError("scratch epoch requires one original external-immutable-RHS contraction")
        self._geometry = {name: contractions[0][name] for name in ("m", "k", "n")}
        arguments = [row["tensor"] for row in cb["kernel_abi"]["args"]]
        if len(set(arguments)) != len(arguments):
            raise ValueError("scratch epoch has aliased logical ABI owners")
        lhs, rhs = ("tensor_" + str(arguments.index(contractions[0][role])) for role in ("lhs", "rhs"))
        selected = source_owner._selected
        geometry = selected.report()["geometry"]
        store = next(row for row in geometry["stores"] if row["name"] == "scratchpad")
        rows = store["banks"] * store["rows_per_bank"]
        fields = selected._documents()[2]["layouts"]["fields"]
        local = next(field for field in fields if field["bundle"] == "PreloadRs" and field["name"] == "local_addr")
        data = next(field for field in local["subfields"] if field["field"] == "data")
        if local["status"] != "derived" or data["offset"] != 0:
            raise ValueError("scratch epoch lacks independently derived selected local-address fields")
        observed = invocation_record.verify(source_owner._paths["trace_invocation"])
        log = Path(observed["stderr"]["path"]).read_text()
        import importlib.util

        from merlin.targetgen.native_model_execution import _pinned_native_file

        framing = source_owner._pins["packet_framing_checker"]
        spec = importlib.util.spec_from_file_location("original_scratch_epoch_framing", framing["path"])
        qualify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qualify)

        native = json.loads(source_owner._paths["result"].read_bytes())
        isa = selected.operand_observer().isa_constants("gemmini")
        _cpu_writes(log, isa["CUSTOM_OPCODE"])  # rejects every observed nonzero hart
        trace, events = qualify._events(
            log, isa, qualify._symbols(_pinned_native_file(native, "elf"), source_owner._paths["nm"])
        )
        commands = iter(trace["instructions"])
        stream = [(event, None if event[0] == "FENCE" else next(commands)) for event in events]
        self._footprint = check_original_footprint(
            stream, lhs=lhs, rhs=rhs, dim=geometry["array_rows"], data_bits=data["width"], physical_rows=rows
        )
        self._stream = _digest(stream)
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_staged_rhs_window",
                "gemmini_staged_rhs_placement",
                "gemmini_source_packet_artifacts",
                "merlin_experiments.phase0.connected_integer_qualification",
                "merlin.llvmlower.source_contraction_bindings",
            )
        }
        self._identity = self._digest()
        self.revalidate()

    def _digest(self):
        return _digest(
            [
                self._owner_identity,
                self._capsule,
                self._checked,
                self._geometry,
                self._footprint,
                self._stream,
                self._sources,
                self._rule,
            ]
        )

    @property
    def identity(self):
        return "gemmini.closed_scratch_epoch.v1:" + self._identity

    def revalidate(self):
        self.source_owner.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.source_owner.identity != self._owner_identity
            or exact_tree_record(self.cap) != self._capsule
            or self._digest() != self._identity
        ):
            raise ValueError("original complete-program scratch ownership or source/runtime evidence changed")

    def record(self):
        self.revalidate()
        return dict(
            identity=self.identity,
            source_owner_identity=self._owner_identity,
            source_geometry=self._geometry,
            footprint=copy.deepcopy(self._footprint),
            actual_stream_sha256=self._stream,
            original_source_execution=copy.deepcopy(self._checked),
            sources=copy.deepcopy(self._sources),
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )

    def assign(self, operations, *, problem, contract):
        self.revalidate()
        if (
            {name: getattr(problem, name) for name in ("m", "k", "n")} != self._geometry
            or contract.hardware != self.source_owner._selected_inputs["facts"]["sha256"]
            or contract.dim != self._footprint["original_operand_windows"][0][1]
            or contract.operand_bits != self._rule["operand_bits"]
            or contract.primitive_bits != self._rule["primitive_mac_bits"]
            or contract.accumulator_bits != self._rule["accumulator_bits"]
        ):
            raise ValueError("scratch assignment requires this complete original source epoch")
        assigned, proof = assigned_window(
            operations,
            problem=problem,
            contract=contract,
            original_owned_rows=self._footprint["maximum_private_physical_rows"],
        )
        self.revalidate()
        return assigned, {**proof, "original_complete_program_owner": self.identity}
