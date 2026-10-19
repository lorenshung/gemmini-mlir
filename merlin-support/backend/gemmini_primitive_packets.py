"""Primitive caller binding to the existing independent packet-product checker.

The generic original logical-pointer renderer owns storage order. This adapter
reopens its exact selected runtime, linked bytes and complete committed trace;
it supplies no compiler lowering or typed model-source certificate.
"""

from __future__ import annotations

import subprocess
from math import prod
from pathlib import Path

import numpy as np
from gemmini_execute_state import ExecuteStateContract
from gemmini_packet_contractions import check_packets
from gemmini_primitive_host_scope import check_host_output_writes
from gemmini_source_packet_artifacts import _cpu_writes
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.runtime import harness_render
from merlin.targetgen import target_registry
from merlin.targetgen.elf_lanes import _section_headers


def check_primitive_packets(owner, arm, contraction, values):
    """Bind current original primitive ABI/bytes to every actual device product."""
    from gemmini_cleanup_issue import checked_qualifier

    observed = invocation_record.verify(Path(arm["issue_invocation"]["path"]))
    log = Path(observed["stderr"]["path"]).read_text()
    if len(log.encode()) > 64 << 20:
        raise ValueError(
            "primitive commit trace exceeds the original bounded trace budget"
        )
    selected, cb = owner.selected, arm["command_buffer"]
    # This is the original loaded renderer, never a second harness. Its actual
    # completed build must name these exact sources. Actual linked input bytes,
    # CPU writes and every DMA/product/output below remain independent checks.
    build_sources = arm["build_record"].get("source_pins", [])
    renderer_pins = {
        name: _pin(module_source_path(name))
        for name in (
            "merlin.runtime.harness_render",
            "merlin.runtime.direct_kernel_harness",
            "merlin.runtime.backends.chipyard_rocc_bound",
        )
    }
    if any(pin not in build_sources for pin in renderer_pins.values()):
        raise ValueError(
            "primitive original completed build does not bind its loaded neutral pointer renderer"
        )
    contract_path = target_registry.resolve("gemmini").capability_contract_path
    if (
        _pin(contract_path)["sha256"]
        != owner.arithmetic.record()["sources"]["software_spec"]["sha256"]
    ):
        raise ValueError(
            "primitive actual caller selects another original software contract"
        )
    contract = target_registry.resolve("gemmini").load_contract()
    harness_render.validate_contract(contract)
    # The retained effective buffer is canonical sorted JSON. Its insertion
    # order is not the original implicit pointer ABI. The independently pinned
    # original third compiler entrypoint supplies that order; only this mapping
    # view is reordered, never the retained buffer or any executable.
    if set(arm["caller_order"]) != set(cb["tensors"]) or len(
        arm["caller_order"]
    ) != len(cb["tensors"]):
        raise ValueError("primitive original caller pointer order is incomplete")
    caller_cb = dict(
        cb, tensors={name: cb["tensors"][name] for name in arm["caller_order"]}
    )
    normalized, _mutable = harness_render._roster(caller_cb)
    abi = {
        f"tensor_{i}": (row["tensor"], cb["tensors"][row["tensor"]])
        for i, row in enumerate(normalized["kernel_abi"]["args"])
    }
    checker = checked_qualifier(owner.cleanup)
    observer = selected.operand_observer()
    symbols = checker._symbols(Path(arm["elf"]), owner.nm)
    isa = observer.isa_constants("gemmini")
    trace, events = checker._events(log, isa, symbols)
    commands = iter(trace["instructions"])
    stream, decoded = [], []
    for event in events:
        command = None if event[0] == "FENCE" else next(commands)
        if command is not None and command["class"] != event[0]:
            raise ValueError("primitive actual command/fence framing changed")
        stream.append({"class": "FENCE"} if command is None else command)
        decoded.append((event, command))
    if next(commands, None) is not None:
        raise ValueError("primitive actual packet framing is incomplete")
    elf_bytes = Path(arm["elf"]).read_bytes()
    if elf_bytes[5] != 1:
        raise ValueError("primitive linked inputs need selected little-endian storage")
    sections = _section_headers(elf_bytes)
    writes = _cpu_writes(log, isa["CUSTOM_OPCODE"])
    text_symbols = subprocess.run(
        [str(owner.nm), "-S", "--defined-only", str(arm["elf"])],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    ).stdout
    text = {}
    for line in text_symbols.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in {"t", "T", "w", "W"}:
            text.setdefault(parts[3], []).append((int(parts[0], 16), int(parts[1], 16)))
    entry_name = contract["harness_abi"]["entry_symbol"]
    if any(
        len(text.get(name, [])) != 1 or text[name][0][1] <= 0
        for name in ("main", entry_name)
    ):
        raise ValueError(
            "primitive original linked caller/entry executable extents are missing or ambiguous"
        )
    output_names = {
        physical
        for physical, (logical, _spec) in abi.items()
        if logical == contraction.output
    }
    output_ranges = [
        (start, size) for start, size, physical in symbols if physical in output_names
    ]
    if len(output_ranges) != 1:
        raise ValueError(
            "primitive complete output has no unique original linked storage"
        )
    host_scope = check_host_output_writes(
        log,
        entry=text[entry_name][0][0],
        main=text["main"][0],
        outputs=output_ranges,
        custom_opcode=isa["CUSTOM_OPCODE"],
    )

    def contents(address, width):
        return [
            elf_bytes[offset + address - start : offset + address - start + width]
            for _name, kind, _flags, offset, size, start in sections
            if kind == 1 and start <= address and address + width <= start + size
        ]

    def written(address, width):
        return any(
            start < address + width and address < start + size for start, size in writes
        )

    for name, value in values.items():
        names = [
            physical for physical, (logical, _spec) in abi.items() if logical == name
        ]
        found = [
            (start, size) for start, size, physical in symbols if physical in names
        ]
        expected = np.asarray(value, dtype=np.int8).tobytes()
        if len(found) != 1 or len(expected) != prod(cb["tensors"][name]["shape"]):
            raise ValueError(
                "primitive input has no unique complete linked pointer ABI"
            )
        start, size = found[0]
        if (
            len(expected) > size
            or written(start, len(expected))
            or contents(start, len(expected)) != [expected]
        ):
            raise ValueError(
                "primitive actual input bytes changed or were written by the host/device"
            )

    def memory(address, width):
        found = [
            (start, size, name)
            for start, size, name in symbols
            if start <= address < start + size
        ]
        if len(found) != 1 or found[0][2] not in abi:
            raise ValueError(
                "primitive DMA lacks unique original linked tensor ownership"
            )
        start, _size, physical = found[0]
        name, spec = abi[physical]
        offset = address - start
        if (
            width != {"i8": 1, "i32": 4}.get(spec["dtype"])
            or offset % width
            or offset // width >= prod(spec["shape"])
        ):
            raise ValueError("primitive DMA exceeds its complete logical tensor")
        return name, offset // width

    def zero(address, width):
        found = [
            (start, size, name)
            for start, size, name in symbols
            if start <= address and address + width <= start + size
        ]
        return (
            len(found) == 1
            and found[0][2] not in abi
            and not written(address, width)
            and contents(address, width) == [bytes(width)]
        )

    fields = {
        (row["bundle"], row["name"]): row
        for row in selected._documents()[2]["layouts"]["fields"]
    }

    def field(bundle, name, raw):
        entry = fields[bundle, name]
        if entry["status"] != "derived":
            raise ValueError("primitive packet field lacks selected source evidence")
        return raw >> entry["offset"] & ((1 << entry["width"]) - 1)

    def shape(bundle, raw):
        return field(bundle, "num_rows", raw), field(bundle, "num_cols", raw)

    geometry = selected.report()["geometry"]
    local = fields["PreloadRs", "local_addr"]
    subfields = {row["field"]: row for row in local["subfields"]}
    ex = ExecuteStateContract(selected)

    def load(command):
        raw, actual = command["rs1"]["raw"], command["decoded"]
        if (
            actual["state_id"] != 0
            or actual["scale_bits"] != 0x3F800000
            or field("ConfigMvinRs1", "shrink", raw) != 0
            or field("ConfigMvinRs1", "pixel_repeats", raw) != 0
            or field("ConfigMvinRs1", "stride", raw) != geometry["array_cols"]
        ):
            raise ValueError("primitive load lacks selected identity/unshrunk effects")

    stores = {row["name"]: row for row in geometry["stores"]}
    checked = check_packets(
        stream,
        contractions=(contraction,),
        dim=geometry["array_rows"],
        spad_rows=stores["scratchpad"]["banks"] * stores["scratchpad"]["rows_per_bank"],
        acc_rows=stores["accumulator"]["banks"]
        * stores["accumulator"]["rows_per_bank"],
        address_bits=local["width"],
        data_bits=subfields["data"]["width"],
        accumulator_flag=1 << subfields["is_acc_addr"]["offset"],
        accumulate_flag=1 << subfields["accumulate"]["offset"],
        full_width_flag=1 << subfields["read_full_acc_row"]["offset"],
        memory_cell=memory,
        zero_cell=zero,
        execute_contract=ex.identity_ws,
        weight_shape=ex.weight_shape,
        compute_shape=lambda raw: shape("ComputeRs", raw),
        preload_shape=lambda raw: shape("PreloadRs", raw),
        load_configuration=load,
    )
    if (
        harness_render._roster(caller_cb)[0] != normalized
        or any(_pin(pin["path"]) != pin for pin in renderer_pins.values())
        or _pin(contract_path)["sha256"]
        != owner.arithmetic.record()["sources"]["software_spec"]["sha256"]
    ):
        raise ValueError(
            "primitive original caller storage/source changed during actual packet proof"
        )
    invocation_record.verify(Path(arm["issue_invocation"]["path"]))
    rhs = [name for name, (logical, _spec) in abi.items() if logical == contraction.rhs]
    if len(rhs) != 1:
        raise ValueError(
            "primitive immutable RHS has ambiguous original linked ABI ownership"
        )
    return {**checked, "original_kernel_host_output_scope": host_scope}, decoded, rhs[0]
