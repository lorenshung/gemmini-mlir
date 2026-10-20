"""Original linked memory and complete commit-stream evidence for packet checks."""

from __future__ import annotations

import importlib.util
from math import prod
from pathlib import Path

import numpy as np
from gemmini_execute_state import ExecuteStateContract
from gemmini_packet_contractions import Contraction, check_packets
from merlin.common import invocation_record
from merlin.llvmlower.source_contraction_bindings import bind_rank_two_contractions
from merlin.targetgen.elf_lanes import _section_headers


def _cpu_writes(log, custom_opcode):
    result = []
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("packet source scope requires one observed hart")
        word = int(parts[4].strip("()"), 16)
        for i, part in enumerate(parts[:-2]):
            if part != "mem" or not all(
                token.startswith("0x") for token in parts[i + 1 : i + 3]
            ):
                continue
            opcode = word & 127
            if opcode == custom_opcode:
                if len(parts[i + 2]) != 4 or int(parts[i + 2], 16) > 255:
                    raise ValueError(
                        "selected receiver custom write is not a complete byte observation"
                    )
                result.append((int(parts[i + 1], 16), 1))
                continue
            scalar = (
                opcode == 0x23
                and word >> 12 & 7 <= 3
                or opcode in {0x27, 0x2F}
                and word >> 12 & 7 in {2, 3}
                or word & 3 in {0, 2}
                and word >> 13 & 7 in {5, 6, 7}
            )
            if not scalar:
                raise ValueError(
                    "unresolved host store width prevents source zero-storage proof"
                )
            result.append(
                (int(parts[i + 1], 16), max(8, (len(parts[i + 2]) - 2 + 1) // 2))
            )
    return result


def check_execution_packets(
    *,
    selected,
    source,
    proof,
    command_buffer,
    elf,
    invocation,
    nm,
    input_values,
    expected_trace_argv,
):
    """Reopen an original completed trace, then reconstruct actual required work."""
    contractions = tuple(
        Contraction(**row)
        for row in bind_rank_two_contractions(source, command_buffer, proof)
    )
    abi = {
        f"tensor_{i}": (row["tensor"], command_buffer["tensors"][row["tensor"]])
        for i, row in enumerate(command_buffer["kernel_abi"]["args"])
    }
    return _check_bound_execution_packets(
        selected=selected,
        elf=elf,
        invocation=invocation,
        nm=nm,
        input_values=input_values,
        expected_trace_argv=expected_trace_argv,
        contractions=contractions,
        abi=abi,
        command_buffer=command_buffer,
    )


def check_direct_execution_packets(
    *,
    selected,
    source,
    proof,
    command_buffer,
    elf,
    invocation,
    nm,
    input_values,
    expected_trace_argv,
    entry_symbol,
    zero_fill=None,
):
    """Use the exact typed direct-source binding; never invent a global plan."""
    from gemmini_load_streams import SelectedLoadStreams
    from merlin.llvmlower.direct_source_contraction import (
        bind_direct_source_contraction,
    )

    load_streams = SelectedLoadStreams(selected)
    if zero_fill is not None:
        from gemmini_zero_fill import SelectedAccumulatorZeroFill

        if (
            type(zero_fill) is not SelectedAccumulatorZeroFill
            or zero_fill.selected is not selected
        ):
            raise TypeError(
                "direct zero fill requires its original live selected owner"
            )
        zero_fill.revalidate()

    contraction = Contraction(
        **bind_direct_source_contraction(source, command_buffer, proof)
    )
    abi = {
        f"tensor_{i}": (name, command_buffer["tensors"][name])
        for i, name in enumerate(command_buffer["compiler_pointer_abi"]["arguments"])
    }
    report = _check_bound_execution_packets(
        selected=selected,
        elf=elf,
        invocation=invocation,
        nm=nm,
        input_values=input_values,
        expected_trace_argv=expected_trace_argv,
        contractions=(contraction,),
        abi=abi,
        command_buffer=command_buffer,
        load_streams=load_streams,
        zero_fill=zero_fill,
    )
    # Complete products/readouts are established first. The same full trace
    # must additionally veto host answer stores during the one real call.
    from gemmini_direct_host_scope import linked_code_extents
    from gemmini_primitive_host_scope import check_host_output_writes

    checker_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/shared_load_configuration/qualify.py"
    )
    spec = importlib.util.spec_from_file_location(
        "original_direct_host_framing", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    symbols = checker._symbols(Path(elf), Path(nm))
    code = linked_code_extents(Path(elf), Path(nm), entry_symbol)

    def symbol(name):
        rows = [(start, size) for start, size, label in symbols if label == name]
        if len(rows) != 1 or rows[0][1] <= 0:
            raise ValueError("direct source scope has no unique complete linked symbol")
        return rows[0]

    trace = invocation_record.verify(Path(invocation))
    report["host_output_scope"] = check_host_output_writes(
        Path(trace["stderr"]["path"]).read_text(),
        entry=code[entry_symbol][0],
        main=code["main"],
        outputs=[
            symbol(physical)
            for physical, (logical, _tensor) in abi.items()
            if logical == contraction.output
        ],
        custom_opcode=selected.operand_observer().isa_constants("gemmini")[
            "CUSTOM_OPCODE"
        ],
    )
    invocation_record.verify(Path(invocation))
    report["selected_load_streams"] = load_streams.record()
    if zero_fill is not None:
        report["selected_accumulator_zero_fill"] = zero_fill.record()
    return report


def _check_bound_execution_packets(
    *,
    selected,
    elf,
    invocation,
    nm,
    input_values,
    expected_trace_argv,
    contractions,
    abi,
    command_buffer,
    load_streams=None,
    zero_fill=None,
):
    declaration = invocation_record.verify(Path(invocation))
    if declaration["argv"] != expected_trace_argv:
        raise ValueError(
            "actual command trace belongs to a different ELF/runtime/observation invocation"
        )
    if Path(declaration["stderr"]["path"]).stat().st_size > 64 << 20:
        raise ValueError("packet trace exceeds its independent byte budget")
    log = Path(declaration["stderr"]["path"]).read_text()
    checker_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/shared_load_configuration/qualify.py"
    )
    spec = importlib.util.spec_from_file_location(
        "original_packet_framing", checker_path
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    observer = selected.operand_observer()
    symbols = checker._symbols(Path(elf), Path(nm))
    trace, events = checker._events(log, observer.isa_constants("gemmini"), symbols)
    stream = []
    instructions = iter(trace["instructions"])
    for event in events:
        if event[0] == "FENCE":
            stream.append({"class": "FENCE"})
        else:
            command = next(instructions)
            if command["class"] != event[0]:
                raise ValueError("actual packet/fence framing changed")
            stream.append(command)
    if next(instructions, None) is not None:
        raise ValueError("actual packet framing is incomplete")
    sections = _section_headers(Path(elf).read_bytes())
    elf_bytes = Path(elf).read_bytes()
    writes = _cpu_writes(log, observer.isa_constants("gemmini")["CUSTOM_OPCODE"])
    observed_zeros = ()
    if load_streams is not None:
        from gemmini_private_zero_storage import observed_private_zero_storage

        observed_zeros = observed_private_zero_storage(
            log,
            symbols=symbols,
            abi_symbols=abi,
            custom_opcode=observer.isa_constants("gemmini")["CUSTOM_OPCODE"],
        )
    if elf_bytes[5] != 1:
        raise ValueError(
            "selected source packet scope requires explicit little-endian ELF data"
        )
    for name, values in input_values.items():
        names = [
            physical for physical, (logical, _tensor) in abi.items() if logical == name
        ]
        found = [
            (start, size) for start, size, physical in symbols if physical in names
        ]
        tensor = command_buffer["tensors"][name]
        dtype = {"i8": "i1", "i32": "<i4"}.get(tensor["dtype"])
        if len(found) != 1 or dtype is None:
            raise ValueError(
                "source input has no unique supported actual linked storage"
            )
        start, size = found[0]
        expected = np.asarray(values, dtype=dtype).tobytes()
        if (
            len(expected) != prod(tensor["shape"]) * np.dtype(dtype).itemsize
            or len(expected) > size
            or any(a < start + len(expected) and start < a + w for a, w in writes)
        ):
            raise ValueError(
                "actual source entry bytes are clobbered or have an incomplete ABI"
            )
        actual = [
            elf_bytes[
                offset + start - address : offset + start - address + len(expected)
            ]
            for _name, kind, _flags, offset, length, address in sections
            if kind == 1
            and address <= start
            and start + len(expected) <= address + length
        ]
        if actual != [expected]:
            raise ValueError(
                "actual linked source inputs differ from independent sealed input values"
            )

    def memory(address, width):
        found = [
            (start, size, name)
            for start, size, name in symbols
            if start <= address < start + size
        ]
        if len(found) != 1 or found[0][2] not in abi:
            raise ValueError("actual DMA lacks unique original linked ABI ownership")
        start, _size, name = found[0]
        logical, tensor = abi[name]
        expected = {"i8": 1, "i32": 4}.get(tensor["dtype"])
        offset = address - start
        if (
            width != expected
            or offset % width
            or offset // width >= prod(tensor["shape"])
        ):
            raise ValueError("actual DMA exceeds the complete original logical tensor")
        return logical, offset // width

    def zero(address, width):
        found = [
            (start, size, name)
            for start, size, name in symbols
            if start <= address and address + width <= start + size
        ]
        if len(found) != 1 or found[0][2] in abi:
            return False
        if found[0] in observed_zeros:
            return True
        if any(
            start < address + width and address < start + size for start, size in writes
        ):
            return False
        for _name, kind, _flags, offset, size, start in sections:
            if kind == 1 and start <= address and address + width <= start + size:
                begin = offset + address - start
                return elf_bytes[begin : begin + width] == bytes(width)
        return False

    ex = ExecuteStateContract(selected)
    fields = {
        (row["bundle"], row["name"]): row
        for row in selected._documents()[2]["layouts"]["fields"]
    }

    def value(bundle, name, raw):
        field = fields[bundle, name]
        if field["status"] != "derived":
            raise ValueError("actual instruction field lacks selected source evidence")
        return raw >> field["offset"] & ((1 << field["width"]) - 1)

    def shape(bundle, raw):
        return value(bundle, "num_rows", raw), value(bundle, "num_cols", raw)

    geometry = selected.report()["geometry"]
    local = fields["PreloadRs", "local_addr"]
    subfields = {row["field"]: row for row in local["subfields"]}

    def flag(name):
        return 1 << subfields[name]["offset"]

    def load(command):
        raw, decoded = command["rs1"]["raw"], command["decoded"]
        if (
            (
                decoded["state_id"] != 0
                if load_streams is None
                else decoded["state_id"] not in load_streams.states
            )
            or decoded["scale_bits"] != 0x3F800000
            or value("ConfigMvinRs1", "shrink", raw) != 0
            or (
                value("ConfigMvinRs1", "pixel_repeats", raw) != 0
                if load_streams is None
                else value("ConfigMvinRs1", "pixel_repeats", raw) not in (0, 1)
            )
            or value("ConfigMvinRs1", "stride", raw) != geometry["array_cols"]
        ):
            raise ValueError(
                "actual load is outside selected identity/unshrunk source effects"
            )

    stores = {row["name"]: row for row in geometry["stores"]}
    report = check_packets(
        stream,
        contractions=contractions,
        dim=geometry["array_rows"],
        spad_rows=stores["scratchpad"]["banks"] * stores["scratchpad"]["rows_per_bank"],
        acc_rows=stores["accumulator"]["banks"]
        * stores["accumulator"]["rows_per_bank"],
        address_bits=local["width"],
        data_bits=subfields["data"]["width"],
        accumulator_flag=flag("is_acc_addr"),
        accumulate_flag=flag("accumulate"),
        full_width_flag=flag("read_full_acc_row"),
        memory_cell=memory,
        zero_cell=zero,
        execute_contract=ex.identity_ws,
        weight_shape=ex.weight_shape,
        compute_shape=lambda raw: shape("ComputeRs", raw),
        preload_shape=lambda raw: shape("PreloadRs", raw),
        load_configuration=load,
        load_streams=None if load_streams is None else load_streams.ports,
        zero_fill=None if zero_fill is None else zero_fill.accepts,
    )
    invocation_record.verify(Path(invocation))
    if load_streams is not None:
        report["observed_private_zero_storage"] = [
            {
                "address": start,
                "bytes": size,
                "symbol": name,
                "evidence": "complete committed zero stores before first device command; no later clobber",
            }
            for start, size, name in observed_zeros
        ]
    return report
