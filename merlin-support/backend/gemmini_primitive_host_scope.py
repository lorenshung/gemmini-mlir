"""Actual primitive call scope, without treating setup writes as execution.

This selected RISC-V caller proof supplies no cycle attribution. Kernel callees
stay inside the invocation until the exact original main-call return address.
"""

from gemmini_source_packet_artifacts import _cpu_writes


def _zero_output_store(word, parts):
    """Exact integer SW/SD or C.SW/C.SD/SP width and committed zero bytes."""
    if word & 3 == 3 and word & 127 == 0x23 and word >> 12 & 7 in (2, 3):
        width = 1 << (word >> 12 & 7)
    elif word & 3 in (0, 2) and word >> 13 & 7 in (6, 7):
        width = 1 << ((word >> 13 & 7) - 4)
    else:
        raise ValueError(
            "primitive output initialization needs exact integer i32/i64 zero stores"
        )
    places = [i for i, token in enumerate(parts) if token == "mem"]
    if len(places) != 1:
        raise ValueError(
            "primitive zero store has no unique complete committed memory value"
        )
    index = places[0]
    if index + 2 >= len(parts):
        raise ValueError("primitive zero store memory observation is incomplete")
    address, value = parts[index + 1 : index + 3]
    if (
        not address.startswith("0x")
        or not value.startswith("0x")
        or len(value) != 2 + width * 2
        or int(value, 16) != 0
    ):
        raise ValueError(
            "primitive output initialization lacks exact complete zero bytes"
        )
    return int(address, 16), width


def check_host_output_writes(
    log,
    *,
    entry,
    main,
    outputs,
    custom_opcode,
    allow_complete_zero_initialization=False,
    commands=None,
):
    # The optional caller must first prove every product and complete readout
    # from this same decoded actual stream. This function only adds CPU order
    # and zero-byte coverage, never source/product or completion authority.
    if type(allow_complete_zero_initialization) is not bool:
        raise ValueError("primitive initialization selection must be explicit")
    if allow_complete_zero_initialization and (
        type(commands) is not list or not commands
    ):
        raise ValueError(
            "primitive zero initialization needs the same complete actual command stream"
        )
    command_iter = iter(commands or ())
    memory_started, zero_stores = False, []
    previous, returning, entries, completed, retired = None, None, 0, 0, 0
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("primitive invocation requires one original observed hart")
        pc, word = int(parts[3], 16), int(parts[4].strip("()"), 16)
        command = None
        if allow_complete_zero_initialization and word & 127 == custom_opcode:
            command = next(command_iter, None)
            if command is None or command.get("funct") != word >> 25:
                raise ValueError("primitive zero-init actual command framing changed")
        if pc == entry:
            entries += 1
            if returning is not None or previous is None or entries != 1:
                raise ValueError(
                    "primitive original entry has an ambiguous or repeated invocation"
                )
            call_pc, call, registers = previous
            if not main[0] <= call_pc < main[0] + main[1]:
                raise ValueError(
                    "primitive entry is not called by the original fixed main"
                )
            if call & 3 == 3 and call & 127 in (0x6F, 0x67) and call >> 7 & 31 == 1:
                return_pc = call_pc + 4
            elif call & 0xF07F == 0x9002 and call >> 7 & 31:
                return_pc = call_pc + 2
            else:
                raise ValueError(
                    "primitive entry lacks its actual original RISC-V link call"
                )
            if registers.count(("x1", return_pc)) != 1:
                raise ValueError(
                    "primitive actual caller link value is absent or inconsistent"
                )
            returning = return_pc
        if returning is not None and pc == returning:
            completed += 1
            returning = None
        elif returning is not None:
            retired += 1
            if command is not None:
                if command["class"] in {
                    "MVIN",
                    "PRELOAD",
                    "COMPUTE_PRELOADED",
                    "COMPUTE_ACCUMULATE",
                    "MVOUT",
                }:
                    memory_started = True
                elif command["class"] not in {
                    "FLUSH",
                    "CONFIG_EX",
                    "CONFIG_LD",
                    "CONFIG_ST",
                }:
                    raise ValueError(
                        "primitive zero-init order includes unsupported device effects"
                    )
            if word & 127 != custom_opcode:
                may_store = (
                    word & 3 == 3
                    and word & 127 in {0x23, 0x27, 0x2F}
                    or word & 3 in {0, 2}
                    and word >> 13 & 7 in {5, 6, 7}
                )
                memory = [i for i, token in enumerate(parts) if token == "mem"]
                if may_store and (
                    len(memory) != 1
                    or memory[0] + 2 >= len(parts)
                    or not all(
                        token.startswith("0x")
                        for token in parts[memory[0] + 1 : memory[0] + 3]
                    )
                ):
                    raise ValueError(
                        "primitive host store has unresolved committed memory effects"
                    )
                for address, width in _cpu_writes(line, custom_opcode):
                    if any(
                        start < address + width and address < start + size
                        for start, size in outputs
                    ):
                        if not allow_complete_zero_initialization or memory_started:
                            raise ValueError(
                                "primitive CPU overwrites a required device output during the actual kernel call"
                            )
                        exact_address, exact_width = _zero_output_store(word, parts)
                        if (
                            exact_address % 4
                            or exact_width not in (4, 8)
                            or not any(
                                start <= exact_address
                                and exact_address + exact_width <= start + size
                                for start, size in outputs
                            )
                        ):
                            raise ValueError(
                                "primitive zero initialization exceeds complete aligned output elements"
                            )
                        zero_stores.append((exact_address, exact_width))
        registers = []
        for index, token in enumerate(parts[:-1]):
            if (
                token.startswith("x")
                and token[1:].isdigit()
                and parts[index + 1].startswith("0x")
            ):
                registers.append((token, int(parts[index + 1], 16)))
        previous = pc, word, registers
    if allow_complete_zero_initialization and next(command_iter, None) is not None:
        raise ValueError("primitive zero-init actual command stream is incomplete")
    if entries != 1 or completed != 1 or returning is not None or retired == 0:
        raise ValueError(
            "primitive original main-to-kernel invocation did not close exactly once"
        )
    if zero_stores:
        intervals = iter(sorted(zero_stores))
        current = next(intervals, None)
        for start, size in sorted(outputs):
            cursor = start
            while current is not None and current[0] < start + size:
                address, width = current
                if address != cursor:
                    raise ValueError(
                        "primitive zero initialization has incomplete or repeated output bytes"
                    )
                cursor += width
                current = next(intervals, None)
            if cursor != start + size:
                raise ValueError(
                    "primitive zero initialization does not cover every complete output byte"
                )
        if current is not None:
            raise ValueError("primitive zero initialization has foreign output storage")
    return {
        "scope": "one actual original main call through the same caller return",
        "kernel_cpu_output_writes": len(zero_stores),
        "complete_pre_device_zero_initialization_bytes": sum(
            width for _address, width in zero_stores
        ),
        "kernel_cpu_result_overwrites": 0,
        "kernel_retired_instructions_observed": retired,
        "target_cycles": None,
        "timing_authority": "NONE",
    }
