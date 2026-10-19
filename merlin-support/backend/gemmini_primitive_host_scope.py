"""Actual primitive call scope, without treating setup writes as execution.

This selected RISC-V caller proof supplies no cycle attribution. Kernel callees
stay inside the invocation until the exact original main-call return address.
"""

from gemmini_source_packet_artifacts import _cpu_writes


def check_host_output_writes(log, *, entry, main, outputs, custom_opcode):
    previous, returning, entries, completed, retired = None, None, 0, 0, 0
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("primitive invocation requires one original observed hart")
        pc, word = int(parts[3], 16), int(parts[4].strip("()"), 16)
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
            if word & 127 != custom_opcode:
                for address, width in _cpu_writes(line, custom_opcode):
                    if any(
                        start < address + width and address < start + size
                        for start, size in outputs
                    ):
                        raise ValueError(
                            "primitive CPU overwrites a required device output during the actual kernel call"
                        )
        registers = []
        for index, token in enumerate(parts[:-1]):
            if (
                token.startswith("x")
                and token[1:].isdigit()
                and parts[index + 1].startswith("0x")
            ):
                registers.append((token, int(parts[index + 1], 16)))
        previous = pc, word, registers
    if entries != 1 or completed != 1 or returning is not None or retired == 0:
        raise ValueError(
            "primitive original main-to-kernel invocation did not close exactly once"
        )
    return {
        "scope": "one actual original main call through the same caller return",
        "kernel_cpu_output_writes": 0,
        "kernel_retired_instructions_observed": retired,
        "target_cycles": None,
        "timing_authority": "NONE",
    }
