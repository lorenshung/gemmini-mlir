"""Separate bounded linked code extents for the original actual-call veto."""

import subprocess
from pathlib import Path

from merlin.targetgen.elf_lanes import _section_headers


def select_code_extents(text, sections, entry_symbol):
    if type(entry_symbol) is not str or not entry_symbol or entry_symbol == "main":
        raise ValueError("direct source entry must name its original distinct kernel")
    symbols = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2] in {"t", "T", "w", "W"}:
            symbols.setdefault(fields[3], []).append(
                (int(fields[0], 16), int(fields[1], 16))
            )
    result = {}
    for name in (entry_symbol, "main"):
        rows = symbols.get(name, [])
        if len(rows) != 1 or rows[0][1] <= 0:
            raise ValueError("direct source code extent is absent or ambiguous")
        address, size = rows[0]
        containing = [
            row
            for row in sections
            if row[1] == 1
            and row[2] & 4
            and row[5] <= address
            and address + size <= row[5] + row[4]
        ]
        if len(containing) != 1:
            raise ValueError(
                "direct source code extent lacks unique complete executable storage"
            )
        result[name] = (address, size)
    return result


def linked_code_extents(elf, nm, entry_symbol):
    text = subprocess.run(
        [str(nm), "-S", "--defined-only", str(elf)],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    ).stdout
    if len(text.encode()) > 4 << 20:
        raise ValueError("direct source code symbol roster exceeds its finite bound")
    return select_code_extents(
        text, _section_headers(Path(elf).read_bytes()), entry_symbol
    )
