"""Complete observed private zero storage; never infer initial NOBITS bytes."""

from gemmini_primitive_host_scope import _zero_output_store
from gemmini_source_packet_artifacts import _cpu_writes


def observed_private_zero_storage(log, *, symbols, abi_symbols, custom_opcode):
    """Require full zero stores before the first device command and no clobber.

    Reads with no stored memory value are not writes. Every possible scalar
    store must nevertheless expose a complete address/value observation.
    This conservative whole-program rule deliberately refuses late setup.
    """
    if type(custom_opcode) is not int or not 0 < custom_opcode < 128:
        raise ValueError("private zero storage needs the original selected opcode")
    private = [
        row for row in symbols if row[2] not in abi_symbols and 0 < row[1] <= 4096
    ]
    if len(private) > 256 or len({(a, n) for a, n, _ in private}) != len(private):
        raise ValueError("private zero-storage symbol roster is ambiguous or oversized")
    covered = {row: set() for row in private}
    bad, started = set(), False
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("private zero storage requires the original single hart")
        word = int(parts[4].strip("()"), 16)
        custom = word & 127 == custom_opcode
        if custom:
            started = True
        else:
            may_store = (
                word & 3 == 3
                and word & 127 in {0x23, 0x27, 0x2F}
                or word & 3 in {0, 2}
                and word >> 13 & 7 in {5, 6, 7}
            )
            memory = [i for i, t in enumerate(parts) if t == "mem"]
            if may_store and (
                len(memory) != 1
                or memory[0] + 2 >= len(parts)
                or not all(
                    t.startswith("0x") for t in parts[memory[0] + 1 : memory[0] + 3]
                )
            ):
                raise ValueError("private zero store has unresolved committed effects")
        for address, width in _cpu_writes(line, custom_opcode):
            for row in private:
                start, size, _name = row
                if address >= start + size or address + width <= start:
                    continue
                if started or custom:
                    bad.add(row)
                    continue
                try:
                    exact, extent = _zero_output_store(word, parts)
                except ValueError:
                    bad.add(row)
                    continue
                if not start <= exact or exact + extent > start + size:
                    bad.add(row)
                else:
                    covered[row].update(range(exact, exact + extent))
    if not started:
        raise ValueError("private zero storage lacks any original device command")
    return tuple(
        row for row in private if row not in bad and len(covered[row]) == row[1]
    )
