"""Selected scratchpad shared-port ceilings, separate from DMA/timing models.

Local address cuts are not actual-command effects or reachable overlap. No saved
record issues authority; original selected files and helper sources stay pinned.
"""

from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin
from merlin.perf.capsule_roofline import compose_service_floor
from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F


def _uint(typ):
    if not isinstance(typ, tuple) or len(typ) != 2 or typ[0] != "uint":
        raise ValueError("bank service requires original unsigned scalar ports")
    return typ[1]


def _one(module, lhs, rhs):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(lhs)]
    if len(rows) != 1 or rows[0].guards or rows[0].rhs != F.parse_expr(rhs):
        raise ValueError("bank service original instance/clock wiring differs")


def _expand(module, expr, depth=0):
    if depth > 128:
        raise ValueError("bank mapping source expression exceeds finite depth")
    if type(expr) is F.Ref and not expr.segments and expr.root in module.nodes:
        return _expand(module, module.nodes[expr.root][0], depth + 1)
    if type(expr) is F.Prim:
        return F.Prim(
            expr.op, tuple(_expand(module, arg, depth + 1) for arg in expr.args)
        )
    return expr


def _form(expr):
    if type(expr) is F.Lit and expr.kind == "UInt":
        return ("UInt", expr.value)
    if type(expr) is F.Prim:
        return (expr.op, *(_form(arg) for arg in expr.args))
    return expr


def _and_atoms(expr):
    if type(expr) is F.Prim and expr.op == "and":
        return [atom for arg in expr.args for atom in _and_atoms(arg)]
    return [_form(expr)]


class BankPortCones:
    """Exhaustive Boolean input cuts for accepted row-read/write exclusion."""

    def __init__(self, module, modules):
        self.observer = C.ScalarCutpoints(module, modules=modules)
        self.row_bits = _uint(F.field_type(module.port_type("io"), ("write", "addr")))
        write_bits = _uint(F.field_type(module.port_type("io"), ("write", "data")))
        if write_bits % 8:
            raise ValueError("bank data width is not a complete byte row")
        self.row_bytes = write_bits // 8
        if any(
            _uint(F.field_type(module.port_type("io"), fields)) != 1
            for fields in (
                ("write", "ready"),
                ("write", "valid"),
                ("read", "req", "ready"),
                ("read", "req", "valid"),
            )
        ):
            raise ValueError("bank service needs single Boolean ready/valid interfaces")
        if (
            _uint(F.field_type(module.port_type("io"), ("read", "req", "bits", "addr")))
            != self.row_bits
            or _uint(
                F.field_type(module.port_type("io"), ("read", "resp", "bits", "data"))
            )
            != self.row_bytes * 8
            or not 1 <= self.row_bits <= 20
            or not 1 <= self.row_bytes <= 1024
        ):
            raise ValueError("bank read/write address/data geometry differs")

    def at(self, *, response_queue_will_empty, read_valid, write_valid):
        values = self.observer.evaluate(
            {
                "q_will_be_empty": response_queue_will_empty,
                "io.read.req.valid": read_valid,
                "io.write.valid": write_valid,
            },
            ["io.read.req.ready", "io.write.ready", "ren"],
        )
        expected_ready = response_queue_will_empty * (1 - write_valid)
        if values != {
            "io.read.req.ready": expected_ready,
            "io.write.ready": 1,
            "ren": read_valid * expected_ready,
        }:
            raise ValueError(
                "selected bank is not checked write-priority shared-port service"
            )
        accepted_read, accepted_write = (
            values["ren"],
            write_valid * values["io.write.ready"],
        )
        if accepted_read + accepted_write > 1:
            raise ValueError(
                "selected bank accepts simultaneous shared-port operations"
            )
        return {
            "read_accepted": accepted_read,
            "write_accepted": accepted_write,
            "read_ready": values["io.read.req.ready"],
            "write_ready": values["io.write.ready"],
        }

    def check(self):
        for bits in range(8):
            self.at(
                response_queue_will_empty=bits & 1,
                read_valid=(bits >> 1) & 1,
                write_valid=(bits >> 2) & 1,
            )
        return {
            "boolean_cut_cases": 8,
            "accepted_reads_plus_writes_per_cycle_upper": 1,
            "queue_reachability_proven": False,
        }


def derive_address_mapping(top, instances, row_bits):
    """Check actual mvin-return bank predicates and row-address driver trees."""
    count = len(instances)
    if not count or count & (count - 1) or count > 16:
        raise ValueError(
            "bank address mapping needs finite power-of-two original banks"
        )
    bank_bits = (count - 1).bit_length()
    if bank_bits == 0:
        raise ValueError(
            "single-bank specialization needs its own explicit source proof"
        )
    source = F.parse_expr("mvin_scale_pixel_repeater.io.resp.bits.laddr.data")
    for index, instance in enumerate(instances):
        _one(top, f"{instance}.clock", "clock")
        for field in ("write.addr", "write.valid", "read.req.valid"):
            _one(top, f"{instance}.io.{field}", f"bank_ios[{index}].{field}")
        _one(top, f"bank_ios[{index}].write.ready", f"{instance}.io.write.ready")
        rows = [
            row
            for row in top.connects
            if row.lhs == F.parse_expr(f"bank_ios[{index}].write.addr")
            and _expand(top, row.rhs)
            == F.Prim("bits", (source, F.IntArg(row_bits - 1), F.IntArg(0)))
        ]
        if len(rows) != 1 or len(rows[0].guards) != 2:
            raise ValueError("bank row mapping lacks exact original mvin-return driver")
        before, selected = rows[0].guards
        if (
            before.positive
            or before.cond != F.parse_expr(f"io.srams.write[{index}].valid")
            or not selected.positive
            or type(selected.cond) is not F.Ref
        ):
            raise ValueError("bank mapping changed original execute-write priority")
        atoms = _and_atoms(_expand(top, selected.cond))
        expected = [
            F.parse_expr("mvin_scale_pixel_repeater.io.resp.valid"),
            F.parse_expr(
                "eq(mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc, UInt<1>(0))"
            ),
            F.Prim(
                "eq",
                (
                    F.Prim(
                        "bits",
                        (
                            source,
                            F.IntArg(row_bits + bank_bits - 1),
                            F.IntArg(row_bits),
                        ),
                    ),
                    F.Lit("UInt", max(1, bank_bits), index),
                ),
            ),
            F.parse_expr(f"bank_ios[{index}].write.ready"),
        ]
        if len(atoms) != len(expected) or set(atoms) != {
            _form(expr) for expr in expected
        }:
            raise ValueError(
                "bank selection differs from original contiguous local row slices"
            )
    return {
        "source": source.text(),
        "bank_bits": [row_bits + bank_bits - 1, row_bits],
        "row_bits": [row_bits - 1, 0],
        "banks": count,
        "rows_per_bank": 1 << row_bits,
        "scope": "mvin-return local row cut; execute writes retain original priority",
    }


class ScratchpadBankService:
    def __init__(self, selected):
        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError("bank service requires original selected source owner")
        selected.revalidate()
        facts, _, _ = selected._documents()
        fir = Path(facts["inputs"]["fir_path"])
        if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
            raise ValueError("selected bank FIRRTL bytes changed")
        top = F.load_modules(fir, {"Scratchpad"})["Scratchpad"]
        names = sorted(
            name
            for name, row in top.decls.items()
            if row[0] == "inst" and name.startswith("spad_mems_")
        )
        if names != [f"spad_mems_{i}" for i in range(len(names))]:
            raise ValueError(
                "original contiguous scratchpad instance roster unavailable"
            )
        modules = F.load_modules(fir, {top.decls[name][1] for name in names})
        child_names = {
            row[1]
            for module in modules.values()
            for row in module.decls.values()
            if row[0] == "inst"
        }
        modules.update(F.load_modules(fir, child_names))
        self.cones = [
            BankPortCones(modules[top.decls[name][1]], modules) for name in names
        ]
        geometry = {(cone.row_bits, cone.row_bytes) for cone in self.cones}
        if len(geometry) != 1:
            raise ValueError(
                "bank service has nonuniform unexplained selected geometry"
            )
        self.checks = [cone.check() for cone in self.cones]
        self.mapping = derive_address_mapping(top, names, self.cones[0].row_bits)
        store = [
            row
            for row in selected.report()["geometry"]["stores"]
            if row["name"] == "scratchpad"
        ]
        if (
            len(store) != 1
            or store[0]["banks"] != len(names)
            or store[0]["rows_per_bank"] != self.mapping["rows_per_bank"]
            or store[0]["row_bytes"] != self.cones[0].row_bytes
        ):
            raise ValueError(
                "bank source geometry differs from original selected facts"
            )
        self.selected = selected
        self.sources = {
            "firrtl": _pin(fir),
            "implementation": _pin(__file__),
            "scalar_observer": _pin(C.__file__),
            "firrtl_parser": _pin(F.__file__),
            "generic_service_floor": _pin(compose_service_floor.__code__.co_filename),
        }

    def revalidate(self):
        self.selected.revalidate()
        if any(_pin(pin["path"]) != pin for pin in self.sources.values()):
            raise ValueError("original bank service source changed")

    def local_row(self, address):
        self.revalidate()
        return self._local_row(address)

    def _local_row(self, address):
        if (
            type(address) is not int
            or not 0 <= address < self.mapping["banks"] * self.mapping["rows_per_bank"]
        ):
            raise ValueError(
                "local row cut is outside selected scratchpad address space"
            )
        bank, row = divmod(address, self.mapping["rows_per_bank"])
        return {
            "bank": bank,
            "row": row,
            "resource": f"storage.scratchpad.bank{bank}",
            "actual_command_bound": False,
        }

    def record(self):
        self.revalidate()
        return {
            "schema": "gemmini.selected_scratchpad_bank_service.v1",
            "sources": self.sources,
            "selected_inputs": self.selected.report()["selected_inputs"],
            "mapping": self.mapping,
            "checks": self.checks,
            "resources": {
                f"storage.scratchpad.bank{i}": {
                    "unit": "accepted_row_operation",
                    "capacity_per_cycle_upper": 1,
                    "provenance": "selected single bank read/write acceptance cones share one service slot",
                }
                for i in range(len(self.cones))
            },
            "unresolved": {
                "actual_commands": "actual load/execute/readout commands are not yet bound through all local pipelines to bank events",
                "completion": "accepted bank service does not establish row consumption or instruction/fence retirement",
                "overlap": "different bank resources do not prove reachable simultaneous issue or downstream overlap",
            },
            "measured_cycles": None,
            "attainment": None,
            "convergence_eligible": False,
        }

    def local_service_floor(self, operations):
        self.revalidate()
        if not isinstance(operations, (list, tuple)) or len(operations) > 100000:
            raise ValueError("bank service needs finite explicit local row cuts")
        events = []
        for index, operation in enumerate(operations):
            if (
                not isinstance(operation, dict)
                or set(operation) != {"address", "direction"}
                or operation["direction"] not in {"read", "write"}
            ):
                raise ValueError(
                    "local bank operation needs explicit row address and read/write direction"
                )
            local = self._local_row(operation["address"])
            events.append(
                {
                    "id": f"local-cut-{index}",
                    "demand": {local["resource"]: 1},
                    "compulsory": False,
                    "provenance": "explicit caller local row cut; not actual instruction or fixed-workload evidence",
                }
            )
        record = self.record()
        return compose_service_floor(
            record["resources"],
            events,
            scope="emitted_schedule",
            unresolved=record["unresolved"],
        )
