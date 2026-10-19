"""Selected DMA packet/service effects from exact unsigned FIRRTL cones.

Request envelopes are source-derived, not observed RTL packets or DRAM traffic.
Retries, translation stalls, arbitration and completion latency are unpriced.
"""

from __future__ import annotations

import math
from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F


def _literal_field(module, reference):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(reference)]
    if (
        len(rows) != 1
        or rows[0].guards
        or type(rows[0].rhs) is not F.Lit
        or rows[0].rhs.kind != "UInt"
    ):
        raise ValueError("packet size needs one original unsigned literal driver")
    return rows[0].rhs.value


def _sizes(module, prefix):
    names = sorted(
        name
        for name, row in module.decls.items()
        if row[0] == "wire"
        and name.startswith(prefix)
        and name[len(prefix) :].isdigit()
    )
    sizes = [_literal_field(module, name + ".size") for name in names]
    if (
        not sizes
        or sizes != sorted(set(sizes))
        or any(size < 1 or size & (size - 1) for size in sizes)
    ):
        raise ValueError(
            "selected packet candidate sizes are absent or unordered powers of two"
        )
    for name, size in zip(names, sizes, strict=True):
        if _literal_field(module, name + ".lg_size") != size.bit_length() - 1:
            raise ValueError("packet size/log-size effects disagree")
    return sizes


def _beat(module, direction):
    channel = "d" if direction == "read" else "a"
    data = F.field_type(module.port_type("auto"), ("out", channel, "bits", "data"))
    valid = F.field_type(module.port_type("auto"), ("out", channel, "valid"))
    if data is None or data[0] != "uint" or data[1] % 8 or valid != ("uint", 1):
        raise ValueError("selected packet data channel width/handshake is unavailable")
    return data[1] // 8


class PacketSelectors:
    """Checked local source packet selectors; no whole-machine timing authority."""

    def __init__(self, reader, writer):
        self.read, self.write = (
            C.ScalarCutpoints(reader, max_depth=256),
            C.ScalarCutpoints(writer, max_depth=256),
        )
        self.sizes = {
            "read": _sizes(reader, "read_packets_"),
            "write": _sizes(writer, "write_packets_"),
        }
        self.beats = {"read": _beat(reader, "read"), "write": _beat(writer, "write")}
        expression = writer.nodes["write_packet"][0]
        if type(expression) is not F.Prim or expression.op != "mux":
            raise ValueError("writer packet has no explicit source hold/bypass mux")
        condition = expression.args[0]
        if type(condition) is F.Ref and not condition.segments:
            condition = writer.nodes[condition.root][0]
        if (
            type(condition) is not F.Prim
            or condition.op != "eq"
            or condition.args[0] != F.Ref("state")
            or type(condition.args[1]) is not F.Lit
        ):
            raise ValueError("writer new-packet state is not source-derived")
        self.new_write_state = condition.args[1].value
        for direction, sizes in self.sizes.items():
            if max(sizes) > 1024 or any(size % self.beats[direction] for size in sizes):
                raise ValueError("packet candidates exceed bounded complete-beat scope")

    def packet(self, direction, address, remaining):
        if (
            direction not in self.sizes
            or type(address) is not int
            or address < 0
            or type(remaining) is not int
        ):
            raise ValueError(
                "packet projection needs explicit unsigned address/length/direction"
            )
        if not 1 <= remaining <= max(self.sizes[direction]):
            raise ValueError("packet length exceeds the selected bounded row proof")
        if direction == "read":
            prefix, payload = "read_packet", "bytes_read"
            cuts = {"req.vaddr": address, "bytesLeft": remaining}
            observer = self.read
        else:
            prefix, payload = "best_write_packet", "bytes_written"
            cuts = {
                "req.vaddr": address,
                "bytesLeft": remaining,
                "state": self.new_write_state,
            }
            observer = self.write
        outputs = [
            prefix + "." + name for name in ("size", "lg_size", "vaddr", payload)
        ]
        outputs += ["untranslated_a.bits.tl_a.size", "next_vaddr"]
        if direction == "write":
            outputs += ["write_beats"]
        values = observer.evaluate(cuts, outputs)
        # The independent arithmetic description is checked against actual source
        # cones, including strict-greater tie breaking and address alignment.
        candidates = [
            (min(size - address % size, remaining), size)
            for size in self.sizes[direction]
        ]
        amount, size = max(candidates, key=lambda row: row[0])
        expected = {
            "size": size,
            "lg_size": size.bit_length() - 1,
            "vaddr": address - address % size,
            payload: amount,
        }
        if any(values[prefix + "." + key] != value for key, value in expected.items()):
            raise ValueError(
                "actual selected packet cone differs from maximum-payload/smaller-tie rule"
            )
        if (
            values["untranslated_a.bits.tl_a.size"] != expected["lg_size"]
            or values["next_vaddr"] != address + amount
        ):
            raise ValueError(
                "actual packet header or request-address progression differs"
            )
        beats = math.ceil(size / self.beats[direction])
        if direction == "write" and values["write_beats"] != beats:
            raise ValueError("actual writer beat count differs from full packet size")
        return {
            "virtual_packet_address": expected["vaddr"],
            "physical_packet_address": None,
            "packet_bytes": size,
            "payload_bytes": amount,
            "data_beats": beats,
            "data_beat_bytes": self.beats[direction],
            "next_virtual_address": address + amount,
        }

    def row(self, direction, address, length):
        if type(length) is not int or not 1 <= length <= max(
            self.sizes.get(direction, [0])
        ):
            raise ValueError(
                "row packet request exceeds bounded selected payload scope"
            )
        packets, remaining = [], length
        while remaining:
            packet = self.packet(direction, address, remaining)
            packets.append(packet)
            remaining -= packet["payload_bytes"]
            address = packet["next_virtual_address"]
            if len(packets) > 1024:
                raise ValueError("row packet inventory exceeds bounded source scope")
        return packets

    def check_all_offsets_and_lengths(self):
        if sum(max(sizes) ** 2 for sizes in self.sizes.values()) > 8192:
            raise ValueError(
                "exhaustive packet proof exceeds explicit 8192-case admission"
            )
        count = 0
        for direction in ("read", "write"):
            limit = max(self.sizes[direction])
            for offset in range(limit):
                for length in range(1, limit + 1):
                    self.packet(direction, offset, length)
                    count += 1
        return {
            "cutpoint_cases": count,
            "scope": "all packet offsets and row lengths through selected maximum",
        }


class DmaServiceContract:
    """Live selected source owner; never recovered from a saved JSON verdict."""

    def __init__(self, selected):
        if type(selected) is not SelectedSchedulingTarget:
            raise ValueError("DMA effects require original selected source owner")
        selected.revalidate()
        self.observer = selected.operand_observer()
        self.isa = self.observer.isa_constants("gemmini")
        facts, _, operands = selected._documents()
        self.fields = {
            (row["bundle"], row["name"]): row for row in operands["layouts"]["fields"]
        }
        fir = Path(facts["inputs"]["fir_path"])
        if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
            raise ValueError("selected packet FIRRTL source changed")
        self.selected = selected
        self.sources = {
            "firrtl": _pin(fir),
            "implementation": _pin(__file__),
            "scalar_observer": _pin(C.__file__),
            "firrtl_parser": _pin(F.__file__),
        }
        self.modules = F.load_modules(
            fir,
            {"StreamReaderCore", "StreamWriter", "LoadController", "StoreController"},
        )
        children = {
            row[1]
            for name in ("LoadController", "StoreController")
            for row in self.modules[name].decls.values()
            if row[0] == "inst"
        }
        self.modules.update(F.load_modules(fir, children))
        self.load = C.ScalarCutpoints(
            self.modules["LoadController"], modules=self.modules
        )
        self.store = C.ScalarCutpoints(
            self.modules["StoreController"], modules=self.modules
        )
        scratchpad = F.load_modules(fir, {"Scratchpad"})["Scratchpad"]
        spad_children = {
            scratchpad.decls[name][1] for name in ("writer", "write_issue_q")
        }
        types = F.load_modules(fir, spad_children)
        self.scratchpad = C.ScalarCutpoints(scratchpad, modules=types)
        self.selector = PacketSelectors(
            self.modules["StreamReaderCore"], self.modules["StreamWriter"]
        )

    def _decode(self, command, expected):
        if not isinstance(command, dict) or command.get("class") != expected:
            raise ValueError("actual packet has wrong selected instruction class")
        kind, fields = self.observer.decode_instruction(
            command["funct"], command["rs1"], command["rs2"], self.isa
        )
        if kind != expected or fields != command.get("decoded"):
            raise ValueError(
                "actual packet metadata differs from original selected operand decoding"
            )
        return fields

    def _field(self, bundle, name, raw):
        field = self.fields[bundle, name]
        if field.get("status") != "derived":
            raise ValueError(
                "configuration field lacks original selected source evidence"
            )
        return raw >> field["offset"] & ((1 << field["width"]) - 1)

    def load_row_requests(self, command, *, configuration):
        """Observe exact row requests under actual selected identity load configuration."""
        self.revalidate()
        self._decode(command, "MVIN")
        config = self._decode(configuration, "CONFIG_LD")
        raw = configuration["rs1"]["raw"]
        dimension = self.selected.report()["geometry"]["array_rows"]
        shrink = self._field("ConfigMvinRs1", "shrink", raw)
        if (
            config["state_id"] != 0
            or config["scale_bits"] != 0x3F800000
            or shrink != 0
            or self._field("ConfigMvinRs1", "pixel_repeats", raw) != 0
            or self._field("ConfigMvinRs1", "stride", raw) != dimension
        ):
            raise ValueError(
                "load mapping needs actual identity unshrunk state-zero configuration"
            )
        stride = config["stride"]
        raw1, raw2 = command["rs1"]["raw"], command["rs2"]["raw"]
        cuts = {
            "cmd_q.io.deq.bits.cmd.rs1": raw1,
            "cmd_q.io.deq.bits.cmd.rs2": raw2,
            "cmd_q.io.deq.bits.cmd.inst.funct": command["funct"],
            "strides[0]": stride,
            "shrinks[0]": shrink,
            "row_counter": 0,
        }
        names = [
            "actual_rows_read",
            "io.dma.req.bits.vaddr",
            "io.dma.req.bits.cols",
            "io.dma.req.bits.repeats",
            "io.dma.req.bits.has_acc_bitwidth",
            "io.dma.req.bits.all_zeros",
        ]
        initial = self.load.evaluate(cuts, names)
        rows, cols = initial["actual_rows_read"], initial["io.dma.req.bits.cols"]
        dimension = self.selected.report()["geometry"]["array_rows"]
        if (
            not 1 <= rows <= dimension
            or not 1 <= cols <= dimension
            or initial["io.dma.req.bits.all_zeros"]
        ):
            raise ValueError(
                "load request geometry exceeds checked nonzero panel scope"
            )
        result = []
        for row in range(rows):
            values = self.load.evaluate({**cuts, "row_counter": row}, names)
            if values["io.dma.req.bits.vaddr"] != raw1 + row * stride:
                raise ValueError(
                    "actual load row address differs from selected stride progression"
                )
            length = self.selector.read.evaluate(
                {
                    "req.len": cols,
                    "req.has_acc_bitwidth": values["io.dma.req.bits.has_acc_bitwidth"],
                    "bytesRequested": 0,
                },
                ["bytesLeft"],
            )["bytesLeft"]
            result.append(
                {
                    "row": row,
                    "virtual_address": values["io.dma.req.bits.vaddr"],
                    "payload_bytes": length,
                    "repeats": values["io.dma.req.bits.repeats"],
                    "packets": self.selector.row(
                        "read", values["io.dma.req.bits.vaddr"], length
                    ),
                }
            )
        return result

    def store_row_requests(self, command, *, configuration):
        """Observe exact plain full-width request geometry; pooling modes refuse."""
        self.revalidate()
        fields = self._decode(command, "MVOUT")
        config = self._decode(configuration, "CONFIG_ST")
        if fields.get("readout") != "i32" or any(
            config[name] != 0
            for name in (
                "acc_act",
                "pool_stride",
                "pool_size",
                "upad",
                "lpad",
                "pool_out_dim",
                "porows",
                "pocols",
                "orows",
                "ocols",
            )
        ):
            raise ValueError(
                "store mapping requires actual plain i32 nonpooling readout"
            )
        stride = config["out_stride_bytes"]
        raw1, raw2 = command["rs1"]["raw"], command["rs2"]["raw"]
        cuts = {
            "cmd_q.io.deq.bits.cmd.rs1": raw1,
            "cmd_q.io.deq.bits.cmd.rs2": raw2,
            "cmd_q.io.deq.bits.cmd.inst.funct": command["funct"],
            "stride": stride,
            "pool_stride": 0,
            "pool_size": 0,
            "row_counter": 0,
            "block_counter": 0,
        }
        address_fields = (
            "is_acc_addr",
            "accumulate",
            "read_full_acc_row",
            "data",
            "garbage_bit",
        )
        names = [
            "io.dma.req.bits.vaddr",
            "io.dma.req.bits.len",
            "io.dma.req.bits.dest",
            "io.dma.req.bits.store_en",
            "io.dma.req.bits.pool_en",
        ]
        names += ["io.dma.req.bits.laddr." + field for field in address_fields]
        dimension = self.selected.report()["geometry"]["array_rows"]
        if not 1 <= fields["rows"] <= dimension or not 1 <= fields["cols"] <= dimension:
            raise ValueError("store geometry exceeds checked panel scope")
        result = []
        for row in range(command["decoded"]["rows"]):
            values = self.store.evaluate({**cuts, "row_counter": row}, names)
            if (
                values["io.dma.req.bits.vaddr"] != raw1 + row * stride
                or values["io.dma.req.bits.dest"]
                or values["io.dma.req.bits.pool_en"]
                or values["io.dma.req.bits.store_en"] != 1
            ):
                raise ValueError(
                    "actual store row has unsupported address/destination/pooling effects"
                )
            queue_cuts = {
                "write_issue_q.io.deq.bits.laddr." + field: values[
                    "io.dma.req.bits.laddr." + field
                ]
                for field in address_fields
            }
            queue_cuts["write_issue_q.io.deq.bits.len"] = values["io.dma.req.bits.len"]
            length = self.scratchpad.evaluate(
                queue_cuts,
                ["writer.io.req.bits.len"],
            )["writer.io.req.bits.len"]
            result.append(
                {
                    "row": row,
                    "virtual_address": values["io.dma.req.bits.vaddr"],
                    "request_elements": values["io.dma.req.bits.len"],
                    "payload_bytes": length,
                    "packets": self.selector.row(
                        "write", values["io.dma.req.bits.vaddr"], length
                    ),
                }
            )
        return result

    def revalidate(self):
        self.selected.revalidate()
        if any(_pin(row["path"]) != row for row in self.sources.values()):
            raise ValueError("original DMA source/effect observer changed")

    def record(self, *, exhaustive=False):
        self.revalidate()
        return {
            "schema": "gemmini.selected_dma_packet_service.v1",
            "sources": self.sources,
            "candidate_packet_bytes": self.selector.sizes,
            "beat_bytes": self.selector.beats,
            "packet_selector_checks": self.selector.check_all_offsets_and_lengths()
            if exhaustive
            else None,
            "observation": "source-derived no-retry client request envelopes; not actual RTL trace or DRAM traffic",
            "physical_data_bytes_observed": None,
            "queue_reachability": "explicit controller/request cutpoints; FIFO/state reachability is not established by a scalar cone",
            "completion_latency": None,
            "resource_overlap": "UNKNOWN",
            "timing_qualified": False,
            "convergence_eligible": False,
        }
