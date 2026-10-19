"""Checked local DMA handshake and source-ID ownership cones.

Explicit cuts do not prove reachability or whole-transfer completion.
"""

from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F


def _uint(typ):
    if not isinstance(typ, tuple) or len(typ) != 2 or typ[0] != "uint":
        raise ValueError("DMA service requires original unsigned scalar types")
    return typ[1]


def _port(module, port, *fields):
    return _uint(F.field_type(module.port_type(port), fields))


def _driver(module, name):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(name)]
    if len(rows) != 1 or rows[0].guards or type(rows[0].rhs) is not F.Ref:
        raise ValueError(
            "DMA ownership needs one original unconditional next-state expression"
        )
    return rows[0].rhs.root


class DmaOwnershipCones:
    """Execute original local cuts; no implicit register history/reachability."""

    def __init__(self, reader, writer, tracker, wrapper, modules, *, new_write_state):
        self.reader = C.ScalarCutpoints(reader)
        self.writer = C.ScalarCutpoints(writer)
        self.tracker = C.ScalarCutpoints(tracker)
        self.wrapper = C.ScalarCutpoints(wrapper, modules=modules)
        entries = F.parse_type(tracker.decls["entries"][1])
        if entries[0] != "vec" or not 1 <= entries[2] <= 64:
            raise ValueError(
                "reader transaction ownership exceeds explicit finite scope"
            )
        self.read_slots = entries[2]
        self.write_slots = _uint(F.parse_type(writer.decls["xactBusy"][1]))
        if not 1 <= self.write_slots <= 64:
            raise ValueError(
                "writer transaction ownership exceeds explicit finite scope"
            )
        self.next_busy = _driver(writer, "xactBusy")
        counter = "io_beatData_bits_last_counter"
        rows = [row for row in reader.connects if row.lhs == F.Ref(counter)]
        if (
            len(rows) != 1
            or len(rows[0].guards) != 1
            or not rows[0].guards[0].positive
            or type(rows[0].guards[0].cond) is not F.Ref
            or type(rows[0].rhs) is not F.Ref
        ):
            raise ValueError("reader response counter lacks one accepted-beat update")
        self.read_counter, self.read_counter_next = counter, rows[0].rhs.root
        self.read_counter_width = _uint(F.parse_type(reader.decls[counter][1]))
        self.read_beat_bytes = _port(reader, "auto", "out", "d", "bits", "data") // 8
        self.read_counter_enable = rows[0].guards[0].cond.root
        if (
            self.reader.evaluate({"state": 0}, ["untranslated_a.bits.tl_a.opcode"])[
                "untranslated_a.bits.tl_a.opcode"
            ]
            != 4
        ):
            raise ValueError("reader request is not the selected Get operation")
        for full, opcode in ((0, 1), (1, 0)):
            if (
                self.writer.evaluate(
                    {"write_packet.is_full": full}, ["untranslated_a.bits.tl_a.opcode"]
                )["untranslated_a.bits.tl_a.opcode"]
                != opcode
            ):
                raise ValueError("writer request is not selected PutPartial/PutFull")
        updates = [
            row
            for row in tracker.connects
            if row.lhs.root == "entries"
            and row.lhs.segments
            and row.lhs.segments[-1] == "valid"
        ]
        allocation = [
            row
            for row in updates
            if row.lhs == F.parse_expr("entries[free_entry].valid")
        ]
        release = [
            row
            for row in updates
            if row.lhs == F.parse_expr("entries[io.peek.xactid].valid")
        ]
        if (
            len(allocation) != 1
            or len(release) != 1
            or len(updates) != self.read_slots + 2
            or allocation[0].rhs != F.Lit("UInt", 1, 1)
            or release[0].rhs != F.Lit("UInt", 1, 0)
            or len(allocation[0].guards) != 1
            or len(release[0].guards) != 1
            or not allocation[0].guards[0].positive
            or not release[0].guards[0].positive
            or release[0].guards[0].cond != F.parse_expr("io.peek.pop")
        ):
            raise ValueError(
                "reader source-ID allocation/pop effects differ from original store updates"
            )
        self.read_allocation_guard = allocation[0].guards[0].cond.root
        self.new_write_state = new_write_state
        self.writer_states = 1 << _uint(F.parse_type(writer.decls["state"][1]))
        if self.writer_states > 8:
            raise ValueError("writer phase state exceeds explicit finite scope")
        samples = {
            state: self.writer.evaluate(
                {"state": state, "xactBusy": 0}, ["io.busy", "untranslated_a.valid"]
            )
            for state in range(self.writer_states)
        }
        idle = [state for state, row in samples.items() if row["io.busy"] == 0]
        active = [
            state for state, row in samples.items() if row["untranslated_a.valid"]
        ]
        if len(idle) != 1 or len(active) != 2 or new_write_state not in active:
            raise ValueError(
                "writer idle/new/continuation phases are not source-derived"
            )
        self.idle_state = idle[0]
        self.active_states = tuple(active)
        if _port(reader, "io", "reserve", "xactid") != _port(
            tracker, "io", "alloc", "xactid"
        ):
            raise ValueError("reader source-ID ownership and allocation widths differ")
        for module, slots in ((reader, self.read_slots), (writer, self.write_slots)):
            for channel in ("a", "d"):
                if (
                    _port(module, "auto", "out", channel, "bits", "source")
                    != (slots - 1).bit_length()
                ):
                    raise ValueError(
                        "client source-ID width differs from transaction storage"
                    )
                if any(
                    _port(module, "auto", "out", channel, signal) != 1
                    for signal in ("ready", "valid")
                ):
                    raise ValueError("client channel lacks one ready/valid handshake")

    def reader_issue(
        self, *, state, reserve_ready, request_ready, response_ready, response_valid
    ):
        cuts = {
            "state": state,
            "io.reserve.ready": reserve_ready,
            "untranslated_a.ready": request_ready,
            "io.beatData.ready": response_ready,
            "nodeOut.d.valid": response_valid,
        }
        result = self.reader.evaluate(
            cuts,
            [
                "untranslated_a.valid",
                "io.reserve.valid",
                "nodeOut.d.ready",
                "io.beatData.valid",
            ],
        )
        if result != {
            "untranslated_a.valid": int(state == 1 and reserve_ready),
            "io.reserve.valid": int(state == 1 and request_ready),
            "nodeOut.d.ready": response_ready,
            "io.beatData.valid": response_valid,
        }:
            raise ValueError(
                "actual reader reserve/request/response handshake cone differs"
            )
        result["allocate"] = result["io.reserve.valid"] & reserve_ready
        result["request_fire"] = result["untranslated_a.valid"] & request_ready
        if result["allocate"] != result["request_fire"]:
            raise ValueError(
                "reader allocation and untranslated request issue lost atomicity"
            )
        return result

    def reader_allocation(self, occupied):
        if type(occupied) is not int or not 0 <= occupied < 1 << self.read_slots:
            raise ValueError("reader occupancy exceeds original transaction store")
        result = self.tracker.evaluate(
            {
                f"entries[{i}].valid": (occupied >> i) & 1
                for i in range(self.read_slots)
            },
            ["io.alloc.ready", "io.alloc.xactid", "io.busy"],
        )
        full = occupied == (1 << self.read_slots) - 1
        free = next(
            (i for i in range(self.read_slots) if not occupied >> i & 1),
            self.read_slots - 1,
        )
        if result != {
            "io.alloc.ready": int(not full),
            "io.alloc.xactid": free,
            "io.busy": int(occupied != 0),
        }:
            raise ValueError("actual reader allocation/source-ID/busy cone differs")
        return result

    def reader_release(self, *, ready, valid, last):
        result = self.wrapper.evaluate(
            {
                "beatPacker.io.in.ready": ready,
                "beatPacker.io.in.valid": valid,
                "core.io.beatData.bits.last": last,
            },
            ["xactTracker.io.peek.pop"],
        )
        if result["xactTracker.io.peek.pop"] != ready * valid * last:
            raise ValueError(
                "reader transaction is not released at accepted final response beat"
            )
        return result["xactTracker.io.peek.pop"]

    def reader_response_step(self, *, packet_bytes, beat_bytes, counter, ready, valid):
        if (
            type(packet_bytes) is not int
            or type(beat_bytes) is not int
            or packet_bytes < beat_bytes
            or beat_bytes != self.read_beat_bytes
            or packet_bytes % beat_bytes
            or packet_bytes & (packet_bytes - 1)
        ):
            raise ValueError(
                "reader response needs an explicit complete-beat power-of-two packet"
            )
        cuts = {
            "nodeOut.d.bits.size": packet_bytes.bit_length() - 1,
            "nodeOut.d.bits.opcode": 1,
            self.read_counter: counter,
            "nodeOut.d.ready": ready,
            "nodeOut.d.valid": valid,
        }
        result = self.reader.evaluate(
            cuts,
            ["io.beatData.bits.last", self.read_counter_enable, self.read_counter_next],
        )
        beats = packet_bytes // beat_bytes
        last = int(counter == 1 or beats == 1)
        next_counter = (
            beats - 1
            if counter == 0
            else (counter - 1) % (1 << self.read_counter_width)
        )
        if (
            result["io.beatData.bits.last"] != last
            or result[self.read_counter_enable] != ready * valid
            or result[self.read_counter_next] != next_counter
        ):
            raise ValueError(
                "reader actual response beat counter differs from packet envelope"
            )
        return {
            "last": last,
            "accepted_final": ready * valid * last,
            "counter_next": next_counter if ready * valid else counter,
        }

    def writer_step(
        self, *, state, occupied, request_ready, response_valid, response_source
    ):
        if type(occupied) is not int or not 0 <= occupied < 1 << self.write_slots:
            raise ValueError("writer occupancy exceeds original transaction store")
        if state not in (self.idle_state, *self.active_states):
            raise ValueError("writer state is outside source-derived protocol phases")
        if (
            type(response_source) is not int
            or not 0 <= response_source < self.write_slots
        ):
            raise ValueError("writer response source exceeds original source IDs")
        if response_valid and not occupied >> response_source & 1:
            raise ValueError("writer response has no outstanding source-ID owner")
        cuts = {
            "state": state,
            "xactBusy": occupied,
            "untranslated_a.ready": request_ready,
            "nodeOut.d.valid": response_valid,
            "nodeOut.d.bits.source": response_source,
        }
        outputs = [
            "xactId",
            "xactBusy_fire",
            "untranslated_a.valid",
            "nodeOut.d.ready",
            "io.busy",
            self.next_busy,
        ]
        result = self.writer.evaluate(cuts, outputs)
        free = next((i for i in range(self.write_slots) if not occupied >> i & 1), 0)
        valid = int(
            state in self.active_states and occupied != (1 << self.write_slots) - 1
        )
        allocate = valid * request_ready * int(state == self.new_write_state)
        expected = (occupied | ((1 << free) if allocate else 0)) & ~(
            (1 << response_source) if response_valid else 0
        )
        if (
            result["untranslated_a.valid"] != valid
            or result["xactBusy_fire"] != allocate
            or result["nodeOut.d.ready"] != int(occupied != 0)
            or result["io.busy"] != int(occupied != 0 or state != self.idle_state)
            or result[self.next_busy] != expected
            or (allocate and result["xactId"] != free)
        ):
            raise ValueError(
                "actual writer allocation/acknowledgement/retirement cone differs"
            )
        return {
            "allocate": allocate,
            "acknowledge": response_valid,
            "occupied_next": expected,
            "source": result["xactId"] if allocate else None,
            "busy": result["io.busy"],
            "request_valid": valid,
            "response_ready": result["nodeOut.d.ready"],
        }

    def check(self):
        count = 0
        for bits in range(32):
            self.reader_issue(
                state=bits & 1,
                reserve_ready=(bits >> 1) & 1,
                request_ready=(bits >> 2) & 1,
                response_ready=(bits >> 3) & 1,
                response_valid=(bits >> 4) & 1,
            )
            count += 1
        for bits in range(8):
            self.reader_release(
                ready=bits & 1, valid=(bits >> 1) & 1, last=(bits >> 2) & 1
            )
            count += 1
        for valid in (0, 1):
            for ready in (0, 1):
                observed = self.tracker.evaluate(
                    {"io.alloc.valid": valid, "io.alloc.ready": ready},
                    [self.read_allocation_guard],
                )
                if observed[self.read_allocation_guard] != valid * ready:
                    raise ValueError(
                        "reader transaction store allocation is not valid/ready atomic"
                    )
                count += 1
        for slots, operation in (
            (self.read_slots, self.reader_allocation),
            (self.write_slots, None),
        ):
            masks = {
                0,
                (1 << slots) - 1,
                *(1 << i for i in range(slots)),
                *((1 << slots) - 1 ^ (1 << i) for i in range(slots)),
            }
            for occupied in sorted(masks):
                if operation:
                    operation(occupied)
                    count += 1
                else:
                    for state in (self.idle_state, *self.active_states):
                        for ready in (0, 1):
                            for source in [
                                None,
                                *(i for i in range(slots) if occupied >> i & 1),
                            ]:
                                self.writer_step(
                                    state=state,
                                    occupied=occupied,
                                    request_ready=ready,
                                    response_valid=int(source is not None),
                                    response_source=source or 0,
                                )
                                count += 1
        return {"local_cutpoint_cases": count, "reachability_proven": False}
