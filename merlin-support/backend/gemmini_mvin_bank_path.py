"""Narrow selected plain-MVIN row/mask path; no physical event timestamps."""

import json
from pathlib import Path
from types import MethodType

from gemmini_bank_service import BankPortCones, ScratchpadBankService, _expand, _one
from gemmini_dma_service import DmaServiceContract
from gemmini_timing_resources import _pin
from merlin.targetgen.rtl import firrtl_scalar_cutpoints as C
from merlin.targetgen.rtl import firrtl_struct as F

_BANK_RECORD = ScratchpadBankService.record
_BANK_REVALIDATE = ScratchpadBankService.revalidate
_CONE_CHECK = BankPortCones.check
_SCALAR_EVALUATE = C.ScalarCutpoints.evaluate
_DMA_ROWS = DmaServiceContract.load_row_requests


def _bound(owner, name, original):
    method = getattr(owner, name, None)
    if (
        type(method) is not MethodType
        or method.__self__ is not owner
        or method.__func__ is not original
    ):
        raise ValueError("bank path lost its original bound source checker")
    return method


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _contains(expression, reference):
    return expression == reference or (
        type(expression) is F.Prim
        and any(_contains(arg, reference) for arg in expression.args)
    )


class CheckedBankService:
    """Freshly derive source state, then refuse changed mapping/checks/callbacks."""

    def __init__(self, owner, selected):
        if type(owner) is not ScratchpadBankService or owner.selected is not selected:
            raise ValueError(
                "bank writes require the same original selected bank owner"
            )
        fresh = ScratchpadBankService(selected)
        self.owner = owner
        self._issued_owner, self._selected = owner, selected
        self.sources = {
            "implementation": _pin(__file__),
            "bank_implementation": _pin(_BANK_RECORD.__code__.co_filename),
        }
        self.record = _canonical(_bound(fresh, "record", _BANK_RECORD)())
        self.state = self._state(fresh)
        self.revalidate()

    @staticmethod
    def _state(owner):
        result = []
        for cone in owner.cones:
            if (
                type(cone) is not BankPortCones
                or type(cone.observer) is not C.ScalarCutpoints
            ):
                raise ValueError("bank source cone owner changed")
            _bound(cone.observer, "evaluate", _SCALAR_EVALUATE)
            result.append(
                (cone.row_bits, cone.row_bytes, _bound(cone, "check", _CONE_CHECK)())
            )
        return _canonical(result)

    def revalidate(self):
        if (
            self.owner is not self._issued_owner
            or self.owner.selected is not self._selected
        ):
            raise ValueError("bank path changed its original owner selection")
        if any(_pin(pin["path"]) != pin for pin in self.sources.values()):
            raise ValueError("bank path implementation changed")
        _bound(self.owner, "revalidate", _BANK_REVALIDATE)()
        if (
            _canonical(_bound(self.owner, "record", _BANK_RECORD)()) != self.record
            or self._state(self.owner) != self.state
        ):
            raise ValueError("bank source mapping/checks/state changed after issuance")

    def local_row(self, address):
        mapping = json.loads(self.record)["mapping"]
        if (
            type(address) is not int
            or not 0 <= address < mapping["banks"] * mapping["rows_per_bank"]
        ):
            raise ValueError("actual local row exceeds selected bank address space")
        bank, row = divmod(address, mapping["rows_per_bank"])
        return {"bank": bank, "row": row, "resource": f"storage.scratchpad.bank{bank}"}


def _rhs(module, name):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(name)]
    if len(rows) != 1:
        raise ValueError("plain MVIN path lacks one original field update")
    return rows[0].rhs


class PlainMvinBankPath:
    def __init__(self, dma, banks):
        if (
            type(dma) is not DmaServiceContract
            or type(banks) is not CheckedBankService
            or dma.selected is not banks.owner.selected
        ):
            raise ValueError(
                "plain MVIN path requires original same-target DMA/bank owners"
            )
        self.dma, self.banks = dma, banks
        fir = Path(dma.sources["firrtl"]["path"])
        top = F.load_modules(fir, {"Scratchpad"})["Scratchpad"]
        roles = {
            name: top.decls[name][1] for name in ("mvin_scale_pixel_repeater", "vsm")
        }
        modules = F.load_modules(
            fir, {"LoadController", "BeatMerger", "StreamReader", *roles.values()}
        )
        self.merger = C.ScalarCutpoints(modules["BeatMerger"])
        self.pixel = C.ScalarCutpoints(
            modules[roles["mvin_scale_pixel_repeater"]], max_depth=256
        )
        self.scale = C.ScalarCutpoints(modules[roles["vsm"]], max_depth=256)
        self.load = dma.load
        self.scratchpad = C.ScalarCutpoints(top, modules=modules)
        _one(
            modules["StreamReader"], "io.resp.bits.addr", "beatPacker.io.out.bits.addr"
        )
        _one(top, "vsm_in_q.io.enq.bits.tag.addr", "reader.io.resp.bits.addr")
        _one(top, "vsm.io.req", "vsm_in_q.io.deq")
        _one(
            modules["StreamReader"], "io.resp.bits.mask", "beatPacker.io.out.bits.mask"
        )
        for lane in range(dma.selected.report()["geometry"]["array_cols"]):
            _one(
                top,
                f"vsm_in_q.io.enq.bits.tag.mask[{lane}]",
                f"reader.io.resp.bits.mask[{lane}]",
            )
            _one(
                top,
                f"mvin_scale_pixel_repeater.io.req.bits.mask[{lane}]",
                f"vsm.io.resp.bits.tag.mask[{lane}]",
            )
        self.normalized_pixel = _rhs(
            modules["LoadController"], "pixel_repeats[state_id]"
        )
        captures = [
            row.rhs
            for row in modules[roles["mvin_scale_pixel_repeater"]].connects
            if row.lhs == F.parse_expr("req.bits.pixel_repeats")
            and _contains(
                _expand(modules[roles["mvin_scale_pixel_repeater"]], row.rhs),
                F.parse_expr("io.req.bits.pixel_repeats"),
            )
        ]
        if len(captures) != 1:
            raise ValueError("pixel capture count normalization is absent or ambiguous")
        self.pixel_capture = captures[0]
        if (
            type(self.normalized_pixel) is not F.Ref
            or type(self.pixel_capture) is not F.Ref
        ):
            raise ValueError(
                "pixel normalization/capture must use original source nodes"
            )
        self.dimension = dma.selected.report()["geometry"]["array_cols"]
        self.sources = {
            "implementation": _pin(__file__),
            "firrtl": _pin(fir),
            "scalar_observer": _pin(C.__file__),
        }
        # Preserve row/mask tags through the selected scale capture. Numerical
        # identity and complete products remain the independent source owner's gate.
        scale = modules[roles["vsm"]]
        self.entries = F.parse_type(scale.decls["out_regs"][1])[2]
        if not 1 <= self.entries <= 8:
            raise ValueError("selected scale output ownership exceeds bounded scope")
        for index in range(self.entries):
            for field, expected in (("row", "in.bits.repeats"), ("tag", "in.bits.tag")):
                if _rhs(scale, f"out_regs[{index}].{field}") != F.parse_expr(expected):
                    raise ValueError(
                        "selected scale path changed row/mask-tag preservation"
                    )
        # Actual bank mask driver must copy the same repeater mask, with the
        # same execute-write priority as the original bank-address proof.
        for bank in range(len(banks.owner.cones)):
            address = [
                r
                for r in top.connects
                if r.lhs == F.parse_expr(f"bank_ios[{bank}].write.addr")
                and type(_expand(top, r.rhs)) is F.Prim
                and _expand(top, r.rhs).op == "bits"
            ]
            address = [
                r
                for r in address
                if _contains(
                    _expand(top, r.rhs),
                    F.parse_expr("mvin_scale_pixel_repeater.io.resp.bits.laddr.data"),
                )
            ]
            if len(address) != 1:
                raise ValueError("bank mask has no unique original address guard")
            for lane in range(self.dimension):
                rows = [
                    r
                    for r in top.connects
                    if r.lhs == F.parse_expr(f"bank_ios[{bank}].write.mask[{lane}]")
                    and r.rhs
                    == F.parse_expr(
                        f"mvin_scale_pixel_repeater.io.resp.bits.mask[{lane}]"
                    )
                ]
                if len(rows) != 1 or rows[0].guards != address[0].guards:
                    raise ValueError(
                        "bank byte mask differs from selected row ownership"
                    )

    def revalidate(self):
        self.dma.revalidate()
        self.banks.revalidate()
        _bound(self.dma, "load_row_requests", _DMA_ROWS)
        _bound(self.banks, "local_row", CheckedBankService.local_row)
        if any(_pin(pin["path"]) != pin for pin in self.sources.values()):
            raise ValueError("plain MVIN row path source changed")

    def mask_path(self, local, cols, *, offset, payload, shift, packet_bytes):
        if type(cols) is not int or not 1 <= cols <= self.dimension:
            raise ValueError("plain MVIN column tail exceeds selected byte row")
        if (
            not 0 <= offset < self.dimension
            or not 0 < payload <= cols
            or offset + payload > cols
        ):
            raise ValueError("packet fragment exceeds actual local row mask")
        cuts = {
            "req.valid": 1,
            "req.bits.has_acc_bitwidth": 0,
            "req.bits.shift": shift,
            "req.bits.bytes_to_read": payload,
            "req.bits.spad_row_offset": offset,
            "req.bits.addr": local,
            "req.bits.block_stride": self.dimension,
            "bytesRead": packet_bytes,
            "bytesSent": 0,
        }
        outputs = [
            "io.out.valid",
            "io.out.bits.addr",
            "io.out.bits.last",
            *(f"io.out.bits.mask[{i}]" for i in range(self.dimension)),
        ]
        merger = self.merger.evaluate(cuts, outputs)
        mask = [int(offset <= i < offset + payload) for i in range(self.dimension)]
        if merger != {
            "io.out.valid": 1,
            "io.out.bits.addr": local,
            "io.out.bits.last": 1,
            **{f"io.out.bits.mask[{i}]": v for i, v in enumerate(mask)},
        }:
            raise ValueError("selected DMA merge changed contiguous row/mask ownership")
        for entry in range(self.entries):
            cuts = {
                "head_oh": 1 << entry,
                f"out_regs[{entry}].row": 0,
                f"out_regs[{entry}].tag.addr": local,
                **{f"out_regs[{entry}].tag.mask[{i}]": v for i, v in enumerate(mask)},
            }
            outputs = [
                "io.resp.bits.row",
                "io.resp.bits.tag.addr",
                *(f"io.resp.bits.tag.mask[{i}]" for i in range(self.dimension)),
            ]
            values = self.scale.evaluate(cuts, outputs)
            if values != {
                "io.resp.bits.row": 0,
                "io.resp.bits.tag.addr": local,
                **{f"io.resp.bits.tag.mask[{i}]": v for i, v in enumerate(mask)},
            }:
                raise ValueError(
                    "scale output changed actual row/address/mask ownership"
                )
        forwarded = self.scratchpad.evaluate(
            {"vsm.io.resp.bits.tag.addr": local, "vsm.io.resp.bits.row": 0},
            ["mvin_scale_pixel_repeater.io.req.bits.laddr.data"],
        )
        if forwarded["mvin_scale_pixel_repeater.io.req.bits.laddr.data"] != local:
            raise ValueError("scale-to-repeater local destination changed")
        cuts = {
            "req.valid": 1,
            "req.bits.pixel_repeats": 0,
            "req.bits.len": cols,
            "req.bits.laddr.data": local,
            "req.bits.laddr.is_acc_addr": 0,
            "req.bits.last": 1,
            **{f"req.bits.mask[{i}]": v for i, v in enumerate(mask)},
        }
        outputs = [
            "io.resp.valid",
            "io.resp.bits.laddr.data",
            "io.resp.bits.last",
            *(f"io.resp.bits.mask[{i}]" for i in range(self.dimension)),
        ]
        pixel = self.pixel.evaluate(cuts, outputs)
        if pixel != {
            "io.resp.valid": 1,
            "io.resp.bits.laddr.data": local,
            "io.resp.bits.last": 1,
            **{f"io.resp.bits.mask[{i}]": v for i, v in enumerate(mask)},
        }:
            raise ValueError(
                "selected single-row repeater changed address/mask effects"
            )
        return mask

    def command_rows(self, command, configuration):
        self.revalidate()
        rows = self.dma.load_row_requests(command, configuration=configuration)
        if any(row["repeats"] != 0 for row in rows):
            raise ValueError("broadcast/repeated MVIN local writes are not mapped")
        fields = command["decoded"]
        if (
            fields["is_acc_addr"]
            or fields["addr"] + fields["rows"]
            > len(self.banks.owner.cones)
            * json.loads(self.banks.record)["mapping"]["rows_per_bank"]
        ):
            raise ValueError("accumulator or out-of-range local MVIN is not mapped")
        raw = configuration["rs1"]["raw"]
        normalized = self.load.evaluate(
            {"cmd_q.io.deq.bits.cmd.rs1": raw}, [self.normalized_pixel.root]
        )[self.normalized_pixel.root]
        captured = self.pixel.evaluate(
            {"io.req.bits.pixel_repeats": normalized}, [self.pixel_capture.root]
        )[self.pixel_capture.root]
        if normalized != 1 or captured != 0:
            raise ValueError(
                "ordinary pixel normalization is not one request/zero remaining repetitions"
            )
        result = []
        for row in rows:
            cuts = {
                "cmd_q.io.deq.bits.cmd.rs1": command["rs1"]["raw"],
                "cmd_q.io.deq.bits.cmd.rs2": command["rs2"]["raw"],
                "cmd_q.io.deq.bits.cmd.inst.funct": command["funct"],
                "strides[0]": configuration["decoded"]["stride"],
                "shrinks[0]": 0,
                "row_counter": row["row"],
            }
            name = "io.dma.req.bits.laddr.data"
            flags = [
                "io.dma.req.bits.laddr." + field
                for field in (
                    "is_acc_addr",
                    "accumulate",
                    "read_full_acc_row",
                    "norm_cmd",
                    "garbage",
                    "garbage_bit",
                )
            ]
            values = self.load.evaluate(cuts, [name, *flags])
            local = values[name]
            if any(values[flag] != 0 for flag in flags):
                raise ValueError("plain scratchpad MVIN has unsupported local flags")
            if (
                local != fields["addr"] + row["row"]
                or row["payload_bytes"] != fields["cols"]
            ):
                raise ValueError("actual plain byte-row load address/extent differs")
            consumed = 0
            for ordinal, packet in enumerate(row["packets"]):
                entry = self.dma.selector.read.evaluate(
                    {
                        "req.spaddr": local,
                        "req.block_stride": self.dimension,
                        "req.has_acc_bitwidth": 0,
                        "bytesRequested": consumed,
                    },
                    ["io.reserve.entry.addr", "io.reserve.entry.spad_row_offset"],
                )
                if entry != {
                    "io.reserve.entry.addr": local,
                    "io.reserve.entry.spad_row_offset": consumed,
                }:
                    raise ValueError(
                        "packet fragment changed selected local destination/offset"
                    )
                shift = (
                    row["virtual_address"] + consumed - packet["virtual_packet_address"]
                )
                mask = self.mask_path(
                    local,
                    fields["cols"],
                    offset=consumed,
                    payload=packet["payload_bytes"],
                    shift=shift,
                    packet_bytes=packet["packet_bytes"],
                )
                result.append(
                    {
                        "local_row": local,
                        **self.banks.local_row(local),
                        "byte_mask": mask,
                        "source_row": row["row"],
                        "packet_ordinal": ordinal,
                        "payload_bytes": packet["payload_bytes"],
                        "physical_event_observed": False,
                    }
                )
                consumed += packet["payload_bytes"]
            if consumed != fields["cols"]:
                raise ValueError(
                    "packet fragments do not completely cover the actual local row"
                )
        return result
