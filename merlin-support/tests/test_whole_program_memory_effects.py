"""Original packet induction through explicit neutral owners, not qualification."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_packet_contractions import banked_stream, command, shape


@pytest.fixture
def issued(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    M = importlib.import_module("gemmini_whole_program_memory_effects")

    class Selected:
        def __init__(self):
            self.inputs = {"facts": "neutral original"}

        def revalidate(self):
            pass

        def report(self):
            return {"selected_inputs": copy.deepcopy(self.inputs)}

    class Arithmetic:
        identity = "neutral original arithmetic"

        def __init__(self, selected):
            self.selected = selected
            self._selected = selected

        def revalidate(self):
            pass

        def record(self):
            return self.selected.report()

    class Loads:
        def __init__(self, selected):
            self.selected = selected
            self.states = (0, 1, 2)
            self.ports = {"MVIN": 0, "MVIN2": 1, "MVIN3": 2}

        def revalidate(self):
            pass

        def record(self):
            return {"ports": dict(self.ports), "states": list(self.states)}

    class Zero:
        def __init__(self, selected, arithmetic):
            self.selected, self.arithmetic = selected, arithmetic
            self.original = True

        def revalidate(self):
            pass

        def record(self):
            return {"neutral_original_zero_contract": self.original}

        def accepts(self, command):
            fields = command["decoded"]
            dim = self.selected.report().get("geometry", {}).get("array_rows", 2)
            return (
                command["class"] == "MVIN3"
                and command["rs1"]["raw"] == 0
                and fields.get("is_acc_addr") is True
                and fields["addr"] == 128
                and 0 < fields["rows"] <= dim
                and 0 < fields["cols"] <= dim
            )

    for name, cls in (
        ("SelectedSchedulingTarget", Selected),
        ("IntegerArithmeticOwner", Arithmetic),
        ("SelectedLoadStreams", Loads),
        ("SelectedAccumulatorZeroFill", Zero),
    ):
        monkeypatch.setattr(M, name, cls)
    # Neutral original-owner stand-ins are explicit; actual RTL is never inferred.
    monkeypatch.setattr(
        M,
        "_CONSTRUCTORS",
        tuple(
            (
                name,
                getattr(M, name),
                getattr(M, name).__init__,
                getattr(M, name).__init__.__code__,
            )
            for name in ("SelectedLoadStreams", "SelectedAccumulatorZeroFill")
        ),
    )
    selected = Selected()
    arithmetic = Arithmetic(selected)
    owner = M.WholeProgramMemoryEffects(selected, arithmetic)
    return M, owner, selected, arithmetic


def test_live_bundle_returns_only_same_selected_original_owners(issued):
    M, owner, selected, _ = issued
    loads, zero = owner.packet_owners(selected)
    assert loads.selected is zero.selected is selected
    result = owner.record()
    result["loads"]["ports"]["MVIN"] = 2
    assert owner.record()["loads"]["ports"]["MVIN"] == 0
    with pytest.raises(ValueError, match="foreign"):
        owner.packet_owners(M.SelectedSchedulingTarget())


@pytest.mark.parametrize(
    "protocol", [True, 1, {}, "unknown", "gemmini.source_observation.v2"]
)
def test_protocol_is_closed_and_not_observation_authority(issued, protocol):
    M, *_ = issued
    with pytest.raises(ValueError, match="unknown"):
        M.require_protocol(protocol)


def test_json_unissued_and_subclassed_bundles_cannot_issue(issued):
    M, _, selected, arithmetic = issued
    forged = object.__new__(M.WholeProgramMemoryEffects)
    with pytest.raises(ValueError, match="not originally issued"):
        forged.record()

    class Foreign(M.WholeProgramMemoryEffects):
        pass

    with pytest.raises(TypeError):
        Foreign(selected, arithmetic)


@pytest.mark.parametrize(
    "fault", ["loads", "zero", "selected", "callback", "record", "source"]
)
def test_live_bundle_state_owner_callback_source_mutants_refuse(
    issued, monkeypatch, fault
):
    M, owner, selected, _ = issued
    state = M._ISSUED[owner]
    if fault == "loads":
        state["loads"].ports["MVIN"] = 2
    elif fault == "zero":
        state["zeros"].arithmetic = object()
    elif fault == "selected":
        selected.inputs["facts"] = "foreign"
    elif fault == "callback":
        monkeypatch.setattr(state["zeros"], "accepts", lambda _: True)
    elif fault == "record":
        state["record"]["protocol"] = "foreign"
    else:
        state["record"]["sources"]["gemmini_load_streams"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        owner.packet_owners(selected)


@pytest.mark.parametrize("name", ["_STATE", "_digest", "_pin", "_revalidate_sources"])
def test_in_memory_helper_rebinding_refuses(issued, monkeypatch, name):
    M, owner, *_ = issued
    monkeypatch.setattr(M, name, lambda *_: None)
    with pytest.raises(ValueError, match="helper changed"):
        owner.record()


def test_in_memory_helper_code_and_constructor_mutations_refuse(issued, monkeypatch):
    M, owner, *_ = issued
    original = M._digest.__code__
    M._digest.__code__ = (lambda _: "forged").__code__
    try:
        with pytest.raises(ValueError, match="helper changed"):
            owner.record()
    finally:
        M._digest.__code__ = original
    monkeypatch.setattr(M.SelectedLoadStreams, "__init__", lambda *a: None)
    with pytest.raises(ValueError, match="constructor changed"):
        owner.record()


def test_failed_final_original_derivation_revokes_partially_issued_bundle(
    issued, monkeypatch
):
    M, _, selected, arithmetic = issued
    count = len(M._ISSUED)

    def failed(self):
        raise ValueError("neutral original final derivation refused")

    monkeypatch.setattr(M.SelectedAccumulatorZeroFill, "revalidate", failed)
    # Keep the exception alive: weak-reference garbage collection alone must
    # not remove a partially issued object retained by its failing traceback.
    with pytest.raises(ValueError, match="final derivation refused") as error:
        M.WholeProgramMemoryEffects(selected, arithmetic)
    assert error.value.__traceback__ is not None
    assert len(M._ISSUED) == count


@pytest.fixture
def packet_route(issued, tmp_path, monkeypatch, request):
    M, owner, selected, _ = issued
    P = importlib.import_module("gemmini_source_packet_artifacts")
    commands = banked_stream()
    m, k, n, dim = 3, 3, 2, 2
    for row in commands:
        if row["class"] == "CONFIG_LD":
            state = row["decoded"]["state_id"]
            row["rs1"]["raw"] = (0x3F800000 << 32) | (2 << 16) | (1 << 8) | state
            row["decoded"]["scale_bits"] = 0x3F800000
        elif row["class"] == "MVIN3":
            row["rs1"]["raw"] = 0
    if getattr(request, "param", None) == "retained_public_35":
        m, k, n, dim = 7, 17, 31, 16
        commands = [command("CONFIG_EX")]
        # Exact raw fields from the retained public35 observation. Tensor data
        # and decoder framing below remain neutral, without target authority.
        for state, raw, stride in (
            (0, 4575657221409472769, 17),
            (1, 4575657221409472777, 31),
            (2, 4575657221409472785, 0),
        ):
            commands.append(
                command(
                    "CONFIG_LD",
                    raw,
                    stride,
                    state_id=state,
                    scale_bits=0x3F800000,
                    stride=stride,
                )
            )
        for col in range(0, n, dim):
            cols = min(dim, n - col)
            commands.append(
                command("MVIN3", 0, rows=m, cols=cols, addr=128, is_acc_addr=True)
            )
            for depth in range(0, k, dim):
                width = min(dim, k - depth)
                commands.extend(
                    [
                        command(
                            "MVIN",
                            1000 + depth,
                            rows=m,
                            cols=width,
                            addr=0,
                            is_acc_addr=False,
                        ),
                        command(
                            "MVIN2",
                            2000 + depth * n + col,
                            rows=width,
                            cols=cols,
                            addr=16,
                            is_acc_addr=False,
                        ),
                        command(
                            "PRELOAD", shape(width, cols, 16), shape(m, cols, 128 | 64)
                        ),
                        command("FENCE"),
                        command(
                            "COMPUTE_PRELOADED",
                            shape(m, width, 0),
                            garbage=True,
                            bd=255,
                        ),
                    ]
                )
            st = next(r for r in banked_stream() if r["class"] == "CONFIG_ST")
            st["decoded"]["out_stride_bytes"] = n * 4
            commands.extend(
                [
                    st,
                    command(
                        "MVOUT",
                        3000 + col * 4,
                        rows=m,
                        cols=cols,
                        addr=128 | 32,
                        readout="i32",
                    ),
                ]
            )
    arrays = {"A": np.ones((m, k), dtype=np.int8), "B": np.ones((k, n), dtype=np.int8)}
    payload = bytearray(64)
    payload[5] = 1
    sections, symbols = [], []
    for i, (name, address, raw) in enumerate(
        (
            ("A", 1000, arrays["A"].tobytes()),
            ("B", 2000, arrays["B"].tobytes()),
            ("C", 3000, bytes(m * n * 4)),
        )
    ):
        sections.append((name, 1, 0, len(payload), len(raw), address))
        symbols.append((address, len(raw), f"tensor_{i}"))
        payload.extend(raw)
    elf = tmp_path / "neutral.elf"
    elf.write_bytes(payload)
    log = tmp_path / "neutral-complete-log"
    log.write_text("core 0: 3 0x80000000 (0x0000006b)\n")
    framing = SimpleNamespace(
        _symbols=lambda *_: symbols,
        _events=lambda *_: (
            {"instructions": [r for r in commands if r["class"] != "FENCE"]},
            [(r["class"], 0, 0) for r in commands],
        ),
    )
    monkeypatch.setattr(
        P.importlib.util,
        "spec_from_file_location",
        lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None)),
    )
    monkeypatch.setattr(P.importlib.util, "module_from_spec", lambda *_: framing)
    monkeypatch.setattr(
        P.invocation_record,
        "verify",
        lambda _: {"argv": ["neutral"], "stderr": {"path": str(log)}},
    )
    monkeypatch.setattr(P, "_section_headers", lambda _: sections)
    monkeypatch.setattr(
        P,
        "bind_rank_two_contractions",
        lambda *_: [
            {
                "ordinal": 7,
                "lhs": "A",
                "rhs": "B",
                "output": "C",
                "m": m,
                "k": k,
                "n": n,
            }
        ],
    )
    ex = SimpleNamespace(
        identity_ws=lambda *_: None,
        weight_shape=lambda raw: (raw >> 16, raw >> 8 & 255),
    )
    monkeypatch.setattr(P, "ExecuteStateContract", lambda _: ex)
    fields = [
        {
            "bundle": "ConfigMvinRs1",
            "name": name,
            "offset": offset,
            "width": width,
            "status": "derived",
        }
        for name, offset, width in (
            (("shrink", 2, 1), ("pixel_repeats", 8, 5), ("stride", 16, 14))
            if dim == 16
            else (("shrink", 4, 1), ("pixel_repeats", 8, 4), ("stride", 16, 8))
        )
    ]
    fields += [
        {
            "bundle": bundle,
            "name": name,
            "offset": offset,
            "width": width,
            "status": "derived",
        }
        for bundle in ("ComputeRs", "PreloadRs")
        for name, offset, width in (("num_rows", 16, 8), ("num_cols", 8, 8))
    ]
    fields.append(
        {
            "bundle": "PreloadRs",
            "name": "local_addr",
            "width": 8,
            "subfields": [
                {"field": n, "offset": o, "width": w}
                for n, o, w in (
                    ("data", 0, 5),
                    ("is_acc_addr", 7, 1),
                    ("accumulate", 6, 1),
                    ("read_full_acc_row", 5, 1),
                )
            ],
        }
    )
    selected._documents = lambda: ({}, {}, {"layouts": {"fields": fields}})
    # Preserve the original report callback while adding neutral geometry data.
    original_report = type(selected).report

    def report(self):
        return dict(
            original_report(self),
            geometry={
                "array_rows": dim,
                "array_cols": dim,
                "stores": [
                    {"name": "scratchpad", "banks": 1, "rows_per_bank": 32},
                    {"name": "accumulator", "banks": 1, "rows_per_bank": 32},
                ],
            },
        )

    monkeypatch.setattr(type(selected), "report", report)
    # Reissue after installing the neutral selected geometry; no actual owner issued.
    owner = M.WholeProgramMemoryEffects(selected, issued[3])
    selected.operand_observer = lambda: SimpleNamespace(
        isa_constants=lambda _: {"CUSTOM_OPCODE": 0x6B}
    )
    cb = {
        "tensors": {
            name: {"dtype": "i32" if name == "C" else "i8", "shape": list(v.shape)}
            for name, v in {**arrays, "C": np.zeros((m, n), dtype=np.int32)}.items()
        },
        "kernel_abi": {"args": [{"tensor": n} for n in ("A", "B", "C")]},
    }
    args = {
        "selected": selected,
        "source": tmp_path / "source",
        "proof": {},
        "command_buffer": cb,
        "elf": elf,
        "invocation": tmp_path / "invocation",
        "nm": tmp_path / "nm",
        "input_values": arrays,
        "expected_trace_argv": ["neutral"],
    }
    return P, owner, commands, args


def test_whole_program_wrapper_checks_complete_banked_packets_old_default_refuses(
    packet_route,
):
    P, owner, _, args = packet_route
    with pytest.raises(ValueError, match="identity/unshrunk"):
        P.check_execution_packets(**args)
    result = P.check_execution_packets(**args, memory_effects=owner)
    assert result["products"] == 18 and result["raw_i32_values"] == 6
    assert result["fences_preserving_pending_preload"] == 4
    assert result["selected_source_memory_effects"]["protocol"].endswith(".v1")


@pytest.mark.parametrize(
    "fault",
    [
        "pixel",
        "scale",
        "shrink",
        "spad_stride",
        "state",
        "no_config",
        "zero",
        "pending",
    ],
)
def test_whole_wrapper_keeps_all_original_identity_ownership_and_completion_guards(
    packet_route, fault
):
    P, owner, commands, args = packet_route
    ld = next(r for r in commands if r["class"] == "CONFIG_LD")
    if fault == "pixel":
        ld["rs1"]["raw"] += 1 << 8
    elif fault == "scale":
        ld["decoded"]["scale_bits"] = 0
    elif fault == "shrink":
        ld["rs1"]["raw"] |= 1 << 4
    elif fault == "spad_stride":
        ld["rs1"]["raw"] += 1 << 16
    elif fault == "state":
        ld["decoded"]["state_id"] = True
    elif fault == "no_config":
        commands.remove(ld)
    elif fault == "zero":
        next(r for r in commands if r["class"] == "MVIN3")["rs1"]["raw"] = 9999
    else:
        commands[:] = commands[
            : next(i for i, r in enumerate(commands) if r["class"] == "PRELOAD") + 1
        ]
    with pytest.raises(ValueError):
        P.check_execution_packets(**args, memory_effects=owner)


def test_foreign_or_json_packet_bundle_refuses_before_work(packet_route):
    P, _, _, args = packet_route
    with pytest.raises(TypeError, match="original live"):
        P.check_execution_packets(**args, memory_effects={"status": "pass"})


def test_whole_wrapper_rechecks_live_selected_effects_after_packet_induction(
    packet_route, monkeypatch
):
    P, owner, _, args = packet_route
    M = importlib.import_module("gemmini_whole_program_memory_effects")
    original = P._check_bound_execution_packets

    def changed(**kwargs):
        result = original(**kwargs)
        M._ISSUED[owner]["loads"].ports["MVIN"] = 2
        return result

    monkeypatch.setattr(P, "_check_bound_execution_packets", changed)
    with pytest.raises(ValueError, match="derivation changed"):
        P.check_execution_packets(**args, memory_effects=owner)


@pytest.mark.parametrize("packet_route", ["retained_public_35"], indirect=True)
def test_exact_public35_load_fields_have_complete_independent_k_n_tail_products(
    packet_route,
):
    P, owner, _, args = packet_route
    with pytest.raises(ValueError, match="identity/unshrunk"):
        P.check_execution_packets(**args)
    result = P.check_execution_packets(**args, memory_effects=owner)
    assert result["products"] == 3689
    assert result["raw_i32_values"] == 217
    assert result["computes"] == 4
