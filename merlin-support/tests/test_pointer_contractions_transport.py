"""Neutral selection and all-output host-veto controls; no target qualification."""
# ruff: noqa: F811 -- imported pytest fixture

import importlib
import json
from types import SimpleNamespace

import pytest
from test_generated_source_diagnostic import issued  # noqa: F401


def test_multiple_dots_require_explicit_original_owner_and_contract(issued, tmp_path):
    M, original, inputs, *_ = issued
    D = importlib.import_module("gemmini_direct_source_execution")
    assert M._STATE(original)["owner_type"] is not D.PointerContractionsExecutionOwner
    selected = dict(inputs, source_binding_protocol=D.POINTER_CONTRACTIONS_PROTOCOL)
    with pytest.raises(ValueError, match="execution contract"):
        M.GeneratedSourceDiagnosticIssuer(selected)
    contract = tmp_path / "pointer-contract"
    contract.write_text("neutral selected contract")
    selected["execution_contract"] = str(contract)
    owner = M.GeneratedSourceDiagnosticIssuer(selected)
    state = M._STATE(owner)
    assert state["owner_type"] is D.PointerContractionsExecutionOwner
    assert "merlin.llvmlower.pointer_source_contractions" in state["sources"]
    record = owner.record()
    assert record["source_binding_protocol"] == D.POINTER_CONTRACTIONS_PROTOCOL
    assert record["timing_authority"] == record["promotion_authority"] == "NONE"
    assert record["hardware_execution_qualified"] is False
    state["owner_type"] = D.DirectSourceExecutionOwner
    with pytest.raises(ValueError):
        owner.revalidate()


@pytest.mark.parametrize("reverse", [False, True])
def test_live_owner_class_cannot_switch_protocol(reverse):
    D = importlib.import_module("gemmini_direct_source_execution")
    first, second = D.DirectSourceExecutionOwner, D.PointerContractionsExecutionOwner
    if reverse:
        first, second = second, first
    owner = object.__new__(first)
    # Neutral control of the constructor's private class binding only.
    D._OWNER_SELECTIONS[owner] = first
    owner.__class__ = second
    with pytest.raises(ValueError, match="originally selected owner type"):
        owner.revalidate()
    with pytest.raises(ValueError, match="originally selected owner type"):
        _ = owner.identity


@pytest.mark.parametrize("multiple", [False, True])
def test_failed_or_subclassed_constructor_never_leaves_a_selected_owner(multiple):
    D = importlib.import_module("gemmini_direct_source_execution")
    selected = (
        D.PointerContractionsExecutionOwner
        if multiple
        else D.DirectSourceExecutionOwner
    )
    owner = object.__new__(selected)
    with pytest.raises(TypeError):
        selected.__init__(owner, execution_contract={})
    assert owner not in D._OWNER_SELECTIONS
    with pytest.raises(ValueError, match="originally selected owner type"):
        owner.revalidate()

    class Foreign(selected):
        pass

    with pytest.raises(TypeError, match="exact original owner"):
        Foreign(execution_contract={})


def test_operator_pointer_protocol_is_explicit_and_never_rewrites_emission():
    from test_public_source_admission import I

    protocol = "gemmini.pointer_source_contractions.v1"
    cb = {
        "compiler_pointer_abi": {"version": 1, "arguments": ["a", "b", "c", "y", "z"]}
    }
    before = json.loads(json.dumps(cb))
    assert I.select_abi(cb, {}, "/contract")[0] == "merlin.native_pointer_source.v1"
    route, inputs = I.select_abi(cb, {"source_binding_protocol": protocol}, "/contract")
    assert route == inputs["source_binding_protocol"] == protocol
    assert cb == before and inputs["execution_contract"] == "/contract"
    with pytest.raises(ValueError, match="requires the emitted pointer ABI"):
        I.select_abi({"kernel_abi": {"kind": "whole_program"}}, inputs, "/contract")
    with pytest.raises(ValueError, match="unknown requested"):
        I.select_abi(cb, {"source_binding_protocol": "foreign"}, "/contract")


def test_new_binder_is_pinned_before_any_factory_restore(tmp_path, monkeypatch):
    F = importlib.import_module("gemmini_generated_source_factory")
    paths = {}

    def source(name):
        if name not in paths:
            paths[name] = tmp_path / (name + ".py")
            paths[name].write_text("# neutral helper\n")
        return paths[name]

    monkeypatch.setattr(F, "module_source_path", source)
    selection = F.original_factory(
        {"source_binding_protocol": "gemmini.pointer_source_contractions.v1"}
    )
    binder = source("merlin.llvmlower.pointer_source_contractions")
    assert str(binder) in {row["path"] for row in selection.record()["sources"]}
    calls = []
    monkeypatch.setattr(
        F, "_original_restore", lambda *args, **kwargs: calls.append("forbidden")
    )
    binder.write_text("# changed before any privileged callback\n")
    with pytest.raises(ValueError, match="factory or inputs changed"):
        selection.create(work_root=tmp_path / "fresh")
    assert not calls


@pytest.fixture
def packet_transport(tmp_path, monkeypatch):
    P = importlib.import_module("gemmini_source_packet_artifacts")
    B = importlib.import_module("merlin.llvmlower.pointer_source_contractions")
    L = importlib.import_module("gemmini_load_streams")
    H = importlib.import_module("gemmini_direct_host_scope")
    arguments = [f"a{i}" for i in range(6)] + ["Y0", "Y3", "Y1", "Y2"]
    cb = {
        "compiler_pointer_abi": {"version": 1, "arguments": arguments},
        "tensors": {name: {} for name in arguments},
    }
    bindings = tuple(
        {
            "ordinal": 2 + 3 * i,
            "lhs": "a0",
            "rhs": f"a{i + 1}",
            "output": name,
            "m": 2,
            "k": 2,
            "n": 2,
        }
        for i, name in enumerate(arguments[6:])
    )
    monkeypatch.setattr(B, "bind_pointer_source_contractions", lambda *args: bindings)
    monkeypatch.setattr(
        L,
        "SelectedLoadStreams",
        lambda _: SimpleNamespace(record=lambda: {"neutral": True}),
    )
    calls = []

    def checked(**kwargs):
        calls.append(kwargs)
        return {"neutral_product_boundary": True}

    monkeypatch.setattr(P, "_check_bound_execution_packets", checked)
    symbols = [(0x3000 + i * 0x100, 16, f"tensor_{i}") for i in range(len(arguments))]
    checker = SimpleNamespace(_symbols=lambda *args: symbols)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(P.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(P.importlib.util, "module_from_spec", lambda _: checker)
    monkeypatch.setattr(
        H,
        "linked_code_extents",
        lambda *args: {"kernel": (0x2000, 16), "main": (0x1000, 32)},
    )
    log = tmp_path / "trace.txt"
    monkeypatch.setattr(
        P.invocation_record, "verify", lambda _: {"stderr": {"path": str(log)}}
    )
    selected = SimpleNamespace(
        operand_observer=lambda: SimpleNamespace(
            isa_constants=lambda _: {"CUSTOM_OPCODE": 0x7B}
        )
    )

    def run(store_index=None):
        middle = "core 0: 3 0x2004 (00000013)"
        if store_index is not None:
            address = symbols[store_index][0]
            middle = f"core 0: 3 0x2004 (00002023) mem {address:#x} 0x00000001"
        log.write_text(
            f"core 0: 3 0x1000 (000000ef) x1 0x1004\ncore 0: 3 0x2000 (00000013)\n{middle}\ncore 0: 3 0x1004 (00000013)\n"
        )
        return P.check_direct_execution_packets(
            selected=selected,
            source=tmp_path / "source",
            proof={},
            command_buffer=cb,
            elf=tmp_path / "elf",
            invocation=tmp_path / "invocation",
            nm=tmp_path / "nm",
            input_values={},
            expected_trace_argv=[],
            entry_symbol="kernel",
            source_binding_protocol=P.POINTER_CONTRACTIONS_PROTOCOL,
        )

    return run, calls, bindings


def test_every_contraction_reaches_packet_checker_and_one_original_call_veto(
    packet_transport,
):
    run, calls, bindings = packet_transport
    report = run()
    assert [row.output for row in calls[0]["contractions"]] == [
        row["output"] for row in bindings
    ]
    assert report["host_output_scope"]["kernel_cpu_result_overwrites"] == 0
    assert report["host_output_scope"]["timing_authority"] == "NONE"


@pytest.mark.parametrize("output_index", range(4))
def test_host_overwrite_of_each_returned_output_is_rejected(
    packet_transport, output_index
):
    run, *_ = packet_transport
    with pytest.raises(ValueError, match="CPU overwrites"):
        run(6 + output_index)
