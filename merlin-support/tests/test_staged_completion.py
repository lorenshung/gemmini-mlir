"""Synthetic control cones exercise refusal; selected RTL is a separate run."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def adapter(monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "backend"
    monkeypatch.syspath_prepend(str(backend))
    spec = importlib.util.spec_from_file_location(
        "staged_under_test", backend / "gemmini_staged_completion.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def modules(
    adapter, *, busy_exception=True, following=True, pending=True, execute_busy=False
):
    busy = "and(eq(empty,UInt<1>(0)),eq(and(eq(utilization,UInt<2>(1)),solitary_preload),UInt<1>(0)))"
    if not busy_exception:
        busy = "eq(empty,UInt<1>(0))"
    suffix = "or(reservation_station.io.busy,spad.io.busy)"
    if execute_busy:
        suffix = "or(" + suffix + ",ex_controller.io.busy)"
    following_gate = "cmd_q.io.deq.valid[1]" if following else "UInt<1>(1)"
    pending_gate = (
        "eq(or(pending_completed_rob_ids[0].valid,pending_completed_rob_ids[1].valid),UInt<1>(0))"
        if pending
        else "UInt<1>(1)"
    )
    bodies = {
        "ReservationStation": "  connect io.busy, " + busy,
        "Gemmini": "  connect io.busy, " + suffix,
        "ExecuteController": (
            "  node DoConfig = eq(cmd_q.io.deq.bits[0].cmd.inst.funct,UInt<1>(0))\n"
            "  node config_drained = and(DoConfig,and(eq(matmul_in_progress,UInt<1>(0)),"
            + pending_gate
            + "))\n"
            "  when eq(control_state,UInt<2>(0)) :\n"
            "    when cmd_q.io.deq.valid[0] :\n"
            "      when config_drained :\n"
            "        connect cmd_q.io.deq.pop, UInt<1>(1)\n"
            "      else :\n"
            "        when and(eq(cmd_q.io.deq.bits[0].cmd.inst.funct,UInt<3>(6)),and("
            + following_gate
            + ",eq(raw_hazard_pre,UInt<1>(0)))) :\n"
            "          connect performing_single_preload, UInt<1>(1)\n"
        ),
    }
    return {
        name: adapter.F.parse_module(
            name, list(enumerate(("module " + name + " :\n" + body).splitlines(), 1))
        )
        for name, body in bodies.items()
    }


def test_lone_preload_can_survive_fence_and_waits_for_following_command(adapter):
    proof = adapter.prove(modules(adapter), config_function=0, preload_function=6)
    assert (
        next(
            row
            for row in proof["reservation_busy"]
            if row["valid_entries"] == 1 and row["solitary_preload"]
        )["busy"]
        is False
    )
    assert (
        next(
            row
            for row in proof["reservation_busy"]
            if row["valid_entries"] == 1 and not row["solitary_preload"]
        )["busy"]
        is True
    )
    assert not any(
        row["start"] for row in proof["preload_start"] if not row["following_valid"]
    )
    assert not any(
        row["accepted"]
        for row in proof["execute_config_accept"]
        if row["pending_completion"] or row["matmul_in_progress"]
    )


@pytest.mark.parametrize(
    "change", ["busy_exception", "following", "pending", "execute_busy"]
)
def test_changed_control_effects_do_not_reuse_original_contract(adapter, change):
    options = {change: change == "execute_busy"}
    with pytest.raises(ValueError):
        adapter.prove(
            modules(adapter, **options), config_function=0, preload_function=6
        )


def test_contract_separates_rs_release_state_visibility_and_fence(adapter):
    contract = adapter.event_contract()
    assert (
        contract["load_store_configuration"]["reservation_release"]
        != contract["load_store_configuration"]["configuration_visible"]
    )
    assert contract["load_store_configuration"]["dma_tracker_allocation"] is False
    assert "configuration.execute" in contract["preload"]["reads_tokens"]
    assert (
        "pending_preload.reservation" in contract["cpu_fence"]["must_preserve_tokens"]
    )
    assert contract["native_implementation_qualified"] is False
