"""Narrow source/count arithmetic tests; these fixtures qualify no hardware."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import gemmini_cpu_retirement_service as S
from merlin.perf.capsule_roofline import compose_service_floor


def source(name, body):
    return S.F.parse_module(
        name, list(enumerate(("module " + name + " :\n" + body).splitlines(), 1))
    )


def fixture():
    trace = "{valid : UInt<1>, insn : UInt<32>}[1]"
    rocket = source(
        "Rocket",
        f"  output io : {{trace : {{insns : {trace}}}}}\n"
        "  inst csr of CSRFile\n"
        "  connect csr.io.retire, wb_valid\n"
        "  connect io.trace.insns, csr.io.trace\n"
        "  connect io.trace.insns, csr.io.trace\n"
        "  node addr = bits(wb_reg_inst, 31, 20)\n"
        "  connect csr.io.rw.addr, addr\n"
        "  node cmd = and(wb_ctrl.csr, not(mux(wb_reg_valid, UInt<1>(0), UInt<3>(4))))\n"
        "  connect csr.io.rw.cmd, cmd\n"
        "  node wb_csr_value = mux(neq(wb_ctrl.csr, UInt<3>(0)), csr.io.rw.rdata, mux(wb_ctrl.mul, wb_reg_wdata, wb_reg_wdata))\n"
        "  node rf_wdata = mux(other, previous, wb_csr_value)\n"
        "  infer mport write_port = rf[address], clock\n"
        "  connect write_port, rf_wdata\n",
    )
    csr = source("CSRFile", f"  output io : {{retire : UInt<1>, trace : {trace}}}\n")
    return rocket, csr


def report(instructions=6):
    binding = {
        "schema": "merlin.spike_scope_binding.v1",
        "elf_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "scope_sha256": "c" * 64,
        "evidence_sha256": "d" * 64,
        "begin_pc": 16,
        "end_pc": 32,
        "occurrence": 0,
    }
    return {
        "schema": S.O.REPORT_V2,
        "feature_engine_sha256": S.FEATURE_ENGINE,
        **{key: binding[key] for key in ("elf_sha256", "input_sha256", "scope_sha256")},
        "execution_completed": True,
        "functional_status": "PASS",
        "raw_functional_exit_code": 0,
        "instructions": instructions + 10,
        "feature_counts": {
            "cpu.integer": instructions - 1,
            "cpu.csr": 1,
            "cpu.system": 0,
        },
        "feature_observation": {
            "schema": S.O.COUNT_SCHEMA,
            "complete": True,
            "scope_instructions": instructions,
            "counted_cpu_instructions": instructions,
            "scope_kind": "counter_pc_pair",
            "window_index": 0,
            "primary_cpu_features": ["cpu.integer", "cpu.csr", "cpu.system"],
            "unresolved": [],
            "qualification": "not_established",
        },
        "scope_binding": binding,
        "counter_windows": [
            {
                "begin_pc": 16,
                "end_pc": 32,
                "instructions": instructions,
                "estimated_cycles": None,
            }
        ],
        "estimated_cycles": None,
        "timing_complete": False,
        "cycle_breakdown": {
            "scope": "whole_trace",
            "resources": [{"busy_unit_cycles": 999999}],
        },
    }


def test_actual_retirement_and_wb_data_effects_required_not_trace_width_only():
    rocket, csr = fixture()
    proof = S.derive_retirement(rocket, csr)
    assert proof["retire_width"] == 1
    assert proof["endpoint_instruction_discount"] == 1
    assert "original selected counter review" in proof["counter_clock_equivalence"]


@pytest.mark.parametrize(
    "mutation",
    [
        "wide",
        "retire",
        "addr",
        "cmd",
        "dead_read",
        "no_write",
        "guard",
        "duplicate_conflict",
    ],
)
def test_unproved_or_decorative_retirement_effects_refuse(mutation):
    rocket, csr = fixture()
    if mutation == "wide":
        kind, body, line = csr.decls["io"]
        csr.decls["io"] = kind, body.replace("[1]", "[2]"), line
    elif mutation == "retire":
        row = next(
            row for row in rocket.connects if row.lhs == S.F.parse_expr("csr.io.retire")
        )
        rocket.connects.remove(row)
    elif mutation == "addr":
        _value, line, locator = rocket.nodes["addr"]
        rocket.nodes["addr"] = (
            S.F.parse_expr("bits(ex_reg_inst, 31, 20)"),
            line,
            locator,
        )
    elif mutation == "cmd":
        _value, line, locator = rocket.nodes["cmd"]
        rocket.nodes["cmd"] = S.F.Ref("id_command"), line, locator
    elif mutation == "dead_read":
        __value, line, locator = rocket.nodes["rf_wdata"]
        rocket.nodes["rf_wdata"] = S.F.Ref("previous"), line, locator
    elif mutation == "no_write":
        rocket.connects = [
            row for row in rocket.connects if row.lhs != S.F.Ref("write_port")
        ]
    else:
        row = next(
            row
            for row in rocket.connects
            if row.lhs == S.F.parse_expr("io.trace.insns")
        )
        rocket.connects.append(
            S.F.Connect(
                row.lhs,
                S.F.Ref("foreign"),
                (S.F.Guard(S.F.Ref("g"), True),) if mutation == "guard" else (),
                99,
                None,
            )
        )
    with pytest.raises(ValueError):
        S.derive_retirement(rocket, csr)


@pytest.mark.parametrize("count,expected", [(1, 0), (2, 1), (1773, 1772)])
def test_strict_interior_demand_no_prices_or_whole_trace_resource_sum(count, expected):
    observed = report(count)
    assert S._scope_count(observed) == expected
    assert observed["estimated_cycles"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "foreign_engine",
        "wrong_elf",
        "whole_trace",
        "window",
        "count_disagreement",
        "counter_reconfiguration",
        "system_control",
    ],
)
def test_missing_foreign_or_inconsistent_scope_never_gets_a_floor(mutation):
    observed = copy.deepcopy(report())
    if mutation == "incomplete":
        observed["feature_observation"]["complete"] = False
    elif mutation == "foreign_engine":
        observed["feature_engine_sha256"] = "e" * 64
    elif mutation == "wrong_elf":
        observed["scope_binding"]["elf_sha256"] = "e" * 64
    elif mutation == "whole_trace":
        observed["feature_observation"]["scope_kind"] = "whole_program"
    elif mutation == "window":
        observed["counter_windows"][0]["begin_pc"] += 4
    elif mutation == "count_disagreement":
        observed["feature_counts"]["cpu.integer"] += 1
    else:
        feature = "cpu.csr" if mutation == "counter_reconfiguration" else "cpu.system"
        observed["feature_counts"][feature] += 1
        observed["feature_counts"]["cpu.integer"] -= 1
    with pytest.raises(ValueError):
        S._scope_count(observed)


def test_emitted_host_waste_cannot_increase_frozen_workload_floor():
    resources = {
        "cpu.scalar_retirement": {
            "unit": "committed instruction",
            "capacity_per_cycle_upper": 1,
            "provenance": "checked structural effect",
        }
    }
    event = {
        "id": "interior",
        "demand": {"cpu.scalar_retirement": S._scope_count(report())},
        "compulsory": False,
        "provenance": "original scoped emitted work",
    }
    result = compose_service_floor(
        resources, [event], scope="emitted_schedule", unresolved={"device": "UNKNOWN"}
    )
    assert result["floor_cycles"] == 5
    assert result["status"] == "partial"
    assert result["convergence_eligible"] is False
    with pytest.raises(ValueError, match="compulsory"):
        compose_service_floor(resources, [event], scope="workload")


def test_saved_success_cannot_construct_source_owner():
    with pytest.raises(TypeError, match="original selected and native source owners"):
        S.RetirementServiceContract({"status": "PASS"}, {"qualified": True})


def test_owner_metadata_returns_defensive_copies_without_issuing_a_source_owner(
    monkeypatch,
):
    # Exercise only metadata copying; bypassing construction here establishes no
    # live source or execution authority and is never a production constructor.
    owner = object.__new__(S.RetirementServiceContract)
    owner._record = {"pins": {"original": {"sha256": "a" * 64}}}
    owner._binding = {"hardware": "b" * 64}
    owner._identity = "c" * 64
    monkeypatch.setattr(owner, "revalidate", lambda: None)
    first = owner.record()
    first["pins"]["original"]["sha256"] = "changed"
    binding = owner.selected_binding()
    binding["hardware"] = "changed"
    assert owner.record()["pins"]["original"]["sha256"] == "a" * 64
    assert owner.selected_binding()["hardware"] == "b" * 64
    assert owner.identity == S.SCHEMA + ":" + "c" * 64
