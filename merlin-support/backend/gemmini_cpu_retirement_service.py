"""Selected scalar retirement service, separate from compulsory workload work.

The interval count excludes its opening CSR and includes its closing CSR. We
discard the closing instruction as well: the resulting interior instructions
need distinct scalar retirement opportunities. This is only a necessary core
clock service bound for this emitted schedule. It neither prices instructions
nor establishes DMA traffic, overlap, a full-call floor or timing accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gemmini_native_timing_semantics import FEATURE_ENGINE, NativeTimingSemanticsOwner
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common import invocation_record as I
from merlin.targetgen.rtl import firrtl_struct as F
from merlin_timing import observations as O

SCHEMA = "gemmini.scalar_retirement_service.v1"


def _driver(module, path):
    rows = [row for row in module.connects if row.lhs == F.parse_expr(path)]
    if not rows or any(row.guards or row.rhs != rows[0].rhs for row in rows):
        raise ValueError("retirement path has no unique unconditional source effect")
    return rows[0].rhs


def _expand(module, expression, active=()):
    if expression in active or len(active) > 32:
        raise ValueError("retirement source cone is cyclic or exceeds its bound")
    if type(expression) is F.Ref and not expression.segments:
        node = module.nodes.get(expression.root)
        if node:
            return _expand(module, node[0], (*active, expression))
    if type(expression) is F.Prim:
        return F.Prim(
            expression.op,
            tuple(
                _expand(module, item, (*active, expression)) for item in expression.args
            ),
        )
    return expression


def _data_nodes(module, root):
    pending, seen = [root], set()
    while pending:
        name = pending.pop()
        if name in seen or name not in module.nodes:
            continue
        seen.add(name)
        if len(seen) > 1024:
            raise ValueError("WB data source cone exceeds its bounded proof scope")
        pending.extend(
            ref.root for ref in F._refs_in(module.nodes[name][0]) if not ref.segments
        )
    return seen


def derive_retirement(rocket, csr):
    """Check actual selected FIR effects; a trace vector alone is insufficient."""
    instances = [row[1] for row in rocket.decls.values() if row[0] == "inst"]
    trace = F.field_type(rocket.port_type("io"), ("trace", "insns"))
    csr_trace = F.field_type(csr.port_type("io"), ("trace",))
    if (
        instances.count("CSRFile") != 1
        or not trace
        or trace[0] != "vec"
        or trace[2] != 1
        or csr_trace != trace
        or F.field_type(csr.port_type("io"), ("retire",)) != ("uint", 1)
        or _driver(rocket, "csr.io.retire") != F.Ref("wb_valid")
        or _driver(rocket, "io.trace.insns") != F.parse_expr("csr.io.trace")
    ):
        raise ValueError("selected CPU has no checked scalar WB retirement path")
    address = _expand(rocket, _driver(rocket, "csr.io.rw.addr"))
    if address != F.parse_expr("bits(wb_reg_inst, 31, 20)"):
        raise ValueError("counter address is not selected from the WB instruction")
    command = _expand(rocket, _driver(rocket, "csr.io.rw.cmd"))
    expected = F.parse_expr(
        "and(wb_ctrl.csr, not(mux(wb_reg_valid, UInt<1>(0), UInt<3>(4))))"
    )
    if command != expected:
        raise ValueError("CSR command is not the checked WB command selection")
    # These explicit selected muxes place the CSR value in the same WB path.
    expected_read = F.parse_expr(
        "mux(neq(wb_ctrl.csr, UInt<3>(0)), csr.io.rw.rdata, "
        "mux(wb_ctrl.mul, wb_reg_wdata, wb_reg_wdata))"
    )
    if not any(
        row.rhs == F.Ref("rf_wdata")
        and rocket.decls.get(row.lhs.root, (None,))[0] == "mport"
        for row in rocket.connects
    ):
        raise ValueError("selected WB data has no actual register-file write port")
    reads = [
        _expand(rocket, rocket.nodes[name][0])
        for name in _data_nodes(rocket, "rf_wdata")
        for row in (rocket.nodes[name],)
        if type(row[0]) is F.Prim
        and row[0].op == "mux"
        and row[0].args[1] == F.parse_expr("csr.io.rw.rdata")
    ]
    if expected_read not in reads:
        raise ValueError("CSR sample has no checked same-stage WB data selection")
    return {
        "retire_width": 1,
        "clock_domain": "selected Rocket core clock",
        "cutpoint": "same-stage WB CSR selection and scalar retirement",
        "counter_interval_count": "opening excluded; closing included",
        "service_demand": "strict interior committed instructions; closing excluded too",
        "endpoint_instruction_discount": 1,
        "pipeline_stalls": "can only increase required core clock service interval",
        "counter_clock_equivalence": "original selected counter review remains required",
    }


def _scope_count(report):
    if (
        report.get("schema") != O.REPORT_V2
        or report.get("feature_engine_sha256") != FEATURE_ENGINE
        or not O.feature_complete(report)
        or report["feature_observation"]["scope_kind"] != "counter_pc_pair"
    ):
        raise ValueError(
            "retirement demand needs a complete source-bound counter-PC report"
        )
    binding = report.get("scope_binding", {})
    if (
        binding.get("schema") != "merlin.spike_scope_binding.v1"
        or any(
            binding.get(key) != report.get(key)
            for key in ("elf_sha256", "input_sha256", "scope_sha256")
        )
        or not binding.get("evidence_sha256")
    ):
        raise ValueError(
            "retirement demand lacks its original ELF/input/counter binding"
        )
    return max(0, report["feature_observation"]["scope_instructions"] - 1)


class RetirementServiceContract:
    """Live structural/effect owner, not a saved verdict or timing qualifier."""

    def __init__(self, selected, native_semantics, *, runtime=None):
        if (
            type(selected) is not SelectedSchedulingTarget
            or type(native_semantics) is not NativeTimingSemanticsOwner
        ):
            raise TypeError(
                "retirement service needs original selected and native source owners"
            )
        from merlin_experiments.phase2.archive_spike_runtime import ArchiveSpikeRuntime

        if type(runtime) is not ArchiveSpikeRuntime:
            raise TypeError(
                "retirement service needs the original selected counter runtime"
            )
        runtime.verify()
        selected.revalidate()
        native_semantics.revalidate()
        binding = native_semantics.selected_binding()
        if (
            binding["hardware"] != _pin(selected.facts_path)["sha256"]
            or binding["software"] != _pin(selected.contract_path)["sha256"]
        ):
            raise ValueError(
                "retirement service owners select different hardware/software"
            )
        facts = selected._documents()[0]
        fir = Path(facts["inputs"]["fir_path"])
        if _pin(fir)["sha256"] != facts["inputs"]["fir_sha256"]:
            raise ValueError("retirement service selected FIR source changed")
        modules = F.load_modules(fir, {"Rocket", "CSRFile"})
        self._proof = derive_retirement(modules["Rocket"], modules["CSRFile"])
        record = native_semantics.record()
        timing = json.loads(runtime.timing_json)
        if (
            _pin(runtime.model_path) != record["pins"]["model"]
            or _pin(runtime.observer) != record["pins"]["observer"]
            or timing.get("counter_asm") != "csrr %0, mcycle"
            or timing.get("counter_semantics", {}).get("unit") != "cycles"
            or timing["counter_semantics"].get("width_bits") != 64
        ):
            raise ValueError(
                "retirement service lacks matching selected core-cycle counter semantics"
            )
        self._pins = {
            "implementation": _pin(__file__),
            "selected_firrtl": _pin(fir),
            "firrtl_reader": _pin(F.__file__),
            "observer": record["pins"]["observer"],
            "observer_source": record["pins"][
                "observer_source:riscv/annotated_timing.cc"
            ],
            "model": record["pins"]["model"],
        }
        self._selected, self._native = selected, native_semantics
        self._runtime = runtime
        self._counter_asm = timing["counter_asm"]
        self._binding = binding
        self.revalidate()

    def revalidate(self):
        self._selected.revalidate()
        self._native.revalidate()
        self._runtime.verify()
        _revalidate_sources(self._pins)
        if self._native.selected_binding() != self._binding:
            raise ValueError("retirement service native source binding changed")

    def record(self):
        self.revalidate()
        return {
            "schema": SCHEMA,
            "selected_binding": dict(self._binding),
            "pins": dict(self._pins),
            "proof": dict(self._proof),
            "original_counter_runtime_sha256": self._runtime.sha256,
            "authority": "necessary emitted-schedule service only; no statistical qualification",
            "objective_floor_eligible": False,
            "convergence_eligible": False,
        }

    def evidence(
        self, report_path, invocation_path, counter_review_path, *, source_sha256
    ):
        """Reopen an actual observer product, not a copied timing/count verdict.

        The caller still owns current source-to-ELF/full-output admission. This
        helper binds that caller's source identity without issuing that proof.
        """
        self.revalidate()
        if (
            type(source_sha256) is not str
            or len(source_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in source_sha256)
        ):
            raise ValueError(
                "retirement evidence needs an explicit current source digest"
            )
        report_path, invocation_path = Path(report_path), Path(invocation_path)
        if report_path.stat().st_size > 16 << 20:
            raise ValueError("retirement count report exceeds its admission bound")
        report = json.loads(report_path.read_bytes())
        demand = _scope_count(report)
        invocation = I.verify(invocation_path)
        pin = _pin(report_path)
        observer = self._pins["observer"]
        argv = invocation.get("argv", [])
        guest = argv[argv.index("--") + 1 :] if "--" in argv else []
        actual_products = [
            row
            for row in invocation.get("outputs", ())
            if row.get("sha256") == pin["sha256"]
        ]
        if (
            invocation.get("stage") != "archive_screen_count"
            or len(actual_products) != 1
            or not any(
                row.get("sha256") == report["elf_sha256"]
                for row in invocation.get("inputs", ())
            )
            or observer not in invocation.get("dependencies", ())
            or self._pins["model"] not in invocation.get("dependencies", ())
            or not guest
            or guest[0] != observer["path"]
            or "--timing-model=" + self._pins["model"]["path"] not in guest
            or "--timing-input-sha256=" + report["input_sha256"] not in guest
            or "--timing-report=" + actual_products[0]["path"] not in guest
            or report.get("model_sha256") != self._binding["runtime"]
        ):
            raise ValueError(
                "retirement count product lacks its original selected observer process"
            )
        review_path = Path(counter_review_path)
        if review_path.stat().st_size > 4 << 20:
            raise ValueError("retirement counter review exceeds its admission bound")
        review = json.loads(review_path.read_bytes())
        review_digest = hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        binding = report["scope_binding"]
        if (
            review_digest != binding["evidence_sha256"]
            or review.get("counter_asm") != self._counter_asm
            or any(
                review.get(key) != binding[key]
                for key in ("elf_sha256", "begin_pc", "end_pc", "occurrence")
            )
        ):
            raise ValueError("retirement count changed its original counter review")
        self.revalidate()
        provenance = (
            "source-checked scalar WB retirement; complete original counter-PC interior"
        )
        return {
            "schema": "merlin.phase2.service_effect_evidence.v1",
            "source_sha256": source_sha256,
            "elf_sha256": report["elf_sha256"],
            "scope_sha256": report["scope_sha256"],
            "device_sha256": report["domain"]["device_sha256"],
            "scope": "emitted_schedule",
            "resources": {
                "cpu.scalar_retirement": {
                    "unit": "committed instruction",
                    "capacity_per_cycle_upper": self._proof["retire_width"],
                    "provenance": provenance,
                }
            },
            "events": [
                {
                    "id": "strict_counter_window_interior",
                    "demand": {"cpu.scalar_retirement": demand},
                    "compulsory": False,
                    "provenance": provenance,
                }
            ],
            "completion_chains": [],
            "unresolved": {
                "complete_call": "packing, dispatch, accelerator completion, shared resources and overlap remain separately bounded or UNKNOWN"
            },
            "provenance": {
                "report": pin,
                "original_process_product": actual_products[0],
                "invocation": _pin(invocation_path),
                "counter_review": _pin(review_path),
                "source_owner": self.record(),
            },
            "interpretation": "emitted work including redundant host work; never a compulsory workload floor",
            "attainment": None,
            "measurement_comparability": "original selected core-cycle counter semantics reopened; no measured target cycles supplied here",
            "convergence_eligible": False,
        }
