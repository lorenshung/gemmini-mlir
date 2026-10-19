"""Original live owner for one exact source/input/executable correspondence.

The constructor selects original artifacts, not serialized success. Every
callback reopens source, compiler outputs, complete native values and actual
issued product/state effects. This admits no timing or general dense backend.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_source_packet_artifacts import check_execution_packets
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import native_model_execution as native
from merlin.targetgen.golden_store import load_golden
from merlin.targetgen.native_model_output_binding import (
    bind_source_outputs,
    source_output_values,
)
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from xdsl.parser import Parser


class SourceExecutionOwner:
    def __init__(
        self,
        *,
        arithmetic,
        selected,
        result,
        command_buffer,
        lowered,
        trace_invocation,
        nm,
    ):
        if (
            type(arithmetic) is not IntegerArithmeticOwner
            or type(selected) is not SelectedSchedulingTarget
        ):
            raise ValueError(
                "source execution needs its original independently derived arithmetic owner"
            )
        self._arithmetic, self._selected = arithmetic, selected
        self._arithmetic_identity = arithmetic.identity
        self._selected_inputs = selected.report()["selected_inputs"]
        self._paths = {
            name: Path(path).resolve(strict=True)
            for name, path in {
                "result": result,
                "command_buffer": command_buffer,
                "lowered": lowered,
                "trace_invocation": trace_invocation,
                "nm": nm,
            }.items()
        }
        self._pins = {name: _pin(path) for name, path in self._paths.items()}
        for name in (
            "gemmini_source_execution",
            "gemmini_source_packet_artifacts",
            "gemmini_packet_contractions",
            "gemmini_execute_state",
            "merlin.llvmlower.source_contraction_bindings",
            "merlin.perf.compiler_plan_evidence",
            "merlin.perf.executed_trace",
            "merlin.runtime.out_b64",
            "merlin.targetgen.native_model_execution",
            "merlin.targetgen.native_model_output_binding",
            "merlin.common.invocation_record",
        ):
            self._pins[name] = _pin(module_source_path(name))
        checker = (
            Path(__file__).resolve().parents[2]
            / "experiments/shared_load_configuration/qualify.py"
        )
        self._pins["packet_framing_checker"] = _pin(checker)
        self._identity = self._digest()
        self.revalidate()

    def _digest(self):
        return hashlib.sha256(
            json.dumps(
                [self._pins, self._arithmetic_identity, self._selected_inputs],
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @property
    def identity(self):
        return "gemmini.current_source_execution.v1:" + self._identity

    def revalidate(self):
        self._arithmetic.revalidate()
        self._selected.revalidate()
        if (
            self._arithmetic.identity != self._arithmetic_identity
            or self._selected.report()["selected_inputs"] != self._selected_inputs
            or any(
                str(path) != self._pins[name]["path"]
                for name, path in self._paths.items()
            )
        ):
            raise ValueError(
                "original live realization owners or artifact bindings changed"
            )
        if (
            self._arithmetic.record()["selected_inputs"]
            != self._selected.report()["selected_inputs"]
        ):
            raise ValueError(
                "source execution and arithmetic selected different original inputs"
            )
        _revalidate_sources(self._pins)
        if self._digest() != self._identity:
            raise ValueError(
                "original source execution evidence/checker selection changed"
            )

    def source_rule(self):
        self.revalidate()
        return self._arithmetic.source_rule()

    def verify_source_execution(self, source_path, proof, capsule_dir):
        self.revalidate()
        source_path, capsule_dir = Path(source_path), Path(capsule_dir)
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != proof["interface_sha256"]:
            raise ValueError(
                "actual source execution differs from current sealed source proof"
            )
        row = json.loads(self._paths["result"].read_bytes())
        cb = json.loads(self._paths["command_buffer"].read_bytes())
        lowered = self._paths["lowered"].read_text()
        interface = native._pinned_native_file(row["source"], "interface")
        if interface.read_bytes() != source:
            raise ValueError(
                "native executable was not built from this exact original source"
            )
        if (
            row["candidate"]["command_buffer_sha256"]
            != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
            or row["candidate"]["lowered_mlir_sha256"]
            != hashlib.sha256(lowered.encode()).hexdigest()
        ):
            raise ValueError(
                "actual native compiler artifacts differ from selected source realization"
            )
        structural = verify_compiler_global_plan(
            source_text=source.decode(),
            lowered_text=lowered,
            command_buffer=cb,
            candidate_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
            parsed_lowered_module=Parser(make_llvm_context(), lowered).parse_module(),
        )
        if (
            structural.get("status") != "verified"
            or structural.get("control_flow", {}).get("status") != "verified"
        ):
            raise ValueError(
                "actual compiler source/task/ABI/control flow correspondence is unresolved"
            )
        for group in (row["source"], row["build_artifacts"]):
            for role in group:
                native._pinned_native_file(group, role)
        for pin in row["host_build_sources"]:
            if _pin(pin["path"]) != pin:
                raise ValueError("original native build implementation changed")
        inputs, binding = native._bind_inputs(
            cb, interface.parent, target="gemmini", rtl_facts=self._selected.facts_path
        )
        if binding != row["binding"]:
            raise ValueError("actual source entry/weight/mutable ownership changed")
        original_cap = native._pinned_native_file(
            row["source"], "capsule_declaration"
        ).parent
        original = yaml.safe_load((original_cap / "capsule.yaml").read_bytes())
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_bytes())
        golden = load_golden(capsule_dir)
        if original["numeric_policy"] != declaration["numeric_policy"]:
            raise ValueError(
                "native realization cannot change original source numerical tolerances"
            )
        oldgolden = load_golden(original_cap)
        if set(oldgolden["outputs"]) != set(golden["outputs"]) or any(
            not np.array_equal(oldgolden["outputs"][name], values)
            for name, values in golden["outputs"].items()
        ):
            raise ValueError(
                "native realization used different independent complete references"
            )
        names = cb["params"]["global_program_plan"]["entry_bindings"]
        attrs = declaration["operation"]["attributes"]
        outputs = attrs.get("outs") or [attrs["out"]]
        input_names = attrs["arg_order"][: -len(outputs)]
        if len(input_names) != len(names):
            raise ValueError(
                "native realization input roster differs from sealed complete source ABI"
            )
        for name, compiled in zip(input_names, names, strict=True):
            expected = golden["oracle_provenance"]["inputs"][name]
            if not np.array_equal(
                np.asarray(inputs[compiled]).reshape(expected["shape"]),
                np.asarray(expected["decoded"]).reshape(expected["shape"]),
            ):
                raise ValueError(
                    "actual executable input differs from independent sealed input"
                )
        trace = invocation_record.verify(self._paths["trace_invocation"])
        elf = native._pinned_native_file(row, "elf")
        runtime = self._arithmetic.record()["sources"]
        citation = row["tiers"]["L2"]["engine_citation"]
        if (
            citation.get("engine") != "spike"
            or citation.get("target") != "gemmini"
            or citation.get("flags")
            != ["--extlib=" + citation["extension"]["path"], "--extension=gemmini"]
        ):
            raise ValueError(
                "source realization has unsupported or unbound functional runtime flags"
            )
        for role, name in (
            ("binary", "runtime_bin/spike"),
            ("extension", "runtime_lib/libgemmini.so"),
        ):
            if _pin(citation[role]["path"]) != runtime[name]:
                raise ValueError(
                    "source realization uses a different functional arithmetic runtime"
                )
        for console in (
            native._pinned_native_file(row["tiers"]["L2"], "console"),
            Path(trace["stdout"]["path"]),
        ):
            actual, decoder = {}, OutB64Decoder()
            for line in console.read_text().splitlines():
                decoder.consume(line.split(), actual)
            decoder.require_closed()
            actual = source_output_values(
                actual, bind_source_outputs(cb, capsule_dir, source_path, golden)
            )
            if set(actual) != set(golden["outputs"]) or any(
                not np.array_equal(
                    np.asarray(actual[name]).reshape(-1), np.asarray(value).reshape(-1)
                )
                for name, value in golden["outputs"].items()
            ):
                raise ValueError(
                    "actual complete native/trace outputs differ from independent sealed reference"
                )
        packets = check_execution_packets(
            selected=self._selected,
            source=source_path,
            proof=proof,
            command_buffer=cb,
            elf=elf,
            invocation=self._paths["trace_invocation"],
            nm=self._paths["nm"],
            input_values=inputs,
            expected_trace_argv=[
                citation["binary"]["path"],
                *citation["flags"],
                "-l",
                "--log-commits",
                str(elf),
            ],
        )
        self.revalidate()
        return {
            "status": "verified_source_execution_correspondence",
            "source_sha256": proof["interface_sha256"],
            "source_rule": self.source_rule(),
            "owner_identity": self.identity,
            "actual_packet_effects": packets,
            "complete_output_values": sum(
                np.asarray(value).size for value in golden["outputs"].values()
            ),
            "scope": "exact original source and signed inputs, actual product/state effects and complete outputs",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "general_dense_adapter_enabled": False,
        }
