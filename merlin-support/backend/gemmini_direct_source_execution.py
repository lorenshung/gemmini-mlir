"""Original actual-effect proof for an explicitly selected direct signed dot.

No global program plan is fabricated. Current source, independent complete
values, real pointer storage and every issued product/readout are rechecked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import yaml
from gemmini_source_execution import SourceExecutionOwner
from gemmini_source_packet_artifacts import check_direct_execution_packets
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.llvmlower.direct_source_contraction import bind_direct_source_contraction
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import native_model_execution as native
from merlin.targetgen.golden_store import load_golden

PROTOCOL = "merlin.native_pointer_source.v1"


class DirectSourceExecutionOwner(SourceExecutionOwner):
    def __init__(self, *, execution_contract, **kwargs):
        if type(self) is not DirectSourceExecutionOwner:
            raise TypeError("direct source proof requires its exact original owner")
        super().__init__(**kwargs)
        if (
            type(execution_contract) is not dict
            or _pin(execution_contract["path"]) != execution_contract
        ):
            raise ValueError(
                "direct source proof needs its original selected execution contract"
            )
        self._pins["execution_contract"] = dict(execution_contract)
        for name in (
            "gemmini_direct_source_execution",
            "gemmini_primitive_host_scope",
            "gemmini_load_streams",
            "gemmini_zero_fill",
            "gemmini_private_zero_storage",
            "merlin.llvmlower.direct_source_contraction",
            "merlin.targetgen.native_pointer_source_binding",
            "merlin.targetgen.native_pointer_source_execution",
        ):
            self._pins[name] = _pin(module_source_path(name))
        self._identity = self._digest()
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.current_direct_source_execution.v1:" + self._identity

    def revalidate(self):
        if type(self) is not DirectSourceExecutionOwner:
            raise ValueError("direct source proof lost its original type")
        if (
            SourceExecutionOwner.revalidate is not _BASE_REVALIDATE
            or _BASE_REVALIDATE.__code__ is not _BASE_REVALIDATE_CODE
        ):
            raise ValueError("original base source verifier changed")
        for name, function, code in _ORIGINAL_API:
            method = getattr(self, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not self
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("direct source proof lost its original callback")
        _BASE_REVALIDATE(self)

    def verify_source_execution(self, source_path, proof, capsule_dir):
        from merlin.targetgen.native_pointer_source_binding import bind_pointer_source

        self.revalidate()
        source_path, capsule_dir = Path(source_path), Path(capsule_dir)
        raw = source_path.read_bytes()
        row = json.loads(self._paths["result"].read_bytes())
        cb = json.loads(self._paths["command_buffer"].read_bytes())
        lowered = self._paths["lowered"].read_bytes()
        if (
            row.get("schema") != "merlin_candidate_native_pointer_source_v1"
            or row.get("numeric", {}).get("status") != "pass"
            or proof.get("interface_sha256") != hashlib.sha256(raw).hexdigest()
            or row["candidate"]["command_buffer_sha256"]
            != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
            or row["candidate"]["lowered_mlir_sha256"]
            != hashlib.sha256(lowered).hexdigest()
        ):
            raise ValueError(
                "direct source proof differs from its complete original native artifacts"
            )
        contraction = bind_direct_source_contraction(source_path, cb, proof)
        interface = native._pinned_native_file(row["source"], "interface")
        if interface.read_bytes() != raw:
            raise ValueError(
                "actual direct executable source differs from this sealed source"
            )
        original_cap = native._pinned_native_file(
            row["source"], "capsule_declaration"
        ).parent
        for group in (row["source"], row["build_artifacts"]):
            for role in group:
                native._pinned_native_file(group, role)
        for pin in row["host_build_sources"]:
            if (
                type(pin) is not dict
                or set(pin)
                not in ({"path", "sha256"}, {"path", "sha256", "size_bytes"})
                or _pin(pin["path"]) != {key: pin[key] for key in ("path", "sha256")}
                or "size_bytes" in pin
                and (
                    type(pin["size_bytes"]) is not int
                    or pin["size_bytes"] != Path(pin["path"]).stat().st_size
                )
            ):
                raise ValueError("direct source original host build source changed")
        inputs, binding = bind_pointer_source(cb, original_cap, interface.parent)
        current_inputs, current_binding = bind_pointer_source(
            cb, capsule_dir, interface.parent
        )
        if (
            binding != row["binding"]
            or binding.get("protocol") != PROTOCOL
            or any(
                current_binding[key] != binding[key]
                for key in ("source_sha256", "inputs", "returns")
            )
            or current_inputs != inputs
        ):
            raise ValueError(
                "actual direct source/input/return storage differs from original native execution"
            )
        original = yaml.safe_load((original_cap / "capsule.yaml").read_bytes())
        current = yaml.safe_load((capsule_dir / "capsule.yaml").read_bytes())
        golden, previous = load_golden(capsule_dir), load_golden(original_cap)
        if (
            original["numeric_policy"] != current["numeric_policy"]
            or set(golden["outputs"]) != set(previous["outputs"])
            or any(
                not np.array_equal(np.asarray(v), np.asarray(previous["outputs"][name]))
                for name, v in golden["outputs"].items()
            )
        ):
            raise ValueError(
                "direct source proof changed independent complete references or numerical policy"
            )
        trace = invocation_record.verify(self._paths["trace_invocation"])
        elf = native._pinned_native_file(row, "elf")
        citation = row["tiers"]["L2"]["engine_citation"]
        runtime = self._arithmetic.record()["sources"]
        if (
            row["tiers"]["L2"].get("status") != "pass"
            or citation.get("engine") != "spike"
            or citation.get("target") != "gemmini"
            or citation.get("flags")
            != ["--extlib=" + citation["extension"]["path"], "--extension=gemmini"]
            or any(
                _pin(citation[role]["path"]) != runtime[name]
                for role, name in (
                    ("binary", "runtime_bin/spike"),
                    ("extension", "runtime_lib/libgemmini.so"),
                )
            )
        ):
            raise ValueError(
                "direct source proof selected a different original functional runtime"
            )
        for console in (
            native._pinned_native_file(row["tiers"]["L2"], "console"),
            Path(trace["stdout"]["path"]),
        ):
            actual, decoder = {}, OutB64Decoder()
            for line in console.read_text().splitlines():
                decoder.consume(line.split(), actual)
            decoder.require_closed()
            mapped = {
                r["source_name"]: actual.get(r["compiled_name"])
                for r in binding["returns"]
            }
            if (
                set(actual) != {r["compiled_name"] for r in binding["returns"]}
                or set(mapped) != set(golden["outputs"])
                or any(
                    not np.array_equal(
                        np.asarray(mapped[name]).reshape(-1),
                        np.asarray(values).reshape(-1),
                    )
                    for name, values in golden["outputs"].items()
                )
            ):
                raise ValueError(
                    "actual complete native/trace values differ from direct source reference"
                )
        selection = row["pointer_execution"]
        from merlin.runtime.harness_render import validate_contract

        harness = validate_contract(
            yaml.safe_load(Path(self._pins["execution_contract"]["path"]).read_bytes())
        ).pointer_abi
        actual_contract = native._pinned_native_file(selection, "selected_contract")
        if (
            selection.get("protocol") != PROTOCOL
            or _pin(actual_contract) != self._pins["execution_contract"]
            or selection.get("entry_symbol") != harness.entry_symbol
            or selection.get("completion_symbol") != harness.completion_symbol
        ):
            raise ValueError(
                "direct source scope has no original selected entry symbol"
            )
        from gemmini_zero_fill import SelectedAccumulatorZeroFill

        packets = check_direct_execution_packets(
            selected=self._selected,
            source=source_path,
            proof=proof,
            command_buffer=cb,
            elf=elf,
            invocation=self._paths["trace_invocation"],
            nm=self._paths["nm"],
            input_values={name: values["values"] for name, values in inputs.items()},
            expected_trace_argv=[
                citation["binary"]["path"],
                *citation["flags"],
                *self._observation_flags,
                str(elf),
            ],
            entry_symbol=selection["entry_symbol"],
            zero_fill=SelectedAccumulatorZeroFill(self._selected, self._arithmetic),
        )
        self.revalidate()
        return {
            "status": "verified_source_execution_correspondence",
            "source_sha256": proof["interface_sha256"],
            "source_rule": self.source_rule(),
            "owner_identity": self.identity,
            "actual_packet_effects": packets,
            "direct_source_binding": contraction,
            "complete_output_values": sum(
                np.asarray(v).size for v in golden["outputs"].values()
            ),
            "scope": "one exact typed direct signed dot, real pointer inputs, complete actual products/readout/outputs and host-store veto",
            "source_binding_protocol": PROTOCOL,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "general_dense_adapter_enabled": False,
        }


_BASE_REVALIDATE = SourceExecutionOwner.revalidate
_BASE_REVALIDATE_CODE = _BASE_REVALIDATE.__code__
_ORIGINAL_API = tuple(
    (
        name,
        getattr(DirectSourceExecutionOwner, name),
        getattr(DirectSourceExecutionOwner, name).__code__,
    )
    for name in (
        "__init__",
        "revalidate",
        "verify_source_execution",
        "source_rule",
        "_digest",
    )
)
