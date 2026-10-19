"""Live original primitive-pair L2 diagnostic, never a final qualification.

Ordinary numerical grading stays untouched, including mandatory pending L3.
This separate owner reopens the original grade/build/caller and freshly checks
actual packets, immutable inputs, all products and outputs for each current ELF.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml
from gemmini_cleanup_issue import check_issued_program
from gemmini_cleanup_layer_artifacts import _ordinary
from gemmini_primitive_context import PrimitiveCompilerContext
from gemmini_primitive_grade import _read_arm
from gemmini_primitive_packets import check_primitive_packets
from gemmini_primitive_source import bind_primitive, check_integer_values
from gemmini_staged_tile_effects import check_output_tile_permutation
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.paths import module_source_path
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.contract.readback_policy import canonical_sha256


class PrimitiveDiagnosticOwner:
    """Explicit additional original effect checker; no JSON qualification loader."""

    def __init__(self, qualification):
        if type(self) is not PrimitiveDiagnosticOwner:
            raise ValueError("primitive subclasses cannot issue an original diagnostic")
        self._initialize_context(PrimitiveCompilerContext.from_zero(qualification))

    @classmethod
    def from_rhs_qualification(cls, qualification):
        """Select only the exact original RHS owner's checked base compiler seed."""
        if cls is not PrimitiveDiagnosticOwner:
            raise ValueError("primitive subclasses cannot issue an original diagnostic")
        context = PrimitiveCompilerContext.from_rhs(qualification)
        selected = object.__new__(cls)
        selected._initialize_context(context)
        return selected

    def _initialize_context(self, context):
        if type(context) is not PrimitiveCompilerContext:
            raise ValueError(
                "primitive diagnostic requires its original live compiler context"
            )
        context.revalidate()
        self.context, self.qualification = context, context.qualification
        self._context_identity = context.identity
        self.selected = context.seed.tile.selected
        self.arithmetic = context.seed.tile.arithmetic
        self.cleanup, self.nm = context.seed.cleanup, context.seed.nm
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                __name__,
                "gemmini_primitive_context",
                "gemmini_primitive_caller",
                "gemmini_primitive_grade",
                "gemmini_primitive_source",
                "gemmini_primitive_packets",
                "gemmini_primitive_host_scope",
                "gemmini_packet_contractions",
                "gemmini_source_packet_artifacts",
                "gemmini_cleanup_issue",
                "gemmini_cleanup_layer_artifacts",
                "gemmini_staged_tile_effects",
                "gemmini_staged_rhs_effects",
                "gemmini_execute_state",
                "gemmini_zero_initializer_capability",
                "merlin.runtime.harness_render",
                "merlin.runtime.direct_kernel_harness",
                "merlin.runtime.backends.chipyard_rocc_bound",
                "merlin.targetgen.contract.interface_emit",
                "merlin.targetgen.capsule_golden",
                "merlin.targetgen.capsule_inputs",
                "merlin.targetgen.executed_program",
            )
        }
        self._identity = canonical_sha256([self._context_identity, self._sources])

    @property
    def identity(self):
        return "gemmini.primitive_diagnostic.v2:" + self._identity

    def revalidate(self):
        if type(self.context) is not PrimitiveCompilerContext:
            raise ValueError("primitive original live compiler context was replaced")
        self.context.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.context.identity != self._context_identity
            or self.qualification is not self.context.qualification
            or canonical_sha256([self._context_identity, self._sources])
            != self._identity
            or self.selected is not self.context.seed.tile.selected
            or self.arithmetic is not self.context.seed.tile.arithmetic
            or self.cleanup is not self.context.seed.cleanup
            or self.nm != self.context.seed.nm
        ):
            raise ValueError(
                "primitive original selection/semantics/checker identity changed"
            )

    def check_package_binding(self, package):
        self.revalidate()
        result = self.context.check_binding(package)
        self.revalidate()
        return result

    def check_pair(
        self,
        *,
        capsule_dir,
        baseline_result,
        baseline_package,
        candidate_result,
        candidate_package,
        work_root,
    ):
        from merlin.targetgen import capsule_golden as CG

        self.revalidate()
        started = time.monotonic()
        capsule_dir, work_root = Path(capsule_dir), Path(work_root)
        if (
            not work_root.is_absolute()
            or work_root.exists()
            or work_root.resolve() != work_root
            or any(p.is_symlink() for p in work_root.parents)
        ):
            raise ValueError(
                "primitive proof requires a new ordinary independent work root"
            )
        pins = {
            name: _pin(_ordinary(capsule_dir / name))
            for name in (
                "capsule.yaml",
                "capsule.interface.mlir",
                "golden.yaml",
                "golden.npz",
            )
        }
        if any(Path(pin["path"]).stat().st_size > 64 << 20 for pin in pins.values()):
            raise ValueError(
                "primitive original source/reference leaves exceed the bounded byte domain"
            )
        capsule = yaml.safe_load(Path(pins["capsule.yaml"]["path"]).read_bytes())
        capsule["__dir__"] = str(capsule_dir)
        source = Path(pins["capsule.interface.mlir"]["path"]).read_text()
        bind_primitive(capsule, source)
        inputs = {
            name: np.asarray(tensor.data).reshape(tensor.shape)
            for name, tensor in CG.materialize_capsule_leaves(capsule).items()
        }
        golden = CG.golden(capsule, capsule_dir)
        arms, streams, rhs_names, costs = {}, [], [], {}
        for label, result, package in (
            ("baseline", baseline_result, baseline_package),
            ("candidate", candidate_result, candidate_package),
        ):
            tick = time.monotonic()
            arm = _read_arm(self, result_path=result, package=package, capsule=capsule)
            if Path(arm["source"]["path"]).read_text() != source:
                raise ValueError(
                    "primitive ordinary grade used another original source"
                )
            contraction = bind_primitive(capsule, source, arm["command_buffer"])
            bind_primitive(capsule, source, arm["original_command_buffer"])
            caller_order = list(arm["original_command_buffer"]["tensors"])
            if caller_order != [contraction.rhs, contraction.lhs, contraction.output]:
                raise ValueError(
                    "primitive original source pointer order differs from resident weight/lhs/output ABI"
                )
            numeric = check_integer_values(
                contraction, inputs, golden, self.arithmetic.source_rule()
            )
            actual, decoder = {}, OutB64Decoder()
            for line in arm["reference_console"].read_text().splitlines():
                decoder.consume(line.split(), actual)
            decoder.require_closed()
            if set(actual) != set(golden) or any(
                not np.array_equal(
                    np.asarray(actual[name]).reshape(-1), np.asarray(value).reshape(-1)
                )
                for name, value in golden.items()
            ):
                raise ValueError(
                    "primitive original actual complete outputs differ from independent reference"
                )
            current = check_issued_program(
                self,
                elf=arm["elf"],
                command_buffer=arm["command_buffer"],
                reference_console=arm["reference_console"],
                work_root=work_root / label,
            )
            current.update(
                elf=arm["elf"],
                build_record=arm["build_record"],
                caller_order=caller_order,
            )
            packets, stream, rhs = check_primitive_packets(
                self, current, contraction, inputs
            )
            arms[label] = {
                "original_result": arm["result"],
                "ordinary_status": arm["ordinary_status"],
                "ordinary_failure": arm["ordinary_failure"],
                "elf": _pin(arm["elf"]),
                "harness": arm["harness"],
                "command_buffer": arm["command_buffer_artifact"],
                "lowered": arm["lowered"],
                "original_execution": arm["original_execution_invocation"],
                "caller_roster_basis": arm["caller_roster_basis"],
                "fresh_actual_issue": current["issue_invocation"],
                "source_arithmetic": numeric,
                "actual_packets": packets,
            }
            streams.append(stream)
            rhs_names.append(rhs)
            costs[label + "_original_custody_and_fresh_packet_proof_wall_seconds"] = (
                time.monotonic() - tick
            )
        if len(set(rhs_names)) != 1:
            raise ValueError(
                "primitive pair changed exact immutable linked RHS ownership"
            )
        geometry = self.selected.report()["geometry"]
        stores = {row["name"]: row for row in geometry["stores"]}
        effects = check_output_tile_permutation(
            *streams,
            rhs=rhs_names[0],
            operand_rows=stores["scratchpad"]["banks"]
            * stores["scratchpad"]["rows_per_bank"],
            dim=geometry["array_rows"],
        )
        self.revalidate()
        _revalidate_sources(pins)
        return {
            "schema": "gemmini.primitive_pair_diagnostic.v1",
            "status": "checked_actual_primitive_L2_pair",
            "owner": self.identity,
            "compiler_context": self.context.record(),
            "source": pins,
            "arms": arms,
            "actual_issue_effects": effects,
            "costs": costs,
            "wall_seconds": time.monotonic() - started,
            "original_required_tiers": capsule["required_oracle_tiers"],
            "L3": "pending; no waiver or copied verdict",
            "scope": "original ordinary primitive arithmetic and actual complete current ELF issue",
            "final_functional_guard_qualified": False,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "performance_promoted": False,
        }
