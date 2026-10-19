"""Actual source/packet inverse for private zero-initializer virtualization.

An original live arithmetic/target owner, fresh complete-source verification and
both original native executions are mandatory. This checker issues no compiler
capability, functional grade, hardware qualification or timing authority.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_staged_source_pair import _arm, check_source_arms
from gemmini_timing_resources import _pin, _revalidate_sources

from merlin.common.paths import module_source_path
from merlin.perf.zero_initializer_evidence import check_zero_initializer_grouping


def canonical_issue_stream(stream, command_buffer):
    """Name checked ABI storage by role; only independently proved zero rows alias.

    This is a comparison primitive, not a zero or source proof. Its original
    caller must FIRST run SourceExecutionOwner on this exact linked stream.
    """
    arguments = command_buffer["kernel_abi"]["args"]
    names = [row["tensor"] for row in arguments]
    if len(set(names)) != len(names):
        raise ValueError("initializer issue inverse has aliased logical ABI owners")
    physical = {f"tensor_{index}": "abi/" + name for index, name in enumerate(names)}
    result = []
    for original, command in stream:
        event = copy.deepcopy(list(original))
        if isinstance(event[1], (tuple, list)):
            name, offset = event[1]
            if type(name) is not str or type(offset) is not int or offset < 0:
                raise ValueError("initializer issue inverse has an unresolved logical storage offset")
            if name in physical:
                event[1] = [physical[name], offset]
            elif (
                event[0] == "MVIN"
                and command is not None
                and command["decoded"].get("is_acc_addr") is True
                and type(offset) is int
                and offset == 0
            ):
                # SourceExecutionOwner separately reads this actual ELF storage,
                # proves every byte zero, and rejects every CPU/DMA clobber.
                event[1] = ["independently_proved_accumulator_zero", 0]
            else:
                raise ValueError("initializer issue inverse found an unowned or nonzero-offset static transfer")
        result.append((event, command))
    return result


class ZeroInitializerRealizationChecker:
    """Live current-source comparison; deliberately not a capability issuer."""

    def __init__(self, *, selected, arithmetic, nm, allow_tile_rewrites=False):
        if (
            type(selected) is not SelectedSchedulingTarget
            or type(arithmetic) is not IntegerArithmeticOwner
            or type(allow_tile_rewrites) is not bool
        ):
            raise ValueError("initializer realization requires original live selected arithmetic/target owners")
        self.selected, self.arithmetic, self.nm = selected, arithmetic, Path(nm)
        self._selected = selected.report()["selected_inputs"]
        self._arithmetic = arithmetic.record()
        self._allow_tiles = allow_tile_rewrites
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_zero_initializer_effects",
                "gemmini_staged_source_pair",
                "gemmini_staged_tile_effects",
                "gemmini_staged_rhs_effects",
                "merlin.perf.zero_initializer_evidence",
                "merlin.compile.scheduling.zero_initializer_groups",
                "merlin.transforms.dense_ir",
                "merlin.frontends.linalg_mlir",
                "merlin.common.dense_float_ir",
                "merlin.targetgen.oot_starterkit.plan",
                "merlin.llvmlower.source_contraction_bindings",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
                "gemmini_packet_contractions",
                "gemmini_execute_state",
            )
        }
        self._sources["nm"] = _pin(self.nm)
        self._identity = self._digest()
        self.revalidate()

    def _digest(self):
        return _digest([self._sources, self._selected, self._arithmetic, self._allow_tiles])

    @property
    def identity(self):
        return "gemmini.zero_initializer_realization.v2:" + self._identity

    def revalidate(self):
        self.selected.revalidate()
        self.arithmetic.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.selected.report()["selected_inputs"] != self._selected
            or self.arithmetic.record() != self._arithmetic
            or self._arithmetic["selected_inputs"] != self._selected
            or self._digest() != self._identity
        ):
            raise ValueError("original initializer realization selection or independent checks changed")

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "sources": copy.deepcopy(self._sources),
            "selected_inputs": copy.deepcopy(self._selected),
            "allow_tile_rewrites": self._allow_tiles,
            "scope": "current source/native/packet comparison only; no compiler capability or functional grade",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def _source_arm(self, arm):
        import yaml
        from merlin_experiments.phase0.connected_integer_qualification import source_proof

        from merlin.targetgen.native_model_execution import _pinned_native_file

        native = json.loads(_leaf(arm, "native_result").read_bytes())
        cap = _pinned_native_file(native["source"], "capsule_declaration").parent
        proof = source_proof(yaml.safe_load((cap / "capsule.yaml").read_bytes()), cap)
        return _arm(self, arm, proof, cap)

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        if execution_domain != "functional_spike":
            raise ValueError("initializer comparison has no qualified target timing domain")
        from merlin.targetgen.native_model_execution import _pinned_native_file

        source_identities = []
        for arm in (before, after):
            sources = json.loads(_leaf(arm, "native_result").read_bytes())["source"]
            for role in sources:
                _pinned_native_file(sources, role)
            source_identities.append({role: (pin["sha256"], pin.get("size_bytes")) for role, pin in sources.items()})
        if source_identities[0] != source_identities[1]:
            raise ValueError(
                "initializer comparison changed original source, inputs, weights, policies or complete references"
            )
        base, candidate, proof_scope = check_source_arms(self, before, after)
        original, original_rhs, base_check, old_cb = base
        rewritten, rewritten_rhs, candidate_check, new_cb = candidate
        native = json.loads(_leaf(before, "native_result").read_bytes())
        source = Path(native["source"]["capsule_declaration"]["path"]).parent / "capsule.interface.mlir"
        grouping = check_zero_initializer_grouping(source.read_text(), old_cb, new_cb)
        original = canonical_issue_stream(original, old_cb)
        rewritten = canonical_issue_stream(rewritten, new_cb)
        if not self._allow_tiles:
            if [event for event, _ in original] != [event for event, _ in rewritten]:
                raise ValueError(
                    "zero initializer candidate changed accelerator commands, fences, settings or logical DMA owners"
                )
            effects = {"status": "identical_checked_accelerator_issue_sequence", "actual_rhs_loads_removed": 0}
        else:
            from gemmini_staged_tile_effects import check_output_tile_permutation

            def logical_rhs(physical, cb):
                index = int(physical.removeprefix("tensor_"))
                return "abi/" + cb["kernel_abi"]["args"][index]["tensor"]

            rhs = logical_rhs(original_rhs, old_cb)
            if rhs != logical_rhs(rewritten_rhs, new_cb):
                raise ValueError("composite initializer/tiling changed its immutable source RHS")
            # Only existing placed writes are retained or reordered; no new
            # allocation or free-window assumption is introduced by this proof.
            rows = [
                command["decoded"]["addr"] + command["decoded"]["rows"]
                for event, command in original
                if event[0] == "MVIN" and command["decoded"].get("is_acc_addr") is False
            ]
            effects = check_output_tile_permutation(
                original,
                rewritten,
                rhs=rhs,
                operand_rows=max(rows),
                dim=self._arithmetic["arithmetic_rule"]["max_primitive_k"],
            )
        self.revalidate()
        return {
            "status": "checked_zero_initializer_source_and_actual_issue",
            "grouping": grouping,
            "actual_issue_effects": effects,
            "original_source_execution": [base_check, candidate_check],
            "original_source_proof_scope": proof_scope,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "compiler_capability_issued": False,
        }
