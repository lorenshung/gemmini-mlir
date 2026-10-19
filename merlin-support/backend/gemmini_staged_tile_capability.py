"""Original checked whole-tile traversal ownership, without saved success grants."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from gemmini_staged_dense_capability import StagedDensePrototype, _digest
from gemmini_timing_resources import _pin, _revalidate_sources


class StagedTilePrototype:
    """Prepare real copied compilers; qualification needs actual sealed pairs."""

    def __init__(self, prototype):
        if type(prototype) is not StagedDensePrototype:
            raise ValueError(
                "tile traversal requires original staged arithmetic/completion owners"
            )
        from merlin.common.paths import module_source_path

        self.prototype = prototype
        self._prototype_identity = prototype.identity
        self._prototype_record = prototype.record()
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_staged_tile_capability",
                "gemmini_staged_tile_hook",
                "gemmini_staged_tile_source",
                "gemmini_staged_tile_effects",
                "gemmini_staged_compile_custody",
                "gemmini_staged_source_pair",
                "merlin.targetgen.compiler_emission_binding",
            )
        }
        self._identity = _digest(
            [self._sources, self._prototype_identity, self._prototype_record]
        )
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.staged_tile_prototype.v1:" + self._identity

    @property
    def cleanup(self):
        return self.prototype.cleanup

    @property
    def library(self):
        return self.prototype.library

    @property
    def nm(self):
        return self.prototype.nm

    @property
    def selected(self):
        return self.prototype.selected

    @property
    def arithmetic(self):
        return self.prototype.arithmetic

    @property
    def contract(self):
        return self.prototype.contract

    def revalidate(self):
        self.prototype.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.prototype.identity != self._prototype_identity
            or _digest(
                [self._sources, self._prototype_identity, self._prototype_record]
            )
            != self._identity
        ):
            raise ValueError("original whole-tile source/effect owner changed")

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "sources": self._sources,
            "prototype": copy.deepcopy(self._prototype_record),
            "scope": "unqualified complete-tile traversal prototype; every final realization required",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def _public(self):
        return {
            "schema": "gemmini.staged_tile_public.v1",
            "contract": asdict(self.contract),
            "library_contract_sha256": self.library.contract.sha256,
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        if (
            type(policy) is not dict
            or set(policy) != {"enabled", "min_m", "min_n", "rewrite_order"}
            or type(policy["enabled"]) is not bool
            or any(
                type(policy[key]) is not int or policy[key] < 0
                for key in ("min_m", "min_n")
            )
            or type(policy["rewrite_order"]) is not list
            or any(type(name) is not str for name in policy["rewrite_order"])
            or set(policy["rewrite_order"]) != {"n_before_m", "rhs_residency"}
            or len(policy["rewrite_order"]) != 2
        ):
            raise ValueError(
                "tile preparation needs explicit Boolean/shape thresholds and two ordered rewrite choices"
            )
        # One selected composite prepares from the original Phase 1 source.
        # The narrower preparation's mechanisms remain disabled in both arms.
        self.prototype.prepare_compiler(
            source, destination, policy={"enabled": False, "min_m": 0}
        )
        for name in ("gemmini_staged_tile_hook", "gemmini_staged_tile_source"):
            shutil.copyfile(self._sources[name]["path"], destination / (name + ".py"))
        (destination / "staged-tile-public.json").write_text(
            json.dumps(self._public(), sort_keys=True)
        )
        (destination / "staged_dense_policy.py").write_text(
            '"""Candidate-owned shape policy and executable rewrite order."""\n'
            f"ENABLED={policy['enabled']!r}\nMIN_M={policy['min_m']!r}\nMIN_N={policy['min_n']!r}\nORDER={tuple(policy['rewrite_order'])!r}\n"
            "def choose_staged_tile_rewrites(shape, contract):\n"
            "    from merlin.compile.scheduling.staged_dense import StagedDenseRewrite, order_output_tiles_n_before_m, eliminate_redundant_rhs_loads\n"
            "    m,k,n=shape\n"
            "    if not (ENABLED and m>=MIN_M and n>=MIN_N and m>contract.dim and n>contract.dim and k<=contract.dim): return ()\n"
            "    implementations={'n_before_m':order_output_tiles_n_before_m,'rhs_residency':eliminate_redundant_rhs_loads}\n"
            "    return tuple(StagedDenseRewrite(name,implementations[name]) for name in ORDER)\n"
        )
        path = destination / "direct_codegen.py"
        before = "from gemmini_staged_dense_hook import rewrite as _staged_statements"
        text = path.read_text()
        if text.count(before) != 1:
            raise ValueError(
                "tile preparation lost original unique staged emitter boundary"
            )
        path.write_text(
            text.replace(
                before,
                "from gemmini_staged_tile_hook import rewrite as _staged_statements",
            )
        )
        self.revalidate()
        return destination

    def check_compiler_binding(self, package):
        self.revalidate()
        checked = self.prototype.check_compiler_binding(package)
        path = Path(package) / "staged-tile-public.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 1 << 20
            or json.loads(path.read_bytes()) != self._public()
        ):
            raise ValueError(
                "tile compiler changed its original bounded public source/effect/library contract"
            )
        self.revalidate()
        return {**checked, "staged_tile_public_contract": _pin(path)}

    def check_pair_artifacts(self, before, after, *, execution_domain):
        if execution_domain != "functional_spike":
            raise ValueError(
                "whole-tile traversal has no qualified target timing domain"
            )
        from gemmini_staged_source_pair import check_source_arms

        base, cand, source_scope = check_source_arms(self.prototype, before, after)
        baseline, rhs, base_source, base_cb = base
        candidate, candidate_rhs, candidate_source, candidate_cb = cand
        from gemmini_cleanup_final_guard import _logical

        if (
            rhs != candidate_rhs
            or base_source["source_sha256"] != candidate_source["source_sha256"]
            or _logical(base_cb) != _logical(candidate_cb)
        ):
            raise ValueError(
                "whole-tile pair changed original complete source/work/RHS ownership"
            )
        from gemmini_staged_tile_effects import check_output_tile_permutation

        checked = check_output_tile_permutation(
            baseline,
            candidate,
            rhs=rhs,
            operand_rows=self.contract.operand_rows,
            dim=self.contract.dim,
        )
        self.revalidate()
        return {
            **checked,
            "original_source_execution": [base_source, candidate_source],
            "original_source_proof_scope": source_scope,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def check_functional_guard(
        self, baseline, candidate, baseline_result, candidate_result, *, grade_root
    ):
        from gemmini_cleanup_final_guard import check_functional_guard

        return check_functional_guard(
            self,
            baseline,
            candidate,
            baseline_result,
            candidate_result,
            grade_root=grade_root,
        )


class QualifiedStagedTileOwner:
    """Original issuance runs actual full source/product/state/output pair checks."""

    def __init__(self, prototype, pairs):
        if (
            type(prototype) is not StagedTilePrototype
            or type(pairs) is not tuple
            or len(pairs) < 4
            or len(pairs) > 32
            or any(
                type(pair) is not dict or set(pair) != {"before", "after"}
                for pair in pairs
            )
        ):
            raise ValueError(
                "tile admission needs an original owner and independent source boundary/control pairs"
            )
        self.prototype = prototype
        self._pairs = copy.deepcopy(pairs)
        from gemmini_staged_compile_custody import check_original_emission

        self._emissions = [
            [check_original_emission(prototype, arm) for arm in pair.values()]
            for pair in pairs
        ]
        self._checks = [
            prototype.check_pair_artifacts(**pair, execution_domain="functional_spike")
            for pair in pairs
        ]
        dim = prototype.contract.dim
        sources = [
            row["original_source_execution"][0]["source_sha256"] for row in self._checks
        ]
        positives = [
            row["original_source_execution"][0]["source_geometry"]
            for row in self._checks
            if row["output_tile_order_changed"] and row["actual_rhs_loads_removed"] > 0
        ]
        controls = [
            row["original_source_execution"][0]["source_geometry"]
            for row in self._checks
            if not row["output_tile_order_changed"]
            and row["actual_rhs_loads_removed"] == 0
        ]
        if (
            len(set(sources)) != len(sources)
            or not {0, 1, dim - 1} <= {shape["m"] % dim for shape in positives}
            or not {0, 1, dim - 1} <= {shape["n"] % dim for shape in positives}
            or not any(
                shape["m"] > dim and shape["n"] > dim and shape["k"] > dim
                for shape in controls
            )
        ):
            raise ValueError(
                "tile admission needs independent M/N boundary positives and a two-axis deep-K no-gain control"
            )
        self._prototype_identity = prototype.identity
        self._prototype_record = prototype.record()
        self._source = _pin(__file__)
        self._identity = _digest(
            [
                self._source,
                self._prototype_identity,
                self._prototype_record,
                self._pairs,
                self._checks,
                self._emissions,
            ]
        )
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.qualified_staged_tile.v1:" + self._identity

    @property
    def cleanup(self):
        return self.prototype.cleanup

    @property
    def nm(self):
        return self.prototype.nm

    def revalidate(self):
        self.prototype.revalidate()
        if (
            _pin(__file__) != self._source
            or self.prototype.identity != self._prototype_identity
            or _digest(
                [
                    self._source,
                    self._prototype_identity,
                    self._prototype_record,
                    self._pairs,
                    self._checks,
                    self._emissions,
                ]
            )
            != self._identity
        ):
            raise ValueError(
                "original tile qualification or independent source evidence changed"
            )
        from merlin.common import invocation_record
        from merlin.targetgen.compiler_emission_binding import (
            revalidate_compiler_emission,
        )
        from merlin.targetgen.native_model_execution import _pinned_native_file

        checked_packages = set()
        for group in self._emissions:
            for record in group:
                revalidate_compiler_emission(record, checked_packages=checked_packages)

        for pair in self._pairs:
            for arm in pair.values():
                for role in (
                    "native_result",
                    "command_buffer_artifact",
                    "lowered",
                    "issue_invocation",
                ):
                    pin = arm[role]
                    if _pin(pin["path"]) != {
                        name: pin[name] for name in ("path", "sha256")
                    }:
                        raise ValueError("original tile qualification artifact changed")
                invocation_record.verify(Path(arm["issue_invocation"]["path"]))
                native = json.loads(Path(arm["native_result"]["path"]).read_bytes())
                for group in (native["source"], native["build_artifacts"]):
                    for role in group:
                        _pinned_native_file(group, role)
                for pin in native["host_build_sources"]:
                    if _pin(pin["path"]) != {
                        key: pin[key] for key in ("path", "sha256")
                    }:
                        raise ValueError(
                            "original tile native-build implementation changed"
                        )

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "prototype": copy.deepcopy(self._prototype_record),
            "qualified_pairs": self._pairs,
            "checks": self._checks,
            "original_compiler_emissions": self._emissions,
            "domain": "functional_spike_exact_source_and_inputs",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        return self.prototype.prepare_compiler(source, destination, policy=policy)

    def check_compiler_binding(self, package):
        self.revalidate()
        return self.prototype.check_compiler_binding(package)

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        return self.prototype.check_pair_artifacts(
            before, after, execution_domain=execution_domain
        )

    def check_functional_guard(self, *args, **kwargs):
        self.revalidate()
        return self.prototype.check_functional_guard(*args, **kwargs)


def restore_registry(inputs, *, work_root):
    if not inputs.get("tile_qualification_pairs"):
        raise ValueError(
            "whole-tile capability disabled: original sealed boundary/control execution pairs unavailable"
        )
    from gemmini_staged_dense_admission import prepare_original_prototype
    from merlin.llvmlower.reviewed_pass_capabilities import (
        PassCapabilityBinding,
        ReviewedPassCapabilities,
        ReviewedPassCapability,
    )

    original = prepare_original_prototype(inputs, work_root=work_root)
    from merlin.common.cpu_pool import shared_compute_pool

    # The cleanup admission above owns a separately leased subprocess. Never
    # hold this parent's lease while that child waits for its own CPU grant.
    with shared_compute_pool(timeout_s=120):
        prototype = StagedTilePrototype(original)
        owner = QualifiedStagedTileOwner(
            prototype, tuple(inputs["tile_qualification_pairs"])
        )
        contract = prototype.contract
        binding = PassCapabilityBinding(
            contract.hardware,
            contract.software,
            contract.numerical_policy,
            contract.execution,
            prototype.library.contract.sha256,
        )
        return ReviewedPassCapabilities(
            (ReviewedPassCapability("staged_dense_tile_order_raw_i32", binding, owner),)
        )
