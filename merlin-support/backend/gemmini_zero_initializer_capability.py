"""Original composite preparation and conditional zero-seed realization owner.

Only the original independent compiler/source/native/packet checks below can
issue this owner. It is functional Spike evidence, never hardware or timing
qualification. The original full functional grade remains a separate gate.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path

from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_staged_tile_capability import StagedTilePrototype
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_initializer_effects import ZeroInitializerRealizationChecker

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_emission_binding import (
    check_compiler_emission,
    revalidate_compiler_emission,
)
from merlin.targetgen.native_model_execution import _pinned_native_file


class ZeroInitializerCompilerPrototype:
    """One original composite creates both policies before retained-source freeze."""

    def __init__(self, tile):
        if type(tile) is not StagedTilePrototype:
            raise ValueError("initializer preparation needs original live staged source/effect owners")
        tile.revalidate()
        self.tile = tile
        self._tile = tile.record()
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_zero_initializer_capability",
                "gemmini_zero_initializer_preparation",
                "gemmini_zero_initializer_final_guard",
                "merlin.compile.scheduling.zero_initializer_groups",
                "merlin.targetgen.compiler_emission_binding",
            )
        }
        self.checker = ZeroInitializerRealizationChecker(
            selected=tile.selected, arithmetic=tile.arithmetic, nm=tile.nm, allow_tile_rewrites=True
        )
        self._checker = self.checker.record()
        self._identity = _digest([self._tile, self._sources, self._checker])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.zero_initializer_compiler_prototype.v1:" + self._identity

    @property
    def library(self):
        return self.tile.library

    @property
    def cleanup(self):
        return self.tile.cleanup

    @property
    def nm(self):
        return self.tile.nm

    def revalidate(self):
        self.tile.revalidate()
        self.checker.revalidate()
        _revalidate_sources(self._sources)
        if (
            self.tile.record() != self._tile
            or self.checker.record() != self._checker
            or _digest([self._tile, self._sources, self._checker]) != self._identity
        ):
            raise ValueError("original initializer preparation/effect selection changed")

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "sources": copy.deepcopy(self._sources),
            "tile_prototype": copy.deepcopy(self._tile),
            "realization_checker": copy.deepcopy(self._checker),
            "scope": "unqualified composite preparation; original final realization required",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        if (
            type(policy) is not dict
            or set(policy) != {"zero_initializer", "tile"}
            or type(policy["zero_initializer"]) is not dict
            or set(policy["zero_initializer"]) != {"enabled"}
            or type(policy["zero_initializer"]["enabled"]) is not bool
        ):
            raise ValueError("initializer composite needs explicit independent zero and tile policies")
        from gemmini_zero_initializer_preparation import prepare_compiler

        destination = Path(destination)
        # The selected composite alone controls this order. Each arm starts
        # from the same immutable raw parent; no existing treatment is patched.
        with tempfile.TemporaryDirectory(prefix=".zero-init-preparation-", dir=destination.parent) as directory:
            staged = Path(directory) / "staged"
            self.tile.prepare_compiler(source, staged, policy=policy["tile"])
            prepare_compiler(
                staged,
                destination,
                enabled=policy["zero_initializer"]["enabled"],
                generic_source=Path(self._sources["merlin.compile.scheduling.zero_initializer_groups"]["path"]),
            )
        self.revalidate()
        return destination

    def check_compiler_binding(self, package):
        self.revalidate()
        record = self.tile.check_compiler_binding(package)
        from merlin.targetgen.compiler_pass_selection import _pin as ordinary_leaf_pin

        leaves = {}
        for name in (
            "mixed_codegen.py",
            "provenance.py",
            "zero_initializer_policy.py",
            "passes/__init__.py",
            "passes/zero_initializer_groups.py",
        ):
            path = Path(package) / name
            pin = ordinary_leaf_pin(path)
            if path.stat().st_size > 1 << 20:
                raise ValueError("initializer compiler component exceeds its original source bound")
            leaves[name] = pin
        self.revalidate()
        return {**record, "candidate_owned_initializer_components": leaves}

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        result = self.checker.check_pair_artifacts(before, after, execution_domain=execution_domain)
        self.revalidate()
        return result


class QualifiedZeroInitializerOwner:
    """Fresh original compiler, complete-source and actual-program admission."""

    def __init__(self, prototype, pairs, *, status_path=None):
        if (
            type(prototype) is not ZeroInitializerCompilerPrototype
            or type(pairs) is not tuple
            or not 7 <= len(pairs) <= 32
            or any(type(pair) is not dict or set(pair) != {"before", "after"} for pair in pairs)
        ):
            raise ValueError(
                "initializer admission needs original owners and independent actual source/composition pairs"
            )
        self.prototype = prototype
        self._prototype = prototype.record()
        self._pairs = copy.deepcopy(pairs)
        self._source = _pin(__file__)
        self._emissions = [
            [
                check_compiler_emission(
                    prototype,
                    arm,
                    argument_templates=(
                        "--convert-iface-to-gemmini",
                        "--emit-command-buffer={output_json}",
                        "--emit-target-artifact",
                        "{input_mlir}",
                    ),
                    required_sources=(
                        "compiler.py",
                        "direct_codegen.py",
                        "selected-shared-library.json",
                        "shared-library-sources.json",
                        "_selected_shared_sources.py",
                        "gemmini_staged_tile_hook.py",
                        "gemmini_staged_tile_source.py",
                        "staged_dense_policy.py",
                        "mixed_codegen.py",
                        "provenance.py",
                        "zero_initializer_policy.py",
                        "passes/zero_initializer_groups.py",
                    ),
                )
                for arm in pair.values()
            ]
            for pair in pairs
        ]
        self._checks = []
        started = time.monotonic()
        for ordinal, pair in enumerate(pairs):
            if status_path is not None:
                Path(status_path).write_text(
                    json.dumps(
                        {
                            "stage": "original_initializer_pair",
                            "ordinal": ordinal,
                            "count": len(pairs),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                )
            self._checks.append(prototype.check_pair_artifacts(**pair, execution_domain="functional_spike"))
        sources, geometries = [], []
        for check in self._checks:
            checked = check["original_source_execution"][0]
            sources.append(checked["source_sha256"])
            geometries.append(checked["source_geometry"])
        dim = prototype.tile.contract.dim
        if (
            len(set(sources)) < 6
            or any(not check["grouping"]["groups"] for check in self._checks)
            or not {0, 1, dim - 1} <= {shape["m"] % dim for shape in geometries}
            or not {0, 1, dim - 1} <= {shape["n"] % dim for shape in geometries}
            or not any(shape["k"] > dim for shape in geometries)
            or not any(
                check["actual_issue_effects"].get("output_tile_order_changed") is True
                and check["actual_issue_effects"]["actual_rhs_loads_removed"] > 0
                for check in self._checks
            )
        ):
            raise ValueError(
                "initializer admission lacks independent M/N tails, deep-K and actual composed tile/residency evidence"
            )
        self._identity = self._digest()
        self.revalidate()
        if status_path is not None:
            Path(status_path).write_text(
                json.dumps(
                    {
                        "stage": "conditional_functional_source_owner_issued",
                        "elapsed_seconds": time.monotonic() - started,
                        "hardware_execution_qualified": False,
                        "timing_authority": "NONE",
                    }
                )
            )

    def _digest(self):
        return _digest([self._source, self._prototype, self._pairs, self._checks, self._emissions])

    @property
    def identity(self):
        return "gemmini.qualified_zero_initializer.v1:" + self._identity

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
            or self.prototype.record() != self._prototype
            or self._digest() != self._identity
        ):
            raise ValueError("original initializer qualification, checks or selected sources changed")
        checked_packages = set()
        for group in self._emissions:
            for emission in group:
                revalidate_compiler_emission(emission, checked_packages=checked_packages)
        for pair in self._pairs:
            for arm in pair.values():
                for role in ("native_result", "command_buffer_artifact", "lowered", "issue_invocation"):
                    _leaf(arm, role)
                invocation_record.verify(Path(arm["issue_invocation"]["path"]))
                native = json.loads(Path(arm["native_result"]["path"]).read_bytes())
                for group in (native["source"], native["build_artifacts"]):
                    for role in group:
                        _pinned_native_file(group, role)
                _revalidate_sources({str(index): pin for index, pin in enumerate(native["host_build_sources"])})

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "prototype": copy.deepcopy(self._prototype),
            "qualified_pairs": copy.deepcopy(self._pairs),
            "checks": copy.deepcopy(self._checks),
            "original_compiler_emissions": copy.deepcopy(self._emissions),
            "domain": "functional_spike_exact_source_and_inputs_conditional_final_pair",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def prepare_compiler(self, *args, **kwargs):
        self.revalidate()
        return self.prototype.prepare_compiler(*args, **kwargs)

    def check_compiler_binding(self, package):
        self.revalidate()
        return self.prototype.check_compiler_binding(package)

    def check_pair_artifacts(self, *args, **kwargs):
        self.revalidate()
        return self.prototype.check_pair_artifacts(*args, **kwargs)

    def check_functional_guard(self, *args, **kwargs):
        self.revalidate()
        from gemmini_zero_initializer_final_guard import check_functional_guard

        return check_functional_guard(self, *args, **kwargs)
