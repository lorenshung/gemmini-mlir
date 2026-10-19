"""Original conditional compiler ownership for finite full-K RHS placement.

This is a separately selected compiler mechanism. Its complete current source,
allocation epoch, products, native outputs and issued effects remain checked;
neither physical capacity nor a serialized successful report issues an owner.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import yaml
from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_staged_rhs_relocation_effects import RhsPlacementRealizationChecker, original_source_capsule
from gemmini_staged_rhs_window import ClosedScratchEpochOwner
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_zero_initializer_capability import ZeroInitializerCompilerPrototype, _check_prepared_binding
from merlin_experiments.phase0.capture_execution_attestation import require_verified_execution
from merlin_experiments.phase2.contracts import exact_tree_record

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_emission_binding import check_compiler_emission, revalidate_compiler_emission
from merlin.targetgen.native_model_execution import _pinned_native_file

_METADATA = contextvars.ContextVar("original_rhs_metadata_construction", default=None)
_METADATA_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class _MetadataOperation:
    owner: object
    prototype: object
    identity: str
    callback: object
    code: object

    def check(self):
        owner = self.owner
        if (
            self not in _METADATA_ISSUED
            or type(owner) is not QualifiedRhsResidencyOwner
            or owner.prototype is not self.prototype
            or owner.identity != self.identity
            or type(owner).revalidate is not self.callback
            or self.callback.__code__ is not self.code
        ):
            raise ValueError("original RHS metadata owner or implementation changed")


def _metadata_verified(owner):
    operation = _METADATA.get()
    if operation is None:
        return False
    if type(operation) is not _MetadataOperation:
        raise ValueError("RHS metadata scope is not its original live operation")
    operation.check()
    return operation.owner is owner


@contextlib.contextmanager
def scoped_original_rhs_metadata(owner):
    """Full original custody checks around ONE trusted metadata construction.

    Repeated getters still check source, structure and identities. The result
    cannot leave this scope before its final full custody check. New source,
    native execution and emitted-program proofs never use this scope.
    """
    if type(owner) is not QualifiedRhsResidencyOwner:
        raise ValueError("RHS metadata construction requires the original live qualified owner")
    current = _METADATA.get()
    if current is not None and type(current) is not _MetadataOperation:
        raise ValueError("RHS metadata scope is not its original live operation")
    if current is not None and current.owner is owner:
        current.check()
        try:
            yield
        finally:
            current.check()
        return
    disabled = _METADATA.set(None)
    try:
        owner.revalidate()
        callback = type(owner).revalidate
        operation = _MetadataOperation(owner, owner.prototype, owner.identity, callback, callback.__code__)
        _METADATA_ISSUED.add(operation)
        token = _METADATA.set(operation)
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            _METADATA.reset(token)
            try:
                operation.check()
                owner.revalidate()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("Original RHS final metadata/custody check failed: " + str(error))
            finally:
                _METADATA_ISSUED.discard(operation)
    finally:
        _METADATA.reset(disabled)


def _verify_original_capsules(originals):
    """One full revalidation verifies each EXACT original attestation once.

    Every capsule still reopens its entire tree and checks its own captured
    source membership. There is no state or verification shared across calls.
    """
    verified = {}
    for original in originals:
        cap = Path(original["path"])
        if exact_tree_record(cap) != original["files"]:
            raise ValueError("RHS original sealed source/input/reference publication changed")
        capsule = yaml.safe_load((cap / "capsule.yaml").read_bytes())
        attestations = capsule.get("capture_execution_attestations") or []
        if not attestations:
            raise ValueError("RHS original sealed capture carries no original attestation")
        models = set()
        for attestation in attestations:
            key = _digest(attestation)
            if key not in verified:
                require_verified_execution(attestation)
                verified[key] = copy.deepcopy(attestation)
            elif verified[key] != attestation:
                raise ValueError("RHS original capture attestation digest collision")
            capture = attestation.get("capture") or {}
            if capture.get("kind") == "session":
                models.update(row.get("model_sha256") for row in capture.get("programs", []))
            else:
                models.add(capture.get("model_sha256"))
        raw = (capsule.get("frontend_trace") or {}).get("capture_mlir_sha256")
        if raw is not None and raw not in models:
            raise ValueError("RHS original captured source is not in its verified complete session")


def qualification_coverage(checks, dim):
    """Check fixed independent geometry/control obligations, never objective weights."""
    shapes, sources = [], []
    for check in checks:
        if check.get("status") not in {
            "checked_original_private_rhs_relocation_and_actual_issue",
            "checked_original_private_panel_fallback",
        }:
            raise ValueError("RHS admission lacks an original current-program realization")
        original = check["original_source_execution"][0]
        shapes.append(original["source_geometry"])
        sources.append(original["source_sha256"])
    tails = {0, 1, dim - 1}
    if (
        len(checks) < 6
        or len(set(sources)) < 6
        or any(not tails <= {shape[axis] % dim for shape in shapes} for axis in ("m", "k", "n"))
        or not any(shape["k"] <= dim and not check["full_k_placement_applied"] for shape, check in zip(shapes, checks))
        or not all(
            any(
                shape["k"] > dim
                and shape[axis] % dim == tail
                and check["full_k_placement_applied"]
                and check["actual_issue_effects"]["actual_rhs_loads_removed"] > 0
                for shape, check in zip(shapes, checks)
            )
            for axis in ("m", "k", "n")
            for tail in tails
        )
    ):
        raise ValueError("RHS admission needs independent deep-K M/N/K tails, actual reuse and a single-panel fallback")
    return dict(
        independent_sources=len(set(sources)),
        shapes=copy.deepcopy(shapes),
        positive_tail_remainders=sorted(tails),
        single_panel_fallback=True,
        hardware_execution_qualified=False,
        timing_authority="NONE",
    )


class RhsResidencyCompilerPrototype:
    """One original composite prepares both arms from the immutable raw parent."""

    def __init__(self, seed, epoch):
        if type(seed) is not ZeroInitializerCompilerPrototype or type(epoch) is not ClosedScratchEpochOwner:
            raise ValueError("RHS preparation requires original compiler and complete-program allocation owners")
        if (
            epoch.source_owner._selected_inputs != seed.tile.selected.report()["selected_inputs"]
            or epoch.source_owner._arithmetic.identity != seed.tile.arithmetic.identity
        ):
            raise ValueError("RHS compiler and allocation epoch selected different original source/effect contracts")
        self.seed, self.epoch = seed, epoch
        self._seed, self._epoch = seed.record(), epoch.record()
        self._binding = copy.deepcopy(seed._binding)
        tile = self._binding["staged-tile-public.json"]
        self._binding["staged-rhs-window-public.json"] = dict(
            schema="gemmini.closed_rhs_program_public.v1",
            original_contract=tile["contract"],
            library_contract_sha256=tile["library_contract_sha256"],
            maximum_private_physical_rows=self._epoch["footprint"]["maximum_private_physical_rows"],
            basis_original_program_epoch=epoch.identity,
            condition="one complete current source/execution epoch; original independent final proof required",
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_staged_rhs_capability",
                "gemmini_staged_rhs_preparation",
                "gemmini_staged_rhs_hook",
                "gemmini_staged_rhs_placement",
                "gemmini_staged_dense_capability",
                "gemmini_timing_resources",
                "gemmini_zero_initializer_capability",
                "merlin.common.invocation_record",
                "merlin.common.paths",
                "merlin.compile.scheduling.staged_rhs_residency",
                "merlin.targetgen.compiler_emission_binding",
                "merlin.targetgen.native_model_execution",
                "merlin_experiments.phase0.capture_execution_attestation",
                "merlin_experiments.phase2.contracts",
            )
        }
        self.checker = RhsPlacementRealizationChecker(
            selected=seed.tile.selected, arithmetic=seed.tile.arithmetic, nm=seed.nm
        )
        self._checker = self.checker.record()
        self._identity = _digest([self._seed, self._epoch, self._binding, self._sources, self._checker])
        self.revalidate()

    @property
    def identity(self):
        return "gemmini.full_k_rhs_compiler_prototype.v1:" + self._identity

    @property
    def library(self):
        return self.seed.library

    @property
    def cleanup(self):
        return self.seed.cleanup

    @property
    def nm(self):
        return self.seed.nm

    def revalidate(self):
        _revalidate_sources(self._sources)
        if (
            self.seed.record() != self._seed
            or self.epoch.record() != self._epoch
            or self.checker.record() != self._checker
            or _digest([self._seed, self._epoch, self._binding, self._sources, self._checker]) != self._identity
        ):
            raise ValueError("original full-K RHS compiler, epoch or selected effects changed")

    def record(self):
        self.revalidate()
        return dict(
            identity=self.identity,
            seed=copy.deepcopy(self._seed),
            original_epoch=copy.deepcopy(self._epoch),
            source_ledgers=copy.deepcopy(self._binding),
            sources=copy.deepcopy(self._sources),
            realization_checker=copy.deepcopy(self._checker),
            scope="unqualified finite source-program RHS placement; every new realization checked",
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        if type(policy) is not dict or set(policy) != {"enabled"} or type(policy["enabled"]) is not bool:
            raise ValueError("RHS preparation needs its explicit independent Boolean policy")
        from gemmini_staged_rhs_preparation import prepare_compiler

        destination = Path(destination)
        with tempfile.TemporaryDirectory(prefix=".rhs-composite-", dir=destination.parent) as directory:
            staged = Path(directory) / "seed"
            self.seed.prepare_compiler(
                source,
                staged,
                policy=dict(
                    zero_initializer=dict(enabled=False),
                    tile=dict(enabled=False, min_m=0, min_n=0, rewrite_order=["n_before_m", "rhs_residency"]),
                ),
            )
            prepare_compiler(
                staged,
                destination,
                epoch=self.epoch,
                generic_source=Path(self._sources["merlin.compile.scheduling.staged_rhs_residency"]["path"]),
                enabled=policy["enabled"],
            )
        self.revalidate()
        return destination

    def check_compiler_binding(self, package):
        self.revalidate()
        checked = _check_prepared_binding(package, self._binding)
        from merlin.targetgen.compiler_pass_selection import _pin as ordinary_pin

        extra = {}
        for name in (
            "rhs_residency_policy.py",
            "passes/staged_rhs_residency.py",
            "gemmini_staged_rhs_hook.py",
            "gemmini_staged_rhs_placement.py",
        ):
            path = Path(package) / name
            extra[name] = ordinary_pin(path)
            if path.stat().st_size > 1 << 20:
                raise ValueError("RHS candidate component exceeds the original source bound")
        self.revalidate()
        return {**checked, "rhs_components": extra}

    def check_pair_artifacts(self, before, after, *, execution_domain):
        self.revalidate()
        result = self.checker.check_pair_artifacts(before, after, execution_domain=execution_domain)
        self.revalidate()
        return result


class QualifiedRhsResidencyOwner:
    """Fresh source, compiler-emission, allocation and actual-program qualification."""

    def __init__(self, prototype, pairs, *, status_path=None):
        if (
            type(prototype) is not RhsResidencyCompilerPrototype
            or type(pairs) is not tuple
            or not 6 <= len(pairs) <= 32
            or any(type(pair) is not dict or set(pair) != {"before", "after"} for pair in pairs)
        ):
            raise ValueError("RHS admission needs original owners and independent actual source/control pairs")
        self.prototype = prototype
        self._prototype, self._pairs = prototype.record(), copy.deepcopy(pairs)
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
                        "mixed_codegen.py",
                        "selected-shared-library.json",
                        "shared-library-sources.json",
                        "_selected_shared_sources.py",
                        "staged-rhs-window-public.json",
                        "gemmini_staged_rhs_hook.py",
                        "gemmini_staged_rhs_placement.py",
                        "passes/staged_rhs_residency.py",
                        "rhs_residency_policy.py",
                    ),
                )
                for arm in pair.values()
            ]
            for pair in pairs
        ]
        self._checks = []
        began = time.monotonic()
        costs = []
        for ordinal, pair in enumerate(pairs):
            if status_path is not None:
                Path(status_path).write_text(
                    json.dumps(
                        dict(
                            stage="original_rhs_source_pair",
                            ordinal=ordinal,
                            count=len(pairs),
                            completed=costs,
                            elapsed_seconds=time.monotonic() - began,
                        )
                    )
                )
            tick, cpu = time.monotonic(), time.process_time()
            self._checks.append(prototype.check_pair_artifacts(**pair, execution_domain="functional_spike"))
            costs.append(
                dict(ordinal=ordinal, wall_seconds=time.monotonic() - tick, cpu_seconds=time.process_time() - cpu)
            )
        self._capsules = []
        for pair in pairs:
            native = json.loads(_leaf(pair["before"], "native_result").read_bytes())
            cap = original_source_capsule(pair["before"], native)
            self._capsules.append(dict(path=str(cap), files=exact_tree_record(cap)))
        self._coverage = qualification_coverage(self._checks, prototype.seed.tile.contract.dim)
        self._identity = self._digest()
        self.revalidate()
        if status_path is not None:
            Path(status_path).write_text(
                json.dumps(
                    dict(
                        stage="conditional_functional_rhs_owner_issued",
                        completed=costs,
                        elapsed_seconds=time.monotonic() - began,
                        hardware_execution_qualified=False,
                        timing_authority="NONE",
                    )
                )
            )

    def _digest(self):
        return _digest(
            [self._source, self._prototype, self._pairs, self._emissions, self._checks, self._capsules, self._coverage]
        )

    @property
    def identity(self):
        return "gemmini.qualified_full_k_rhs.v1:" + self._identity

    @property
    def cleanup(self):
        return self.prototype.cleanup

    @property
    def nm(self):
        return self.prototype.nm

    def revalidate(self):
        if (
            _pin(__file__) != self._source
            or self.prototype.record() != self._prototype
            or self._digest() != self._identity
        ):
            raise ValueError("original RHS qualification or current source custody changed")
        if _metadata_verified(self):
            return
        packages = set()
        for group in self._emissions:
            for emission in group:
                revalidate_compiler_emission(emission, checked_packages=packages)
        for pair in self._pairs:
            for arm in pair.values():
                for name in ("native_result", "command_buffer_artifact", "lowered", "issue_invocation"):
                    _leaf(arm, name)
                if "source_capsule_declaration" in arm:
                    _leaf(arm, "source_capsule_declaration")
                invocation_record.verify(Path(arm["issue_invocation"]["path"]))
                native = json.loads(Path(arm["native_result"]["path"]).read_bytes())
                for group in (native["source"], native["build_artifacts"]):
                    for name in group:
                        _pinned_native_file(group, name)
                _revalidate_sources({str(i): pin for i, pin in enumerate(native["host_build_sources"])})
        _verify_original_capsules(self._capsules)

    def record(self):
        self.revalidate()
        return dict(
            identity=self.identity,
            prototype=copy.deepcopy(self._prototype),
            qualified_pairs=copy.deepcopy(self._pairs),
            original_compiler_emissions=copy.deepcopy(self._emissions),
            original_source_capsules=copy.deepcopy(self._capsules),
            checks=copy.deepcopy(self._checks),
            coverage=copy.deepcopy(self._coverage),
            domain="functional_spike_current_complete_single_contraction_and_private_epoch",
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )

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
        from gemmini_cleanup_final_guard import check_functional_guard

        return check_functional_guard(self, *args, **kwargs)
