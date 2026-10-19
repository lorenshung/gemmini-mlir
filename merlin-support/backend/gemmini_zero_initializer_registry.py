"""Original composite selection; only authenticated workers inherit qualification.

Ordinary host selection always performs original independent admission. The
separate sealed-parent protocol can transport that live qualification, never a
current candidate's source, native outputs, trace or emitted-program proof.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

from gemmini_source_execution_issuer_v3 import SourceExecutionIssuerV3
from gemmini_staged_dense_capability import _digest
from gemmini_zero_initializer_capability import QualifiedZeroInitializerOwner, ZeroInitializerCompilerPrototype

from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import _pin

PROTOCOL = "merlin.original_parent_pass_qualification.v1"
METADATA = "merlin.pass_metadata_operation.v1"
SNAPSHOT = "gemmini.original_parent_zero_initializer_qualification.v1"
MECHANISM = "zero_initializer_groups_raw_i32"
DIAGNOSTIC = "source_execution_diagnostic"


def _binding(owner):
    contract = owner.prototype.tile.contract
    return PassCapabilityBinding(
        contract.hardware,
        contract.software,
        contract.numerical_policy,
        contract.execution,
        owner.prototype.library.contract.sha256,
    )


def from_live_owner(owner):
    """The exact original constructor must already have issued this owner."""
    if type(owner) is not QualifiedZeroInitializerOwner:
        raise ValueError("initializer registry needs its original live qualified composite owner")
    owner.revalidate()
    binding = _binding(owner)
    return ReviewedPassCapabilities(
        (
            ReviewedPassCapability(MECHANISM, binding, owner),
            ReviewedPassCapability(DIAGNOSTIC, binding, SourceExecutionIssuerV3(owner)),
        )
    )


def _original_selection(registry):
    if type(registry) is not ReviewedPassCapabilities:
        raise ValueError("initializer parent qualification needs its original live registry")
    with scoped_registry_metadata_verification(registry):
        rows = registry.record()["capabilities"]
        if len(rows) != 2 or {row["mechanism"] for row in rows} != {MECHANISM, DIAGNOSTIC}:
            raise ValueError("initializer parent qualification has a foreign capability roster")
        binding = PassCapabilityBinding(**next(row for row in rows if row["mechanism"] == MECHANISM)["binding"])
        capability = registry.resolve(MECHANISM, binding)
        diagnostic = registry.resolve(DIAGNOSTIC, binding)
        if (
            capability is None
            or diagnostic is None
            or type(capability.owner) is not QualifiedZeroInitializerOwner
            or type(diagnostic.owner) is not SourceExecutionIssuerV3
            or diagnostic.owner._qualification is not capability.owner
        ):
            raise ValueError("initializer parent needs the same original compiler and current-source owners")
        owner = capability.owner
    return binding, owner


def export_qualification(registry):
    """Original host export of an already-issued live qualification, not issuance."""
    with scoped_registry_metadata_verification(registry):
        binding, owner = _original_selection(registry)
        cleanup = owner.prototype.tile.cleanup
        snapshot = dict(
            schema=SNAPSHOT,
            binding=asdict(binding),
            registry=registry.record(),
            qualified_owner=owner.record(),
            original_cleanup_admission=dict(
                receipt=_pin(cleanup.path), invocation=_pin(cleanup.invocation), record=cleanup.record()
            ),
        )
        payload = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        if len(payload.encode()) > 3 << 20:
            raise ValueError("initializer qualification exceeds its unchanged bounded parent transport")
    return json.loads(payload)


def _active_context(snapshot, token):
    # The generic implementation rejects JSON/copied descriptors and verifies
    # original callback/code/source/factory inputs before entering this method.
    context = token.require(snapshot)
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    if active_original_parent_pass_qualification() is not token:
        raise ValueError("initializer inheritance needs its active original sealed-parent token")
    if (
        type(context) is not dict
        or set(context) != {"registry", "factory_inputs"}
        or type(snapshot) is not dict
        or set(snapshot) != {"schema", "binding", "registry", "qualified_owner", "original_cleanup_admission"}
        or snapshot["schema"] != SNAPSHOT
        or context["registry"] != snapshot["registry"]
        or context["factory_inputs"].get("parent_delegation_protocol") != PROTOCOL
        or context["factory_inputs"].get("metadata_verification_protocol") != METADATA
    ):
        raise ValueError("initializer inheritance differs from its exact original parent context")
    return context


def _restore_original_owner(snapshot, inputs, *, token):
    context = _active_context(snapshot, token)
    if context["factory_inputs"] != inputs:
        raise ValueError("initializer inheritance changed its original factory inputs")
    from gemmini_arithmetic_scope import IntegerArithmeticOwner
    from gemmini_cleanup_admission import _ISSUED, OriginalCleanupAdmission
    from gemmini_compiler_passes import CompilerCleanupOwner
    from gemmini_delegated_source_execution import _library
    from gemmini_selected_scheduling import SelectedSchedulingTarget
    from gemmini_staged_dense_admission import StagedCompletionOwner
    from gemmini_staged_dense_capability import StagedDensePrototype
    from gemmini_staged_tile_capability import StagedTilePrototype

    admission = snapshot["original_cleanup_admission"]
    if type(admission) is not dict or set(admission) != {"receipt", "invocation", "record"}:
        raise ValueError("initializer inherited cleanup custody is incomplete")
    for role in ("receipt", "invocation"):
        pin = admission[role]
        if type(pin) is not dict or set(pin) != {"path", "sha256"} or _pin(Path(pin["path"])) != pin:
            raise ValueError("initializer inherited original receipt or invocation changed")
    library = _library(inputs)
    selected = SelectedSchedulingTarget.read(Path(inputs["facts"]), Path(inputs["observed_contract"]))
    cleanup = OriginalCleanupAdmission(
        Path(admission["receipt"]["path"]),
        admission["record"],
        Path(admission["invocation"]["path"]),
        selected,
        library,
        _issuer=_ISSUED,
    )
    compiler = CompilerCleanupOwner(
        cleanup, library, original_host_roots=(Path(inputs["merlin"]) / "src",), nm=Path(inputs["nm"])
    )
    arithmetic = IntegerArithmeticOwner(
        selected, **{key: Path(inputs[key]) for key in ("software_spec", "source_root", "receiver_source", "runtime")}
    )
    completion = StagedCompletionOwner(selected, source_root=Path(inputs["source_root"]))
    prototype = ZeroInitializerCompilerPrototype(
        StagedTilePrototype(StagedDensePrototype(compiler, arithmetic, completion))
    )
    record = snapshot["qualified_owner"]
    if (
        record["prototype"] != prototype.record()
        or record["qualified_pairs"] != inputs["zero_initializer_qualification_pairs"]
    ):
        raise ValueError("initializer inherited qualification differs from current exact source and SDK inputs")
    # Only the authenticated original live parent establishes these historical
    # semantic checks. Current bytes, source/ELF/trace/issue/invocation ownership
    # are reopened below; newly emitted candidate programs are never inherited.
    owner = object.__new__(QualifiedZeroInitializerOwner)
    owner.prototype = prototype
    owner._prototype = prototype.record()
    owner._pairs = copy.deepcopy(record["qualified_pairs"])
    owner._checks = copy.deepcopy(record["checks"])
    owner._emissions = copy.deepcopy(record["original_compiler_emissions"])
    from merlin.common.paths import module_source_path

    owner._source = _pin(module_source_path("gemmini_zero_initializer_capability"))
    prefix = "gemmini.qualified_zero_initializer.v1:"
    identity = record["identity"]
    if type(identity) is not str or not identity.startswith(prefix):
        raise ValueError("initializer inherited qualification has a foreign original identity")
    owner._identity = identity[len(prefix) :]
    owner.revalidate()
    if owner.record() != record:
        raise ValueError("initializer inherited source owner differs from original live parent qualification")
    return owner


def redeem_qualification(snapshot, *, token, work_root):
    """Only the original sealed-child redemption context can use this decoder."""
    context = _active_context(snapshot, token)
    with shared_compute_pool(timeout_s=120):
        owner = _restore_original_owner(snapshot, context["factory_inputs"], token=token)
        if _binding(owner) != PassCapabilityBinding(**snapshot["binding"]):
            raise ValueError("initializer inherited exact capability binding changed")
        registry = from_live_owner(owner)
        with scoped_registry_metadata_verification(registry):
            if registry.record() != snapshot["registry"]:
                raise ValueError("initializer inherited registry differs from original parent")
        work = Path(work_root)
        if not work.is_absolute() or work.is_symlink() or work.resolve() != work:
            raise ValueError("initializer inheritance requires its original canonical worker directory")
        work.mkdir(parents=True, exist_ok=True)
        receipt = dict(
            schema="gemmini.inherited_original_initializer_qualification.v1",
            status="inherited_original_parent_qualification_revalidated",
            registry_identity=registry.identity,
            qualified_owner_identity=owner.identity,
            parent_snapshot_sha256=_digest(snapshot),
            fresh_seven_pair_semantic_replay=False,
            original_artifact_source_sdk_runtime_custody="reopened",
            current_candidate_execution="fresh native trace/source/products/state/complete outputs required",
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )
        with (work / "inherited-initializer-qualification.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
        return registry


def restore_registry(inputs, *, work_root):
    """Full host admission; authenticated original-child inheritance only."""
    if (
        inputs.get("parent_delegation_protocol") != PROTOCOL
        or inputs.get("metadata_verification_protocol") != METADATA
        or not inputs.get("zero_initializer_qualification_pairs")
    ):
        raise ValueError("initializer selection disabled: explicit original protocols and actual source pairs required")
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    token = active_original_parent_pass_qualification()
    if token is not None:
        return token.redeem(restore_registry, inputs, decoder=redeem_qualification, work_root=work_root)
    from gemmini_staged_dense_admission import prepare_original_prototype
    from gemmini_staged_tile_capability import StagedTilePrototype

    # The cleanup child owns its leaf CPU lease. Only lease independent parent
    # arithmetic/source admission AFTER it returns, never wait under that lease.
    staged = prepare_original_prototype(inputs, work_root=Path(work_root) / "original-admission")
    with shared_compute_pool(timeout_s=120):
        prototype = ZeroInitializerCompilerPrototype(StagedTilePrototype(staged))
        owner = QualifiedZeroInitializerOwner(
            prototype,
            tuple(inputs["zero_initializer_qualification_pairs"]),
            status_path=Path(work_root) / "original-initializer-status.json",
        )
        return from_live_owner(owner)
