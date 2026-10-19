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

from gemmini_source_execution_issuer_v4 import SourceExecutionIssuerV4
from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_staged_rhs_capability import (
    QualifiedRhsResidencyOwner,
    RhsResidencyCompilerPrototype,
    scoped_original_rhs_metadata,
)
from gemmini_staged_rhs_relocation_effects import original_source_capsule
from gemmini_staged_rhs_window import ClosedScratchEpochOwner
from gemmini_zero_initializer_capability import ZeroInitializerCompilerPrototype

from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import _pin, _tree

PROTOCOL = "merlin.original_parent_pass_qualification.v1"
METADATA = "merlin.pass_metadata_operation.v1"
SNAPSHOT = "gemmini.original_parent_full_k_rhs_qualification.v1"
MECHANISM = "full_k_rhs_residency_raw_i32"
DIAGNOSTIC = "source_execution_diagnostic"


def _base_packages(inputs):
    packages = inputs.get("base_compiler_preflight_packages", [])
    if (
        type(packages) is not list
        or len(packages) > 8
        or any(
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or type(row["path"]) is not str
            or not Path(row["path"]).is_absolute()
            or type(row["sha256"]) is not str
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
            for row in packages
        )
    ):
        raise ValueError("RHS base preflight requires bounded explicit original compiler tree pins")
    return packages


def _preflight_base_packages(seed, packages):
    """Early source/ledger refusal only; never compiler or execution admission."""
    if type(seed) is not ZeroInitializerCompilerPrototype:
        raise ValueError("RHS base preflight needs its original live base compiler owner")
    checked = []
    for row in packages:
        if _tree(Path(row["path"])) != row["sha256"]:
            raise ValueError("RHS base preflight compiler bytes differ from their original pin")
        checked.append(dict(package=copy.deepcopy(row), binding=seed.check_compiler_binding(Path(row["path"]))))
        if _tree(Path(row["path"])) != row["sha256"]:
            raise ValueError("RHS base preflight compiler changed during its original source check")
    return checked


def _binding(owner):
    contract = owner.prototype.seed.tile.contract
    return PassCapabilityBinding(
        contract.hardware,
        contract.software,
        contract.numerical_policy,
        contract.execution,
        owner.prototype.library.contract.sha256,
    )


def from_live_owner(owner):
    """The exact original constructor must already have issued this owner."""
    if type(owner) is not QualifiedRhsResidencyOwner:
        raise ValueError("RHS registry needs its original live qualified composite owner")
    with scoped_original_rhs_metadata(owner):
        binding = _binding(owner)
        return ReviewedPassCapabilities(
            (
                ReviewedPassCapability(MECHANISM, binding, owner),
                ReviewedPassCapability(DIAGNOSTIC, binding, SourceExecutionIssuerV4(owner)),
            )
        )


def _original_selection(registry):
    if type(registry) is not ReviewedPassCapabilities:
        raise ValueError("RHS parent qualification needs its original live registry")
    with scoped_registry_metadata_verification(registry):
        rows = registry.record()["capabilities"]
        if len(rows) != 2 or {row["mechanism"] for row in rows} != {MECHANISM, DIAGNOSTIC}:
            raise ValueError("RHS parent qualification has a foreign capability roster")
        binding = PassCapabilityBinding(**next(row for row in rows if row["mechanism"] == MECHANISM)["binding"])
        capability = registry.resolve(MECHANISM, binding)
        diagnostic = registry.resolve(DIAGNOSTIC, binding)
        if (
            capability is None
            or diagnostic is None
            or type(capability.owner) is not QualifiedRhsResidencyOwner
            or type(diagnostic.owner) is not SourceExecutionIssuerV4
            or diagnostic.owner._qualification is not capability.owner
        ):
            raise ValueError("RHS parent needs the same original compiler and current-source owners")
        owner = capability.owner
    return binding, owner


def export_qualification(registry):
    """Original host export of an already-issued live qualification, not issuance."""
    with scoped_registry_metadata_verification(registry):
        binding, owner = _original_selection(registry)
        cleanup = owner.prototype.seed.tile.cleanup
        snapshot = dict(
            schema=SNAPSHOT,
            binding=asdict(binding),
            registry=registry.record(),
            qualified_owner=owner.record(),
            original_epoch_state=_export_epoch(owner.prototype.epoch),
            original_cleanup_admission=dict(
                receipt=_pin(cleanup.path), invocation=_pin(cleanup.invocation), record=cleanup.record()
            ),
        )
        payload = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        if len(payload.encode()) > 3 << 20:
            raise ValueError("RHS qualification exceeds its unchanged bounded parent transport")
    return json.loads(payload)


_EPOCH_FIELDS = (
    "_owner_identity",
    "_capsule",
    "_proof",
    "_checked",
    "_rule",
    "_source_rows",
    "_geometry",
    "_footprint",
    "_stream",
    "_sources",
    "_identity",
)


def _export_epoch(epoch):
    if type(epoch) is not ClosedScratchEpochOwner:
        raise ValueError("RHS parent needs its original live complete-program epoch")
    epoch.revalidate()
    return dict(
        cap=str(epoch.cap),
        source_owner_artifacts={name: str(path) for name, path in epoch.source_owner._paths.items()},
        state={name: copy.deepcopy(getattr(epoch, name)) for name in _EPOCH_FIELDS},
        record=epoch.record(),
    )


def _restore_epoch(snapshot, seed, *, original_snapshot, token):
    """Private decoder called only after original token.require authentication."""
    from gemmini_source_execution import SourceExecutionOwner

    _active_context(original_snapshot, token)
    if original_snapshot["original_epoch_state"] != snapshot:
        raise ValueError("RHS inherited epoch is not the actual original parent payload")
    if type(snapshot) is not dict or set(snapshot) != {"cap", "source_owner_artifacts", "state", "record"}:
        raise ValueError("RHS inherited epoch is incomplete")
    paths, state = snapshot["source_owner_artifacts"], snapshot["state"]
    if (
        type(paths) is not dict
        or set(paths) != {"result", "command_buffer", "lowered", "trace_invocation", "nm"}
        or paths["nm"] != str(seed.nm)
        or type(state) is not dict
        or set(state) != set(_EPOCH_FIELDS)
    ):
        raise ValueError("RHS inherited epoch changed its original executable/checker membership")
    source = SourceExecutionOwner(arithmetic=seed.tile.arithmetic, selected=seed.tile.selected, **paths)
    epoch = object.__new__(ClosedScratchEpochOwner)
    epoch.source_owner, epoch.cap = source, Path(snapshot["cap"])
    for name in _EPOCH_FIELDS:
        setattr(epoch, name, copy.deepcopy(state[name]))
    epoch.revalidate()
    if epoch.record() != snapshot["record"]:
        raise ValueError("RHS inherited epoch differs from the original current-byte parent proof")
    return epoch


def _active_context(snapshot, token):
    # The generic implementation rejects JSON/copied descriptors and verifies
    # original callback/code/source/factory inputs before entering this method.
    context = token.require(snapshot)
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    if active_original_parent_pass_qualification() is not token:
        raise ValueError("RHS inheritance needs its active original sealed-parent token")
    if (
        type(context) is not dict
        or set(context) != {"registry", "factory_inputs"}
        or type(snapshot) is not dict
        or set(snapshot)
        != {"schema", "binding", "registry", "qualified_owner", "original_cleanup_admission", "original_epoch_state"}
        or snapshot["schema"] != SNAPSHOT
        or context["registry"] != snapshot["registry"]
        or context["factory_inputs"].get("parent_delegation_protocol") != PROTOCOL
        or context["factory_inputs"].get("metadata_verification_protocol") != METADATA
    ):
        raise ValueError("RHS inheritance differs from its exact original parent context")
    return context


def _restore_original_owner(snapshot, inputs, *, token):
    context = _active_context(snapshot, token)
    if context["factory_inputs"] != inputs:
        raise ValueError("RHS inheritance changed its original factory inputs")
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
        raise ValueError("RHS inherited cleanup custody is incomplete")
    for role in ("receipt", "invocation"):
        pin = admission[role]
        if type(pin) is not dict or set(pin) != {"path", "sha256"} or _pin(Path(pin["path"])) != pin:
            raise ValueError("RHS inherited original receipt or invocation changed")
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
    seed = ZeroInitializerCompilerPrototype(StagedTilePrototype(StagedDensePrototype(compiler, arithmetic, completion)))
    prototype = RhsResidencyCompilerPrototype(
        seed, _restore_epoch(snapshot["original_epoch_state"], seed, original_snapshot=snapshot, token=token)
    )
    record = snapshot["qualified_owner"]
    if (
        record["prototype"] != prototype.record()
        or record["qualified_pairs"] != inputs["rhs_residency_qualification_pairs"]
    ):
        raise ValueError("RHS inherited qualification differs from current exact source and SDK inputs")
    # Only the authenticated original live parent establishes these historical
    # semantic checks. Current bytes, source/ELF/trace/issue/invocation ownership
    # are reopened below; newly emitted candidate programs are never inherited.
    owner = object.__new__(QualifiedRhsResidencyOwner)
    owner.prototype = prototype
    owner._prototype = prototype.record()
    owner._pairs = copy.deepcopy(record["qualified_pairs"])
    owner._checks = copy.deepcopy(record["checks"])
    owner._emissions = copy.deepcopy(record["original_compiler_emissions"])
    owner._capsules = copy.deepcopy(record["original_source_capsules"])
    owner._coverage = copy.deepcopy(record["coverage"])
    from merlin.common.paths import module_source_path

    owner._source = _pin(module_source_path("gemmini_staged_rhs_capability"))
    prefix = "gemmini.qualified_full_k_rhs.v1:"
    identity = record["identity"]
    if type(identity) is not str or not identity.startswith(prefix):
        raise ValueError("RHS inherited qualification has a foreign original identity")
    owner._identity = identity[len(prefix) :]
    owner.revalidate()
    if owner.record() != record:
        raise ValueError("RHS inherited source owner differs from original live parent qualification")
    return owner


def redeem_qualification(snapshot, *, token, work_root):
    """Only the original sealed-child redemption context can use this decoder."""
    context = _active_context(snapshot, token)
    with shared_compute_pool(timeout_s=120):
        owner = _restore_original_owner(snapshot, context["factory_inputs"], token=token)
        if _binding(owner) != PassCapabilityBinding(**snapshot["binding"]):
            raise ValueError("RHS inherited exact capability binding changed")
        registry = from_live_owner(owner)
        with scoped_registry_metadata_verification(registry):
            if registry.record() != snapshot["registry"]:
                raise ValueError("RHS inherited registry differs from original parent")
        work = Path(work_root)
        if not work.is_absolute() or work.is_symlink() or work.resolve() != work:
            raise ValueError("RHS inheritance requires its original canonical worker directory")
        work.mkdir(parents=True, exist_ok=True)
        receipt = dict(
            schema="gemmini.inherited_original_rhs_qualification.v1",
            status="inherited_original_parent_qualification_revalidated",
            registry_identity=registry.identity,
            qualified_owner_identity=owner.identity,
            parent_snapshot_sha256=_digest(snapshot),
            fresh_source_pair_semantic_replay=False,
            original_artifact_source_sdk_runtime_custody="reopened",
            current_candidate_execution="fresh native trace/source/products/state/complete outputs required",
            hardware_execution_qualified=False,
            timing_authority="NONE",
        )
        with (work / "inherited-rhs-qualification.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
        return registry


def restore_registry(inputs, *, work_root):
    """Full host admission; authenticated original-child inheritance only."""
    if (
        type(inputs) is not dict
        or inputs.get("parent_delegation_protocol") != PROTOCOL
        or inputs.get("metadata_verification_protocol") != METADATA
        or type(inputs.get("rhs_allocation_epoch_source")) is not dict
        or type(inputs.get("rhs_residency_qualification_pairs")) is not list
        or not 6 <= len(inputs["rhs_residency_qualification_pairs"]) <= 32
        or any(
            type(pair) is not dict or set(pair) != {"before", "after"}
            for pair in inputs["rhs_residency_qualification_pairs"]
        )
    ):
        raise ValueError("RHS selection disabled: explicit original protocols and actual source pairs required")
    base_packages = _base_packages(inputs)
    allocation = inputs["rhs_allocation_epoch_source"]
    for role in ("native_result", "command_buffer_artifact", "lowered", "issue_invocation"):
        _leaf(allocation, role)
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
        from gemmini_source_execution import SourceExecutionOwner

        seed = ZeroInitializerCompilerPrototype(StagedTilePrototype(staged))
        preflight = _preflight_base_packages(seed, base_packages)
        if preflight:
            (Path(work_root) / "original-base-preflight.json").write_text(
                json.dumps(dict(status="checked_base_sources_and_ledgers_only", packages=preflight), indent=2)
            )
        pairs = tuple(inputs["rhs_residency_qualification_pairs"])
        # Preparation has its own original observed allocation basis. A fresh
        # compiler invocation at another path is independently checked below;
        # identical bytes cannot silently replace that original custody owner.
        first = allocation
        native = json.loads(Path(first["native_result"]["path"]).read_bytes())
        source = SourceExecutionOwner(
            selected=staged.selected,
            arithmetic=staged.arithmetic,
            nm=staged.nm,
            **{
                name: first[role]["path"]
                for name, role in (
                    ("result", "native_result"),
                    ("command_buffer", "command_buffer_artifact"),
                    ("lowered", "lowered"),
                    ("trace_invocation", "issue_invocation"),
                )
            },
        )
        epoch = ClosedScratchEpochOwner(source, capsule_dir=original_source_capsule(first, native))
        prototype = RhsResidencyCompilerPrototype(seed, epoch)
        owner = QualifiedRhsResidencyOwner(prototype, pairs, status_path=Path(work_root) / "original-rhs-status.json")
        return from_live_owner(owner)
