"""Opt-in original live-parent qualification, transported by sealed worker FD.

Normal host selection always runs the original full admission. Only the
generic grader's live, authenticated redemption callback may reuse the exact
parent qualification. Current candidate execution is never inherited.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

from gemmini_source_execution_issuer import SourceExecutionIssuer
from gemmini_source_execution_issuer_v2 import SourceExecutionIssuerV2
from gemmini_staged_dense_capability import _digest
from gemmini_staged_tile_capability import QualifiedStagedTileOwner
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import _pin

PROTOCOL = "merlin.original_parent_pass_qualification.v1"
SNAPSHOT = "gemmini.original_parent_source_qualification.v1"
MECHANISM = "staged_dense_tile_order_raw_i32"
DIAGNOSTIC = "source_execution_diagnostic"


def _original_selection(registry):
    if type(registry) is not ReviewedPassCapabilities:
        raise ValueError(
            "parent qualification needs its original live reviewed registry"
        )
    registry.revalidate()
    rows = registry.record()["capabilities"]
    if len(rows) not in (1, 2) or {row["mechanism"] for row in rows} not in (
        {MECHANISM},
        {MECHANISM, DIAGNOSTIC},
    ):
        raise ValueError(
            "parent qualification has a foreign original capability roster"
        )
    main = next(row for row in rows if row["mechanism"] == MECHANISM)
    binding = PassCapabilityBinding(**main["binding"])
    capability = registry.resolve(MECHANISM, binding)
    owner = capability.owner
    if type(owner) is not QualifiedStagedTileOwner:
        raise ValueError("parent qualification lacks its original checked source owner")
    if len(rows) == 2:
        diagnostic = registry.resolve(DIAGNOSTIC, binding)
        if diagnostic is None or type(diagnostic.owner) not in (
            SourceExecutionIssuer,
            SourceExecutionIssuerV2,
        ):
            raise ValueError("parent diagnostic uses a foreign issuer or exact binding")
        issuer = diagnostic.owner
        if type(issuer) is SourceExecutionIssuerV2:
            issuer = issuer._original
        if issuer._qualification is not owner:
            raise ValueError(
                "parent compiler and diagnostic selected different original owners"
            )
    return capability, binding, owner


def from_live_registry(registry):
    """Wrap already issued live owners; never construct one from saved success."""
    capability, binding, owner = _original_selection(registry)
    return ReviewedPassCapabilities(
        (
            capability,
            ReviewedPassCapability(DIAGNOSTIC, binding, SourceExecutionIssuerV2(owner)),
        )
    )


def export_qualification(registry):
    """The original parent calls this while it still owns the live registry."""
    _capability, binding, owner = _original_selection(registry)
    diagnostic = registry.resolve(DIAGNOSTIC, binding)
    if diagnostic is None or type(diagnostic.owner) is not SourceExecutionIssuerV2:
        raise ValueError("parent export requires the explicit v2 live issuer")
    cleanup = owner.prototype.prototype.cleanup_owner.cleanup
    snapshot = {
        "schema": SNAPSHOT,
        "binding": asdict(binding),
        "registry": registry.record(),
        "qualified_owner": owner.record(),
        "original_cleanup_admission": {
            "receipt": _pin(cleanup.path),
            "invocation": _pin(cleanup.invocation),
            "record": cleanup.record(),
        },
    }
    payload = json.dumps(snapshot, sort_keys=True, allow_nan=False)
    if len(payload.encode()) > 3 << 20:
        raise ValueError(
            "original parent source qualification exceeds its bounded transport"
        )
    registry.revalidate()
    return json.loads(payload)


def _library(inputs):
    from merlin.targetgen.compiler_library import (
        CompilerLibraryContract,
        CompilerLibrarySelection,
        LibraryMember,
    )

    declaration = json.loads(Path(inputs["library_selection"]).read_bytes())
    record = declaration["selection"]
    contract = record["contract"]
    selected = CompilerLibrarySelection(
        CompilerLibraryContract(
            contract["review_id"],
            tuple(contract["public_modules"]),
            tuple(LibraryMember(**row) for row in contract["members"]),
        ),
        Path(record["root"]),
    )
    if selected.record() != record:
        raise ValueError("inherited qualification selected a different exact SDK")
    return selected


def _restore_original_owner(snapshot, inputs, *, token):
    # This private helper is reached only after token.require and active-token
    # identity checks below. It does not accept a public serialized verdict.
    context = token.require(snapshot)
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    if (
        active_original_parent_pass_qualification() is not token
        or context["factory_inputs"] != inputs
    ):
        raise ValueError(
            "original owner restoration requires authenticated live parent context"
        )
    from gemmini_arithmetic_scope import IntegerArithmeticOwner
    from gemmini_cleanup_admission import _ISSUED, OriginalCleanupAdmission
    from gemmini_compiler_passes import CompilerCleanupOwner
    from gemmini_selected_scheduling import SelectedSchedulingTarget
    from gemmini_staged_dense_admission import StagedCompletionOwner
    from gemmini_staged_dense_capability import StagedDensePrototype
    from gemmini_staged_tile_capability import StagedTilePrototype

    admission = snapshot["original_cleanup_admission"]
    if type(admission) is not dict or set(admission) != {
        "receipt",
        "invocation",
        "record",
    }:
        raise ValueError("inherited original cleanup custody is incomplete")
    for role in ("receipt", "invocation"):
        pin = admission[role]
        if (
            type(pin) is not dict
            or set(pin) != {"path", "sha256"}
            or _pin(Path(pin["path"])) != pin
        ):
            raise ValueError(
                "inherited original admission receipt or invocation changed"
            )
    library = _library(inputs)
    selected = SelectedSchedulingTarget.read(
        Path(inputs["facts"]), Path(inputs["observed_contract"])
    )
    cleanup = OriginalCleanupAdmission(
        Path(admission["receipt"]["path"]),
        admission["record"],
        Path(admission["invocation"]["path"]),
        selected,
        library,
        _issuer=_ISSUED,
    )
    compiler = CompilerCleanupOwner(
        cleanup,
        library,
        original_host_roots=(Path(inputs["merlin"]) / "src",),
        nm=Path(inputs["nm"]),
    )
    arithmetic = IntegerArithmeticOwner(
        selected,
        **{
            key: Path(inputs[key])
            for key in ("software_spec", "source_root", "receiver_source", "runtime")
        },
    )
    completion = StagedCompletionOwner(
        selected, source_root=Path(inputs["source_root"])
    )
    prototype = StagedTilePrototype(
        StagedDensePrototype(compiler, arithmetic, completion)
    )
    record = snapshot["qualified_owner"]
    if (
        record["prototype"] != prototype.record()
        or record["qualified_pairs"] != inputs["tile_qualification_pairs"]
    ):
        raise ValueError(
            "inherited original source qualification differs from current selected inputs"
        )
    # Only the authenticated live parent establishes these semantic checks.
    # The existing original revalidator reopens every actual program, source,
    # compiler invocation, runtime/build source and trace pin in this state.
    owner = object.__new__(QualifiedStagedTileOwner)
    owner.prototype = prototype
    owner._pairs = copy.deepcopy(record["qualified_pairs"])
    owner._checks = copy.deepcopy(record["checks"])
    owner._emissions = copy.deepcopy(record["original_compiler_emissions"])
    owner._prototype_identity = prototype.identity
    owner._prototype_record = prototype.record()
    from merlin.common.paths import module_source_path

    owner._source = _pin(module_source_path("gemmini_staged_tile_capability"))
    prefix = "gemmini.qualified_staged_tile.v1:"
    identity = record["identity"]
    if type(identity) is not str or not identity.startswith(prefix):
        raise ValueError("inherited original qualification has a foreign identity")
    owner._identity = identity[len(prefix) :]
    owner.revalidate()
    if owner.record() != record:
        raise ValueError(
            "inherited source owner differs from the original live parent qualification"
        )
    return owner


def redeem_qualification(snapshot, *, token, work_root):
    """Called only inside authenticated original sealed-parent redemption."""
    context = token.require(snapshot)
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    if active_original_parent_pass_qualification() is not token:
        raise ValueError(
            "inherited qualification requires the active original sealed-parent token"
        )
    if (
        type(snapshot) is not dict
        or set(snapshot)
        != {
            "schema",
            "binding",
            "registry",
            "qualified_owner",
            "original_cleanup_admission",
        }
        or snapshot["schema"] != SNAPSHOT
        or type(context) is not dict
        or set(context) != {"registry", "factory_inputs"}
        or context["registry"] != snapshot["registry"]
        or context["factory_inputs"].get("parent_delegation_protocol") != PROTOCOL
    ):
        raise ValueError(
            "inherited source qualification differs from the exact original parent context"
        )
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool(timeout_s=120):
        owner = _restore_original_owner(
            snapshot, context["factory_inputs"], token=token
        )
        binding = PassCapabilityBinding(**snapshot["binding"])
        contract = owner.prototype.contract
        if binding != PassCapabilityBinding(
            contract.hardware,
            contract.software,
            contract.numerical_policy,
            contract.execution,
            owner.prototype.library.contract.sha256,
        ):
            raise ValueError(
                "inherited source qualification changed its exact selected capability binding"
            )
        registry = from_live_registry(
            ReviewedPassCapabilities(
                (ReviewedPassCapability(MECHANISM, binding, owner),)
            )
        )
        if registry.record() != snapshot["registry"]:
            raise ValueError("inherited registry differs from the original live parent")
        work = Path(work_root)
        if not work.is_absolute() or work.is_symlink() or work.resolve() != work:
            raise ValueError(
                "inherited qualification needs an ordinary original worker directory"
            )
        work.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": "gemmini.inherited_original_source_qualification.v1",
            "status": "inherited_original_parent_qualification_revalidated",
            "registry_identity": registry.identity,
            "qualified_owner_identity": owner.identity,
            "parent_snapshot_sha256": _digest(snapshot),
            "fresh_six_pair_semantic_replay": False,
            "original_artifact_source_sdk_runtime_custody": "reopened",
            "current_candidate_execution": "fresh native trace, source/product/state/complete-output proof still required",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }
        with (work / "inherited-source-qualification.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
        return registry


def restore_registry(inputs, *, work_root):
    """Full original admission on host; authenticated opt-in reuse in child."""
    if inputs.get("parent_delegation_protocol") != PROTOCOL:
        raise ValueError(
            "source parent delegation requires explicit original factory opt-in"
        )
    from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
        active_original_parent_pass_qualification,
    )

    token = active_original_parent_pass_qualification()
    if token is not None:
        return token.redeem(
            restore_registry, inputs, decoder=redeem_qualification, work_root=work_root
        )
    from gemmini_source_execution_registry import restore_registry as original_restore
    from merlin.common.cpu_pool import shared_compute_pool

    registry = original_restore(inputs, work_root=work_root)
    with shared_compute_pool(timeout_s=120):
        return from_live_registry(registry)
