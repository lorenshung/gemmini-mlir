"""Original sealed public source -> generated compiler -> native/effects -> writer.

Operator-only experiment entry. External aggregate CPU/RSS/disk supervision is
mandatory; this entry does not create or certify that resource boundary.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import yaml
from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen import (
    application_inventory,
)
from merlin.targetgen import (
    capsule_runner as CR,
)
from merlin.targetgen import (
    capsule_source as CS,
)
from merlin.targetgen import (
    package_runtime as PK,
)
from merlin.targetgen.corpus_spec import CorpusBinding
from merlin.targetgen.native_model_policy import _frozen_model_policy
from merlin.targetgen.rtl.facts import observed_facts
from merlin.targetgen.target_registry import observed_contract
from merlin_experiments.phase0 import (
    connected_integer_qualification as Q,
)
from merlin_experiments.phase0 import (
    evidence,
    freeze,
    requirements,
    staging,
    writer,
)
from merlin_experiments.phase0 import (
    public_programs as PP,
)
from merlin_experiments.phase0 import (
    public_source_roles as R,
)
from merlin_experiments.phase0 import (
    sealed_native_materialization as NM,
)
from merlin_experiments.phase0.instruction_roles import (
    derive_role_taxonomy,
    require_enforceable,
    resolve_policy,
)
from source_admission_inputs import (
    copy_compiler,
    native_loader_python_preflight,
    read_declaration,
    select_abi,
)
from source_admission_source import check

LOGGER = logging.getLogger(__name__)


def _pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def require_purpose(declaration, intake):
    purpose = intake.document(declaration["purpose"])
    limits = purpose["limits"]
    names = {
        "aggregate_cpu_seconds": 120,
        "wall_seconds": 300,
        "rss_bytes": 1 << 30,
        "file_bytes": 64 << 20,
        "disk_bytes": 128 << 20,
    }
    if any(
        type(limits.get(key)) is not int or not 0 < limits[key] <= maximum
        for key, maximum in names.items()
    ):
        raise ValueError("source admission requires finite unchanged per-job ceilings")
    now = datetime.datetime.now(datetime.timezone.utc)
    start, end = [
        datetime.datetime.fromisoformat(purpose[key])
        for key in ("start_utc", "deadline_utc")
    ]
    if (
        start.tzinfo is None
        or end.tzinfo is None
        or not start <= now < end
        or (end - start).total_seconds() > limits["wall_seconds"]
    ):
        raise ValueError(
            "source admission purpose is not within its original fixed clock"
        )
    if (
        purpose.get("identity", declaration["source"]["identity"])
        != declaration["source"]["identity"]
    ):
        raise ValueError("purpose selects another original source member")
    return purpose


def prepare(declaration, selected, output):
    """Compose original bytes and role-bound profile; no target work yet."""
    member = selected["member"]
    attestation = selected["attestation"]
    contract = Path(declaration["execution_contract"]["path"])
    inputs = selected["factory_inputs"]
    target = "gemmini"  # This experiment selects this OOT backend, not a shared compiler policy.
    composed = output / "composed"
    NM.materialize(
        attestation,
        program=declaration["source"]["program"],
        source_member=declaration["source"]["source_member"],
        destination=composed,
        max_payload_bytes=16 << 20,
    )
    label = member["request"]["declaration"]["name"]
    demand = application_inventory.application_demand_inventory(
        {label: composed / "model.mlir"},
        target,
        detailed=True,
        capability_contract=yaml.safe_load(contract.read_bytes()),
        software_spec=yaml.safe_load(Path(inputs["software_spec"]).read_bytes()),
        application_metadata={
            label: {
                "workload_id": label,
                "workload_role": "iteration",
                "coverage_scope": "full_capture",
            }
        },
    )
    raw = json.dumps(
        demand, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    policy = {
        "schema": PP.INTEGER_POLICY_SCHEMA,
        "source_mode": "native_integer_v1",
        "selection": "declared_performance_applications",
        "max_members": 8,
        "max_capture_bytes": 16 << 20,
        "max_input_elements": 1_000_000,
        "max_output_elements": 1_000_000,
    }
    entries, payloads = requirements._materialized_iteration_capsules(
        demand,
        hashlib.sha256(raw).hexdigest(),
        loader_snapshots={
            label: (selected["loader"], _pin(selected["loader"])["sha256"])
        },
        performance_selection=policy,
        performance_forms={},
        facts_sha256=_pin(inputs["facts"])["sha256"],
        capture_attestations={label: attestation},
        source_role_selection=selected["roles"],
        source_role_members={label: declaration["source"]["identity"]},
    )
    if len(entries) != 1:
        raise ValueError(
            "one original selected source must produce exactly one complete profile member"
        )
    prepared = output / "prepared"
    evidence._materialize_evidence(prepared, payloads)
    (prepared / "synth.yaml").write_text(
        yaml.safe_dump({"capsules": entries}, sort_keys=False)
    )
    freeze._materialized_inputs(
        {"inputs": {"synth_profile": str(prepared / "synth.yaml")}}
    )
    entry = entries[0]
    for key in ("path", "loader_path", "profile_path"):
        if key in entry["materialized_capture"]:
            entry["materialized_capture"][key] = str(
                prepared / entry["materialized_capture"][key]
            )
    binding = CorpusBinding(
        target="source_only",
        tile_dim=None,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        atol=0.0,
        rtol=0.0,
    )
    artifact = CS.materialized_model_artifacts(entry["materialized_capture"])
    unqualified = CS.write_model_capsule(
        entry, binding, output / "unqualified-source", artifact=artifact
    )
    cap = yaml.safe_load((unqualified / "capsule.yaml").read_bytes())
    if entry.get("public_source_role") is not None:
        with R.scoped_selection(selected["roles"]):
            R.stamp_written(entry, cap, unqualified)
        (unqualified / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
    PP.verify_written(cap, unqualified)
    selected["private"].check_member(
        unqualified
    )  # Original eligibility BEFORE expensive target work.
    return entry, binding, unqualified


def diagnostic(declaration, selected, unqualified, output):
    """Only stronger requested tiers/classes differ from the original source tree."""
    import shutil

    from gemmini_selected_scheduling import SelectedSchedulingTarget

    inputs = selected["factory_inputs"]
    target = SelectedSchedulingTarget.read(
        Path(inputs["facts"]), Path(inputs["observed_contract"])
    )
    target.revalidate()
    target.operand_observer()
    facts, contract, _ = target._documents()
    with (
        observed_contract("gemmini", contract),
        observed_facts("gemmini", facts, Path(inputs["facts"])),
    ):
        taxonomy = derive_role_taxonomy("gemmini")
    policy = require_enforceable(resolve_policy(["loop_descriptor"], taxonomy))
    classes = sorted(
        {
            name
            for role in ("memory", "weight_load", "matmul", "acc_readout")
            for name in contract["corpus_instruction_roles"]["by_role"][role]
        }
    )
    capsule = output / "experiment/corpus/model/public_0000"
    capsule.parent.mkdir(parents=True)
    shutil.copytree(unqualified, capsule)
    (output / "experiment/corpus/isa").mkdir()
    cap = yaml.safe_load((capsule / "capsule.yaml").read_bytes())
    cap["expected"] = {**cap["expected"], "instruction_classes": classes}
    cap["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    (capsule / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
    coverage = yaml.safe_load(
        (capsule / "expected_instruction_coverage.yaml").read_bytes()
    )
    coverage["instruction_classes"] = classes
    (capsule / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(coverage, sort_keys=False)
    )
    manifest = {
        "schema": "merlin.phase2.development_instruction_policy.v1",
        "instruction_policy": policy,
        "instruction_policy_taxonomy": taxonomy,
        "selected_inputs": target.report()["selected_inputs"],
        "status": "functional_diagnostic_only",
        "timing_qualified": False,
        "certified": False,
        "objective_members": [],
        "public_program_source_basis": "not_established",
        "private_validation_exclusion": "not_established",
        "hardware_execution_qualification": "not_established",
    }
    (output / "experiment/corpus/MANIFEST.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    descriptor = output / "experiment/target_experiment.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "gemmini",
                "rtl": {"via": "mlc"},
                "hardware_spec": {
                    "target_contract": declaration["execution_contract"]["path"]
                },
                "capsule_corpus": str(output / "experiment/corpus/isa"),
                "toolchain": {"sim_via": "chipyard"},
            },
            sort_keys=False,
        )
    )
    changed = {
        p.relative_to(unqualified).as_posix()
        for p in unqualified.rglob("*")
        if p.is_file()
        and p.read_bytes() != (capsule / p.relative_to(unqualified)).read_bytes()
    }
    if changed != {"capsule.yaml", "expected_instruction_coverage.yaml"}:
        raise ValueError("diagnostic selection changed original source/reference bytes")
    os.environ.update(
        MERLIN_TARGET_EXPERIMENT=str(descriptor),
        MERLIN_TARGET_CONTRACT=declaration["execution_contract"]["path"],
    )
    _frozen_model_policy(capsule, target="gemmini")
    return capsule


def execute(declaration, intake, selected, output, report, save):
    purpose = require_purpose(declaration, intake)
    report["native_loader_python"] = native_loader_python_preflight()

    def stage(name):
        require_purpose(declaration, intake)
        report["stage"] = name
        save()
        print(name, flush=True)

    stage("original_materialization_and_early_publication_eligibility")
    with shared_compute_pool(timeout_s=60):
        entry, binding, unqualified = prepare(declaration, selected, output)
        capsule = diagnostic(declaration, selected, unqualified, output)
    stage("unchanged_original_compiler")
    with shared_compute_pool(timeout_s=60):
        baseline = output / "baseline-package"
        copy_compiler(declaration, intake, baseline)
        package = PK.load_package(baseline)
        PK.integrity_scan(package)
        compiled = output / "compiler"
        compiled.mkdir()
        emitted = PK.run_entrypoint(
            package,
            "emit_analysis_bundle",
            capsule / "capsule.interface.mlir",
            compiled / "command_buffer.json",
            timeout=45,
            invocation_directory=compiled / "ordinary-invocation",
        )
        (compiled / "lowered.llvm.mlir").write_text(emitted.stdout)
        (compiled / "compiler.stderr").write_text(emitted.stderr)
        if emitted.returncode:
            raise ValueError(
                "unchanged selected compiler refused: " + emitted.stderr[-2000:]
            )
    cb = json.loads((compiled / "command_buffer.json").read_bytes())
    route, inputs = select_abi(
        cb, selected["factory_inputs"], declaration["execution_contract"]["path"]
    )
    report["selected_original_ABI_route"] = route
    stage("original_live_source_factory")
    from gemmini_generated_source_factory import original_factory

    registry = original_factory(inputs).create(work_root=output / "source-registry")
    capability = registry._capabilities[0]
    issuer = registry.resolve("source_execution_diagnostic", capability.binding).owner
    stage("original_native_complete_output_execution")
    if route == "whole_program":
        from merlin.targetgen.native_model_execution import (
            execute_candidate_model as native,
        )
    else:
        from merlin.targetgen.native_pointer_source_execution import (
            execute_candidate_pointer_source as native,
        )
    cap = yaml.safe_load((capsule / "capsule.yaml").read_bytes())
    cap["__dir__"] = str(capsule)
    with (
        shared_compute_pool(timeout_s=60),
        CR._model_runtime_bundle(
            cap, timeout=60, artifact_root=output / "runtime-bundle"
        ) as (bundle, _, verify),
    ):
        verify()
        result = native(
            command_buffer=cb,
            lowered_mlir_text=emitted.stdout,
            capsule_dir=capsule,
            capture_bundle=bundle,
            target="gemmini",
            out_dir=output / "native",
            simulator="spike",
            timeout=60,
            oracle_tier_ceiling="L2",
            complete_call_timing=False,
            rtl_facts=Path(inputs["facts"]),
            board_config="FireSimGemminiRocketConfig",
        )
        verify()
    if result.get("failure") or result.get("numeric", {}).get("status") != "pass":
        raise ValueError(
            "original complete native execution refused: " + str(result.get("failure"))
        )
    report["complete_native_numeric"] = result["numeric"]
    stage("original_live_source_owner_and_complete_trace")
    remaining = (
        datetime.datetime.fromisoformat(purpose["deadline_utc"])
        - datetime.datetime.now(datetime.timezone.utc)
    ).total_seconds()
    owner = issuer.create_source_execution_owner(
        native_result=output / "native/result.json",
        command_buffer=compiled / "command_buffer.json",
        lowered=compiled / "lowered.llvm.mlir",
        work_root=output / "actual-trace",
        remaining_s=min(90, max(1, remaining - 20)),
    )
    stage("original_writer_and_final_private_exclusion")
    with shared_compute_pool(timeout_s=60):
        with R.scoped_selection(selected["roles"]), Q.scoped_realization_owner(owner):
            published = staging.write_staged(
                lambda root: writer._write_capsule(entry, binding, root),
                output / "publication",
                member=entry["cat"] + "/" + entry["name"],
                operator_exclusion=selected["private"],
            )
        PP.verify_written(
            yaml.safe_load((published / "capsule.yaml").read_bytes()), published
        )
        selected["private"].check_member(published)
    intake.finish()
    report.update(
        status="original_public_writer_admitted",
        publication=str(published),
        capsule=_pin(published / "capsule.yaml"),
        owner_identity=owner.identity,
    )


def require_output(declaration, intake, output):
    output = Path(output).absolute()
    attestation = intake.document(declaration["source"]["attestation"])
    capture = attestation["capture"]
    captured = (
        Path(capture["capture_path"])
        if capture.get("kind") == "session"
        else Path(capture["model_path"]).parent
    )
    protected = (
        Path(declaration["source"]["manifest"]["path"]).parent,
        Path(declaration["compiler"]["root"]),
        captured,
    )
    if (
        output.exists()
        or any(p.is_symlink() for p in (output, *output.parents))
        or any(output.is_relative_to(root) for root in protected)
    ):
        raise ValueError(
            "output must be fresh and outside immutable source, capture and compiler trees"
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    declaration, intake = read_declaration(args.declaration, args.sha256)
    output = require_output(declaration, intake, args.output)
    output.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    report = {
        "declaration": {
            "path": str(args.declaration.absolute()),
            "sha256": args.sha256,
        },
        "schema": "gemmini.public_source_admission_report.v1",
        "status": "active",
        "stage": "input_verification",
        "execution_requested": args.execute,
        "new_capture": False,
        "timing_qualified": False,
        "hardware_qualified": False,
        "formal_cohort_frozen": False,
        "external_resource_supervisor_required": True,
        "internal_resource_supervision": False,
    }

    def save():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        with shared_compute_pool(timeout_s=60):
            report["native_loader_python"] = native_loader_python_preflight()
            selected = check(declaration, intake)
        report.update(
            identity=declaration["source"]["identity"],
            source_manifest=declaration["source"]["manifest"],
            compiler_sha256=selected["compiler_sha256"],
            complete_reference_tensors=len(selected["expected"]),
            complete_reference_values=sum(v.size for v in selected["expected"]),
            source_products=selected["member"]["description"]["scalar_products"],
            original_prospective_role=selected["planned"]["prospective_role"],
        )
        if args.execute:
            execute(declaration, intake, selected, output, report, save)
        else:
            report.update(
                status="original_inputs_checked_no_target_execution",
                publication_eligibility="not yet checked on materialized source; original E still required",
            )
        intake.finish()
    except Exception as error:
        LOGGER.exception("original source admission refused")
        report.update(
            status="precise_original_refusal",
            error_type=type(error).__name__,
            reason=str(error),
        )
    finally:
        save()
    print(json.dumps(report, indent=2))
    return (
        0
        if report["status"]
        in {
            "original_public_writer_admitted",
            "original_inputs_checked_no_target_execution",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
