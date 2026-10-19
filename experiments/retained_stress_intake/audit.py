"""Reopen retained public stress outputs; never transfer their labels to new ELFs."""

import json
import os
import time
from pathlib import Path

import numpy as np
from helpers import check_arm_context, compare_complete_integer_outputs

T = Path("/work/merlin/tasks/phase2-target-adapters-20261019")
P = Path("/work/merlin/tasks/phase1-harness-v2")
R = Path("/work/merlin/tasks/phase2-el4-explore-20261018")
OUT = T / "out/artifacts/target-adapters/gemmini/retained-stress-audit-v3"
OUT.mkdir(exist_ok=False)
facts = P / "merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/input/v6/facts.json"
contract = (
    P
    / "merlin/out/artifacts/probes/gemmini/v1/probes_gemmini_v1_20261018T090833Z_580ba4a/target_contract.json"
)
os.environ.update(
    TMPDIR=str(T / ".tmp"),
    MERLIN_COMPUTE_CPUS="6,7",
    MERLIN_COMPUTE_CPU_SLOTS=str(P / ".tmp/phase1-compute-slots"),
    MERLIN_RTL_FACTS=str(facts),
    MERLIN_TARGET_CONTRACT=str(contract),
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)
from merlin.common.cpu_pool import shared_compute_pool
from merlin.runtime.reference import reference_outputs
from merlin.targetgen.rtl import facts as RTL
from merlin_experiments.phase1.feedback import native_output_readback as READ
from merlin_experiments.phase2 import calibration_validation as CV
from merlin_experiments.phase2.contracts import sha256_file, write_json


def pin(p):
    return {"path": str(p), "sha256": sha256_file(p)}


start = time.monotonic()
files = sorted(
    f
    for base in [R / "out/build/phase2-campaign/build", R / "out/phase2-campaign/build"]
    for f in base.glob("stress__*/**/measurements/gsim*.json")
)
assert len(files) <= 64
report = {
    "schema": "merlin.phase2.retained_public_stress_audit.v3",
    "stage": "waiting_for_shared_readback_audit_slot",
    "records": [],
    "actual_record_count": len(files),
    "new_compiles": 0,
    "new_simulations": 0,
    "new_gsim_runs": 0,
    "new_fits": 0,
    "measured_now": False,
    "promotion_authority": "NONE",
    "retained_labels_not_rebound": True,
    "token_usage": {
        "uncached_input": None,
        "cached_input": None,
        "output": None,
        "reasoning_subset_of_output": None,
    },
}


def save():
    report["wall_seconds"] = time.monotonic() - start
    write_json(OUT / "report.json", report)


save()
references = {}
with (
    shared_compute_pool(timeout_s=180),
    RTL.observed_facts("gemmini", json.loads(facts.read_bytes()), facts),
):
    active = time.monotonic()
    active_cpu = time.process_time()
    report["queue_seconds"] = active - start
    for f in files:
        source = json.loads(f.read_bytes())
        work = f.parent.parent
        row = {
            "original_record": pin(f),
            "member": source["member"],
            "arm": source["arm"],
            "repeat": source["repeat"],
            "source_sha256": source["source_sha256"],
            "elf_sha256": source["build"]["elf_sha256"],
            "historical_cycles": source["cycles"],
            "original_exact_label": source.get("exact"),
            "original_problems": source.get("problems"),
            "status": "UNKNOWN",
            "native_prediction": None,
            "qualification_issued": False,
        }
        report["records"].append(row)
        report["stage"] = "original_output_reopen_" + source["member"]
        save()
        try:
            if time.process_time() - active_cpu > 50:
                raise ValueError(
                    "bounded audit 50CPU-second admission cutoff; row preserved NOT_RUN"
                )
            ctx = source["build"]["context"]
            assert ctx["facts_sha256"] == sha256_file(facts) and ctx[
                "contract_sha256"
            ] == sha256_file(contract)
            own_build = json.loads((work / "build-receipt.json").read_bytes())
            compiler_identity = check_arm_context(source, own_build)
            assert (
                sha256_file(work / "lowering_recipe.json")
                == own_build["lowering_recipe_sha256"]
            )
            assert sha256_file(work / "inputs.npz") == own_build["inputs_sha256"]
            elf = Path(source["build"]["elf"])
            assert sha256_file(elf) == source["build"]["elf_sha256"]
            memory = source["memory_evidence"]
            dump = Path(memory["output_path"])
            assert sha256_file(dump) == memory["artifact_sha256"]
            sub = dump.parent
            admission = READ.admit_coherent_output_layout(
                submission=sub,
                command_buffer_member="command_buffer.json",
                target="gemmini",
                facts_path=facts,
                elf_path=elf,
                expected_elf_sha256=source["build"]["elf_sha256"],
            )
            fixed = READ.admit_fixed_address_output_layout(
                admission=admission,
                submission=sub,
                command_buffer_member="command_buffer.json",
                target="gemmini",
                facts_path=facts,
                elf_path=elf,
            )
            decoded = READ.decode_coherent_output_dump(
                admission=admission,
                fixed_address_admission=fixed,
                submission=sub,
                command_buffer_member="command_buffer.json",
                target="gemmini",
                facts_path=facts,
                elf_path=elf,
                dump_path=dump,
            )
            cbpath = work / "commands.json"
            cb = json.loads(cbpath.read_bytes())
            inputpath = work / "inputs.npz"
            key = (sha256_file(cbpath), sha256_file(inputpath))
            if key not in references:
                with np.load(inputpath, allow_pickle=False) as inputs:
                    values = {
                        name: np.asarray(inputs[name]).reshape(-1).tolist()
                        for name in inputs.files
                    }
                references[key] = reference_outputs(cb, inputs=values)
            expected = references[key]
            actual = decoded["outputs"]
            assert set(expected) == set(actual)
            output_count = compare_complete_integer_outputs(expected, actual)
            row.update(
                status="saved_complete_outputs_recomputed_exact",
                outputs=output_count,
                original_arm_compiler_sha256=compiler_identity,
                compiler_context_authority="historical own-build receipt only; no current compiler qualification",
                output_names=list(expected),
                saved_dump=pin(dump),
                actual_layout_rederived=True,
                input=pin(inputpath),
                command_buffer=pin(cbpath),
                timing_intake="UNKNOWN: no original v3 archive record/count provenance for this independent development roster; no label transfer",
                physical_dram_traffic="UNKNOWN",
                new_execution=False,
            )
        except (
            ValueError,
            AssertionError,
            TypeError,
            KeyError,
            OSError,
            RuntimeError,
        ) as error:
            row.update(reason=str(error)[:500], refusal_type=type(error).__name__)
    report["active_audit_seconds"] = time.monotonic() - active
    report["active_audit_cpu_seconds"] = time.process_time() - active_cpu
planpath = (
    R / "out/artifacts/perf-studies/gemmini/el4-20261018/capacity-calibration-plan.json"
)
plan = json.loads(planpath.read_bytes())
report["capacity_validation"] = {
    "original_plan": pin(planpath),
    "points": len(plan["points"]),
    "observations": {},
    "decision": CV.decide(plan, {}),
    "reason": "Retained stress identities and schedule alternatives do not supply the frozen plan pairs. No row relabelled, omitted, or assigned copied Spike predictions.",
}
report["status_counts"] = {
    s: sum(r["status"] == s for r in report["records"])
    for s in {r["status"] for r in report["records"]}
}
report["actual_unique_ELFs"] = len({r["elf_sha256"] for r in report["records"]})
report["actual_source_groups"] = len({r["source_sha256"] for r in report["records"]})
report["source_fact_pins"] = {
    "facts": pin(facts),
    "observed_contract": pin(contract),
    "reference": pin(Path(reference_outputs.__code__.co_filename)),
    "readback": pin(Path(READ.__file__)),
}
report["stage"] = "retained_stress_complete_output_and_frozen_plan_audit_complete"
save()
print(
    json.dumps(
        {
            k: report[k]
            for k in [
                "stage",
                "actual_record_count",
                "actual_unique_ELFs",
                "actual_source_groups",
                "status_counts",
                "wall_seconds",
            ]
        },
        indent=2,
    )
)
