"""Explicit target-bound publication through the unchanged ordinary writer.

V1 source-only publication remains unchanged. This selection derives additional
target requirements; the original live realization, exclusion and staging owners
still provide all execution/publication authority.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import yaml
from source_admission_inputs import TARGET_SCHEMA


def selected(declaration):
    return declaration.get("schema") == TARGET_SCHEMA


@contextmanager
def target_binding(declaration, selected_inputs, source_binding, diagnostic, intake):
    from gemmini_selected_scheduling import SelectedSchedulingTarget
    from merlin.targetgen import capsule_source as CS
    from merlin.targetgen.corpus_spec import derive_binding
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.target_registry import observed_contract

    if not selected(declaration):
        yield source_binding, None
        return
    if declaration.get("publication") != {"scope": "selected_target_v1"}:
        raise ValueError("target publication needs its explicit closed selection")
    contract_pin = declaration["execution_contract"]
    if os.environ.get("MERLIN_TARGET_CONTRACT") != contract_pin["path"]:
        raise ValueError(
            "target publication differs from the original selected execution contract"
        )
    descriptor = load_target_experiment(Path(os.environ["MERLIN_TARGET_EXPERIMENT"]))
    if descriptor.target != "gemmini":
        raise ValueError("target publication selects another OOT target")
    if str(descriptor.declared_contract) != contract_pin["path"]:
        raise ValueError(
            "target publication differs from its original descriptor contract"
        )
    original = selected_inputs["factory_inputs"]
    if original.get("execution_contract") != contract_pin["path"]:
        raise ValueError(
            "target publication differs from the original source-factory contract"
        )
    target = SelectedSchedulingTarget.read(
        Path(original["facts"]), Path(original["observed_contract"])
    )
    target.revalidate()
    facts, _, _ = target._documents()
    raw = intake.read(contract_pin)
    contract = yaml.safe_load(raw)
    capsule = yaml.safe_load((Path(diagnostic) / "capsule.yaml").read_bytes())
    with (
        observed_contract("gemmini", contract),
        observed_facts("gemmini", facts, Path(original["facts"])),
    ):
        binding = derive_binding(
            descriptor,
            {
                "operand_dtype": source_binding.operand_dtype,
                "compare": source_binding.compare,
                "atol": source_binding.atol,
                "rtol": source_binding.rtol,
                "required_oracle_tiers": capsule["required_oracle_tiers"],
            },
            contract=contract,
            facts=facts,
        )
        if not binding.integer or binding.cap_dtype(
            binding.accum_dtype
        ) != source_binding.cap_dtype(source_binding.accum_dtype):
            raise ValueError(
                "selected target publication changes the original accumulator contract"
            )
        program = (Path(diagnostic) / "capsule.interface.mlir").read_text()
        family, classes = CS.model_accelerator_demand(program, binding)
        if not family or not classes:
            raise ValueError(
                "selected target publication has no derived accelerator instruction demand"
            )
        evidence = {
            "scope": "selected_target_v1",
            "target": binding.target,
            "tile_dim": binding.tile_dim,
            "expected_instruction_classes": classes,
            "semantic_family": family,
            "required_oracle_tiers": list(binding.tiers),
            "contract_sha256": contract_pin["sha256"],
            "qualification_issued": False,
        }

        def finish():
            target.revalidate()
            if intake.read(contract_pin) != raw:
                raise ValueError(
                    "selected target publication contract changed during writing"
                )

        try:
            yield binding, evidence
        except BaseException as error:
            try:
                finish()
            except BaseException as final_error:  # noqa: BLE001 — preserve the primary failure
                error.add_note(
                    "target publication final revalidation also failed: "
                    f"{type(final_error).__name__}: {final_error}"
                )
            raise
        else:
            finish()


def write(
    entry, source_binding, root, *, declaration, selected_inputs, diagnostic, intake
):
    """Close input/policy checks before original staging can publish its member."""
    from merlin.targetgen.native_model_policy import _frozen_model_policy
    from merlin_experiments.phase0 import writer

    with target_binding(
        declaration, selected_inputs, source_binding, diagnostic, intake
    ) as (binding, evidence):
        directory = Path(writer._write_capsule(entry, binding, root))
        cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        original = yaml.safe_load((Path(diagnostic) / "capsule.yaml").read_bytes())
        if (
            (directory / "capsule.interface.mlir").read_bytes()
            != (Path(diagnostic) / "capsule.interface.mlir").read_bytes()
            or cap["inputs"] != original["inputs"]
            or cap["numeric_policy"] != original["numeric_policy"]
            or cap["operation"] != original["operation"]
            or any(
                cap.get(key) != original.get(key)
                for key in ("reference_closure", "public_source_role", "performance")
            )
        ):
            raise ValueError(
                "target publication changed complete source, ABI or numerical policy"
            )
        policy = _frozen_model_policy(directory, target=binding.target)
        if (
            (cap.get("semantic") or {}).get("must_accelerate") is not True
            or (cap.get("semantic") or {}).get("semantic_family")
            != evidence["semantic_family"]
            or policy["expected_instruction_classes"]
            != evidence["expected_instruction_classes"]
            or policy["required_tiers"] != evidence["required_oracle_tiers"]
        ):
            raise ValueError(
                "target publication lost original derived instruction or oracle obligations"
            )
        return directory
