"""Independently recheck retained actual outputs/effects before selecting an owner.

Operator only: input receipts and this checker must stay outside candidate
writes. This admits a conditional functional source mechanism, not a device
backend or any latency prediction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np


def prepare(*, selected, effect_contract, completion_contract, reports, nm, output):
    import gemmini_load_cleanup_capability as capability
    import gemmini_quiescent_load_config as cleanup
    import qualify as checker
    from gemmini_timing_resources import _pin, _revalidate_sources
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.runtime.out_b64 import OutB64Decoder
    from merlin.runtime.reference import reference_outputs

    selected.revalidate()
    effect = effect_contract.revalidate()
    completion = json.loads(Path(completion_contract).read_bytes())
    if (
        completion.get("schema") != "gemmini.staged_completion_contract.v1"
        or completion["selected_inputs"] != selected.report()["selected_inputs"]
    ):
        raise ValueError("completion sidecar differs from the original selected target")
    _revalidate_sources(completion["sources"])
    if effect["selected_inputs"] != completion["selected_inputs"]:
        raise ValueError(
            "load effects and completion contract have different selected inputs"
        )
    library = shared_library_sources()
    observer = selected.operand_observer()
    pins = {
        "effect_contract": _pin(effect_contract.path),
        "completion_contract": _pin(completion_contract),
        "checker": _pin(checker.__file__),
        "admission_checker": _pin(__file__),
        "symbol_inspector": _pin(nm),
        "adapter": _pin(cleanup.__file__),
        "completion_implementation": completion["implementation"],
    }
    _revalidate_sources({"completion_implementation": completion["implementation"]})
    records, runtime, original_identity = [], None, None
    for report_index, report_path in enumerate(reports):
        report = json.loads(Path(report_path).read_bytes())
        if (
            report.get("schema") != "gemmini.quiescent_load_configuration_diagnostic.v2"
            or report["selected_target"]["selected_inputs"]
            != completion["selected_inputs"]
            or report["effect_contract_sha256"] != effect_contract.identity
            or report["shared_library_sources"] != library
            or report["adapter_implementation"] != pins["adapter"]
        ):
            raise ValueError(
                "paired execution does not qualify this exact adapter/library/input selection"
            )
        if runtime is not None and (
            runtime != report["runtime_inputs"]
            or original_identity != report["original_compiler_sha256"]
        ):
            raise ValueError(
                "paired evidence mixes runtime or original compiler identities"
            )
        runtime, original_identity = (
            report["runtime_inputs"],
            report["original_compiler_sha256"],
        )
        pins["report_" + str(report_index)] = _pin(report_path)
        for case_index, case in enumerate(report["cases"]):
            before_after, expected_logical = {}, None
            complete_outputs = 0
            for arm in ("baseline", "candidate"):
                record = case["arms"][arm]
                build = record["build"]
                elf = Path(build["elf"])
                work = elf.parent
                closure = {
                    "elf": elf,
                    "inputs": work / "inputs.npz",
                    "commands": work / "commands.json",
                    "stdout": work / "functional.stdout",
                    "trace": work / "functional.commit.log",
                    "counter_audit": work / "counter-protocol-audit.json",
                }
                for name, path in closure.items():
                    pins[f"pair{report_index}_{case_index}_{arm}_{name}"] = _pin(path)
                if (
                    _pin(elf)["sha256"] != build["elf_sha256"]
                    or _pin(closure["inputs"])["sha256"] != build["inputs_sha256"]
                    or _pin(closure["trace"])["sha256"] != record["trace_sha256"]
                    or _pin(closure["counter_audit"])["sha256"]
                    != build["counter_audit_sha256"]
                ):
                    raise ValueError("actual paired execution artifact changed")
                data = dict(np.load(closure["inputs"], allow_pickle=False))
                cb = json.loads(closure["commands"].read_bytes())
                expected = reference_outputs(
                    cb, {name: values.tolist() for name, values in data.items()}
                )
                actual, decoder = {}, OutB64Decoder()
                for line in closure["stdout"].read_text().splitlines():
                    decoder.consume(line.split(), actual)
                decoder.require_closed()
                if set(actual) != set(expected) or any(
                    not np.array_equal(
                        np.asarray(actual[name]).reshape(-1),
                        np.asarray(value).reshape(-1),
                    )
                    for name, value in expected.items()
                ):
                    raise ValueError(
                        "complete independent output comparison failed at capability admission"
                    )
                complete_outputs += sum(
                    np.asarray(value).size for value in expected.values()
                )
                symbols = checker._symbols(elf, nm)
                trace, events = checker._events(
                    closure["trace"].read_text(),
                    observer.isa_constants("gemmini"),
                    symbols,
                )
                checker._check_transfer_windows(trace, symbols, cb)
                before_after[arm] = events
                logical = build["logical_command_buffer_sha256"]
                if expected_logical is not None and expected_logical != logical:
                    raise ValueError("paired compiler changes required logical work")
                expected_logical = logical
            removed = checker._check_omissions(
                before_after["baseline"],
                before_after["candidate"],
                effect["encoding"]["config_rs1"],
            )
            records.append(
                {
                    "source": case["member"]["source_sha256"],
                    "complete_outputs_rechecked": complete_outputs,
                    "actual_setters_removed": removed,
                    "other_commands_and_fences_unchanged": True,
                }
            )
    if (
        len({row["source"] for row in records}) < 3
        or sum(row["actual_setters_removed"] > 0 for row in records) < 2
    ):
        raise ValueError(
            "capability needs independent public shapes and multiple actual applications"
        )
    pins.update({"runtime_" + name: pin for name, pin in runtime.items()})
    _revalidate_sources(pins)
    selected.revalidate()
    if shared_library_sources() != library:
        raise ValueError("shared release changed during independent admission")
    payload = {
        "schema": capability.SCHEMA,
        "scope": capability.SCOPE,
        "admission": "conditional_final_pair_required",
        "hardware_execution_qualification": "not_established",
        "timing_authority": "NONE",
        "dense_v1_enabled": False,
        "owner_implementation": _pin(capability.__file__),
        "selected_inputs": completion["selected_inputs"],
        "effect_contract": _pin(effect_contract.path),
        "completion_contract": _pin(completion_contract),
        "checker": pins["checker"],
        "runtime_inputs": runtime,
        "original_compiler_sha256": original_identity,
        "shared_library_sources": library,
        "pinned_inputs": pins,
        "qualified_pairs": records,
        "checks": {
            "complete_independent_outputs": True,
            "actual_linked_issue_operands": True,
            "unchanged_required_work": True,
            "only_redundant_fenced_state_zero_setters_removed": True,
            "logical_dma_windows": True,
            "exact_original_runtime_and_library": True,
        },
        "conditions": {
            **cleanup.SEMANTICS,
            "scope": "bounded_source_equivalence_in_selected_functional_spike",
            "new_realization": "independent_full_outputs_and_final_issue_effect_check_required",
            "input_guards_physical_beats_hardware_completion": "not_qualified",
            "fence": "load_side_drain_only; solitary_preload_may_survive",
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return capability.LoadCleanupOwner.read(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "facts",
        "observed-contract",
        "effect-contract",
        "completion-contract",
        "nm",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    args = parser.parse_args()
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "merlin-support/backend")
    )
    scheduling = importlib.import_module("gemmini_selected_scheduling")
    cleanup = importlib.import_module("gemmini_quiescent_load_config")
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool():
        owner = prepare(
            selected=scheduling.SelectedSchedulingTarget.read(
                args.facts, args.observed_contract
            ),
            effect_contract=cleanup.LoadConfigurationContract.read(
                args.effect_contract
            ),
            completion_contract=args.completion_contract,
            reports=args.report,
            nm=args.nm,
            output=args.output,
        )
    print(json.dumps(owner.record(), sort_keys=True))


if __name__ == "__main__":
    main()
