"""Diagnostic copied-compiler integration using existing public build/runtime tools.

This runs no author campaign and records no qualified latency. The original
compiler remains read-only. Both copies receive the same shared algorithms.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _symbols(elf, nm):
    result = subprocess.run(
        [str(nm), "-S", "--defined-only", str(elf)],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[2].lower() in {"b", "d", "r"}:
            address, size = int(fields[0], 16), int(fields[1], 16)
            if size:
                rows.append((address, size, fields[3]))
    return rows


def _memory_name(address, symbols):
    found = [
        (name, address - start, size)
        for start, size, name in symbols
        if start <= address < start + size
    ]
    if len(found) != 1:
        raise ValueError("observed DRAM pointer lacks a unique linked data symbol")
    return found[0]


def _events(log, isa, symbols):
    from merlin.perf.executed_trace import executed_trace

    trace = executed_trace(log.splitlines(), isa=isa, max_retired=2_000_000)
    if trace.get("status") != "measured" or not trace["instructions"]:
        raise ValueError(
            "actual linked execution lacks a complete bounded commit trace"
        )
    commands = iter(trace["instructions"])
    events = []
    for line in log.splitlines():
        tokens = line.split()
        if len(tokens) < 5 or tokens[0] != "core" or not tokens[2].isdigit():
            continue
        word = int(tokens[4].strip("()"), 16)
        if word & 127 == isa["CUSTOM_OPCODE"]:
            row = next(commands)
            first = row["rs1"]["raw"]
            if row["class"] in {"MVIN", "MVOUT"}:
                first = _memory_name(first, symbols)[:2]
            events.append((row["class"], first, row["rs2"]["raw"]))
        elif word & 127 == 15 and (word >> 12) & 7 == 0:
            events.append(("FENCE", 0, 0))
    if next(commands, None) is not None:
        raise ValueError("actual command trace and instruction framing differ")
    return trace, events


def _check_omissions(before, after, raw_config):
    """Independent subsequence/state induction over actual issued operands."""
    cursor, current, quiescent, removed = 0, None, False, 0
    for event in before:
        if cursor < len(after) and event == after[cursor]:
            cursor += 1
        elif (
            event[0] == "CONFIG_LD"
            and event[1] == raw_config
            and quiescent
            and current == event[1:]
        ):
            removed += 1
        else:
            raise ValueError(
                "actual target command changed beyond a redundant fenced load setter"
            )
        if event[0] == "CONFIG_LD" and event[1] == raw_config and quiescent:
            current = event[1:]
            quiescent = False
        elif event[0] == "FENCE":
            quiescent = True
        elif event[0] == "MVIN":
            quiescent = False
        else:
            current, quiescent = None, False
    if cursor != len(after):
        raise ValueError("actual target inserted command or changed fence sequence")
    return removed


def _check_transfer_windows(trace, symbols, command_buffer):
    """Check logical DMA spans against actual linked objects, independently."""
    outputs = {
        "tensor_" + str(index)
        for index, tensor in enumerate(command_buffer["tensors"].values())
        if tensor.get("role") == "output"
    }
    load_stride, store_stride, reads, writes = None, None, 0, 0
    allowed = {
        "CONFIG_EX",
        "CONFIG_LD",
        "CONFIG_ST",
        "MVIN",
        "MVOUT",
        "PRELOAD",
        "COMPUTE_PRELOADED",
        "COMPUTE_ACCUMULATE",
        "FLUSH",
    }
    for instruction in trace["instructions"]:
        kind, fields = instruction["class"], instruction["decoded"]
        if kind not in allowed:
            raise ValueError("actual issued command has unsupported memory effects")
        if kind == "FLUSH":
            if reads or writes:
                raise ValueError(
                    "logical transfer check refuses a flush after DMA begins"
                )
            load_stride = store_stride = None
        if kind == "CONFIG_LD":
            if fields.get("state_id") != 0:
                raise ValueError(
                    "actual transfer does not use selected load state zero"
                )
            load_stride = fields.get("stride")
        elif kind == "CONFIG_ST":
            store_stride = fields.get("out_stride_bytes")
        elif kind in {"MVIN", "MVOUT"}:
            stride = load_stride if kind == "MVIN" else store_stride
            rows, cols = fields.get("rows"), fields.get("cols")
            if (
                any(
                    type(value) is not int or value < 0
                    for value in (stride, rows, cols)
                )
                or not rows
                or not cols
            ):
                raise ValueError("actual transfer dimensions/stride are unavailable")
            if (
                (kind == "MVIN" and type(fields.get("is_acc_addr")) is not bool)
                or kind == "MVOUT"
                and fields.get("readout") not in {"i8", "i32"}
            ):
                raise ValueError("actual transfer element width remains unavailable")
            width = (
                (4 if fields.get("is_acc_addr") else 1)
                if kind == "MVIN"
                else (4 if fields.get("readout") == "i32" else 1)
            )
            symbol, offset, capacity = _memory_name(instruction["rs1"]["raw"], symbols)
            span = (rows - 1) * stride + cols * width
            if offset + span > capacity:
                raise ValueError(
                    "actual logical DMA span exceeds its linked source/destination object"
                )
            if kind == "MVOUT":
                if symbol not in outputs:
                    raise ValueError(
                        "actual accelerator store writes an input or other nonoutput object"
                    )
                writes += 1
            else:
                reads += 1
    if not reads or not writes:
        raise ValueError(
            "diagnostic did not execute accelerator input/output transfers"
        )
    return {
        "logical_reads": reads,
        "logical_writes": writes,
        "physical_beat_overread_and_guards": "not_qualified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--observed-contract", required=True, type=Path)
    parser.add_argument("--baseline-package", required=True, type=Path)
    parser.add_argument("--rocket-core", required=True, type=Path)
    parser.add_argument("--capsule", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    support = Path(__file__).resolve().parents[2] / "merlin-support/backend"
    sys.path.insert(0, str(support))
    sys.path.insert(0, str(args.merlin / "examples/gemmini/phase2/el4_20261018"))
    scheduling = importlib.import_module("gemmini_selected_scheduling")
    cleanup = importlib.import_module("gemmini_quiescent_load_config")
    evaluator = importlib.import_module("evaluate")
    common = evaluator.C
    common.OUT = args.output / "build"
    if (
        Path(os.environ["MERLIN_RTL_FACTS"]).resolve() != args.facts.resolve()
        or Path(os.environ["MERLIN_TARGET_CONTRACT"]).resolve()
        != args.observed_contract.resolve()
    ):
        raise ValueError(
            "existing builder environment differs from explicit selected inputs"
        )
    import numpy as np
    from merlin.common import invocation_record
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.perf.executed_trace import trace_invocation
    from merlin.runtime.backends.base import get_backend

    args.output.mkdir(parents=True, exist_ok=False)
    library_identity = shared_library_sources()
    driver_identity = common.file_hash(__file__)
    selected = scheduling.SelectedSchedulingTarget.read(
        args.facts, args.observed_contract
    )
    original_identity, original_files = common.tree_hash(args.baseline_package)
    manifest = json.loads(
        (args.baseline_package.parent / "package-files.json").read_text()
    )
    if (
        original_identity != manifest["package_sha256"]
        or len(original_files) != manifest["n_files"]
    ):
        raise ValueError("preserved compiler identity differs")
    with shared_compute_pool():
        contract = cleanup.prepare_contract(
            selected,
            runtime=args.baseline_package / "device_runtime.py",
            rocket_core=args.rocket_core,
            output=args.output / "load-config-contract.json",
        )
        observer = selected.operand_observer()
    packages = {}
    for arm, enabled in (("baseline", False), ("candidate", True)):
        package = args.output / "compilers" / arm
        shutil.copytree(args.baseline_package, package)
        shutil.copyfile(
            support / "gemmini_quiescent_load_config.py",
            package / "gemmini_quiescent_load_config.py",
        )
        shutil.copyfile(contract.path, package / "load-config-contract.json")
        common.save(package / "shared-library-sources.json", library_identity)
        direct = package / "direct_codegen.py"
        source = direct.read_text()
        marker = "return declarations, statements"
        if source.count(marker) != 1:
            raise ValueError("copied compiler lacks a unique direct schedule boundary")
        source += "\nfrom gemmini_quiescent_load_config import rewrite_copied_compiler_statements\n"
        source = source.replace(
            marker,
            f"return declarations, rewrite_copied_compiler_statements(statements, enabled={enabled})",
        )
        direct.write_text(source)
        packages[arm] = package
    records = []
    for capsule in args.capsule:
        source = capsule / "capsule.interface.mlir"
        member = {
            "identity": "adapter_check/" + capsule.name,
            "source": str(source),
            "source_sha256": common.file_hash(source),
        }
        pair = {}
        for arm, package in packages.items():
            elf, cb, inputs, build = evaluator.build(
                member, package, arm, seed=1, engine="spike"
            )
            expected = evaluator.reference(cb, inputs)
            invocation = trace_invocation("gemmini", elf)
            if invocation is None:
                raise ValueError("selected functional runtime provides no issue trace")
            argv, env = invocation
            trace_started = time.monotonic()
            with shared_compute_pool():
                trace_admitted = time.monotonic()
                result = invocation_record.run(
                    argv,
                    directory=elf.parent,
                    stage="adapter_functional_trace",
                    inputs=(elf,),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
            trace_finished = time.monotonic()
            (elf.parent / "functional.stdout").write_text(result.stdout)
            (elf.parent / "functional.commit.log").write_text(result.stderr)
            if result.returncode:
                raise ValueError(
                    "selected functional trace failed: " + result.stderr[-1000:]
                )
            actual, _metrics = get_backend("gemmini").parse_output(result.stdout)
            if set(actual) != set(expected) or any(
                not np.array_equal(
                    np.asarray(actual[name]).reshape(-1), np.asarray(value).reshape(-1)
                )
                for name, value in expected.items()
            ):
                raise ValueError("complete independent public output comparison failed")
            nm = evaluator.OBJDUMP.with_name("riscv64-unknown-elf-nm")
            symbols = _symbols(elf, nm)
            trace, events = _events(
                result.stderr, observer.isa_constants("gemmini"), symbols
            )
            windows = _check_transfer_windows(trace, symbols, cb)
            pair[arm] = {
                "build": build,
                "exact_outputs": sum(np.asarray(x).size for x in expected.values()),
                "events": events,
                "accelerator_commands": len(trace["instructions"]),
                "commands_by_class": dict(
                    collections.Counter(row["class"] for row in trace["instructions"])
                ),
                "trace_sha256": common.file_hash(elf.parent / "functional.commit.log"),
                "logical_transfer_windows": windows,
                "cost": {
                    "compile_link_and_admission_s": build["compile_link_s"],
                    "trace_cpu_wait_s": trace_admitted - trace_started,
                    "functional_trace_s": trace_finished - trace_admitted,
                },
            }
        if (
            pair["baseline"]["build"]["logical_command_buffer_sha256"]
            != pair["candidate"]["build"]["logical_command_buffer_sha256"]
        ):
            raise ValueError(
                "copied compiler changed logical work or numerical contracts"
            )
        removed = _check_omissions(
            pair["baseline"]["events"],
            pair["candidate"]["events"],
            contract.revalidate()["encoding"]["config_rs1"],
        )
        records.append(
            {
                "member": member,
                "arms": pair,
                "actual_redundant_load_setters_removed": removed,
            }
        )
        print(
            json.dumps({"member": capsule.name, "exact": True, "removed": removed}),
            flush=True,
        )
    if common.tree_hash(args.baseline_package) != (original_identity, original_files):
        raise ValueError("preserved compiler changed")
    selected.revalidate()
    if (
        shared_library_sources() != library_identity
        or common.file_hash(__file__) != driver_identity
    ):
        raise ValueError("shared library or diagnostic changed during paired execution")
    report = {
        "schema": "gemmini.quiescent_load_configuration_diagnostic.v2",
        "original_compiler_sha256": original_identity,
        "selected_target": selected.report(),
        "effect_contract_sha256": contract.identity,
        "cases": records,
        "scope": "copied compiler, actual linked issue operands and complete public signed outputs",
        "guards_and_input_memory_readback": "not_performed",
        "hardware_execution_qualification": "not_established",
        "timing_qualification": "not_established",
        "formal_phase2_admission": False,
        "shared_library_sources": library_identity,
        "runtime_inputs": {
            "spike": cleanup._pin(
                Path(common.environment()["MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME"])
                / "bin/spike"
            ),
            "functional_extension": cleanup._pin(
                Path(common.environment()["MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME"])
                / "lib/libgemmini.so"
            ),
        },
        "adapter_implementation": cleanup._pin(
            support / "gemmini_quiescent_load_config.py"
        ),
        "driver_sha256": driver_identity,
        "wall_s": time.monotonic() - started,
    }
    common.save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
