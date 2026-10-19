"""Original completed artifact integration, opt-in without simulator execution.

The fixture selects original operator-owned evidence. This test runs the live
independent checks; no saved verdict is accepted as a grant. Mutants are local
copies and never change the retained source, reference, compiler or runtime.
"""

import copy
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def original():
    declaration = os.environ.get("MERLIN_SOURCE_EXECUTION_TEST_FIXTURE")
    if not declaration:
        pytest.skip("explicit original completed source/runtime evidence not selected")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    import yaml
    from gemmini_arithmetic_scope import IntegerArithmeticOwner
    from gemmini_selected_scheduling import SelectedSchedulingTarget
    from gemmini_source_execution import SourceExecutionOwner
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin_experiments.phase0.connected_integer_qualification import source_proof

    row = json.loads(Path(declaration).read_bytes())
    with shared_compute_pool(timeout_s=120):
        selected = SelectedSchedulingTarget.read(row["facts"], row["observed_contract"])
        arithmetic = IntegerArithmeticOwner(
            selected,
            **{
                key: Path(row[key])
                for key in (
                    "software_spec",
                    "source_root",
                    "receiver_source",
                    "runtime",
                )
            },
        )
        cap = Path(row["capsule"])
        proof = source_proof(yaml.safe_load((cap / "capsule.yaml").read_bytes()), cap)
        args = {
            key: Path(row[key])
            for key in ("result", "command_buffer", "lowered", "trace_invocation", "nm")
        }
        yield SourceExecutionOwner, arithmetic, selected, cap, proof, args


def test_original_complete_packet_and_output_correspondence(original):
    cls, arithmetic, selected, cap, proof, args = original
    owner = cls(arithmetic=arithmetic, selected=selected, **args)
    result = owner.verify_source_execution(cap / "capsule.interface.mlir", proof, cap)
    assert result["status"] == "verified_source_execution_correspondence"
    assert result["actual_packet_effects"]["products"] > 0
    assert result["complete_output_values"] > 0
    assert result["general_dense_adapter_enabled"] is False
    assert result["hardware_execution_qualified"] is False
    assert result["timing_authority"] == "NONE"


def test_stale_native_data_cannot_keep_live_owner(original, tmp_path):
    cls, arithmetic, selected, _cap, _proof, args = original
    args = dict(args)
    copied = tmp_path / "result.json"
    copied.write_bytes(args["result"].read_bytes())
    args["result"] = copied
    owner = cls(arithmetic=arithmetic, selected=selected, **args)
    copied.write_text("{}")
    with pytest.raises(ValueError):
        owner.revalidate()


def test_artifact_path_replacement_cannot_redirect_live_owner(original, tmp_path):
    cls, arithmetic, selected, _cap, _proof, args = original
    owner = cls(arithmetic=arithmetic, selected=selected, **args)
    replacement = tmp_path / "other.json"
    replacement.write_bytes(args["result"].read_bytes())
    owner._paths["result"] = replacement
    with pytest.raises(ValueError, match="artifact bindings changed"):
        owner.revalidate()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_outputs",
        "wrong_elf",
        "stale_source",
        "retained_without_preload",
        "runtime_flags",
    ],
)
def test_saved_success_cannot_substitute_actual_complete_effects(
    original, tmp_path, mutation
):
    from merlin.common import invocation_record
    from merlin.targetgen.native_model_execution import _digest

    cls, arithmetic, selected, cap, proof, args = original
    args = dict(args)
    proof = copy.deepcopy(proof)
    native = json.loads(args["result"].read_bytes())
    source = cap / "capsule.interface.mlir"
    if mutation == "missing_outputs":
        console = tmp_path / "console.txt"
        console.write_text(
            "OUT_B64_BEGIN v1 Y0 1 1 4 s\nOUT_B64_CHUNK 00000000 0004 AAAAAA==\nOUT_B64_END\n"
        )
        native["tiers"]["L2"]["console"] = _digest(console)
    elif mutation == "wrong_elf":
        native["elf"]["sha256"] = "0" * 64
    elif mutation == "stale_source":
        source = tmp_path / "different.mlir"
        source.write_bytes((cap / "capsule.interface.mlir").read_bytes() + b"\n")
    elif mutation == "runtime_flags":
        native["tiers"]["L2"]["engine_citation"]["flags"].append("--extension=foreign")
        receipt = json.loads(args["trace_invocation"].read_bytes())
        receipt["argv"].insert(-3, "--extension=foreign")
        invocation = tmp_path / "invocation.json"
        invocation.write_text(json.dumps(receipt))
        args["trace_invocation"] = invocation
    else:
        receipt = json.loads(args["trace_invocation"].read_bytes())
        lines = Path(receipt["stderr"]["path"]).read_text().splitlines(keepends=True)
        isa = selected.operand_observer().isa_constants("gemmini")
        code = next(
            code
            for code, name in isa["FUNCT_CLASS"].items()
            if name == "COMPUTE_PRELOADED"
        )
        retained = next(
            code
            for code, name in isa["FUNCT_CLASS"].items()
            if name == "COMPUTE_ACCUMULATE"
        )
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
                continue
            word = int(parts[4].strip("()"), 16)
            if word & 127 == isa["CUSTOM_OPCODE"] and word >> 25 == code:
                changed = word & ((1 << 25) - 1) | retained << 25
                lines[i] = line.replace(parts[4], f"(0x{changed:08x})", 1)
                break
        else:
            raise AssertionError(
                "original fixture contains no fresh weight preload consumption"
            )
        trace = tmp_path / "mutated.trace"
        trace.write_text("".join(lines))
        receipt["stderr"] = invocation_record._pin(trace)
        invocation = tmp_path / "invocation.json"
        invocation.write_text(json.dumps(receipt))
        args["trace_invocation"] = invocation
    record = tmp_path / "native.json"
    record.write_text(json.dumps(native))
    args["result"] = record
    owner = cls(arithmetic=arithmetic, selected=selected, **args)
    with pytest.raises(ValueError):
        owner.verify_source_execution(source, proof, cap)
