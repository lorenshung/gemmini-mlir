"""One original sealed-source proof within a bounded actual pair check.

No proof is retained across calls. Both actual ELFs, traces, target runtime,
arithmetic and complete outputs are checked independently. Different source
owners or capture/reference leaves use the original per-arm proof path.
"""

import copy
import json
from pathlib import Path

import yaml
from gemmini_source_execution import SourceExecutionOwner
from gemmini_staged_dense_capability import _digest, _leaf
from gemmini_timing_resources import _pin
from merlin.common import invocation_record
from merlin.llvmlower.source_contraction_bindings import bind_rank_two_contractions
from merlin.targetgen.native_model_execution import _pinned_native_file
from merlin_experiments.phase0.connected_integer_qualification import source_proof


def _same_source(before, after):
    rows = [
        json.loads(_leaf(arm, "native_result").read_bytes()) for arm in (before, after)
    ]
    if rows[0]["source"] != rows[1]["source"]:
        return None
    sources = copy.deepcopy(rows[0]["source"])
    for role in sources:
        _pinned_native_file(sources, role)
    return sources


def _arm(owner, arm, proof, cap):
    result = _leaf(arm, "native_result")
    cb_path, lowered, invocation = (
        _leaf(arm, key)
        for key in ("command_buffer_artifact", "lowered", "issue_invocation")
    )
    native = json.loads(result.read_bytes())
    cb = json.loads(cb_path.read_bytes())
    source = cap / "capsule.interface.mlir"
    contractions = bind_rank_two_contractions(source, cb, proof)
    if (
        len(contractions) != 1
        or contractions[0]["rhs"]
        not in cb["params"]["global_program_plan"]["entry_bindings"]
    ):
        raise ValueError(
            "whole-tile source requires one contraction and an immutable original external RHS"
        )
    checked = SourceExecutionOwner(
        arithmetic=owner.arithmetic,
        selected=owner.selected,
        result=result,
        command_buffer=cb_path,
        lowered=lowered,
        trace_invocation=invocation,
        nm=owner.nm,
    ).verify_source_execution(source, proof, cap)
    checked["source_geometry"] = {
        name: contractions[0][name] for name in ("m", "k", "n")
    }
    rhs = [
        "tensor_" + str(i)
        for i, arg in enumerate(cb["kernel_abi"]["args"])
        if arg["tensor"] == contractions[0]["rhs"]
    ]
    if len(rhs) != 1:
        raise ValueError(
            "whole-tile source has no unique original immutable RHS ABI owner"
        )
    import qualify

    observed = invocation_record.verify(invocation)
    trace, events = qualify._events(
        Path(observed["stderr"]["path"]).read_text(),
        owner.selected.operand_observer().isa_constants("gemmini"),
        qualify._symbols(_pinned_native_file(native, "elf"), owner.nm),
    )
    commands = iter(trace["instructions"])
    decoded = [
        (event, None if event[0] == "FENCE" else next(commands)) for event in events
    ]
    return decoded, rhs[0], checked, cb


def check_source_arms(owner, before, after):
    sources = _same_source(before, after)
    if sources is None:
        return (
            owner._source_arm(before),
            owner._source_arm(after),
            "distinct original source leaves: fresh proof per arm",
        )
    cap = _pinned_native_file(sources, "capsule_declaration").parent
    proof = source_proof(yaml.safe_load((cap / "capsule.yaml").read_bytes()), cap)
    original_proof = _digest(proof)
    checked = []
    for arm in (before, after):
        owner.revalidate()
        checked.append(_arm(owner, arm, proof, cap))
        if _digest(proof) != original_proof:
            raise ValueError(
                "actual pair checker changed its original sealed source proof"
            )
        if json.loads(_leaf(arm, "native_result").read_bytes())["source"] != sources:
            raise ValueError(
                "original source pair binding changed during actual checks"
            )
        for role, pin in sources.items():
            if _pin(_pinned_native_file(sources, role)) != {
                key: pin[key] for key in ("path", "sha256")
            }:
                raise ValueError(
                    "original pair source/reference/input/attestation custody changed"
                )
    owner.revalidate()
    return (
        *checked,
        "one fresh proof within this pair; exact same source/input/reference/attestation leaves",
    )
