"""Consume original completed ordinary-queue products; never launch a target."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from gemmini_direct_source_execution import DirectSourceExecutionOwner
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common import invocation_record
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import native_model_execution as native
from merlin.targetgen.native_pointer_source_audit import audit_pointer_source_tiers
from merlin_experiments.phase0.connected_integer_qualification import source_proof


def _read(path, pins):
    path = Path(path)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > 64 << 20
    ):
        raise ValueError("finite pair requires exact bounded ordinary product leaves")
    pin = _pin(path)
    record = json.loads(path.read_bytes())
    if _pin(path) != pin:
        raise ValueError("finite ordinary product changed during read")
    pins[str(path)] = pin
    return record


def _one(parent, pattern):
    # Exact original producer subdirectory, never a first-match recursive scan.
    matches = tuple(Path(parent).glob(pattern))
    if len(matches) != 1:
        raise ValueError("finite pair has no unique complete original producer")
    return matches[0]


def _source_invocation(chain, pins):
    declared = chain.get("compiler_source_invocation")
    if (
        type(declared) is not dict
        or set(declared) != {"path", "sha256"}
        or type(declared["path"]) is not str
    ):
        raise ValueError("finite pair requires the original source invocation pin")
    path = Path(declared["path"])
    before = _read(path, pins)
    if pins[str(path)] != declared:
        raise ValueError("finite source invocation bytes differ from their pin")
    verified = invocation_record.verify(path)
    if verified != before or _pin(path) != declared:
        raise ValueError("finite source invocation changed during verification")
    return path, verified


def _arm(owner, state, workspace, pins, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("finite actual-pair deadline expired before product intake")
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace.resolve() != workspace:
        raise ValueError("finite pair needs its exact original ordinary workspace")
    observation = _read(workspace / "ordinary_observation.json", pins)
    root = workspace / "capsule_runs/runs/gemmini-capsule-bench/functional"
    grade = _read(root / "capsule_result.json", pins)
    if (
        observation.get("functional_status") != "PASS"
        or observation.get("original_grade") != grade
        or grade.get("checked_source_status")
        != "complete_source_L0_L1_L2_diagnostic_pass"
        or grade.get("checked_source_diagnostic", {}).get("binding")
        != asdict(owner._binding)
        or grade["checked_source_diagnostic"].get("effect_capability")
        != owner._source.identity
        or any(
            grade.get("tiers", {}).get(tier, {}).get("status") != "pass"
            for tier in ("L0", "L1", "L2")
        )
    ):
        raise ValueError(
            "finite pair lacks this original complete diagnostic grade/selection"
        )
    result_path = root / "candidate_native/result.json"
    result = _read(result_path, pins)
    if result != grade.get("candidate_native_execution"):
        raise ValueError("finite pair native product differs from original grade")
    emission = grade["candidate_emission"]
    command_buffer = native._pinned_native_file(emission, "command_buffer")
    lowered = native._pinned_native_file(emission, "lowered_mlir")
    source = native._pinned_native_file(emission, "source_interface")
    for path in (command_buffer, lowered, source):
        pins[str(path)] = _pin(path)
    chain_path = _one(
        root / "generated", "emission-custody-*/emission_proof_chain.json"
    )
    chain = _read(chain_path, pins)
    _, producer = _source_invocation(chain, pins)
    expected = (_pin(command_buffer), _pin(lowered))
    if (
        producer.get("stage") != "ordinary_source_emission"
        or producer.get("callable") != "merlin.targetgen.capsule_common.lower_interface"
        or producer.get("inputs") != [_pin(source)]
        or any(pin not in producer.get("outputs", ()) for pin in expected)
    ):
        raise ValueError("finite pair lacks original ordinary source/emission custody")
    compilers = [
        Path(pin["path"]).parent
        for pin in producer.get("dependencies", ())
        if Path(pin["path"]).name == "compiler.py"
    ]
    if len(compilers) != 1:
        raise ValueError("finite pair lost its unique actual compiler dependency")
    package = compilers[0]
    binding = owner.check_compiler_binding(package)
    if hash_tree(package)["sha256"] != grade.get("original_compiler_child", {}).get(
        "candidate_sha256"
    ):
        raise ValueError("finite pair differs from the actual retained compiler source")
    trace_path = _one(
        root / "checked_source_realization/invocations", "*/invocation.json"
    )
    pins[str(trace_path)] = _pin(trace_path)
    invocation_record.verify(trace_path)
    source_owner = DirectSourceExecutionOwner(
        arithmetic=state["arithmetic"],
        selected=state["selected"],
        result=result_path,
        command_buffer=command_buffer,
        lowered=lowered,
        trace_invocation=trace_path,
        nm=Path(state["nm"]["path"]),
        observation_protocol=state["protocol"],
        execution_contract=state["execution_contract"],
    )
    tiers = audit_pointer_source_tiers(
        emission,
        result,
        target="gemmini",
        entry_symbol=result["pointer_execution"]["entry_symbol"],
    )
    if any(
        tiers["tiers"].get(tier, {}).get("status") != "pass"
        for tier in ("L0", "L1", "L2")
    ):
        raise ValueError("finite pair original current L0/L1/L2 audit refused")
    if time.monotonic() >= deadline:
        raise TimeoutError("finite actual-pair deadline expired after current audits")
    return source_owner, binding, source, result_path


def _pair_choices(arms):
    baseline, candidate = (arm[1]["policy"] for arm in arms)
    if baseline != {
        "enabled": True,
        "rewrite_policy": "verify_only",
    } or candidate not in (
        {"enabled": True, "rewrite_policy": "verify_only"},
        {"enabled": True, "rewrite_policy": "n_before_m_rhs"},
    ):
        raise ValueError(
            "finite pair needs original finite baseline and explicit bounded candidate"
        )
    return copy.deepcopy((baseline, candidate))


def _check_source_pair(
    owner, state, *, baseline_workspace, candidate_workspace, capsule, remaining_s
):
    if (
        type(remaining_s) not in (int, float)
        or not math.isfinite(remaining_s)
        or remaining_s <= 0
    ):
        raise ValueError(
            "finite actual pair requires a positive finite remaining budget"
        )
    if type(capsule) is not dict or "__dir__" not in capsule:
        raise ValueError(
            "finite actual pair needs its current original private capsule"
        )
    deadline = time.monotonic() + remaining_s
    before_capsule, pins = copy.deepcopy(capsule), {}
    arms = [
        _arm(owner, state, workspace, pins, deadline)
        for workspace in (baseline_workspace, candidate_workspace)
    ]
    baseline_policy, candidate_policy = _pair_choices(arms)
    capdir = Path(capsule["__dir__"])
    source = capdir / "capsule.interface.mlir"
    pins[str(source)] = _pin(source)
    if any(arm[2].read_bytes() != source.read_bytes() for arm in arms):
        raise ValueError("finite actual pair changed its selected complete source")
    proof = source_proof(capsule, capdir)
    relation = owner.check_actual_pair(
        baseline_owner=arms[0][0],
        candidate_owner=arms[1][0],
        source=source,
        proof=proof,
        capsule_dir=capdir,
    )
    _revalidate_sources(pins)
    if capsule != before_capsule:
        raise ValueError("finite actual pair changed the original capsule selection")
    if time.monotonic() >= deadline:
        raise TimeoutError("finite actual-pair deadline expired before publication")
    return {
        "schema": "gemmini.v26_finite_original_queue_pair.v1",
        "status": "PASS",
        "actual_pair": relation,
        "compiler_bindings": [arm[1] for arm in arms],
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "ordinary_products": pins,
        "shared_pass_invocation_proven": False,
        "hardware_execution_qualified": False,
        "timing_authority": "NONE",
        "promotion_authority": "NONE",
    }


def _implementation():
    if _pin(__file__) != _SOURCE or any(
        globals()[name] is not function or function.__code__ is not code
        for name, function, code in _FUNCTIONS
    ):
        raise ValueError("finite original queue consumer implementation changed")


def check_source_pair(owner, state, **arguments):
    _implementation()
    try:
        return _check_source_pair(owner, state, **arguments)
    finally:
        _implementation()


_SOURCE = _pin(__file__)
_FUNCTIONS = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "_read",
        "_one",
        "_source_invocation",
        "_arm",
        "_pair_choices",
        "_check_source_pair",
        "check_source_pair",
        "_implementation",
        "source_proof",
        "audit_pointer_source_tiers",
        "_pin",
        "_revalidate_sources",
        "hash_tree",
    )
)
