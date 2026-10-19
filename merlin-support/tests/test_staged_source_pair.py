"""Bounded original proof reuse, never a persisted qualification fixture."""

import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    helper = importlib.import_module("gemmini_staged_source_pair")
    cap = tmp_path / "capsule"
    cap.mkdir()
    (cap / "capsule.yaml").write_text(
        "capture_execution_attestations: [original-exact-attestation]\n"
    )
    (cap / "capsule.interface.mlir").write_text("exact source fixture")
    (cap / "inputs.npz").write_text("exact inputs fixture")
    (cap / "golden.npz").write_text("exact independent complete references")
    source = {
        name: {**helper._pin(cap / file), "size_bytes": (cap / file).stat().st_size}
        for name, file in (
            ("capsule_declaration", "capsule.yaml"),
            ("interface", "capsule.interface.mlir"),
            ("inputs", "inputs.npz"),
            ("golden_arrays", "golden.npz"),
        )
    }
    arms = []
    for name in ("baseline", "candidate"):
        p = tmp_path / (name + ".json")
        p.write_text(json.dumps({"source": source}))
        arms.append({"native_result": helper._pin(p)})
    calls = []

    def proof(cap, directory):
        calls.append("original_sealed_proof")
        return {"original_source": "unchanged exact fixture"}

    def arm(owner, side, proof, cap):
        calls.append("independent_runtime_arithmetic_packets_outputs")
        return "original actual arm fixture"

    monkeypatch.setattr(helper, "source_proof", proof)
    monkeypatch.setattr(helper, "_arm", arm)

    class Owner:
        def revalidate(self):
            calls.append("original_owner_revalidated")

        def _source_arm(self, arm):
            calls.append("fresh_per_arm_fallback")
            return "separate original arm fixture"

    return helper, Owner(), arms, calls, cap


def test_one_fresh_original_source_proof_per_exact_pair_and_no_reuse_across_calls(pair):
    helper, owner, arms, calls, _cap = pair
    before, after, scope = helper.check_source_arms(owner, *arms)
    assert before == after == "original actual arm fixture"
    assert calls.count("original_sealed_proof") == 1
    assert calls.count("independent_runtime_arithmetic_packets_outputs") == 2
    assert "within this pair" in scope
    helper.check_source_arms(owner, *arms)
    assert calls.count("original_sealed_proof") == 2


@pytest.mark.parametrize(
    "role", ["capsule_declaration", "interface", "inputs", "golden_arrays"]
)
def test_any_distinct_capture_source_inputs_or_references_use_independent_original_fallback(
    pair, role
):
    helper, owner, arms, calls, cap = pair
    native = Path(arms[1]["native_result"]["path"])
    record = json.loads(native.read_bytes())
    different = cap / (role + "-other")
    different.write_bytes(Path(record["source"][role]["path"]).read_bytes())
    record["source"][role] = helper._pin(different)
    native.write_text(json.dumps(record))
    arms[1]["native_result"] = helper._pin(native)
    before, after, scope = helper.check_source_arms(owner, *arms)
    assert before == after == "separate original arm fixture"
    assert calls.count("original_sealed_proof") == 0
    assert calls.count("fresh_per_arm_fallback") == 2
    assert "fresh proof per arm" in scope


@pytest.mark.parametrize("damage", ["proof", "inputs", "attestation", "native_binding"])
def test_mutation_during_the_fixed_pair_checks_cannot_reuse_original_proof(
    pair, monkeypatch, damage
):
    helper, owner, arms, _calls, cap = pair
    original = helper._arm

    def mutate(owner, arm, proof, directory):
        result = original(owner, arm, proof, directory)
        if damage == "proof":
            proof["original_source"] = "mutated"
        elif damage == "inputs":
            (cap / "inputs.npz").write_text("changed inputs")
        elif damage == "attestation":
            (cap / "capsule.yaml").write_text("changed original attestation")
        else:
            p = Path(arm["native_result"]["path"])
            record = json.loads(p.read_bytes())
            record["source"] = copy.deepcopy(record["source"])
            record["source"].pop("inputs")
            p.write_text(json.dumps(record))
            arm["native_result"] = helper._pin(p)
        return result

    monkeypatch.setattr(helper, "_arm", mutate)
    with pytest.raises(ValueError):
        helper.check_source_arms(owner, *arms)
