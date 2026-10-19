"""Original final-consumer admission controls; fixtures qualify no execution."""

import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "backend"))
    return importlib.import_module("gemmini_cleanup_final_guard")


def test_pending_original_grade_never_launches_final_measurements(
    consumer, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        consumer, "check_native_pair", lambda *args, **kw: calls.append(kw)
    )
    with pytest.raises(ValueError, match="pending"):
        consumer.check_functional_guard(
            None,
            tmp_path,
            tmp_path,
            {"status": "offending"},
            {},
            grade_root=tmp_path / "grade",
        )
    assert not calls


@pytest.fixture
def original_roster(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "compiler.py").write_text("# neutral source-only fixture\n")
    roots = (tmp_path / "grade_library_baseline", tmp_path / "grade")
    results = []
    for root in roots:
        import shutil

        root.mkdir()
        shutil.copytree(package, root / "compiler")
        run = root / "runs/neutral-suite/public-source"
        (run / "candidate_native").mkdir(parents=True)
        native = {"synthetic_original_grade_projection": True}
        (run / "candidate_native/result.json").write_text(json.dumps(native))
        pins = {}
        for role in ("command_buffer", "lowered_mlir"):
            path = run / role
            path.write_text("original fixture data\n")
            import hashlib

            pins[role] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        record = {
            "capsule": "public-source",
            "status": "pass",
            "candidate_native_execution": native,
            "candidate_emission": pins,
        }
        (run / "capsule_result.json").write_text(json.dumps(record))
        results.append(
            {
                "status": "clean",
                "capsules": 1,
                "full_output_regrade": {
                    "status": "pass",
                    "evidence_root": str(root),
                    "score": {
                        "integrity_status": "clean",
                        "per_capsule": [
                            {
                                "capsule": "public-source",
                                "status": "pass",
                                "numeric": "pass",
                            }
                        ],
                    },
                },
            }
        )
    return package, roots, results


def test_final_consumer_joins_exact_original_rosters_to_actual_checker(
    consumer, original_roster, monkeypatch
):
    package, roots, results = original_roster
    calls = []

    def checked(owner, **kwargs):
        calls.append(kwargs)
        return {"synthetic_consumer_routing_only": True}

    monkeypatch.setattr(consumer, "check_native_pair", checked)
    report = consumer.check_functional_guard(
        None, package, package, *results, grade_root=roots[1]
    )
    assert set(report) == {"public-source"}
    assert len(calls) == 1
    assert calls[0]["baseline"]["native_result"].is_relative_to(roots[0])
    assert calls[0]["candidate"]["native_result"].is_relative_to(roots[1])


@pytest.mark.parametrize(
    "damage",
    [
        "pending",
        "cert",
        "root",
        "duplicate",
        "missing",
        "foreign",
        "compiler",
        "receipt",
        "emission",
        "family",
        "integrity",
        "count",
        "unsafe_name",
    ],
)
def test_incomplete_or_foreign_original_grade_never_substitutes_saved_success(
    consumer, original_roster, monkeypatch, damage
):
    package, roots, results = original_roster
    candidate = results[1]
    run = roots[1] / "runs/neutral-suite/public-source"
    path = run / "capsule_result.json"
    row = json.loads(path.read_bytes())
    if damage == "pending":
        candidate["full_output_regrade"]["status"] = "unavailable"
    elif damage == "cert":
        candidate["full_output_regrade"]["score"]["per_capsule"][0]["cert_verdict"] = {
            "would_be_status": "incomplete"
        }
    elif damage == "root":
        candidate["full_output_regrade"]["evidence_root"] = str(roots[0])
    elif damage == "duplicate":
        candidate["full_output_regrade"]["score"]["per_capsule"] *= 2
    elif damage == "missing":
        path.unlink()
    elif damage == "foreign":
        row["capsule"] = "other-source"
    elif damage == "compiler":
        (roots[1] / "compiler/compiler.py").write_text("changed source")
    elif damage == "receipt":
        (run / "candidate_native/result.json").write_text("{}")
    elif damage == "emission":
        (run / "lowered_mlir").write_text("changed bytes")
    elif damage == "integrity":
        candidate["full_output_regrade"]["score"]["integrity_status"] = "uncertain"
    elif damage == "count":
        candidate["capsules"] = 2
    elif damage == "unsafe_name":
        candidate["full_output_regrade"]["score"]["per_capsule"][0]["capsule"] = ".."
    else:
        row.pop("candidate_native_execution")
    if damage != "missing":
        path.write_text(json.dumps(row))
    calls = []
    monkeypatch.setattr(
        consumer, "check_native_pair", lambda *args, **kw: calls.append(kw)
    )
    with pytest.raises(ValueError):
        consumer.check_functional_guard(
            None, package, package, *results, grade_root=roots[1]
        )
    assert not calls


@pytest.mark.parametrize("damage", ["schema", "numeric", "tier", "buffer", "llvm"])
def test_native_arm_refuses_unbound_complete_output_claim(consumer, tmp_path, damage):
    import hashlib

    cb = tmp_path / "buffer.json"
    lowered = tmp_path / "module.mlir"
    result = tmp_path / "result.json"
    cb.write_text("{}")
    lowered.write_text("module {}")
    row = {
        "schema": "merlin_candidate_native_model_execution_v1",
        "numeric": {"status": "pass"},
        "tiers": {"L2": {"status": "pass"}},
        "candidate": {
            "command_buffer_sha256": hashlib.sha256(b"{}").hexdigest(),
            "lowered_mlir_sha256": hashlib.sha256(lowered.read_bytes()).hexdigest(),
        },
    }
    if damage == "schema":
        row["schema"] = "copied_success"
    elif damage == "numeric":
        row["numeric"]["status"] = "incomplete"
    elif damage == "tier":
        row["tiers"]["L2"]["status"] = "unavailable"
    elif damage == "buffer":
        row["candidate"]["command_buffer_sha256"] = "0" * 64
    else:
        row["candidate"]["lowered_mlir_sha256"] = "0" * 64
    result.write_text(json.dumps(row))

    class OriginalFixture:
        def revalidate(self):
            return {}

    owner = OriginalFixture()
    owner.cleanup = OriginalFixture()
    with pytest.raises(ValueError, match="complete graded outputs"):
        consumer.check_native_arm(
            owner,
            native_result=result,
            command_buffer=cb,
            lowered=lowered,
            work_root=tmp_path / "unused",
        )
    assert not (tmp_path / "unused").exists()


def test_only_final_instruction_position_annotations_are_comparable(consumer):
    before = {
        "params": {
            "global_program_plan": {
                "schedule_instruction_count": 4,
                "tasks": [
                    {"instruction_start": 0, "instruction_end": 4, "source": "exact"}
                ],
            }
        },
        "commands": [{"opcode": "COMMIT", "scale": 1}],
    }
    after = copy.deepcopy(before)
    after["params"]["global_program_plan"]["schedule_instruction_count"] = 3
    assert consumer._logical(before) == consumer._logical(after)
    after["commands"][0]["scale"] = 2
    assert consumer._logical(before) != consumer._logical(after)
