"""Neutral source-custody tests and actual original-emitter host proposals.

These fixtures issue no target qualification. No object, ELF or simulator runs.
"""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import MethodType

import pytest
from merlin.llvmlower.reviewed_pass_capabilities import (
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection
from test_v26_staged_source import PACKAGE
from test_v26_staged_source import inputs as staged_inputs


@pytest.fixture
def selected(tmp_path, monkeypatch):
    generated = importlib.import_module("gemmini_generated_source_diagnostic")
    module = importlib.import_module("gemmini_v26_finite_preparation")
    calls = []
    for name in (
        "facts",
        "observed_contract",
        "software_spec",
        "nm",
        "execution_contract",
    ):
        (tmp_path / name).write_text("neutral " + name)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib").mkdir()
    (runtime / "bin/spike").write_text("neutral")
    (runtime / "lib/libgemmini.so").write_text("neutral")
    pins = {
        name: {
            "path": str(tmp_path / name),
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in ("facts", "observed_contract")
    }
    declaration = {
        "root": str(tmp_path),
        "contract_sha256": "neutral-library",
        "contract": {
            "review_id": "neutral",
            "members": [],
            "public_modules": [
                "merlin.compile.scheduling.direct_source",
                "merlin.compile.scheduling.source_tokens",
                "merlin.compile.scheduling.staged_dense",
            ],
        },
    }
    (tmp_path / "library_selection").write_text(json.dumps({"selection": declaration}))

    class Target:
        @classmethod
        def read(cls, *_):
            return cls()

        def revalidate(self):
            calls.append("target")

        def report(self):
            return {
                "selected_inputs": pins,
                "geometry": {
                    "array_rows": 16,
                    "array_cols": 16,
                    "stores": [
                        {"name": "scratchpad", "total_rows": 64},
                        {"name": "accumulator", "total_rows": 32},
                    ],
                },
            }

    class Arithmetic:
        identity = "neutral-arithmetic"

        def __init__(self, selected, **_):
            self.selected = selected

        def revalidate(self):
            calls.append("arithmetic")

        def source_rule(self):
            return {
                "max_primitive_k": 16,
                "operand_bits": 8,
                "primitive_mac_bits": 20,
                "accumulator_bits": 32,
            }

        def record(self):
            return {"rule": self.source_rule()}

    class Completion:
        identity = "neutral-completion"

        def __init__(self, selected, **_):
            self.selected = selected

        def revalidate(self):
            calls.append("completion")

        def record(self):
            return {
                "proof": "neutral",
                "selected_inputs": pins,
                "dense_v1_admission": {"enabled": False},
            }

    class Contract:
        sha256 = "neutral-library"

        def __init__(self, *_):
            pass

    class Library:
        def __init__(self, contract, *_):
            self.contract = contract

        def record(self):
            calls.append("library")
            return copy.deepcopy(declaration)

    for name, value in (
        ("SelectedSchedulingTarget", Target),
        ("IntegerArithmeticOwner", Arithmetic),
        ("StagedCompletionOwner", Completion),
        ("CompilerLibraryContract", Contract),
        ("CompilerLibrarySelection", Library),
    ):
        monkeypatch.setattr(generated, name, value)
    inputs = {
        name: str(tmp_path / name)
        for name in generated._FIELDS - {"observation_protocol"}
    }
    inputs.update(
        source_root=str(tmp_path),
        receiver_source=str(tmp_path),
        observation_protocol=generated.COMMIT_ONLY_PROTOCOL,
        source_binding_protocol="merlin.native_pointer_source.v1",
        execution_contract=str(tmp_path / "execution_contract"),
    )
    source = generated.GeneratedSourceDiagnosticIssuer(inputs)
    cap = ReviewedPassCapability(
        "source_execution_diagnostic", source.selected_binding(), source
    )
    registry = ReviewedPassCapabilities((cap,))
    joined = module.with_finite_preparation(registry)
    return module, generated, source, joined, joined._capabilities[1].owner, calls


def test_original_source_cap_is_preserved_and_preparation_is_not_qualification(
    selected,
):
    module, _, source, registry, owner, _ = selected
    assert registry._capabilities[0].owner is source
    assert registry._capabilities[1].mechanism == module.MECHANISM
    assert registry._capabilities[1].binding == registry._capabilities[0].binding
    row = owner.record()
    assert row["scheduling_qualification"] == row["timing_authority"] == "NONE"
    assert row["hardware_execution_qualified"] is False
    assert not hasattr(owner, "check_functional_guard")


@pytest.mark.parametrize("shape", [(32, 16, 32), (31, 15, 17)])
def test_real_prepared_hook_executes_original_emitter_and_public_shared_passes(
    selected, tmp_path, monkeypatch, shape
):
    module, _, _, registry, owner, _ = selected
    chosen = CompilerPassSelection(
        registry, registry._capabilities[0].binding, mechanisms=(module.MECHANISM,)
    )
    package = tmp_path / "prepared"
    chosen.prepare_compiler(
        module.MECHANISM,
        PACKAGE,
        package,
        policy={"enabled": True, "rewrite_policy": "verify_only"},
    )
    original = staged_inputs(tmp_path, *shape)
    monkeypatch.syspath_prepend(str(package))
    for name in ("finite_staged_policy", "finite_staged_emitter", "finite_staged_hook"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    hook = importlib.import_module("finite_staged_hook")
    before = hook.emit_if_selected(original["source"], original["command_buffer"])
    (package / "finite_staged_policy.py").write_text(
        "ENABLED = True\nREWRITE_POLICY = 'n_before_m_rhs'\n"
    )
    monkeypatch.delitem(sys.modules, "finite_staged_policy")
    after = hook.emit_if_selected(original["source"], original["command_buffer"])
    assert before != after
    assert before.count('asm sideeffect "fence"') == after.count(
        'asm sideeffect "fence"'
    )
    assert (
        owner.check_compiler_binding(package)["policy"]["rewrite_policy"]
        == "n_before_m_rhs"
    )
    assert (PACKAGE / "compiler.py").read_bytes() != (
        package / "compiler.py"
    ).read_bytes()
    for name in ("builder.py", "contraction.py"):
        assert (PACKAGE / name).read_bytes() == (package / name).read_bytes()


@pytest.mark.parametrize(
    "damage", ["binding", "public", "sources", "source_owner", "callback", "helper"]
)
def test_mutated_live_owners_helpers_and_contracts_refuse(
    selected, monkeypatch, damage
):
    module, _, _, _, owner, _ = selected
    if damage == "binding":
        owner._binding = copy.copy(owner._binding)
    elif damage == "public":
        owner._public["contract"]["operand_rows"] = 2048
    elif damage == "sources":
        owner._sources.clear()
    elif damage == "source_owner":
        owner._source = copy.copy(owner._source)
    elif damage == "callback":
        monkeypatch.setattr(
            owner, "prepare_compiler", MethodType(lambda self, *_a, **_k: None, owner)
        )
    else:
        monkeypatch.setattr(module, "_contract", lambda *_: owner._contract)
    with pytest.raises(ValueError, match="original|changed"):
        owner.revalidate()


@pytest.mark.parametrize(
    "damage",
    ["extra", "builder", "hook", "public", "policy_code", "policy_unknown", "library"],
)
def test_actual_prepared_package_mutations_refuse(selected, tmp_path, damage):
    _, _, _, _, owner, _ = selected
    package = tmp_path / "prepared"
    owner.prepare_compiler(
        PACKAGE, package, policy={"enabled": False, "rewrite_policy": "verify_only"}
    )
    if damage == "extra":
        (package / "extra.py").write_text("pass")
    elif damage == "builder":
        with (package / "builder.py").open("a") as stream:
            stream.write("\n# changed\n")
    elif damage == "hook":
        (package / "finite_staged_hook.py").write_text("raise RuntimeError()")
    elif damage == "public":
        (package / "finite-staged-public.json").write_text("{}")
    elif damage == "policy_code":
        (package / "finite_staged_policy.py").write_text(
            "ENABLED=True\nREWRITE_POLICY=__import__('os')\n"
        )
    elif damage == "policy_unknown":
        (package / "finite_staged_policy.py").write_text(
            "ENABLED=True\nREWRITE_POLICY='all'\n"
        )
    else:
        (package / "selected-shared-library.json").write_text("{}")
    with pytest.raises((ValueError, SyntaxError)):
        owner.check_compiler_binding(package)


def test_default_policy_keeps_original_emitter_and_unissued_or_json_refuses(
    selected, tmp_path, monkeypatch
):
    module, _, _, _, owner, _ = selected
    package = tmp_path / "prepared"
    owner.prepare_compiler(
        PACKAGE, package, policy={"enabled": False, "rewrite_policy": "verify_only"}
    )
    monkeypatch.syspath_prepend(str(package))
    for name in ("finite_staged_policy", "finite_staged_hook"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert (
        importlib.import_module("finite_staged_hook").emit_if_selected(
            tmp_path / "missing-source", {}
        )
        is None
    )
    with pytest.raises(ValueError):
        object.__new__(module.FiniteV26PreparationOwner).revalidate()
    with pytest.raises((TypeError, ValueError)):
        module.with_finite_preparation({"qualified": True})


def test_new_original_source_calls_and_final_mutation_remain_fresh(
    selected, monkeypatch
):
    _, _, _, _, owner, calls = selected
    initial = len(calls)
    owner.revalidate()
    one = len(calls) - initial
    owner.revalidate()
    assert len(calls) - initial == 2 * one and one > 0


def test_no_current_pair_or_trace_runs_from_a_saved_success(selected):
    _, _, _, _, owner, _ = selected
    with pytest.raises(ValueError, match="original current source"):
        owner.check_actual_pair(
            baseline_owner={"pass": True}, candidate_owner={"pass": True}
        )


def test_actual_compiler_entrypoint_consumes_the_selected_hook(
    selected, tmp_path, monkeypatch, capsys
):
    _, _, _, _, owner, _ = selected
    package = tmp_path / "prepared"
    owner.prepare_compiler(
        PACKAGE, package, policy={"enabled": True, "rewrite_policy": "verify_only"}
    )
    values = staged_inputs(tmp_path)
    monkeypatch.syspath_prepend(str(package))
    for name in (
        "frontend",
        "linalg_frontend",
        "mixed_frontend",
        "mixed_codegen",
        "mixed_scalar",
        "builder",
        "backend",
        "contraction",
        "conv",
        "pooling",
        "elementwise",
        "finite_staged_hook",
        "finite_staged_emitter",
        "finite_staged_policy",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "actual_prepared_v26_compiler", package / "compiler.py"
    )
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    translated = []

    def import_llvm(argv, *, input, **kwargs):
        assert argv[-1] == "--import-llvm"
        translated.append(input)
        return type(
            "HostImportOnlyFixture",
            (),
            {"returncode": 0, "stdout": input, "stderr": ""},
        )()

    monkeypatch.setattr(compiler.subprocess, "run", import_llvm)
    output = tmp_path / "actual-command-buffer.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(package / "compiler.py"),
            "--emit-target-artifact",
            "--emit-command-buffer=" + str(output),
            str(values["source"]),
        ],
    )
    compiler.main()
    assert len(translated) == 1 and capsys.readouterr().out == translated[0]
    assert " br " not in translated[0]  # finite control selected by actual main
    assert json.loads(output.read_bytes())["compiler_pointer_abi"]["arguments"] == [
        "a",
        "b",
        "Y0",
    ]


def test_prepared_candidate_imports_only_the_explicit_public_sdk(selected, tmp_path):
    from merlin.compile.scheduling.export import export_shared_scheduling
    from merlin.targetgen.compiler_library import freeze_compiler_library
    from merlin.targetgen.package_runtime import integrity_scan, load_package

    _, _, _, _, owner, _ = selected
    package = tmp_path / "prepared"
    owner.prepare_compiler(
        PACKAGE, package, policy={"enabled": True, "rewrite_policy": "verify_only"}
    )
    sdk = tmp_path / "sdk"
    release = export_shared_scheduling(sdk, source_binding=True)
    contract = freeze_compiler_library(
        sdk,
        review_id="host-test-only",
        public_modules=tuple(release["public_modules"]),
        sources=tuple((row["path"], row["module"]) for row in release["members"])
        + (("shared-library.json", None),),
    )
    integrity_scan(
        load_package(package), compiler_library=contract, compiler_library_root=sdk
    )


@pytest.mark.parametrize("remaining", [0, -1, True, float("nan"), float("inf")])
def test_queue_pair_budget_refuses_before_reading_saved_products(
    selected, tmp_path, remaining
):
    _, _, _, _, owner, _ = selected
    with pytest.raises(ValueError, match="positive finite"):
        owner.check_source_pair(
            baseline_workspace=tmp_path / "b",
            candidate_workspace=tmp_path / "c",
            capsule={"__dir__": str(tmp_path)},
            remaining_s=remaining,
        )


def test_queue_pair_requires_current_grade_selection_and_unique_trace(
    selected, tmp_path
):
    from gemmini_v26_finite_pair import _arm, _one

    _, generated, source, _, owner, _ = selected
    (tmp_path / "ordinary_observation.json").write_text(
        json.dumps(
            {
                "functional_status": "PASS",
                "original_grade": {
                    "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass"
                },
            }
        )
    )
    root = tmp_path / "capsule_runs/runs/gemmini-capsule-bench/functional"
    root.mkdir(parents=True)
    (root / "capsule_result.json").write_text(
        json.dumps(
            {"checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass"}
        )
    )
    with pytest.raises(
        ValueError, match="original complete diagnostic grade/selection"
    ):
        _arm(owner, generated._STATE(source), tmp_path, {}, float("inf"))
    with pytest.raises(ValueError, match="unique complete original producer"):
        _one(tmp_path, "*/invocation.json")
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "invocation.json").write_text("{}")
    with pytest.raises(ValueError, match="unique complete original producer"):
        _one(tmp_path, "*/invocation.json")


def test_queue_pair_private_consumer_function_mutation_refuses(
    selected, tmp_path, monkeypatch
):
    import gemmini_v26_finite_pair as pair

    _, _, _, _, owner, _ = selected
    monkeypatch.setattr(pair, "_arm", lambda *_: {"status": "PASS"})
    with pytest.raises(ValueError, match="consumer implementation changed"):
        owner.check_source_pair(
            baseline_workspace=tmp_path / "b",
            candidate_workspace=tmp_path / "c",
            capsule={"__dir__": str(tmp_path)},
            remaining_s=10,
        )


def test_original_factory_reconstructs_source_and_separate_preparation_without_saved_pass(
    selected, tmp_path
):
    import gemmini_v26_finite_factory as factory
    from merlin.targetgen.compiler_pass_selection import restore_original_factory

    module, generated, source, _, _, _ = selected
    original = factory.original_factory(generated._STATE(source)["inputs"])
    restored = restore_original_factory(original.record())
    registry = restored.create(work_root=tmp_path / "fresh-original-factory")
    assert restored.record() == original.record()
    assert (
        type(registry._capabilities[0].owner)
        is generated.GeneratedSourceDiagnosticIssuer
    )
    assert type(registry._capabilities[1].owner) is module.FiniteV26PreparationOwner
    assert registry._capabilities[1].mechanism == module.MECHANISM
    assert {Path(pin["path"]).name for pin in original.record()["sources"]} >= {
        "gemmini_v26_finite_pair.py",
        "gemmini_v26_finite_hook.py",
        "gemmini_generated_source_trace.py",
    }


def test_class_callback_replacement_cannot_issue_an_original_preparer(
    selected, monkeypatch
):
    module, _, _, registry, _, _ = selected
    monkeypatch.setattr(
        module.FiniteV26PreparationOwner, "record", lambda self: {"pass": True}
    )
    with pytest.raises(ValueError, match="original implementation changed"):
        module.FiniteV26PreparationOwner(type(registry)((registry._capabilities[0],)))


def test_own_imported_source_pin_remains_required_before_new_issuance(
    selected, monkeypatch
):
    module, _, _, registry, _, _ = selected
    monkeypatch.setattr(
        module, "_IMPORTED_SOURCE", {"path": __file__, "sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="original implementation changed"):
        module.FiniteV26PreparationOwner(type(registry)((registry._capabilities[0],)))


@pytest.mark.parametrize("candidate", ["verify_only", "n_before_m_rhs"])
def test_original_control_and_retained_policy_are_explicit_pair_choices(candidate):
    from gemmini_v26_finite_pair import _pair_choices

    b = {"enabled": True, "rewrite_policy": "verify_only"}
    c = {"enabled": True, "rewrite_policy": candidate}
    assert _pair_choices([(None, {"policy": b}), (None, {"policy": c})]) == (b, c)


@pytest.mark.parametrize(
    "fault",
    [
        "baseline_rewrite",
        "baseline_disabled",
        "candidate_disabled",
        "candidate_unknown",
        "extra",
    ],
)
def test_missing_or_unbounded_actual_pair_choices_refuse(fault):
    from gemmini_v26_finite_pair import _pair_choices

    b = {"enabled": True, "rewrite_policy": "verify_only"}
    c = {"enabled": True, "rewrite_policy": "n_before_m_rhs"}
    if fault == "baseline_rewrite":
        b["rewrite_policy"] = "n_before_m_rhs"
    elif fault == "baseline_disabled":
        b["enabled"] = False
    elif fault == "candidate_disabled":
        c["enabled"] = False
    elif fault == "candidate_unknown":
        c["rewrite_policy"] = "unknown"
    else:
        c["foreign"] = True
    with pytest.raises(ValueError, match="explicit bounded candidate"):
        _pair_choices([(None, {"policy": b}), (None, {"policy": c})])
