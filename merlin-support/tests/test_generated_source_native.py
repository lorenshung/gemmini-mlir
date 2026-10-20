"""Neutral structural binding controls; no native timing qualification."""

import copy
import importlib
from dataclasses import asdict, replace

import pytest
from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding as Binding
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability as Cap
from test_generated_source_diagnostic import (
    issued,  # noqa: F401 -- original neutral source fixture
)


@pytest.fixture
def joined(issued, monkeypatch):  # noqa: F811
    _, source, _, _, _ = issued
    M = importlib.import_module("gemmini_generated_source_native")
    state = M._STATE(source)
    native = Binding(
        state["binding"].hardware,
        state["selected_inputs"]["observed_contract"]["sha256"],
        "native features",
        "native model",
        "native libraries",
    )

    class Count:
        identity = "original neutral count"

        def __init__(self):
            self.inputs = copy.deepcopy(state["selected_inputs"])
            self.correspondence = self
            self.checks = 0
            self.on_check = None
            self.extension = copy.deepcopy(state["runtime"]["functional_extension"])

        def revalidate(self):
            self.checks += 1
            if self.on_check is not None:
                self.on_check(self)
            return {
                "selected_inputs": copy.deepcopy(self.inputs),
                "pins": {"functional_extension": self.extension},
            }

        def selected_binding(self):
            self.revalidate()
            return asdict(native)

    monkeypatch.setattr(M, "CountHardwareCorrespondenceOwner", Count)
    count = Count()
    source_cap = Cap("source_execution_diagnostic", source.selected_binding(), source)
    count_cap = Cap("timing_model_correspondence", native, count)
    owner = M.GeneratedSourceNativeCorrespondenceOwner.bind(source_cap, count_cap)
    return M, owner, source_cap, count_cap, source, count


def test_original_composite_and_native_digests_related_not_equated(joined):
    _, owner, source, count, *_ = joined
    assert source.binding.software != count.binding.software
    owner.verify_compiler_binding(source.binding, native_binding=count.binding)
    assert owner.selected_binding() == asdict(count.binding)
    assert owner.record()["promotion_authority"] == "NONE"
    assert owner.record()["timing_qualification"] is False


@pytest.mark.parametrize(
    "field", ["hardware", "software", "numerical_policy", "runtime", "library"]
)
def test_all_actual_compiler_fields_must_match(joined, field):
    _, owner, source, count, *_ = joined
    with pytest.raises(ValueError, match="full original binding"):
        owner.verify_compiler_binding(
            replace(source.binding, **{field: "foreign"}), native_binding=count.binding
        )


@pytest.mark.parametrize("role", ["facts", "observed_contract", "operand_sidecar"])
def test_foreign_raw_selected_inputs_refuse(joined, role):
    _, owner, _, _, _, count = joined
    count.inputs[role] = {"path": "/foreign", "sha256": "foreign"}
    with pytest.raises(ValueError, match="facts/runtime differ"):
        owner.revalidate()


def test_foreign_native_runtime_refuses(joined):
    _, owner, _, _, _, count = joined
    count.extension["sha256"] = "foreign"
    with pytest.raises(ValueError, match="facts/runtime differ"):
        owner.revalidate()


def test_saved_receipts_cannot_issue(joined):
    M, _, *_ = joined
    with pytest.raises(TypeError):
        M.GeneratedSourceNativeCorrespondenceOwner.bind({}, {})
    with pytest.raises(TypeError):
        M.GeneratedSourceNativeCorrespondenceOwner()


def test_replaced_implementation_callback_refuses(joined, monkeypatch):
    _, owner, *_ = joined
    monkeypatch.setattr(owner, "verify_compiler_binding", lambda *a, **k: None)
    with pytest.raises(ValueError, match="callback changed"):
        owner.revalidate()


def test_fresh_original_count_checks_run_on_every_binding(joined):
    _, owner, source, count, _, original = joined
    before = original.checks
    owner.verify_compiler_binding(source.binding, native_binding=count.binding)
    first = original.checks
    owner.verify_compiler_binding(source.binding, native_binding=count.binding)
    assert first > before and original.checks > first


def test_final_original_owner_mutation_cannot_escape_scope(joined):
    _, owner, source, count, _, original = joined
    compiler, native = source.binding, count.binding
    before = original.checks

    def final(current):
        if current.checks > before + 2:
            source._binding = replace(compiler, software="foreign")

    original.on_check = final
    with pytest.raises(ValueError):
        owner.verify_compiler_binding(compiler, native_binding=native)


def test_own_source_pin_drift_refuses(joined):
    M, owner, *_ = joined
    M._ISSUED[owner]["sources"]["gemmini_generated_source_native"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        owner.revalidate()
