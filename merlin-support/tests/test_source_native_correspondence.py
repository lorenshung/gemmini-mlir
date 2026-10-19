"""Neutral namespace/custody controls, not actual target qualification."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import gemmini_source_native_correspondence as M
import pytest
from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding as Binding
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability as Cap


class Selected:
    def __init__(self, inputs):
        self.inputs = inputs

    def report(self):
        return {"selected_inputs": self.inputs}


class Completion:
    def __init__(self, selected):
        self.selected = selected
        self.proof = {
            "selected_inputs": copy.deepcopy(selected.inputs),
            "proof": {"actual": "separate original staged proof"},
        }

    def revalidate(self):
        pass

    def record(self):
        return copy.deepcopy(self.proof)


class Prototype:
    def __init__(self):
        self.selected = Selected(
            {
                "facts": {"path": "/original/facts", "sha256": "a" * 64},
                "observed_contract": {"path": "/original/observed", "sha256": "b" * 64},
            }
        )
        self.completion = Completion(self.selected)
        self.contract = SimpleNamespace(
            hardware="a" * 64,
            software=M._digest([self.selected.inputs, self.completion.proof["proof"]]),
            numerical_policy="c" * 64,
            execution="d" * 64,
        )
        self.library = SimpleNamespace(contract=SimpleNamespace(sha256="e" * 64))

    def revalidate(self):
        pass


class Source:
    def __init__(self):
        self._prototype = Prototype()
        self.identity = "neutral-original-source"

    def revalidate(self):
        pass


class Count:
    def __init__(self, inputs):
        self.inputs = copy.deepcopy(inputs)
        self.identity = "neutral-original-count"
        self.correspondence = self

    def revalidate(self):
        return {"selected_inputs": copy.deepcopy(self.inputs)}

    def selected_binding(self):
        return asdict(Binding("a" * 64, "b" * 64, "f" * 64, "1" * 64, "2" * 64))


@pytest.fixture
def owners(monkeypatch):
    # Exact neutral stand-ins exercise registry binding, never target issuance.
    for name, value in [
        ("PublicSourceDiagnosticIssuer", Source),
        ("CountHardwareCorrespondenceOwner", Count),
        ("StagedDensePrototype", Prototype),
        ("StagedCompletionOwner", Completion),
    ]:
        monkeypatch.setattr(M, name, value)
    source = Source()
    p = source._prototype
    c = p.contract
    count = Count(p.selected.inputs)
    source_cap = Cap(
        "source_execution_diagnostic",
        Binding(
            c.hardware,
            c.software,
            c.numerical_policy,
            c.execution,
            p.library.contract.sha256,
        ),
        source,
    )
    count_cap = Cap(
        "timing_model_correspondence", Binding(**count.selected_binding()), count
    )
    owner = M.SourceNativeCorrespondenceOwner.bind(source_cap, count_cap)
    return owner, source_cap, count_cap, source, count


def test_composite_identity_is_related_not_equated(owners):
    owner, s, c, _, _ = owners
    assert s.binding.software != c.binding.software
    owner.verify_compiler_binding(s.binding, native_binding=c.binding)
    assert owner.selected_binding() == asdict(c.binding)
    assert owner.record()["promotion_authority"] == "NONE"


@pytest.mark.parametrize(
    "field", ["hardware", "software", "numerical_policy", "runtime", "library"]
)
def test_actual_compiler_full_binding_must_match(owners, field):
    owner, s, c, _, _ = owners
    with pytest.raises(ValueError):
        owner.verify_compiler_binding(
            replace(s.binding, **{field: "foreign"}), native_binding=c.binding
        )


def test_actual_native_binding_must_match(owners):
    owner, s, c, _, _ = owners
    with pytest.raises(ValueError):
        owner.verify_compiler_binding(
            s.binding, native_binding=replace(c.binding, software=s.binding.software)
        )


@pytest.mark.parametrize("which", ["facts", "observed_contract"])
def test_foreign_raw_selected_inputs_refuse(owners, which):
    owner, _, _, source, _ = owners
    source._prototype.selected.inputs[which]["sha256"] = "3" * 64
    with pytest.raises(ValueError):
        owner.revalidate()


def test_changed_completion_proof_refuses(owners):
    owner, _, _, source, _ = owners
    source._prototype.completion.proof["proof"]["actual"] = "foreign"
    with pytest.raises(ValueError):
        owner.revalidate()


def test_swapped_prototype_refuses(owners):
    owner, _, _, source, _ = owners
    source._prototype = Prototype()
    with pytest.raises(ValueError):
        owner.revalidate()


def test_borrowed_original_owner_callback_refuses(owners):
    owner, _, _, source, _ = owners
    source.revalidate = lambda: None
    with pytest.raises(ValueError):
        owner.revalidate()


def test_replaced_bridge_callback_refuses(owners):
    owner, *_ = owners
    owner.verify_compiler_binding = lambda *a, **k: None
    with pytest.raises(ValueError):
        owner.revalidate()


def test_implementation_byte_drift_refuses(owners, monkeypatch):
    owner, *_ = owners
    original = M._pin
    monkeypatch.setattr(M, "_pin", lambda p: (original(p)[0], "changed"))
    with pytest.raises(ValueError):
        owner.revalidate()


def test_json_cannot_issue():
    with pytest.raises(TypeError):
        M.SourceNativeCorrespondenceOwner.bind({}, {})
    with pytest.raises(TypeError):
        M.SourceNativeCorrespondenceOwner()
