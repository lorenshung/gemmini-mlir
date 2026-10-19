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
        self.checks = 0
        self.valid = True
        self.on_check = None

    def revalidate(self):
        self.checks += 1
        if self.on_check is not None:
            self.on_check(self)
        if not self.valid:
            raise ValueError("original source became invalid")


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


def test_issuance_does_not_keep_a_dropped_owner_alive(owners):
    import gc
    import weakref

    _, source, count, _, _ = owners
    separate = M.SourceNativeCorrespondenceOwner.bind(source, count)
    ref = weakref.ref(separate)
    del separate
    gc.collect()
    assert ref() is None


def test_metadata_getters_coalesce_only_inside_each_original_operation(owners):
    owner, _, _, source, _ = owners
    before = source.checks
    owner.record()
    assert source.checks == before + 2
    owner.selected_binding()
    assert source.checks == before + 4  # no cross-call qualification cache


def test_postcheck_refuses_mutation_during_getter(owners, monkeypatch):
    owner, _, _, source, _ = owners
    selected = source._prototype.selected
    original = selected.report

    def changed():
        result = original()
        source.valid = False
        return result

    monkeypatch.setattr(selected, "report", changed)
    with pytest.raises(ValueError, match="became invalid"):
        owner.record()
    assert M._METADATA._ACTIVE.get() is None


def test_exception_still_checks_original_owner_and_revokes_scope(owners, monkeypatch):
    owner, _, _, source, _ = owners
    before = source.checks

    def failure():
        raise RuntimeError("current independent proof refused")

    monkeypatch.setattr(source._prototype.selected, "report", failure)
    with pytest.raises(RuntimeError, match="independent proof"):
        owner.record()
    assert source.checks == before + 2
    assert M._METADATA._ACTIVE.get() is None


def test_own_callback_mutation_during_operation_refuses(owners, monkeypatch):
    owner, _, _, source, _ = owners
    selected = source._prototype.selected
    original = selected.report

    def changed():
        result = original()
        owner.verify_compiler_binding = lambda *a, **kw: None
        return result

    monkeypatch.setattr(selected, "report", changed)
    with pytest.raises(ValueError, match="bound callback"):
        owner.record()


def test_registry_roster_mutation_refuses(owners):
    owner, *_ = owners
    registry = M._ISSUED[owner][12]
    registry._capabilities = registry._capabilities[::-1]
    with pytest.raises(ValueError):
        owner.revalidate()


def test_metadata_callback_replacement_refuses(owners, monkeypatch):
    owner, *_ = owners
    import contextlib

    @contextlib.contextmanager
    def unverified(_):
        yield

    monkeypatch.setattr(
        M._METADATA, "scoped_registry_metadata_verification", unverified
    )
    with pytest.raises(ValueError, match="metadata callback changed"):
        owner.revalidate()


def test_failed_final_check_cannot_issue_owner(owners, monkeypatch):
    _, source_cap, count_cap, source, _ = owners
    original = source.revalidate.__func__
    calls = 0

    def failing(self):
        nonlocal calls
        original(self)
        calls += 1
        if calls == 2:
            self.valid = False
            raise ValueError("postcheck refused issuance")

    # Original reviewed capability rejects the changed callback even before issue.
    monkeypatch.setattr(type(source), "revalidate", failing)
    size = len(M._ISSUED)
    with pytest.raises(ValueError):
        M.SourceNativeCorrespondenceOwner.bind(source_cap, count_cap)
    assert len(M._ISSUED) == size


def test_own_custody_reopens_after_original_final_owner_check(owners):
    owner, _, _, source, _ = owners
    final = source.checks + 2

    def mutate_on_final(current):
        if current.checks == final:
            owner.verify_compiler_binding = lambda *a, **kw: None

    source.on_check = mutate_on_final
    with pytest.raises(ValueError, match="bound callback"):
        owner.record()
