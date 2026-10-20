"""Source-only live admission independent of any earlier compiler/pass report.

Derive actual arithmetic, completion and runtime facts. Every emitted program
still needs its own full source/packet/output checks. No saved success can issue
this owner and no compiler transformation is authorized by it.
"""

from __future__ import annotations

import copy
import math
import weakref
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_generated_source_trace import observe_current_source
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_source_execution import COMMIT_ONLY_PROTOCOL
from gemmini_source_packet_artifacts import POINTER_CONTRACTIONS_PROTOCOL
from gemmini_staged_dense_admission import StagedCompletionOwner
from gemmini_staged_dense_capability import _digest
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_library import (
    CompilerLibraryContract,
    CompilerLibrarySelection,
    LibraryMember,
)

_ISSUED = weakref.WeakKeyDictionary()
_FIELDS = frozenset(
    {
        "facts",
        "observed_contract",
        "software_spec",
        "source_root",
        "receiver_source",
        "runtime",
        "nm",
        "library_selection",
        "observation_protocol",
    }
)


class GeneratedSourceDiagnosticIssuer:
    def __init__(self, inputs):
        if (
            type(self) is not GeneratedSourceDiagnosticIssuer
            or type(inputs) is not dict
        ):
            raise TypeError(
                "generated source diagnostics need original explicit inputs"
            )
        optional = {
            "metadata_verification_protocol",
            "source_binding_protocol",
            "execution_contract",
            "source_memory_effects_protocol",
        }
        if not _FIELDS <= set(inputs) or set(inputs) - _FIELDS - optional:
            raise ValueError(
                "generated source diagnostics reject missing or foreign input fields"
            )
        if inputs["observation_protocol"] != COMMIT_ONLY_PROTOCOL:
            raise ValueError(
                "generated source diagnostics require explicit complete-commit V2 observation"
            )
        binding_protocol = inputs.get("source_binding_protocol", "whole_program")
        if binding_protocol not in {
            "whole_program",
            "merlin.native_pointer_source.v1",
            POINTER_CONTRACTIONS_PROTOCOL,
        }:
            raise ValueError("unknown generated-source binding protocol")
        memory_effects_protocol = inputs.get("source_memory_effects_protocol")
        if memory_effects_protocol is not None:
            from gemmini_whole_program_memory_effects import require_protocol

            require_protocol(memory_effects_protocol)
            if binding_protocol != "whole_program":
                raise ValueError(
                    "selected source memory effects require whole_program binding"
                )
        if (binding_protocol == "whole_program") != (
            "execution_contract" not in inputs
        ):
            raise ValueError(
                "direct source binding requires its explicit original execution contract"
            )
        from gemmini_source_execution import SourceExecutionOwner

        owner_type = SourceExecutionOwner
        if binding_protocol != "whole_program":
            from gemmini_direct_source_execution import (
                DirectSourceExecutionOwner,
                PointerContractionsExecutionOwner,
            )

            owner_type = (
                PointerContractionsExecutionOwner
                if binding_protocol == POINTER_CONTRACTIONS_PROTOCOL
                else DirectSourceExecutionOwner
            )
        if (
            inputs.get(
                "metadata_verification_protocol", "merlin.pass_metadata_operation.v1"
            )
            != "merlin.pass_metadata_operation.v1"
        ):
            raise ValueError(
                "generated source diagnostics reject unknown metadata protocol"
            )
        paths = {
            name: Path(inputs[name]) for name in _FIELDS - {"observation_protocol"}
        }
        if "execution_contract" in inputs:
            paths["execution_contract"] = Path(inputs["execution_contract"])
        for path in paths.values():
            if (
                not path.is_absolute()
                or path.resolve() != path
                or any(p.is_symlink() for p in (path, *path.parents))
            ):
                raise ValueError(
                    "generated source diagnostics need original canonical input paths"
                )
        import json

        declaration = json.loads(paths["library_selection"].read_bytes())["selection"]
        contract = declaration["contract"]
        library = CompilerLibrarySelection(
            CompilerLibraryContract(
                contract["review_id"],
                tuple(contract["public_modules"]),
                tuple(LibraryMember(**row) for row in contract["members"]),
            ),
            Path(declaration["root"]),
        )
        if library.record() != declaration:
            raise ValueError("generated source selected library bytes changed")
        selected = SelectedSchedulingTarget.read(
            paths["facts"], paths["observed_contract"]
        )
        arithmetic = IntegerArithmeticOwner(
            selected,
            **{
                name: paths[name]
                for name in (
                    "software_spec",
                    "source_root",
                    "receiver_source",
                    "runtime",
                )
            },
        )
        memory_effects = None
        if memory_effects_protocol is not None:
            from gemmini_whole_program_memory_effects import WholeProgramMemoryEffects

            memory_effects = WholeProgramMemoryEffects(selected, arithmetic)
        completion = StagedCompletionOwner(selected, source_root=paths["source_root"])
        runtime = {
            "spike": _pin(paths["runtime"] / "bin/spike"),
            "functional_extension": _pin(paths["runtime"] / "lib/libgemmini.so"),
        }
        sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_generated_source_diagnostic",
                "gemmini_generated_source_factory",
                "gemmini_generated_source_trace",
                "gemmini_arithmetic_scope",
                "gemmini_staged_dense_admission",
                "gemmini_staged_completion",
                "gemmini_source_execution",
                "gemmini_source_execution_issuer",
                "gemmini_source_packet_artifacts",
                "merlin.targetgen.native_model_execution",
                "merlin.common.invocation_record",
            )
        }
        if binding_protocol != "whole_program":
            sources.update(
                {
                    name: _pin(module_source_path(name))
                    for name in (
                        "gemmini_direct_source_execution",
                        "merlin.llvmlower.direct_source_contraction",
                    )
                }
            )
            if binding_protocol == POINTER_CONTRACTIONS_PROTOCOL:
                sources["merlin.llvmlower.pointer_source_contractions"] = _pin(
                    module_source_path("merlin.llvmlower.pointer_source_contractions")
                )
        if memory_effects_protocol is not None:
            sources.update(
                {
                    name: _pin(module_source_path(name))
                    for name in (
                        "gemmini_whole_program_memory_effects",
                        "gemmini_load_streams",
                        "gemmini_zero_fill",
                        "gemmini_private_zero_storage",
                    )
                }
            )
        source_inputs, rule, proof = (
            selected.report()["selected_inputs"],
            arithmetic.source_rule(),
            completion.record(),
        )
        binding = PassCapabilityBinding(
            source_inputs["facts"]["sha256"],
            _digest([source_inputs, proof["proof"]]),
            _digest([rule, arithmetic.record()]),
            _digest(
                runtime
                if memory_effects_protocol is None
                else [
                    runtime,
                    {
                        "source_memory_effects_protocol": memory_effects_protocol,
                        "owner_identity": memory_effects.identity,
                    },
                ]
            )
            if binding_protocol == "whole_program"
            else _digest([runtime, _pin(paths["execution_contract"])]),
            library.contract.sha256,
        )
        state = {
            "selected": selected,
            "arithmetic": arithmetic,
            "completion": completion,
            "library": library,
            "library_record": declaration,
            "selected_inputs": source_inputs,
            "arithmetic_identity": arithmetic.identity,
            "completion_identity": completion.identity,
            "runtime": runtime,
            "nm": _pin(paths["nm"]),
            "sources": sources,
            "binding": binding,
            "protocol": COMMIT_ONLY_PROTOCOL,
            "inputs": copy.deepcopy(inputs),
            "input_pins": {
                name: _pin(path) for name, path in paths.items() if path.is_file()
            },
            "trace": observe_current_source,
            "trace_code": observe_current_source.__code__,
            "binding_protocol": binding_protocol,
            "memory_effects_protocol": memory_effects_protocol,
            "memory_effects": memory_effects,
            "memory_effects_identity": None
            if memory_effects is None
            else memory_effects.identity,
            "owner_type": owner_type,
            "owner_init": owner_type.__init__,
            "owner_init_code": owner_type.__init__.__code__,
            "execution_contract": None
            if binding_protocol == "whole_program"
            else _pin(paths["execution_contract"]),
        }
        state["callbacks"] = tuple(
            (
                owner,
                name,
                getattr(type(owner), name),
                getattr(type(owner), name).__code__,
            )
            for owner, names in (
                (selected, ("revalidate", "report")),
                (arithmetic, ("revalidate", "record", "source_rule")),
                (completion, ("revalidate", "record")),
                (library, ("record",)),
            )
            for name in names
        )
        if memory_effects is not None:
            state["callbacks"] += tuple(
                (
                    memory_effects,
                    name,
                    getattr(type(memory_effects), name),
                    getattr(type(memory_effects), name).__code__,
                )
                for name in ("revalidate", "record", "packet_owners")
            )
        state["identity"] = _digest(
            [
                state["input_pins"],
                sources,
                source_inputs,
                arithmetic.identity,
                completion.identity,
                declaration,
                runtime,
                state["nm"],
                asdict(binding),
                binding_protocol,
            ]
            + (
                []
                if memory_effects_protocol is None
                else [
                    {
                        "source_memory_effects_protocol": memory_effects_protocol,
                        "owner_identity": memory_effects.identity,
                    }
                ]
            )
        )
        _ISSUED[self] = state
        try:
            self.revalidate()
        except BaseException:
            del _ISSUED[self]
            raise

    def _original_state(self):
        state = _ISSUED.get(self)
        if type(self) is not GeneratedSourceDiagnosticIssuer or state is None:
            raise ValueError("generated source issuer was not originally issued")
        for name, function, code in _ORIGINAL_API:
            method = getattr(self, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not self
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("original generated-source callback changed")
        _revalidate_sources(state["sources"])
        for owner, name, function, code in state["callbacks"]:
            method = getattr(owner, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not owner
                or method.__func__ is not function
                or function.__code__ is not code
            ):
                raise ValueError("original generated-source semantic callback changed")
        if (
            type(state["selected"]) is not SelectedSchedulingTarget
            or type(state["arithmetic"]) is not IntegerArithmeticOwner
            or type(state["completion"]) is not StagedCompletionOwner
            or type(state["library"]) is not CompilerLibrarySelection
            or _digest(
                [
                    state["input_pins"],
                    state["sources"],
                    state["selected_inputs"],
                    state["arithmetic_identity"],
                    state["completion_identity"],
                    state["library_record"],
                    state["runtime"],
                    state["nm"],
                    asdict(state["binding"]),
                    state["binding_protocol"],
                ]
                + (
                    []
                    if state["memory_effects_protocol"] is None
                    else [
                        {
                            "source_memory_effects_protocol": state[
                                "memory_effects_protocol"
                            ],
                            "owner_identity": state["memory_effects_identity"],
                        }
                    ]
                )
            )
            != state["identity"]
        ):
            raise ValueError("original generated-source state or binding changed")
        if (
            observe_current_source is not state["trace"]
            or observe_current_source.__code__ is not state["trace_code"]
        ):
            raise ValueError("original generated-source trace algorithm changed")
        if (
            state["owner_type"].__init__ is not state["owner_init"]
            or state["owner_init"].__code__ is not state["owner_init_code"]
        ):
            raise ValueError("original generated-source effect constructor changed")
        from gemmini_source_execution import SourceExecutionOwner

        expected_type = SourceExecutionOwner
        if state["binding_protocol"] != "whole_program":
            from gemmini_direct_source_execution import (
                DirectSourceExecutionOwner,
                PointerContractionsExecutionOwner,
            )

            expected_type = (
                PointerContractionsExecutionOwner
                if state["binding_protocol"] == POINTER_CONTRACTIONS_PROTOCOL
                else DirectSourceExecutionOwner
            )
        if (
            state["owner_type"] is not expected_type
            or state["execution_contract"]
            != state["input_pins"].get("execution_contract")
            or state["protocol"] != COMMIT_ONLY_PROTOCOL
            or state["memory_effects_protocol"]
            != state["inputs"].get("source_memory_effects_protocol")
            or (
                state["memory_effects_protocol"] is not None
                and state["binding_protocol"] != "whole_program"
            )
        ):
            raise ValueError("original generated-source execution selection changed")
        if state["memory_effects_protocol"] is None:
            if (
                state["memory_effects"] is not None
                or state["memory_effects_identity"] is not None
            ):
                raise ValueError("unselected source memory-effects owner appeared")
        else:
            from gemmini_whole_program_memory_effects import WholeProgramMemoryEffects

            if (
                type(state["memory_effects"]) is not WholeProgramMemoryEffects
                or state["memory_effects"].identity != state["memory_effects_identity"]
            ):
                raise ValueError("original source memory-effects owner changed")
        return state

    @property
    def identity(self):
        return (
            "gemmini.generated_current_source_diagnostic.v1:" + _STATE(self)["identity"]
        )

    def revalidate(self):
        with shared_compute_pool(timeout_s=120):
            state = _STATE(self)
            state["selected"].revalidate()
            state["arithmetic"].revalidate()
            state["completion"].revalidate()
            if state["memory_effects"] is not None:
                state["memory_effects"].packet_owners(state["selected"])
            _revalidate_sources(
                {**state["input_pins"], **state["runtime"], "nm": state["nm"]}
            )
            if (
                state["selected"].report()["selected_inputs"]
                != state["selected_inputs"]
                or state["arithmetic"].identity != state["arithmetic_identity"]
                or state["completion"].identity != state["completion_identity"]
                or state["library"].record() != state["library_record"]
            ):
                raise ValueError(
                    "original generated-source inputs or live semantic owners changed"
                )
            _STATE(self)

    def source_rule(self):
        self.revalidate()
        return _STATE(self)["arithmetic"].source_rule()

    def selected_binding(self):
        self.revalidate()
        return _STATE(self)["binding"]

    def record(self):
        self.revalidate()
        state = _STATE(self)
        record = {
            "identity": self.identity,
            "selected_inputs": copy.deepcopy(state["selected_inputs"]),
            "sources": copy.deepcopy(state["sources"]),
            "runtime_inputs": copy.deepcopy(state["runtime"]),
            "source_observation_protocol": state["protocol"],
            "source_binding_protocol": state["binding_protocol"],
            "scheduling_qualification": "NONE",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "promotion_authority": "NONE",
        }
        if state["memory_effects_protocol"] is not None:
            record["source_memory_effects_protocol"] = state["memory_effects_protocol"]
            record["source_memory_effects"] = state["memory_effects"].record()
        return record

    def create_source_execution_owner(
        self, *, native_result, command_buffer, lowered, work_root, remaining_s
    ):
        if (
            type(remaining_s) not in (int, float)
            or not math.isfinite(remaining_s)
            or remaining_s <= 0
        ):
            raise ValueError(
                "generated source diagnostics require a positive finite remaining budget"
            )
        return observe_current_source(
            self,
            native_result=native_result,
            command_buffer=command_buffer,
            lowered=lowered,
            work_root=work_root,
            remaining_s=remaining_s,
        )


_STATE = GeneratedSourceDiagnosticIssuer._original_state
_ORIGINAL_API = tuple(
    (
        name,
        getattr(GeneratedSourceDiagnosticIssuer, name),
        getattr(GeneratedSourceDiagnosticIssuer, name).__code__,
    )
    for name in (
        "_original_state",
        "revalidate",
        "record",
        "source_rule",
        "selected_binding",
        "create_source_execution_owner",
    )
)


def restore_registry(inputs, *, work_root):
    # This factory runs only source derivations. It launches no target program.
    work = Path(work_root)
    if (
        not work.is_absolute()
        or work.resolve() != work
        or work.exists()
        or any(p.is_symlink() for p in (work, *work.parents))
    ):
        raise ValueError(
            "generated source factory requires fresh original output custody"
        )
    with shared_compute_pool(timeout_s=120):
        owner = GeneratedSourceDiagnosticIssuer(inputs)
        capability = ReviewedPassCapability(
            "source_execution_diagnostic", owner.selected_binding(), owner
        )
        work.mkdir(parents=True, exist_ok=False)
        return ReviewedPassCapabilities((capability,))
