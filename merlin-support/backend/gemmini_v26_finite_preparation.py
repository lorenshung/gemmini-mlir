"""Explicit owned-copy V26 preparation, separate from final pair qualification."""

from __future__ import annotations

import ast
import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary

import gemmini_generated_source_diagnostic as generated
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_v26_finite_pair import check_source_pair as _consume_source_pair
from gemmini_v26_staged_actual_pair import check_actual_staged_pair
from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import module_source_path
from merlin.common.tree_hash import hash_tree
from merlin.compile.scheduling.staged_dense import StagedDenseContract
from merlin.llvmlower.reviewed_metadata import scoped_registry_metadata_verification
from merlin.llvmlower.reviewed_pass_capabilities import (
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import (
    _digest,
    _tree,
    copy_compiler_source,
)

MECHANISM = "diagnostic_v26_finite_preparation"
ORIGINAL_SHA256 = "29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24"
_ISSUED = WeakKeyDictionary()
_ORIGINAL_BOUNDARY = "        llvm_ir = emit(buffer) if llvm_ir is None else llvm_ir"
_SELECTED_BOUNDARY = """        if llvm_ir is None:
            from finite_staged_hook import emit_if_selected
            llvm_ir = emit_if_selected(Path(inputs[0]), buffer)
        llvm_ir = emit(buffer) if llvm_ir is None else llvm_ir"""
_LEAVES = {
    "finite_staged_hook.py": "gemmini_v26_finite_hook",
    "finite_staged_emitter.py": "gemmini_v26_staged_source",
}


def _methods(owner):
    result = []
    for name in (
        "revalidate",
        "record",
        "selected_binding",
        "prepare_compiler",
        "check_compiler_binding",
        "check_actual_pair",
        "check_source_pair",
    ):
        callback = getattr(owner, name)
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise ValueError("finite preparation lost its original bound callback")
        result.append((name, callback.__func__, callback.__func__.__code__))
    return tuple(result)


def _source_snapshot(registry):
    with scoped_registry_metadata_verification(registry):
        if (
            type(registry) is not ReviewedPassCapabilities
            or len(registry._capabilities) != 1
        ):
            raise ValueError(
                "finite preparation needs its exact single-source registry"
            )
        capability = registry._capabilities[0]
        if (
            capability.mechanism != "source_execution_diagnostic"
            or type(capability.owner) is not generated.GeneratedSourceDiagnosticIssuer
        ):
            raise ValueError(
                "finite preparation needs the original generated-source issuer"
            )
        record = capability.record()
        state = generated._STATE(capability.owner)
        if state["binding_protocol"] != "merlin.native_pointer_source.v1":
            raise ValueError(
                "finite preparation requires explicit original direct-source binding"
            )
        binding = capability.binding
    return capability, record, state, binding


def _contract(state, binding):
    selected, arithmetic, completion = (
        state[name] for name in ("selected", "arithmetic", "completion")
    )
    report, rule, completed = (
        selected.report(),
        arithmetic.source_rule(),
        completion.record(),
    )
    geometry = report["geometry"]
    stores = {row["name"]: row for row in geometry["stores"]}
    dim = geometry["array_rows"]
    if (
        dim != geometry["array_cols"]
        or dim != rule["max_primitive_k"]
        or stores["scratchpad"]["total_rows"] < 2 * dim
        or stores["accumulator"]["total_rows"] < dim
        or completed["selected_inputs"] != report["selected_inputs"]
        or completed["dense_v1_admission"]["enabled"] is not False
    ):
        raise ValueError(
            "finite preparation lacks selected arithmetic/storage/completion premises"
        )
    return StagedDenseContract(
        binding.hardware,
        binding.software,
        binding.numerical_policy,
        binding.runtime,
        dim,
        2 * dim,
        dim,
        rule["operand_bits"],
        rule["primitive_mac_bits"],
        rule["accumulator_bits"],
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        False,
    )


def _policy(value):
    if (
        type(value) is not dict
        or set(value) != {"enabled", "rewrite_policy"}
        or type(value["enabled"]) is not bool
        or type(value["rewrite_policy"]) is not str
        or value["rewrite_policy"] not in {"verify_only", "n_before_m_rhs"}
    ):
        raise ValueError("finite preparation needs explicit bounded policy choices")
    return "def choose_finite_policy():\n    return " + repr(value) + "\n"


def _read_policy(path):
    raw = Path(path).read_bytes()
    if len(raw) > 1024:
        raise ValueError("finite candidate policy exceeds its bound")
    module = ast.parse(raw)
    if len(module.body) != 1 or type(module.body[0]) is not ast.FunctionDef:
        raise ValueError("finite candidate policy has executable or foreign statements")
    function = module.body[0]
    if (
        function.name != "choose_finite_policy"
        or function.decorator_list
        or function.returns
        or function.args.posonlyargs
        or function.args.args
        or function.args.vararg is not None
        or function.args.kwonlyargs
        or function.args.kwarg is not None
        or function.args.defaults
        or function.args.kw_defaults
        or getattr(function, "type_params", ())
        or len(function.body) != 1
        or type(function.body[0]) is not ast.Return
    ):
        raise ValueError("finite candidate policy changed its original choices")
    policy = ast.literal_eval(function.body[0].value)
    _policy(policy)
    return policy


def _original(source):
    source = Path(source)
    _tree(source)
    if any(
        {"build", "__pycache__", ".git"} & set(path.relative_to(source).parts)
        for path in source.rglob("*")
    ):
        raise ValueError("finite original source contains generated state")
    inventory = hash_tree(source)
    if inventory["sha256"] != ORIGINAL_SHA256 or inventory["n_files"] != 16:
        raise ValueError(
            "finite preparation requires its exact immutable original 16-file V26 source"
        )
    files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    text = files["compiler.py"].decode()
    if text.count(_ORIGINAL_BOUNDARY) != 1:
        raise ValueError("finite preparation lost the unique original emitter boundary")
    files["compiler.py"] = text.replace(_ORIGINAL_BOUNDARY, _SELECTED_BOUNDARY).encode()
    return files


class FiniteV26PreparationOwner:
    def __init__(self, registry):
        _implementation()
        if type(self) is not FiniteV26PreparationOwner:
            raise TypeError("finite preparation requires its original owner")
        with shared_compute_pool(timeout_s=120):
            capability, record, state, binding = _source_snapshot(registry)
            self._registry, self._source, self._binding = registry, capability, binding
            self._source_record = record
            self._contract = _contract(state, binding)
            self._sources = {
                name: _pin(module_source_path(name))
                for name in (
                    "gemmini_v26_finite_preparation",
                    "gemmini_v26_finite_factory",
                    "gemmini_v26_finite_hook",
                    "gemmini_v26_finite_pair",
                    "gemmini_v26_staged_source",
                    "gemmini_v26_staged_actual_pair",
                    "gemmini_v26_staged_pair_effects",
                    "gemmini_generated_source_diagnostic",
                    "merlin.targetgen.compiler_pass_selection",
                    "merlin.llvmlower.reviewed_metadata",
                    "merlin.compile.scheduling.direct_source",
                    "merlin.compile.scheduling.source_tokens",
                    "merlin.compile.scheduling.staged_dense",
                    "merlin.compile.scheduling.staged_dense_verification",
                    "merlin.common.tree_hash",
                    "merlin.targetgen.native_pointer_source_audit",
                    "merlin_experiments.phase0.connected_integer_qualification",
                )
            }
            library = state["library"].record()
            required = {
                "merlin.compile.scheduling.direct_source",
                "merlin.compile.scheduling.source_tokens",
                "merlin.compile.scheduling.staged_dense",
            }
            if (
                not required <= set(library["contract"]["public_modules"])
                or library["contract_sha256"] != binding.library
            ):
                raise ValueError(
                    "finite preparation requires the explicit equal public source SDK"
                )
            self._public = {
                "schema": "gemmini.v26_finite_public.v1",
                "contract": asdict(self._contract),
                "library_contract_sha256": binding.library,
                "original_compiler_sha256": ORIGINAL_SHA256,
                "authority": "proposal only; independent current source/packet/output pair required",
            }
            self._identity = _digest([record, self._public, self._sources])
            _ISSUED[self] = (
                registry,
                capability,
                binding,
                copy.deepcopy(
                    (
                        record,
                        self._contract,
                        self._sources,
                        self._public,
                        self._identity,
                    )
                ),
                _methods(self),
            )
            try:
                self.revalidate()
            except BaseException:
                _ISSUED.pop(self, None)
                raise

    @property
    def identity(self):
        return "gemmini.v26_finite_preparation.v1:" + self._identity

    def revalidate(self):
        with shared_compute_pool(timeout_s=120):
            _implementation()
            issued = _ISSUED.get(self)
            if type(self) is not FiniteV26PreparationOwner or issued is None:
                raise ValueError("finite preparation was not originally issued")

            def check():
                if (
                    self._registry is not issued[0]
                    or self._source is not issued[1]
                    or self._binding is not issued[2]
                    or (
                        self._source_record,
                        self._contract,
                        self._sources,
                        self._public,
                        self._identity,
                    )
                    != issued[3]
                    or _methods(self) != issued[4]
                ):
                    raise ValueError(
                        "finite preparation original source/binding/callback changed"
                    )
                _revalidate_sources(self._sources)

            check()
            _implementation()
            source, record, state, binding = _source_snapshot(self._registry)
            if (
                source is not self._source
                or record != self._source_record
                or binding != self._binding
                or _contract(state, binding) != self._contract
                or _digest([record, self._public, self._sources]) != self._identity
            ):
                raise ValueError(
                    "finite preparation selected live semantics/library changed"
                )
            check()

    def selected_binding(self):
        self.revalidate()
        return self._binding

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "mechanism": MECHANISM,
            "binding": asdict(self._binding),
            "sources": copy.deepcopy(self._sources),
            "public": copy.deepcopy(self._public),
            "scheduling_qualification": "NONE",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "promotion_authority": "NONE",
        }

    def prepare_compiler(self, source, destination, *, policy):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            chosen = _policy(policy)
            files = _original(source)
            before = _tree(Path(source))
            destination = copy_compiler_source(Path(source), Path(destination))
            (destination / "compiler.py").write_bytes(files["compiler.py"])
            for name, module in _LEAVES.items():
                shutil.copyfile(self._sources[module]["path"], destination / name)
            (destination / "finite_staged_policy.py").write_text(chosen)
            (destination / "finite-staged-public.json").write_text(
                json.dumps(self._public, sort_keys=True) + "\n"
            )
            (destination / "selected-shared-library.json").write_text(
                json.dumps(
                    generated._STATE(self._source.owner)["library"].record(),
                    sort_keys=True,
                )
                + "\n"
            )
            self.check_compiler_binding(destination)
            if _tree(Path(source)) != before:
                raise ValueError("finite preparation changed its immutable origin")
            self.revalidate()
            return destination

    def check_compiler_binding(self, package):
        with shared_compute_pool(timeout_s=120):
            self.revalidate()
            package = Path(package)
            _tree(package)
            actual = {
                path.relative_to(package).as_posix(): path
                for path in package.rglob("*")
                if path.is_file()
            }
            expected = (
                set(_original_files())
                | set(_LEAVES)
                | {
                    "finite_staged_policy.py",
                    "finite-staged-public.json",
                    "selected-shared-library.json",
                }
            )
            if set(actual) != expected:
                raise ValueError("finite compiler has missing/foreign components")
            original = _original_files()
            original["compiler.py"] = (
                original["compiler.py"]
                .decode()
                .replace(_ORIGINAL_BOUNDARY, _SELECTED_BOUNDARY)
                .encode()
            )
            if any(
                actual[name].read_bytes() != raw for name, raw in original.items()
            ) or any(
                actual[name].read_bytes()
                != Path(self._sources[module]["path"]).read_bytes()
                for name, module in _LEAVES.items()
            ):
                raise ValueError(
                    "finite compiler changed original emitter/candidate hook bytes"
                )
            if (
                json.loads(actual["finite-staged-public.json"].read_bytes())
                != self._public
                or json.loads(actual["selected-shared-library.json"].read_bytes())
                != generated._STATE(self._source.owner)["library"].record()
            ):
                raise ValueError(
                    "finite compiler changed its selected public source/library contract"
                )
            policy = _read_policy(actual["finite_staged_policy.py"])
            self.revalidate()
            return {
                "schema": "gemmini.v26_finite_compiler_binding.v1",
                "prepared_sha256": _tree(package),
                "policy": policy,
                "scheduling_qualification": "NONE",
                "timing_authority": "NONE",
            }

    def check_actual_pair(self, **arguments):
        # Deliberately no metadata transaction or CPU lease spans current proof.
        self.revalidate()
        state = generated._STATE(self._source.owner)
        for owner in (
            arguments.get("baseline_owner"),
            arguments.get("candidate_owner"),
        ):
            if (
                getattr(owner, "_selected", None) is not state["selected"]
                or getattr(owner, "_arithmetic", None) is not state["arithmetic"]
            ):
                raise ValueError(
                    "finite actual pair needs its original current source owners"
                )
        try:
            return check_actual_staged_pair(**arguments, completion=state["completion"])
        finally:
            self.revalidate()

    def check_source_pair(
        self, *, baseline_workspace, candidate_workspace, capsule, remaining_s
    ):
        # This consumes completed original process products only. Neither a
        # metadata scope nor a fresh target process is introduced here.
        self.revalidate()
        state = generated._STATE(self._source.owner)
        try:
            return _consume_source_pair(
                self,
                state,
                baseline_workspace=baseline_workspace,
                candidate_workspace=candidate_workspace,
                capsule=capsule,
                remaining_s=remaining_s,
            )
        finally:
            self.revalidate()


def _original_files():
    # This immutable public source is shipped with this OOT, not a receipt or
    # another host's private package path. Both preparation and binding rehash it.
    source = (
        Path(__file__).resolve().parents[2]
        / "phase1_generated/el4_fire_aligned_v26_original_29213759/package"
    )
    _original(source)
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def with_finite_preparation(registry):
    with shared_compute_pool(timeout_s=120):
        source, _, _, binding = _source_snapshot(registry)
        owner = FiniteV26PreparationOwner(registry)
        return ReviewedPassCapabilities(
            (source, ReviewedPassCapability(MECHANISM, binding, owner))
        )


def _implementation():
    if (
        _pin(__file__) != _IMPORTED_SOURCE
        or _digest(
            [
                MECHANISM,
                ORIGINAL_SHA256,
                _ORIGINAL_BOUNDARY,
                _SELECTED_BOUNDARY,
                _LEAVES,
            ]
        )
        != _CONSTANTS
        or generated._STATE is not _SOURCE_STATE
        or _SOURCE_STATE.__code__ is not _SOURCE_CODE
        or FiniteV26PreparationOwner.identity.fget is not _IDENTITY
        or _IDENTITY.__code__ is not _IDENTITY_CODE
        or any(
            globals()[name] is not function or function.__code__ is not code
            for name, function, code in _FUNCTIONS
        )
        or any(
            getattr(FiniteV26PreparationOwner, name) is not function
            or function.__code__ is not code
            for name, function, code in _ORIGINAL_API
        )
    ):
        raise ValueError("finite preparation original implementation changed")


_SOURCE_STATE, _SOURCE_CODE = generated._STATE, generated._STATE.__code__
_CONSTANTS = _digest(
    [MECHANISM, ORIGINAL_SHA256, _ORIGINAL_BOUNDARY, _SELECTED_BOUNDARY, _LEAVES]
)
_IMPORTED_SOURCE = _pin(__file__)
_ORIGINAL_API = tuple(
    (
        name,
        getattr(FiniteV26PreparationOwner, name),
        getattr(FiniteV26PreparationOwner, name).__code__,
    )
    for name in (
        "__init__",
        "revalidate",
        "record",
        "selected_binding",
        "prepare_compiler",
        "check_compiler_binding",
        "check_actual_pair",
        "check_source_pair",
    )
)
_IDENTITY = FiniteV26PreparationOwner.identity.fget
_IDENTITY_CODE = _IDENTITY.__code__
_FUNCTIONS = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "_methods",
        "_source_snapshot",
        "_contract",
        "_policy",
        "_read_policy",
        "_original",
        "_original_files",
        "with_finite_preparation",
        "_implementation",
        "check_actual_staged_pair",
        "_consume_source_pair",
        "hash_tree",
        "copy_compiler_source",
        "_pin",
        "_revalidate_sources",
        "_tree",
    )
)
