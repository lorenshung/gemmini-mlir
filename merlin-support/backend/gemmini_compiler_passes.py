"""Live cleanup preparation before the ordinary compiler source/edit freeze.

Only exact reviewed public leaves enter the copied compiler. Original host
qualification owners and private evidence remain outside candidate access.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class CompilerCleanupOwner:
    def __init__(self, cleanup, library, *, original_host_roots):
        from gemmini_cleanup_admission import OriginalCleanupAdmission
        from gemmini_timing_resources import _pin
        from merlin.common.paths import module_source_path
        from merlin.targetgen.compiler_library import CompilerLibrarySelection

        if (
            type(cleanup) is not OriginalCleanupAdmission
            or type(library) is not CompilerLibrarySelection
        ):
            raise ValueError(
                "compiler preparation needs original live cleanup/library owners"
            )
        self.cleanup, self.library = cleanup, library
        self._host_roots = tuple(Path(path) for path in original_host_roots)
        if any(
            not path.is_absolute()
            or path.is_symlink()
            or path.resolve(strict=True) != path
            for path in self._host_roots
        ):
            raise ValueError(
                "compiler preparation requires explicit canonical original dependency roots"
            )
        self._cleanup_identity, self._library = cleanup.identity, library.record()
        self._sources = {
            "owner": _pin(__file__),
            "admission_bridge": _pin(module_source_path("gemmini_cleanup_admission")),
            "hook": _pin(Path(__file__).with_name("gemmini_compiler_pass_hook.py")),
            "cleanup": _pin(module_source_path("gemmini_quiescent_load_config")),
            "startup": _pin(
                module_source_path("merlin.targetgen.selected_library_imports")
            ),
        }
        self.identity = "gemmini.compiler_cleanup.v1:" + _digest(
            [
                self._cleanup_identity,
                self._library,
                self._sources,
                list(map(str, self._host_roots)),
            ]
        )
        self.revalidate()

    def revalidate(self):
        from gemmini_timing_resources import _pin

        self.cleanup.revalidate()
        if (
            self.cleanup.identity != self._cleanup_identity
            or self.library.record() != self._library
        ):
            raise ValueError("original cleanup or selected library changed")
        if any(_pin(pin["path"]) != pin for pin in self._sources.values()):
            raise ValueError("original compiler preparation source changed")
        if (
            self.cleanup.selected_binding()["compiler_library_selection"]
            != self._library
        ):
            raise ValueError("cleanup qualification selected a different library")

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "sources": self._sources,
            "original_host_roots": list(map(str, self._host_roots)),
            "cleanup": self.cleanup.record(),
            "library": self._library,
            "scope": "pre-freeze compiler preparation; independent final realization required",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        if (
            set(policy) != {"enabled", "min_macs"}
            or type(policy["enabled"]) is not bool
            or type(policy["min_macs"]) is not int
            or policy["min_macs"] < 0
        ):
            raise ValueError(
                "cleanup policy requires enabled and a nonnegative shape threshold"
            )
        digest = hashlib.sha256()
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("original compiler contains a linked source")
            if path.is_file():
                digest.update(
                    path.relative_to(source).as_posix().encode()
                    + b"\0"
                    + path.read_bytes()
                    + b"\0"
                )
        qualified = self.cleanup.revalidate()
        if digest.hexdigest() != qualified["original_compiler_sha256"]:
            raise ValueError(
                "compiler preparation did not receive its original qualified Phase 1 compiler"
            )
        shutil.copytree(source, destination)
        contract = json.loads(Path(qualified["effect_contract"]["path"]).read_bytes())
        resources = destination / "cleanup-public-leaves"
        resources.mkdir()
        size = 0

        def project(pin):
            nonlocal size
            path = Path(pin["path"])
            payload = path.read_bytes()
            size += len(payload)
            if size > 1 << 20 or hashlib.sha256(payload).hexdigest() != pin["sha256"]:
                raise ValueError(
                    "public cleanup source changed or exceeds its projection bound"
                )
            name = pin["sha256"] + "-" + path.name
            (resources / name).write_bytes(payload)
            return {"path": "cleanup-public-leaves/" + name, "sha256": pin["sha256"]}

        public = {
            **contract,
            "runtime": project(contract["runtime"]),
            "selected_inputs": {
                name: project(pin) for name, pin in contract["selected_inputs"].items()
            },
            "sources": [project(pin) for pin in contract["sources"]],
        }
        (destination / "cleanup-public-inputs.json").write_text(
            json.dumps(
                {
                    "schema": "gemmini.cleanup_public_inputs.v1",
                    "origin_contract_sha256": qualified["effect_contract"]["sha256"],
                    "contract": public,
                },
                sort_keys=True,
            )
        )
        for key, name in (
            ("hook", "gemmini_compiler_pass_hook.py"),
            ("cleanup", "gemmini_quiescent_load_config.py"),
            ("startup", "_selected_shared_sources.py"),
        ):
            shutil.copyfile(self._sources[key]["path"], destination / name)
        # The original package facade is an implementation namespace, not an
        # additional-library public API. Preserve the algorithm while naming
        # its already reviewed exact public leaves for integrity admission.
        helper = destination / "gemmini_quiescent_load_config.py"
        text = helper.read_text()
        original_import = "from merlin.sched.ir import Kernel, call, evaluate"
        if text.count(original_import) != 1:
            raise ValueError("original cleanup import boundary changed")
        helper.write_text(
            text.replace(
                original_import,
                "from merlin.sched.ir.kernel import Kernel, call\nfrom merlin.sched.ir.expr import evaluate",
            )
        )
        (destination / "selected-shared-library.json").write_text(
            json.dumps(self.library.record(), sort_keys=True)
        )
        (destination / "shared-library-sources.json").write_text(
            json.dumps(qualified["shared_library_sources"], sort_keys=True)
        )
        (destination / "cleanup_policy.py").write_text(
            '"""Candidate-owned shape policy; independent checks remain fixed."""\n'
            f"ENABLED = {policy['enabled']!r}\nMIN_MACS = {policy['min_macs']!r}\n"
            "def choose_cleanup(shape):\n    m, k, n = shape\n    return ENABLED and m * k * n >= MIN_MACS\n"
        )
        driver = destination / "compiler.py"
        text = driver.read_text()
        marker = "\ndef main():\n"
        if text.count(marker) != 1:
            raise ValueError("original compiler has no unique startup")
        roots = ",".join(f"Path({str(path)!r})" for path in self._host_roots)
        startup = (
            '\nfrom _selected_shared_sources import activate_selected_sources\nactivate_selected_sources(json.loads(Path(__file__).with_name("selected-shared-library.json").read_text()), original_host_roots=('
            + roots
            + ",))\n"
        )
        driver.write_text(text.replace(marker, startup + marker))
        direct = destination / "direct_codegen.py"
        text = direct.read_text()
        marker = "return declarations, statements"
        if text.count(marker) != 1:
            raise ValueError("original compiler has no unique direct schedule boundary")
        direct.write_text(
            text.replace(
                marker,
                "return declarations, _cleanup_statements(statements, shape=(M,K,N))",
            )
            + "\nfrom gemmini_compiler_pass_hook import rewrite as _cleanup_statements\n"
        )
        self.revalidate()
        return destination

    def check_actual_pair(self, before, after, *, execution_domain):
        self.revalidate()
        result = self.cleanup.check_actual_pair(
            before, after, execution_domain=execution_domain
        )
        self.revalidate()
        return result


def capability_binding(owner):
    from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding

    owner.revalidate()
    row = owner.cleanup.revalidate()
    return PassCapabilityBinding(
        row["selected_inputs"]["facts"]["sha256"],
        _digest([row["selected_inputs"]["observed_contract"], row["effect_contract"]]),
        "unchanged_original_compiler:" + row["original_compiler_sha256"],
        _digest(row["runtime_inputs"]),
        owner.library.contract.sha256,
    )


def restore_registry(inputs, *, work_root):
    """Original host factory: rerun full independent admission, never read a verdict."""
    import sys

    support = Path(__file__).resolve().parent
    sys.path.insert(0, str(support))
    sys.path.insert(
        0, str(support.parents[1] / "experiments/shared_load_configuration")
    )
    from gemmini_cleanup_admission import run_original_admission
    from merlin.llvmlower.reviewed_pass_capabilities import (
        ReviewedPassCapabilities,
        ReviewedPassCapability,
    )
    from merlin.targetgen.compiler_library import (
        CompilerLibraryContract,
        CompilerLibrarySelection,
        LibraryMember,
    )

    # Original host verification uses its fixed checked implementation. Do not
    # replace an evaluator's already loaded namespace with compiler SDK code.
    # Admission independently compares every shared source digest; only copied
    # compiler startup loads the exact selected snapshot before compiler imports.
    declaration = json.loads(Path(inputs["library_selection"]).read_bytes())
    record = declaration["selection"]
    contract = record["contract"]
    library = CompilerLibrarySelection(
        CompilerLibraryContract(
            contract["review_id"],
            tuple(contract["public_modules"]),
            tuple(LibraryMember(**row) for row in contract["members"]),
        ),
        Path(record["root"]),
    )
    if library.record() != record:
        raise ValueError("original reviewed library selection changed")
    cleanup = run_original_admission(inputs, work_root=work_root, library=library)
    owner = CompilerCleanupOwner(
        cleanup, library, original_host_roots=(Path(inputs["merlin"]) / "src",)
    )
    return ReviewedPassCapabilities(
        (
            ReviewedPassCapability(
                "configuration_cleanup", capability_binding(owner), owner
            ),
        )
    )
