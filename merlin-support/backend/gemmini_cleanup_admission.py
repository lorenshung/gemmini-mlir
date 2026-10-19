"""Live original admission subprocess; saved records cannot issue this owner."""

from __future__ import annotations

import json
from pathlib import Path

_ISSUED = object()


class OriginalCleanupAdmission:
    def __init__(self, receipt, record, invocation, selected, library, *, _issuer):
        if _issuer is not _ISSUED:
            raise ValueError("rerun original independent cleanup admission")
        from gemmini_timing_resources import _pin

        self.path, self.invocation = Path(receipt), Path(invocation)
        self.selected, self.library = selected, library
        self.payload, self._record = self.path.read_bytes(), record
        self.identity = record["identity"]
        self._pins = (_pin(self.path), _pin(self.invocation))
        self.revalidate()

    def revalidate(self):
        from gemmini_timing_resources import _pin, _revalidate_sources
        from merlin.common import invocation_record

        self.selected.revalidate()
        invocation = invocation_record.verify(self.invocation)
        if (
            tuple(_pin(path) for path in (self.path, self.invocation)) != self._pins
            or self.path.read_bytes() != self.payload
            or json.loads(Path(invocation["stdout"]["path"]).read_bytes())
            != self._record
            or self._record["live_host_admission"]["compiler_library_selection"]
            != self.library.record()
        ):
            raise ValueError("original issued cleanup admission changed")
        row = json.loads(self.payload)
        if (
            row["schema"] != "gemmini.fenced_load_cleanup_capability.v1"
            or row["admission"] != "conditional_final_pair_required"
            or row["hardware_execution_qualification"] != "not_established"
            or row["timing_authority"] != "NONE"
            or row["dense_v1_enabled"] is not False
            or row["compiler_library_selection"] != self.library.record()
            or row["selected_inputs"] != self.selected.report()["selected_inputs"]
            or not row["checks"]
            or not all(value is True for value in row["checks"].values())
        ):
            raise ValueError("original independent cleanup admission is incomplete")
        _revalidate_sources(row["pinned_inputs"])
        return row

    def selected_binding(self):
        self.revalidate()
        return {"compiler_library_selection": self.library.record()}

    def record(self):
        self.revalidate()
        return self._record

    def check_actual_pair(self, before, after, *, execution_domain):
        if execution_domain != "functional_spike":
            raise ValueError(
                "cleanup realization requires its selected functional domain"
            )
        import qualify
        from gemmini_quiescent_load_config import LoadConfigurationContract

        row = self.revalidate()
        contract = LoadConfigurationContract.read(row["effect_contract"]["path"])
        removed = qualify._check_omissions(
            before, after, contract.revalidate()["encoding"]["config_rs1"]
        )
        self.revalidate()
        return {
            "actual_setters_removed": removed,
            "other_issued_commands_and_fences_unchanged": True,
            "numerical_and_memory_checks": "required independently for this realization",
            "timing_authority": "NONE",
        }


def run_original_admission(inputs, *, work_root, library):
    """Run the unchanged SDK-isolated checker, outside any parent CPU lease."""
    import os
    import sys

    from gemmini_selected_scheduling import SelectedSchedulingTarget
    from merlin.common import cpu_pool, invocation_record

    if cpu_pool.current_thread_owns_cpu_lease():
        raise ValueError(
            "release parent compute lease before original admission child acquires its own"
        )
    work_root.mkdir(parents=True, exist_ok=False)
    receipt = work_root / "independently-rechecked-receipt.json"
    driver = (
        Path(__file__).resolve().parents[2]
        / "experiments/shared_load_configuration/admit.py"
    )
    argv = [
        sys.executable,
        "-B",
        str(driver),
        "--merlin",
        inputs["merlin"],
        "--compiler-library-selection",
        inputs["library_selection"],
        "--output",
        str(receipt),
    ]
    for key, flag in (
        ("facts", "facts"),
        ("observed_contract", "observed-contract"),
        ("effect_contract", "effect-contract"),
        ("completion_contract", "completion-contract"),
        ("nm", "nm"),
    ):
        argv += ["--" + flag, inputs[key]]
    for report in inputs["reports"]:
        argv += ["--report", report]
    completed = invocation_record.run(
        argv,
        directory=work_root,
        stage="original_cleanup_independent_admission",
        inputs=tuple(
            Path(inputs[key])
            for key in (
                "facts",
                "observed_contract",
                "effect_contract",
                "completion_contract",
                "library_selection",
            )
        )
        + tuple(Path(path) for path in inputs["reports"]),
        dependencies=(driver, Path(__file__)),
        outputs=(receipt,),
        env=cpu_pool.child_environment(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode:
        raise ValueError(
            "original isolated cleanup admission refused: " + completed.stderr[-1500:]
        )
    invocations = tuple((work_root / "invocations").glob("*/invocation.json"))
    if len(invocations) != 1:
        raise ValueError("original admission lacks its unique actual subprocess issue")
    return OriginalCleanupAdmission(
        receipt,
        json.loads(completed.stdout),
        invocations[0],
        SelectedSchedulingTarget.read(
            Path(inputs["facts"]), Path(inputs["observed_contract"])
        ),
        library,
        _issuer=_ISSUED,
    )
