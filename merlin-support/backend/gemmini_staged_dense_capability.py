"""Original staged raw-i32 capability, conditional on every final realization.

Source-derived arithmetic/completion constrain a prototype; only original
packet/source/output checks can issue its functional capability. Saved success
records or compiler-authored plans cannot construct this owner.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml
from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_compiler_passes import CompilerCleanupOwner
from gemmini_source_execution import SourceExecutionOwner
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.compile.scheduling.staged_dense import StagedDenseContract
from merlin.llvmlower.source_contraction_bindings import bind_rank_two_contractions
from merlin_experiments.phase0.connected_integer_qualification import source_proof


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _leaf(arm, key):
    row = arm.get(key)
    if type(row) is not dict or _pin(row["path"]) != {
        name: row[name] for name in ("path", "sha256")
    }:
        raise ValueError("staged final pair lacks exact original artifact " + key)
    return Path(row["path"])


class StagedDensePrototype:
    """Explicit original-host preparation, without a serialized qualification."""

    def __init__(self, cleanup, arithmetic, completion):
        from gemmini_staged_dense_admission import StagedCompletionOwner

        if (
            type(cleanup) is not CompilerCleanupOwner
            or type(arithmetic) is not IntegerArithmeticOwner
            or type(completion) is not StagedCompletionOwner
        ):
            raise ValueError(
                "staged source needs original checked cleanup/arithmetic owners"
            )
        self.cleanup_owner, self.cleanup, self.arithmetic = (
            cleanup,
            cleanup.cleanup,
            arithmetic,
        )
        self.library, self.selected = cleanup.library, cleanup.cleanup.selected
        self.completion = completion
        self._completion = completion.record()
        self._sources = {
            name: _pin(module_source_path(name))
            for name in (
                "gemmini_staged_dense_capability",
                "gemmini_staged_dense_hook",
                "gemmini_staged_dense_source",
                "gemmini_staged_rhs_effects",
                "gemmini_source_execution",
                "gemmini_source_packet_artifacts",
                "gemmini_packet_contractions",
                "merlin.llvmlower.source_contraction_bindings",
                "merlin_experiments.phase0.connected_integer_qualification",
            )
        }
        self._cleanup_record, self._arithmetic_record = (
            cleanup.record(),
            arithmetic.record(),
        )
        rule, geometry = arithmetic.source_rule(), self.selected.report()["geometry"]
        stores = {row["name"]: row for row in geometry["stores"]}
        dim = geometry["array_rows"]
        if (
            geometry["array_cols"] != dim
            or rule["max_primitive_k"] != dim
            or stores["scratchpad"]["total_rows"] < 2 * dim
            or stores["accumulator"]["total_rows"] < dim
            or self._completion.get("schema") != "gemmini.staged_completion_contract.v1"
            or self._completion.get("selected_inputs")
            != self.selected.report()["selected_inputs"]
            or self._completion.get("dense_v1_admission", {}).get("enabled")
            is not False
        ):
            raise ValueError(
                "selected staged source geometry/completion correspondence is unavailable"
            )
        # These are the original emitter's assigned temporary windows, not all
        # hardware rows declared free. The transformation allocates no new rows.
        self.contract = StagedDenseContract(
            self.selected.report()["selected_inputs"]["facts"]["sha256"],
            _digest(
                [self.selected.report()["selected_inputs"], self._completion["proof"]]
            ),
            _digest([rule, self._arithmetic_record]),
            _digest(self.cleanup.revalidate()["runtime_inputs"]),
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
        self._contract = asdict(self.contract)
        self._identity = _digest(
            [
                self._sources,
                self._cleanup_record,
                self._arithmetic_record,
                self._completion,
                self._contract,
            ]
        )
        self.revalidate()

    @property
    def nm(self):
        return self.cleanup_owner.nm

    @property
    def identity(self):
        return "gemmini.staged_dense_prototype.v1:" + self._identity

    def revalidate(self):
        self.cleanup_owner.revalidate()
        self.arithmetic.revalidate()
        self.completion.revalidate()
        _revalidate_sources(self._sources)
        _revalidate_sources(self._completion["sources"])
        if (
            self.cleanup_owner.record() != self._cleanup_record
            or self.arithmetic.record() != self._arithmetic_record
            or asdict(self.contract) != self._contract
            or self.completion.record() != self._completion
            or _digest(
                [
                    self._sources,
                    self._cleanup_record,
                    self._arithmetic_record,
                    self._completion,
                    self._contract,
                ]
            )
            != self._identity
        ):
            raise ValueError(
                "original staged source/effect/arithmetic selection changed"
            )

    def record(self):
        self.revalidate()
        return {
            "identity": self.identity,
            "sources": self._sources,
            "contract": self._contract,
            "library": self.library.record(),
            "scope": "unqualified prototype; every final native source/packet/output pair required",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
            "dense_v1_enabled": False,
        }

    def prepare_compiler(self, source, destination, *, policy):
        self.revalidate()
        if (
            set(policy) != {"enabled", "min_m"}
            or type(policy["enabled"]) is not bool
            or type(policy["min_m"]) is not int
            or policy["min_m"] < 0
        ):
            raise ValueError(
                "staged policy requires Boolean enable and nonnegative source M threshold"
            )
        # Isolate this mechanism: the same cleanup library is available, with
        # its policy disabled in both treatments. Do not mix two changes here.
        self.cleanup_owner.prepare_compiler(
            source, destination, policy={"enabled": False, "min_macs": 0}
        )
        for module in ("gemmini_staged_dense_hook", "gemmini_staged_dense_source"):
            shutil.copyfile(
                self._sources[module]["path"], destination / (module + ".py")
            )
        (destination / "staged-dense-public.json").write_text(
            json.dumps(
                {
                    "schema": "gemmini.staged_dense_public.v1",
                    "contract": self._contract,
                    "library_contract_sha256": self.library.contract.sha256,
                },
                sort_keys=True,
            )
        )
        (destination / "staged_dense_policy.py").write_text(
            '"""Candidate-owned source/resource policy; original final checks stay fixed."""\n'
            f"ENABLED={policy['enabled']!r}\nMIN_M={policy['min_m']!r}\n"
            "def choose_staged_dense(shape, contract):\n    m,k,n=shape\n    return ENABLED and m>=MIN_M and m>contract.dim and k<=contract.dim and n<=contract.dim\n"
        )
        path = destination / "direct_codegen.py"
        text = path.read_text()
        before = "return declarations, _cleanup_statements(statements, shape=(M,K,N))"
        if text.count(before) != 1:
            raise ValueError(
                "staged preparation lost its unique original actual emitter boundary"
            )
        path.write_text(
            text.replace(
                before,
                "return declarations, _cleanup_statements(_staged_statements(statements, declarations=declarations, lhs=a, rhs=b, output=c, shape=(M,K,N)), shape=(M,K,N))",
            )
            + "\nfrom gemmini_staged_dense_hook import rewrite as _staged_statements\n"
        )
        self.revalidate()
        return destination

    def _source_arm(self, arm):
        self.revalidate()
        result = _leaf(arm, "native_result")
        cb_path, lowered, invocation = (
            _leaf(arm, key)
            for key in ("command_buffer_artifact", "lowered", "issue_invocation")
        )
        row, cb = json.loads(result.read_bytes()), json.loads(cb_path.read_bytes())
        from merlin.targetgen.native_model_execution import _pinned_native_file

        cap = _pinned_native_file(row["source"], "capsule_declaration").parent
        proof = source_proof(yaml.safe_load((cap / "capsule.yaml").read_bytes()), cap)
        source = cap / "capsule.interface.mlir"
        contractions = bind_rank_two_contractions(source, cb, proof)
        if (
            len(contractions) != 1
            or contractions[0]["rhs"]
            not in cb["params"]["global_program_plan"]["entry_bindings"]
        ):
            raise ValueError(
                "initial staged capability requires one contraction and original immutable external RHS"
            )
        checked = SourceExecutionOwner(
            arithmetic=self.arithmetic,
            selected=self.selected,
            result=result,
            command_buffer=cb_path,
            lowered=lowered,
            trace_invocation=invocation,
            nm=self.nm,
        ).verify_source_execution(source, proof, cap)
        checked["source_geometry"] = {
            name: contractions[0][name] for name in ("m", "k", "n")
        }
        args = cb["kernel_abi"]["args"]
        owners = [
            "tensor_" + str(i)
            for i, item in enumerate(args)
            if item["tensor"] == contractions[0]["rhs"]
        ]
        if len(owners) != 1:
            raise ValueError(
                "staged RHS has no unique original immutable physical ABI owner"
            )
        import qualify

        observed = invocation_record.verify(invocation)
        elf = _pinned_native_file(row, "elf")
        trace, events = qualify._events(
            Path(observed["stderr"]["path"]).read_text(),
            self.selected.operand_observer().isa_constants("gemmini"),
            qualify._symbols(elf, self.nm),
        )
        decoded, commands = [], iter(trace["instructions"])
        for event in events:
            decoded.append((event, None if event[0] == "FENCE" else next(commands)))
        return decoded, owners[0], checked, cb

    def check_pair_artifacts(self, before, after, *, execution_domain):
        if execution_domain != "functional_spike":
            raise ValueError(
                "staged final realization has no qualified nonfunctional timing domain"
            )
        baseline, rhs, baseline_source, base_cb = self._source_arm(before)
        candidate, candidate_rhs, candidate_source, cand_cb = self._source_arm(after)
        if (
            rhs != candidate_rhs
            or baseline_source["source_sha256"] != candidate_source["source_sha256"]
        ):
            raise ValueError("staged pair changed original source or RHS ownership")
        from gemmini_cleanup_final_guard import _logical

        if _logical(base_cb) != _logical(cand_cb):
            raise ValueError(
                "staged pair changed original complete logical work/numerical contract"
            )
        from gemmini_staged_rhs_effects import check_rhs_omissions

        actual = check_rhs_omissions(
            baseline,
            candidate,
            rhs=rhs,
            operand_rows=self.contract.operand_rows,
            dim=self.contract.dim,
        )
        self.revalidate()
        return {
            "status": "checked_actual_staged_rhs_pair",
            "actual_rhs_loads_removed": actual["actual_rhs_loads_removed"],
            "original_source_execution": [baseline_source, candidate_source],
            "all_other_commands_fences_and_settings_unchanged": True,
            "rhs_immutability": "original complete source entry bytes and observed CPU writes rechecked",
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }

    def check_functional_guard(
        self, baseline, candidate, baseline_result, candidate_result, *, grade_root
    ):
        from gemmini_cleanup_final_guard import check_functional_guard

        return check_functional_guard(
            self,
            baseline,
            candidate,
            baseline_result,
            candidate_result,
            grade_root=grade_root,
        )
