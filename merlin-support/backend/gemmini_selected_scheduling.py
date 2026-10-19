"""Explicit selected-input boundary for shared scheduling target adapters.

This is a trusted OOT intake/decoder, not a compiler seed or a qualified device
adapter. It consumes the selected observed contract and its operand sidecar,
never the curated backend's implicit target selection. Structural widths and
observed operands do not establish instruction effects or completion semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


class SchedulingUnavailable(ValueError):
    """Selected inputs cannot establish a requested scheduling capability."""


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(path, limit=4 << 20):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("selected scheduling input exceeds its byte limit")
    document = json.loads(payload)
    if type(document) is not dict:
        raise ValueError("selected scheduling input must be a JSON object")
    return path, payload, document


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _unique(rows, name):
    selected = [row for row in rows if row.get("name") == name]
    if len(selected) != 1:
        raise ValueError(f"selected facts require exactly one {name}")
    return selected[0]


_MECHANISMS = {
    "dense_blocking_reuse": (
        "dense",
        "typed dense arithmetic, initialization, exact continuation and completion",
    ),
    "temporary_storage_copy_elimination": (
        "storage",
        "physical backing aliases, byte addressing, reservations and actual memory effects",
    ),
    "readout_fusion": (
        "fusion",
        "complete scalar arithmetic/layout and actual readout operand/effect bindings",
    ),
    "configuration_cleanup": (
        "configuration",
        "ordered persistent writes with no pulse, enqueue, completion or other effect",
    ),
    "device_spills": (
        "spills",
        "exact synchronous device save/reload and actual byte provenance",
    ),
    "complete_row_reduction": (
        "row_reduction",
        "selected complete-row numerical algorithm and executable phase primitives",
    ),
    "final_program_limits": (
        "program",
        "final linked-machine decoding, address/branch limits and continuation interfaces",
    ),
    "asynchronous_overlap": (
        None,
        "dependency/completion effects, queue ownership and a checked asynchronous scheduler",
    ),
}


@dataclass(frozen=True)
class SelectedSchedulingTarget:
    """Immutable selected facts, observed contract and exact operand sidecar."""

    facts_path: Path
    facts_bytes: bytes
    contract_path: Path
    contract_bytes: bytes
    operand_path: Path
    operand_bytes: bytes
    _implementation_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, "_implementation_sha256", _sha(Path(__file__).read_bytes())
        )
        self._validate(*self._documents())

    @classmethod
    def read(cls, facts_path, contract_path):
        facts_path, facts_bytes, _ = _read(facts_path, 64 << 20)
        contract_path, contract_bytes, contract = _read(contract_path)
        sidecar = contract.get("rocc_operand_selection", {}).get("sidecar", {})
        if (
            type(sidecar) is not dict
            or not sidecar.get("path")
            or not sidecar.get("sha256")
        ):
            raise ValueError(
                "observed contract requires its exact operand sidecar selection"
            )
        operand_path, operand_bytes, _ = _read(sidecar["path"])
        if _sha(operand_bytes) != sidecar["sha256"]:
            raise ValueError(
                "selected operand sidecar digest differs from observed contract"
            )
        selected = cls(
            facts_path,
            facts_bytes,
            contract_path,
            contract_bytes,
            operand_path,
            operand_bytes,
        )
        return selected

    def _documents(self):
        return tuple(
            json.loads(payload)
            for payload in (self.facts_bytes, self.contract_bytes, self.operand_bytes)
        )

    def _validate(self, facts, contract, operands):
        hardware = facts.get("facts", {})
        if (
            contract.get("name") != "gemmini"
            or hardware.get("target") != "gemmini"
            or operands.get("target") != "gemmini"
        ):
            raise ValueError("selected target identities differ")
        if facts.get("source_consistency", {}).get("status") != "verified":
            raise ValueError("selected facts lack verified source consistency")
        selected_fir = facts.get("inputs", {}).get("fir_sha256")
        if not selected_fir or selected_fir != contract.get("rtl_source", {}).get(
            "artifact_sha256", {}
        ).get("fir"):
            raise ValueError(
                "selected hardware and observed contract FIRRTL identities differ"
            )
        if facts.get("source_consistency", {}).get("config") != contract.get(
            "rtl_source", {}
        ).get("config"):
            raise ValueError("selected hardware configuration names differ")
        original = (
            contract.get("rocc_operand_selection", {})
            .get("original_facts", {})
            .get("sha256")
        )
        if (
            not original
            or original != operands.get("original_facts_sha256")
            or original != _sha(self.facts_bytes)
        ):
            raise ValueError("operand sidecar original-facts binding differs")
        layout = operands.get("layouts", {})
        if layout.get("source_sha256") != selected_fir:
            raise ValueError("operand field extraction belongs to different FIRRTL")
        table = _unique(hardware.get("interfaces", []), "funct_decode_table")
        declared = (
            operands.get("protocol", {}).get("encoding", {}).get("semantic_class", {})
        )
        if set(map(int, declared)) != set(table.get("legal_funct", [])):
            raise ValueError(
                "operand protocol and observed legal-function roster differ"
            )
        _integer(table.get("custom_opcode"), "selected custom opcode", minimum=1)
        self._geometry(hardware)

    def revalidate(self):
        if _sha(Path(__file__).read_bytes()) != self._implementation_sha256:
            raise ValueError("selected scheduling adapter changed; reselect before use")
        for path, payload in (
            (self.facts_path, self.facts_bytes),
            (self.contract_path, self.contract_bytes),
            (self.operand_path, self.operand_bytes),
        ):
            if path.read_bytes() != payload:
                raise ValueError(
                    "selected scheduling inputs changed; reselect before use"
                )

    @staticmethod
    def _geometry(hardware):
        mesh = _unique(hardware.get("arrays", []), "mesh")
        rows, cols = (
            _integer(mesh.get(axis), "mesh " + axis, minimum=1)
            for axis in ("rows", "cols")
        )
        stores = []
        for name in ("scratchpad", "accumulator"):
            memory = _unique(hardware.get("memory_address_spaces", []), name)
            banks = _integer(memory.get("banks"), name + " banks", minimum=1)
            depth = _integer(memory.get("depth"), name + " depth", minimum=1)
            capacity = _integer(memory.get("bytes"), name + " bytes", minimum=1)
            if capacity % (banks * depth):
                raise ValueError(
                    "selected memory capacity is not a complete number of physical rows"
                )
            stores.append(
                {
                    "name": name,
                    "banks": banks,
                    "rows_per_bank": depth,
                    "total_rows": banks * depth,
                    "row_bytes": capacity // (banks * depth),
                    "capacity_bytes": capacity,
                    "physical_aliases_status": memory.get(
                        "physical_aliases_status", "unknown"
                    ),
                }
            )
        return {
            "array_rows": rows,
            "array_cols": cols,
            "stores": stores,
            "qualification": "structural_dimensions_only; not usable allocation or timing",
        }

    def operand_observer(self):
        """Use Merlin's existing source-rederived observer, without target discovery.

        Its original loader verifies controller/ISA source pins and re-derives
        every selected FIRRTL field. This deliberately does not duplicate that
        parser or reinterpret reserved packing slots as actual field widths.
        """
        from merlin.targetgen.rocc.selected_operand_protocol import (
            bind_original_operand_selection,
        )
        from merlin.targetgen.rocc.semantics import BoundSemantics

        self.revalidate()
        facts, contract, _ = self._documents()
        original = BoundSemantics(target="gemmini", contract=contract, facts=facts)
        observer = bind_original_operand_selection(
            original,
            original_facts_bytes=self.facts_bytes,
            selection_bytes=self.operand_bytes,
            expected_selection_sha256=_sha(self.operand_bytes),
        )
        self.revalidate()
        return observer

    def decode_observation(self, instruction_word, rs1, rs2):
        """Decode an actual instruction plus its observed register values.

        A RoCC instruction word does not contain its scalar operand values. The
        caller must supply the values observed at issue; this is not a static
        ELF evaluator, complete machine decoder or execution qualification.
        RISC-V instruction field locations are architectural encoding here;
        the selected custom opcode and function classes come from RTL facts.
        """
        self.revalidate()
        instruction_word = _integer(instruction_word, "instruction word")
        if instruction_word >= 1 << 32:
            raise ValueError("instruction word exceeds architectural width")
        facts, _, operands = self._documents()
        table = _unique(facts["facts"]["interfaces"], "funct_decode_table")
        if instruction_word & ((1 << 7) - 1) != table["custom_opcode"]:
            raise ValueError("instruction does not use the selected custom opcode")
        function = instruction_word >> 25
        name = operands["protocol"]["encoding"]["semantic_class"].get(str(function))
        if name is None:
            raise SchedulingUnavailable(
                "instruction function is outside the selected observed roster"
            )
        bits = _integer(
            operands["layouts"].get("word_bits"), "operand word width", minimum=1
        )
        for value in (rs1, rs2):
            _integer(value, "observed register value")
            if value >= 1 << bits:
                raise ValueError("observed register value exceeds selected width")
        observer = self.operand_observer()
        semantic_class, decoded = observer.decode_instruction(
            function,
            {"raw": rs1, "kind": "const"},
            {"raw": rs2, "kind": "const"},
            observer.isa_constants("gemmini"),
        )
        self.revalidate()
        return {
            "instruction_word": instruction_word,
            "function": function,
            "semantic_class": semantic_class,
            "register_usage": (instruction_word >> 12) & 7,
            "rs1": rs1,
            "rs2": rs2,
            "decoded": decoded,
            "effects": "UNKNOWN",
            "completion": "UNKNOWN",
        }

    def report(self):
        self.revalidate()
        facts, contract, operands = self._documents()
        mechanisms = []
        for name, (section, needed) in _MECHANISMS.items():
            mechanisms.append(
                {
                    "mechanism": name,
                    "enabled": False,
                    "status": "UNAVAILABLE",
                    "required_section": section,
                    "reasons": [
                        "observed target contract is not a reviewed merlin.software_spec.v1 scheduling contract",
                        "missing " + needed,
                        "no selected independent execution qualification for this shared adapter",
                    ],
                }
            )
        return {
            "schema": "gemmini.selected_scheduling_readiness.v1",
            "target": "gemmini",
            "hardware": {
                "config": contract["rtl_source"]["config"],
                "firrtl_sha256": facts["inputs"]["fir_sha256"],
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": self._implementation_sha256,
            },
            "selected_inputs": {
                key: {"path": str(path), "sha256": _sha(payload)}
                for key, path, payload in (
                    ("facts", self.facts_path, self.facts_bytes),
                    ("observed_contract", self.contract_path, self.contract_bytes),
                    ("operand_sidecar", self.operand_path, self.operand_bytes),
                )
            },
            "geometry": self._geometry(facts["facts"]),
            "structural_dataflow": facts["facts"]
            .get("timing_structure", {})
            .get("dataflow", {"status": "unknown"}),
            "observed_instruction_classes": operands["protocol"]["encoding"][
                "semantic_class"
            ],
            "operand_semantics_status": operands["layouts"].get(
                "command_acceptance_and_semantic_roles", "unproven"
            ),
            "mechanisms": mechanisms,
            "enabled_mechanisms": [],
            "numerical_policy": "unchanged; supplied by original compiler/cohort admission",
            "execution_qualification": "not_established",
            "timing": "UNKNOWN",
        }

    def require(self, mechanism):
        if mechanism not in _MECHANISMS:
            raise ValueError("unknown shared scheduling mechanism")
        self.revalidate()
        raise SchedulingUnavailable(
            "selected target cannot enable "
            + mechanism
            + ": "
            + "; ".join(
                next(
                    row["reasons"]
                    for row in self.report()["mechanisms"]
                    if row["mechanism"] == mechanism
                )
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--observed-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--check-operands",
        action="store_true",
        help="rederive selected operand widths from pinned sources",
    )
    parser.add_argument(
        "--require", action="append", default=[], choices=sorted(_MECHANISMS)
    )
    args = parser.parse_args()
    selected = SelectedSchedulingTarget.read(args.facts, args.observed_contract)
    report = selected.report()
    if args.check_operands:
        from merlin.common.cpu_pool import shared_compute_pool

        with shared_compute_pool():
            selected.operand_observer()
        report["operand_source_check"] = {
            "status": "source_rederived",
            "device_execution": "not_run",
            "effects_and_completion": "unproven",
        }
    selected.revalidate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for mechanism in args.require:
        try:
            selected.require(mechanism)
        except SchedulingUnavailable as error:
            print(str(error), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
