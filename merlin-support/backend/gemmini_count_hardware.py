"""Source-derived correspondence for an original count model and functional ABI.

Geometry and receiver association are checked independently of calibration.
This does not establish accelerator completion, physical equivalence or timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from gemmini_selected_scheduling import SelectedSchedulingTarget, _unique
from gemmini_timing_resources import _pin, _revalidate_sources
from merlin.targetgen.rtl import firrtl_struct as F

SCHEMA = "gemmini.count_hardware_correspondence.v1"


def _defines(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if (
            fields
            and fields[0] == "#define"
            and len(fields) >= 2
            and "(" not in fields[1]
        ):
            if fields[1] in result:
                raise ValueError("ambiguous receiver macro")
            result[fields[1]] = fields[2:]
    return result


def _integer_macro(macros, name):
    value = macros.get(name)
    if value is None or len(value) != 1:
        raise ValueError("receiver needs one literal integer macro: " + name)
    return int(value[0], 0)


def _dtype_bytes(text, alias):
    matches = [
        line.split()[1]
        for line in text.splitlines()
        if line.split()[:1] == ["typedef"] and line.split()[-1:] == [alias + ";"]
    ]
    if len(matches) != 1 or matches[0] not in {"int8_t", "int16_t", "int32_t"}:
        raise ValueError("receiver needs an exact signed integer typedef")
    return int(matches[0][3:-2]) // 8


def _reset(fir):
    rows = F.read_module_bodies(fir, {"StoreController"})["StoreController"]
    found = []
    for line_number, raw in rows:
        text, locator = F.split_locator(raw)
        fields = text.strip().split(None, 1)
        if len(fields) != 2 or fields[0] != "regreset":
            continue
        name, _, rest = fields[1].partition(":")
        if name.strip() != "activation":
            continue
        parts = []
        for _ in range(3):
            first, rest = F._split_top_comma(rest)
            parts.append(first)
        value = F.parse_expr(rest.strip())
        if type(value) is not F.Lit or value.value < 0:
            raise ValueError("selected store activation reset is unresolved")
        found.append((value.value, {"firrtl_line": line_number, "locator": locator}))
    if len(found) != 1:
        raise ValueError("selected store activation reset is absent or ambiguous")
    return found[0]


def _structural_flags(fir, accumulator_row_bytes):
    modules = F.load_modules(
        fir,
        {"Scratchpad", "Gemmini", "PixelRepeater", "PixelRepeater_1", "StreamWriter"},
    )
    scratchpad = modules["Scratchpad"]
    instances = {
        name: row[1] for name, row in scratchpad.decls.items() if row[0] == "inst"
    }
    if instances.get("writer") != "StreamWriter":
        raise ValueError("selected scratchpad writer implementation differs")
    full = F.field_type(
        modules["StreamWriter"].port_type("io"), ("req", "bits", "data")
    )
    if full != ("uint", accumulator_row_bytes * 8):
        raise ValueError("selected DMA writer does not carry a full accumulator row")
    repeaters = []
    for name in ("zero_writer_pixel_repeater", "mvin_scale_pixel_repeater"):
        target = modules.get(instances.get(name))
        if target is None or target.decls.get("req", (None,))[0] != "reg":
            raise ValueError("selected pixel repeater is absent or passthrough")
        typ = F.parse_type(target.decls["req"][1])
        if F.field_type(typ, ("bits", "pixel_repeats")) != ("uint", 8):
            raise ValueError("selected pixel repeater state differs")
        writes = [
            row
            for row in target.connects
            if row.lhs == F.parse_expr("req.bits.pixel_repeats")
        ]
        if len(writes) != 2 or not all(row.guards for row in writes):
            raise ValueError("selected pixel repeat state transitions are unresolved")
        repeaters.append(
            {
                "instance": name,
                "module": target.name,
                "state_write_lines": [x.line for x in writes],
            }
        )
    opcode = F.field_type(
        modules["Gemmini"].port_type("io"), ("cmd", "bits", "inst", "opcode")
    )
    if not opcode or opcode[0] != "uint":
        raise ValueError("selected instruction opcode width is unresolved")
    return {"full_width_accumulator_readout": 1, "first_layer_optimizations": 1}, {
        "writer_data_bits": full[1],
        "repeaters": repeaters,
        "opcode_bits": opcode[1],
    }


def derive(*, model_path, selected, runtime, receiver, functional_spike_source):
    selected.revalidate()
    selected.operand_observer()  # Reopen source pins and rederive selected operand slices.
    facts, observed, operands = selected._documents()
    model_path, runtime, receiver, functional_spike_source = map(
        Path, (model_path, runtime, receiver, functional_spike_source)
    )
    model = json.loads(model_path.read_bytes())
    deployment_path = runtime / "deployment.json"
    deployment = json.loads(deployment_path.read_bytes())
    if deployment.get("schema") != "merlin.functional_spike_deployment.v1":
        raise ValueError("functional runtime lacks its original deployment provenance")
    critical = deployment["sources"]["sources"]["libgemmini"]["critical_files"]
    source_pins = {
        "receiver_" + name: _pin(receiver / name)
        for name in ("gemmini_params.h", "gemmini.h", "gemmini.cc")
    }
    if any(
        source_pins["receiver_" + name]["sha256"] != critical[name] for name in critical
    ):
        raise ValueError("receiver source differs from retained functional build")
    rocc = functional_spike_source / "riscv/rocc.h"
    source_pins["functional_rocc"] = _pin(rocc)
    if (
        source_pins["functional_rocc"]["sha256"]
        != deployment["sources"]["sources"]["spike"]["critical_files"]["riscv/rocc.h"]
    ):
        raise ValueError(
            "receiver opcode source differs from retained functional build"
        )
    source_pins.update(
        {
            "deployment": _pin(deployment_path),
            "functional_extension": _pin(runtime / "lib/libgemmini.so"),
            "model": _pin(model_path),
        }
    )
    if (
        source_pins["functional_extension"]["sha256"]
        != deployment["artifacts"]["lib/libgemmini.so"]["sha256"]
    ):
        raise ValueError(
            "functional extension differs from its original source/build association"
        )
    if (
        not model["libraries"]
        or model["libraries"][0] != source_pins["functional_extension"]["sha256"]
    ):
        raise ValueError("count model selects a different functional extension")
    if model["cpu"]["isa"] != deployment["profile"]["launch"]["isa"]:
        raise ValueError(
            "count model CPU ISA differs from the original functional runtime"
        )
    params = (receiver / "gemmini_params.h").read_text()
    macros, rocc_macros = _defines(params), _defines(rocc.read_text())
    geometry = selected.report()["geometry"]
    stores = {row["name"]: row for row in geometry["stores"]}
    dim = _integer_macro(macros, "DIM")
    if geometry["array_rows"] != dim or geometry["array_cols"] != dim:
        raise ValueError(
            "receiver dimension differs from exact selected array geometry"
        )
    expected = {
        "dim": dim,
        "element_bytes": _dtype_bytes(params, "elem_t"),
        "accumulator_bytes": _dtype_bytes(params, "acc_t"),
        "scratchpad_rows": stores["scratchpad"]["total_rows"],
        "accumulator_rows": stores["accumulator"]["total_rows"],
    }
    formats = {row["name"]: row["dtype"] for row in facts["facts"]["datapaths"]}
    if formats != {
        "input": "i" + str(8 * expected["element_bytes"]),
        "accumulator": "i" + str(8 * expected["accumulator_bytes"]),
    }:
        raise ValueError(
            "receiver signed element/accumulator formats differ from selected RTL"
        )
    if (
        expected["scratchpad_rows"]
        != _integer_macro(macros, "BANK_NUM") * _integer_macro(macros, "BANK_ROWS")
        or expected["accumulator_rows"] != _integer_macro(macros, "ACC_ROWS")
        or stores["scratchpad"]["row_bytes"] != dim * expected["element_bytes"]
        or stores["accumulator"]["row_bytes"] != dim * expected["accumulator_bytes"]
    ):
        raise ValueError(
            "receiver memory rows/widths differ from selected address-space views"
        )
    local = next(
        row
        for row in operands["layouts"]["fields"]
        if row["bundle"] == "MvinRs2"
        and row["name"] == "local_addr"
        and row["status"] == "derived"
    )
    expected["address_bits"] = local["width"]
    if expected["address_bits"] != _integer_macro(macros, "ADDR_LEN"):
        raise ValueError(
            "receiver local address width differs from selected operand structure"
        )
    table = _unique(facts["facts"]["interfaces"], "funct_decode_table")
    expected["custom_opcode"] = table["custom_opcode"]
    expected["custom_opcode_mask"] = _integer_macro(rocc_macros, "ROCC_OPCODE_MASK")
    custom = "ROCC_OPCODE" + str(_integer_macro(macros, "XCUSTOM_ACC"))
    if _integer_macro(rocc_macros, custom) != expected["custom_opcode"]:
        raise ValueError(
            "receiver opcode differs from selected software/RTL command boundary"
        )
    receiver_text = "".join((receiver / "gemmini.cc").read_text().split())
    if (
        "push_custom_insn(insns,"
        + custom
        + ",ROCC_OPCODE_MASK,ILLEGAL_INSN_FUNC,gemmini_custom);"
        not in receiver_text
    ):
        raise ValueError("receiver registration does not use the selected opcode/mask")
    fir = Path(facts["inputs"]["fir_path"])
    source_pins["selected_firrtl"] = _pin(fir)
    if source_pins["selected_firrtl"]["sha256"] != facts["inputs"]["fir_sha256"]:
        raise ValueError("selected FIRRTL changed")
    flags, witness = _structural_flags(fir, stores["accumulator"]["row_bytes"])
    for key, macro in (
        ("first_layer_optimizations", "HAS_FIRST_LAYER_OPTIMIZATIONS"),
        ("full_width_accumulator_readout", "ACC_READ_FULL_WIDTH"),
    ):
        if (macro in macros) != bool(flags[key]):
            raise ValueError(
                "receiver feature differs from selected elaborated structure"
            )
    if expected["custom_opcode_mask"] != (1 << witness["opcode_bits"]) - 1:
        raise ValueError(
            "receiver opcode mask differs from selected instruction field width"
        )
    expected.update(flags)
    expected["reset_accumulator_activation"], witness["store_activation_reset"] = (
        _reset(fir)
    )
    if model["target"]["hardware"] != expected:
        raise ValueError(
            "original count model hardware fields differ from source-derived selection"
        )
    if observed["rtl_source"]["config"] != deployment["profile"]["target_config"]:
        raise ValueError(
            "runtime and selected observed contract name different configurations"
        )
    _revalidate_sources(source_pins)
    selected.revalidate()
    return {
        "schema": SCHEMA,
        "selected_inputs": selected.report()["selected_inputs"],
        "hardware": expected,
        "pins": source_pins,
        "structural_witness": witness,
        "implementation": _pin(__file__),
        "scope": "count_feature_hardware_and_original_functional_build_association",
        "hardware_execution_qualification": "not_established",
        "native_event_timing_qualified": False,
        "limits": [
            "functional receiver reset leaves acc_act initialization unestablished; hardware reset is bound separately",
            "resource/completion, physical aliases, DMA beats and timing are not qualified by this correspondence",
        ],
    }


@dataclass(frozen=True)
class CountHardwareBinding:
    path: Path
    payload: bytes

    @classmethod
    def read(cls, path):
        path = Path(path).resolve(strict=True)
        owner = cls(path, path.read_bytes())
        owner.revalidate()
        return owner

    @property
    def identity(self):
        return SCHEMA + ":" + hashlib.sha256(self.payload).hexdigest()

    def revalidate(self):
        row = json.loads(self.payload)
        if (
            self.path.read_bytes() != self.payload
            or row["schema"] != SCHEMA
            or row["implementation"] != _pin(__file__)
            or row["native_event_timing_qualified"] is not False
            or row["hardware_execution_qualification"] != "not_established"
        ):
            raise ValueError(
                "count hardware correspondence changed or declares unsupported authority"
            )
        _revalidate_sources(row["pins"])
        _revalidate_sources(row["selected_inputs"])
        if (
            json.loads(Path(row["pins"]["model"]["path"]).read_bytes())["target"][
                "hardware"
            ]
            != row["hardware"]
        ):
            raise ValueError("count model differs from derived hardware")
        return row

    def verify(
        self,
        *,
        model_path,
        facts_path,
        observed_contract_path,
        functional_extension_path,
    ):
        row = self.revalidate()
        for path, pin in (
            (model_path, row["pins"]["model"]),
            (facts_path, row["selected_inputs"]["facts"]),
            (observed_contract_path, row["selected_inputs"]["observed_contract"]),
            (functional_extension_path, row["pins"]["functional_extension"]),
        ):
            if _pin(path)["sha256"] != pin["sha256"]:
                raise ValueError(
                    "ordinary queue supplied a different count-model/target/runtime binding"
                )
        return self.identity


@dataclass(frozen=True)
class CountHardwareCorrespondenceOwner:
    correspondence: CountHardwareBinding
    model_path: Path
    facts_path: Path
    observed_contract_path: Path
    functional_extension_path: Path

    @classmethod
    def bind(
        cls,
        sidecar,
        *,
        model_path,
        facts_path,
        observed_contract_path,
        functional_extension_path,
    ):
        correspondence = CountHardwareBinding.read(sidecar)
        row = correspondence.revalidate()
        paths = tuple(
            Path(p).resolve(strict=True)
            for p in (
                model_path,
                facts_path,
                observed_contract_path,
                functional_extension_path,
            )
        )
        model, facts, observed, extension = paths
        # Saved status is not admission. Independently reconstruct the exact
        # selected geometry/receiver/source association before creating owner.
        rebuilt = derive(
            model_path=model,
            selected=SelectedSchedulingTarget.read(facts, observed),
            runtime=Path(row["pins"]["deployment"]["path"]).parent,
            receiver=Path(row["pins"]["receiver_gemmini_params.h"]["path"]).parent,
            functional_spike_source=Path(
                row["pins"]["functional_rocc"]["path"]
            ).parents[1],
        )
        if rebuilt != row:
            raise ValueError(
                "saved count correspondence differs from fresh source derivation"
            )
        owner = cls(correspondence, model, facts, observed, extension)
        owner.revalidate()
        return owner

    @property
    def identity(self):
        payload = (
            self.correspondence.identity,
            *(
                str(path)
                for path in (
                    self.model_path,
                    self.facts_path,
                    self.observed_contract_path,
                    self.functional_extension_path,
                )
            ),
        )
        return (
            SCHEMA
            + ":owner:"
            + hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        )

    def revalidate(self):
        self.correspondence.verify(
            model_path=self.model_path,
            facts_path=self.facts_path,
            observed_contract_path=self.observed_contract_path,
            functional_extension_path=self.functional_extension_path,
        )

    def selected_binding(self):
        """Five explicit registry identities; compiler numeric gates stay separate."""
        self.revalidate()
        model = json.loads(self.model_path.read_bytes())
        libraries = hashlib.sha256(
            json.dumps(
                model["libraries"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        return {
            "hardware": _pin(self.facts_path)["sha256"],
            "software": _pin(self.observed_contract_path)["sha256"],
            "numerical_policy": model["domain"]["feature_schema_sha256"],
            "runtime": _pin(self.model_path)["sha256"],
            "library": libraries,
        }

    def record(self):
        return {
            "identity": self.identity,
            "selected_binding": self.selected_binding(),
            "correspondence": _pin(self.correspondence.path),
            "scope": "source_derived_count_hardware_correspondence_only",
            "hardware_execution_qualification": "not_established",
            "native_event_timing_qualified": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "facts",
        "observed-contract",
        "model",
        "runtime",
        "receiver",
        "functional-spike-source",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool():
        report = derive(
            model_path=args.model,
            selected=SelectedSchedulingTarget.read(args.facts, args.observed_contract),
            runtime=args.runtime,
            receiver=args.receiver,
            functional_spike_source=args.functional_spike_source,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    CountHardwareBinding.read(args.output)


if __name__ == "__main__":
    main()
