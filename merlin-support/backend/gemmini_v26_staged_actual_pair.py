"""Original live source owners plus independent V26 actual-program relation.

No target run is launched here. Existing completed native/commit invocations
are reopened by the original owners and parser. This function never restores
an owner from JSON or issues a compiler-pass capability or timing authority.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MethodType

from gemmini_arithmetic_scope import IntegerArithmeticOwner
from gemmini_direct_source_execution import DirectSourceExecutionOwner
from gemmini_execute_state import ExecuteStateContract
from gemmini_load_streams import SelectedLoadStreams
from gemmini_selected_scheduling import SelectedSchedulingTarget
from gemmini_staged_dense_admission import StagedCompletionOwner
from gemmini_timing_resources import _pin, _revalidate_sources
from gemmini_v26_staged_pair_effects import compare_decoded_pair

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
)
from merlin.compile.scheduling.staged_dense_verification import verify_staged_dense
from merlin.targetgen import native_model_execution as native


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _original_methods(owner):
    expected = _METHODS.get(type(owner))
    if expected is None:
        raise TypeError(
            "actual staged pair requires its exact original live owner types"
        )
    for name, function, code in expected:
        method = getattr(owner, name)
        if (
            type(method) is not MethodType
            or method.__self__ is not owner
            or method.__func__ is not function
            or function.__code__ is not code
        ):
            raise ValueError("actual staged pair original bound callback changed")


def _current_implementation():
    if _pin(__file__) != _IMPORTED_SOURCE:
        raise ValueError("actual staged pair checker source changed after import")
    _revalidate_sources(_IMPORTED_SOURCES)
    for name, function, code in _FUNCTIONS:
        if globals()[name] is not function or function.__code__ is not code:
            raise ValueError("actual staged pair checker callback changed")


def _transport(log, opcode):
    """Retain FENCE words and RoCC transport other than source-register names."""
    result = []
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "core" or not parts[2].isdigit():
            continue
        if parts[1] != "0:":
            raise ValueError("actual staged pair needs the original single-hart trace")
        word = int(parts[4].strip("()"), 16)
        if word & 127 == opcode:
            result.append(word & ~((31 << 15) | (31 << 20)))
        elif word & 127 == 15 and word >> 12 & 7 == 0:
            result.append(word)
    return result


def _pointer_fields(selected, ports):
    classes = selected._documents()[2]["protocol"]["roles"]["classes"]
    result = {}
    for kind in (*ports, "MVOUT"):
        fields = tuple(
            row["name"]
            for row in classes[kind]["fields"]
            if row.get("as") == "operand" and row.get("operand") == "rs1"
        )
        if (
            not fields
            or len(fields) > 8
            or len(set(fields)) != len(fields)
            or any(type(name) is not str or not name for name in fields)
        ):
            raise ValueError(
                "actual staged DMA lacks original declared rs1 pointer-copy fields"
            )
        result[kind] = fields
    return result


def _normalized_pointer_copies(command, first, fields):
    if (
        type(first) is not tuple
        or len(first) != 2
        or type(first[0]) is not str
        or type(first[1]) is not int
        or first[1] < 0
    ):
        raise ValueError("actual staged pointer has no unique linked symbol/offset")
    result = copy.deepcopy(command)
    for name in fields:
        if result["decoded"].get(name) != result["rs1"]:
            raise ValueError(
                "actual staged copied pointer differs from original raw rs1"
            )
        result["decoded"][name] = {
            "kind": "linked_symbol_offset",
            "symbol": first[0],
            "offset": first[1],
        }
    return result


def _decode_arm(owner, binding):
    observed = invocation_record.verify(owner._paths["trace_invocation"])
    log_path = Path(observed["stderr"]["path"])
    if log_path.stat().st_size > 64 << 20:
        raise ValueError(
            "actual staged pair exceeds its original complete-trace byte bound"
        )
    row = json.loads(owner._paths["result"].read_bytes())
    cb = json.loads(owner._paths["command_buffer"].read_bytes())
    elf = native._pinned_native_file(row, "elf")
    path = Path(owner._pins["packet_framing_checker"]["path"])
    if _pin(path) != owner._pins["packet_framing_checker"]:
        raise ValueError("actual staged pair lost its original packet parser")
    spec = importlib.util.spec_from_file_location(
        "original_v26_actual_pair_framing", path
    )
    framing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(framing)
    symbols = framing._symbols(elf, owner._paths["nm"])
    isa = owner._selected.operand_observer().isa_constants("gemmini")
    log = log_path.read_text()
    trace, events = framing._events(log, isa, symbols)
    transports = _transport(log, isa["CUSTOM_OPCODE"])
    if len(transports) != len(events):
        raise ValueError(
            "actual staged pair lost complete command/fence transport framing"
        )
    ports = dict(owner._selected.operand_observer().load_states.ports)
    pointer_fields = _pointer_fields(owner._selected, ports)
    abi = {
        f"tensor_{i}": name
        for i, name in enumerate(cb["compiler_pointer_abi"]["arguments"])
    }
    commands, stream = iter(trace["instructions"]), []
    for event, transport in zip(events, transports, strict=True):
        if event[0] == "FENCE":
            stream.append(
                (
                    ("FENCE", 0, 0),
                    {"class": "FENCE", "transport": transport, "decoded": {}},
                )
            )
            continue
        command = copy.deepcopy(next(commands))
        if command["class"] != event[0]:
            raise ValueError("actual staged pair original decoded event order changed")
        first = command["rs1"]["raw"]
        if event[0] in ports or event[0] == "MVOUT":
            name, offset, _size = framing._memory_name(first, symbols)
            if name in abi:
                first = abi[name], offset
            elif event[0] in ports and command["decoded"].get("is_acc_addr") is True:
                # Each original source owner already re-proved every zero cell.
                first = "private_zero", offset
            else:
                raise ValueError(
                    "actual staged pair DMA lacks original ABI/zero ownership"
                )
            command = _normalized_pointer_copies(
                command, first, pointer_fields[event[0]]
            )
        command["transport"] = transport
        stream.append(((event[0], first, command["rs2"]["raw"]), command))
    if next(commands, None) is not None:
        raise ValueError("actual staged pair left decoded commands unconsumed")
    invocation_record.verify(owner._paths["trace_invocation"])
    return (
        tuple(stream),
        cb,
        {
            "native_result": dict(owner._pins["result"]),
            "command_buffer": dict(owner._pins["command_buffer"]),
            "lowered": dict(owner._pins["lowered"]),
            "complete_trace_invocation": dict(owner._pins["trace_invocation"]),
            "elf": _pin(elf),
            "source_binding": binding,
        },
    )


def _field_reader(selected):
    fields = {
        (row["bundle"], row["name"]): row
        for row in selected._documents()[2]["layouts"]["fields"]
    }
    local = fields["PreloadRs", "local_addr"]
    sub = {row["field"]: row for row in local["subfields"]}
    if sub["data"]["offset"] != 0 or local["status"] != "derived":
        raise ValueError(
            "actual staged pair local-address source mapping is unsupported"
        )
    data_mask = (1 << sub["data"]["width"]) - 1
    continuation = 1 << sub["accumulate"]["offset"]

    def shape(bundle, raw):
        result = []
        for name in ("num_rows", "num_cols"):
            field = fields[bundle, name]
            if field["status"] != "derived":
                raise ValueError("actual staged shape lacks original derived packing")
            result.append(raw >> field["offset"] & ((1 << field["width"]) - 1))
        return tuple(result)

    return shape, data_mask, continuation


def _logical(stream, *, binding, contract, shape, data_mask, continuation, ex, ports):
    effects, started = [], False
    for event, command in stream:
        kind, first, second = event
        fields = command["decoded"]
        raw1 = command.get("rs1", {}).get("raw", 0)
        if kind == "FENCE":
            effects.append(CompletionFence())
        elif kind == "CONFIG_EX":
            ex.identity_ws(raw1, second)
        elif kind in {"CONFIG_LD", "CONFIG_ST"}:
            continue  # Original source owner checks every actual setting.
        elif kind == "FLUSH":
            if started:
                raise ValueError("actual staged projection cannot erase a live drain")
        elif kind in ports:
            started = True
            rows, cols, address = (
                fields["rows"],
                fields["cols"],
                fields["addr"] & data_mask,
            )
            if fields["is_acc_addr"]:
                if first[0] != "private_zero":
                    raise ValueError(
                        "actual staged initialization is not independently private zero"
                    )
                effects.append(AccumulatorZero(address, rows, cols))
            else:
                if first[0] == binding["lhs"]:
                    role, stride = "lhs", binding["k"]
                elif first[0] == binding["rhs"]:
                    role, stride = "rhs", binding["n"]
                else:
                    raise ValueError(
                        "actual staged operand lacks current source ownership"
                    )
                row, col = divmod(first[1], stride)
                effects.append(OperandLoad(role, row, col, address, rows, cols))
        elif kind == "PRELOAD":
            k, n = ex.weight_shape(raw1)
            m, out_n = shape("PreloadRs", second)
            if n != out_n:
                raise ValueError("actual staged descriptor source/return widths differ")
            effects.append(
                DescriptorIssue(
                    raw1 & data_mask,
                    second & data_mask,
                    m,
                    k,
                    n,
                    bool(second & continuation),
                )
            )
        elif kind == "COMPUTE_PRELOADED":
            m, k = shape("ComputeRs", raw1)
            effects.append(ComputeConsume(raw1 & data_mask, m, k))
        elif kind == "MVOUT":
            if first[0] != binding["output"] or first[1] % 4:
                raise ValueError(
                    "actual staged readout changed complete raw output ownership"
                )
            row, col = divmod(first[1] // 4, binding["n"])
            effects.append(
                AccumulatorReadout(
                    fields["addr"] & data_mask, row, col, fields["rows"], fields["cols"]
                )
            )
        else:
            raise ValueError("actual staged projection has unsupported command effects")
    problem = StagedDenseProblem(*(binding[name] for name in ("m", "k", "n")))
    return verify_staged_dense(tuple(effects), problem=problem, contract=contract)


def check_actual_staged_pair(
    *, baseline_owner, candidate_owner, source, proof, capsule_dir, completion
):
    """Freshly check two actual programs; no historical result is an input grant.

    Both owners must have been constructed normally from their current native
    results, emitted CB/LLVM, actual ELF and complete original trace. This call
    re-runs their complete source/output/product/host checks. It performs no
    target run, compiler build, model fitting or capability issuance.
    """
    _current_implementation()
    if (
        type(baseline_owner) is not DirectSourceExecutionOwner
        or type(candidate_owner) is not DirectSourceExecutionOwner
        or type(completion) is not StagedCompletionOwner
    ):
        raise TypeError(
            "actual staged pair needs two original live direct owners and completion owner"
        )
    selected, arithmetic = baseline_owner._selected, baseline_owner._arithmetic
    owners = (baseline_owner, candidate_owner, completion, selected, arithmetic)
    for owner in owners:
        _original_methods(owner)
        owner.revalidate()
    if (
        candidate_owner._selected is not selected
        or candidate_owner._arithmetic is not arithmetic
        or completion.selected is not selected
        or baseline_owner._pins["execution_contract"]
        != candidate_owner._pins["execution_contract"]
        or baseline_owner._observation_flags != candidate_owner._observation_flags
    ):
        raise ValueError(
            "actual staged pair selected different original source/effect/runtime owners"
        )
    identities = (
        baseline_owner.identity,
        candidate_owner.identity,
        completion.identity,
        arithmetic.identity,
    )
    sources = {
        name: _pin(module_source_path(name))
        for name in (
            "gemmini_v26_staged_actual_pair",
            "gemmini_v26_staged_pair_effects",
            "gemmini_direct_source_execution",
            "gemmini_load_streams",
            "gemmini_execute_state",
            "gemmini_staged_dense_admission",
            "merlin.compile.scheduling.staged_dense",
            "merlin.compile.scheduling.staged_dense_verification",
        )
    }
    source = Path(source)
    before_source, before_proof = _pin(source), copy.deepcopy(proof)
    try:
        checks = [
            owner.verify_source_execution(source, proof, Path(capsule_dir))
            for owner in (baseline_owner, candidate_owner)
        ]
        binding = checks[0]["direct_source_binding"]
        if checks[1]["direct_source_binding"] != binding or any(
            row["source_sha256"] != before_source["sha256"]
            or row["complete_output_values"] != binding["m"] * binding["n"]
            for row in checks
        ):
            raise ValueError("actual staged pair changed complete typed source/outputs")
        load_streams = SelectedLoadStreams(selected)
        ex = ExecuteStateContract(selected)
        for owner in (load_streams, ex):
            _original_methods(owner)
            owner.revalidate()
        owners = (*owners, load_streams, ex)
        rule, arithmetic_record, completion_record = (
            arithmetic.source_rule(),
            arithmetic.record(),
            completion.record(),
        )
        inputs, geometry = (
            selected.report()["selected_inputs"],
            selected.report()["geometry"],
        )
        dim = geometry["array_rows"]
        stores = {row["name"]: row for row in geometry["stores"]}
        if (
            dim != geometry["array_cols"]
            or dim != rule["max_primitive_k"]
            or stores["scratchpad"]["total_rows"] < 2 * dim
            or stores["accumulator"]["total_rows"] < dim
            or completion_record["selected_inputs"] != inputs
            or completion_record["dense_v1_admission"]["enabled"] is not False
        ):
            raise ValueError(
                "actual staged pair lacks original private storage/completion mapping"
            )
        runtime = {
            key: arithmetic_record["sources"][role]
            for key, role in (
                ("spike", "runtime_bin/spike"),
                ("functional_extension", "runtime_lib/libgemmini.so"),
            )
        }
        contract = StagedDenseContract(
            inputs["facts"]["sha256"],
            _digest([inputs, completion_record["proof"]]),
            _digest([rule, arithmetic_record]),
            _digest([runtime, baseline_owner._pins["execution_contract"]]),
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
        arms = [
            _decode_arm(owner, binding) for owner in (baseline_owner, candidate_owner)
        ]
        if arms[0][1] != arms[1][1]:
            raise ValueError(
                "actual staged pair changed its original complete command buffer"
            )
        shape, mask, flag = _field_reader(selected)
        logical = [
            _logical(
                arm[0],
                binding=binding,
                contract=contract,
                shape=shape,
                data_mask=mask,
                continuation=flag,
                ex=ex,
                ports=load_streams.ports,
            )
            for arm in arms
        ]
        relation = compare_decoded_pair(
            arms[0][0],
            arms[1][0],
            ports=load_streams.ports,
            rhs=binding["rhs"],
            operand_rows=contract.operand_rows,
            dim=dim,
        )
        result = {
            "schema": "gemmini.v26_original_actual_staged_pair.v1",
            "status": "checked_actual_source_packet_output_pair_diagnostic",
            "source": before_source,
            "source_checks": checks,
            "shared_independent_checks": logical,
            "relation": relation,
            "artifacts": [arm[2] for arm in arms],
            "load_streams": load_streams.record(),
            "completion_identity": completion.identity,
            "sources": sources,
            "shared_pass_invocation_proven": False,
            "target_capability_issued": False,
            "hardware_execution_qualified": False,
            "timing_authority": "NONE",
        }
        if _pin(source) != before_source or proof != before_proof:
            raise ValueError(
                "actual staged pair source/proof changed during current checks"
            )
    finally:
        for owner in owners:
            _original_methods(owner)
            owner.revalidate()
        if (
            any(
                owner._selected is not selected or owner._arithmetic is not arithmetic
                for owner in (baseline_owner, candidate_owner)
            )
            or completion.selected is not selected
            or identities
            != (
                baseline_owner.identity,
                candidate_owner.identity,
                completion.identity,
                arithmetic.identity,
            )
        ):
            raise ValueError(
                "actual staged pair replaced an original owner during current checks"
            )
        _revalidate_sources(sources)
        _current_implementation()
    return result


_METHODS = {
    cls: tuple(
        (name, getattr(cls, name), getattr(cls, name).__code__) for name in names
    )
    for cls, names in (
        (
            DirectSourceExecutionOwner,
            ("revalidate", "verify_source_execution", "source_rule"),
        ),
        (
            SelectedSchedulingTarget,
            ("revalidate", "report", "operand_observer", "_documents"),
        ),
        (IntegerArithmeticOwner, ("revalidate", "record", "source_rule")),
        (StagedCompletionOwner, ("revalidate", "record")),
        (SelectedLoadStreams, ("revalidate", "record")),
        (ExecuteStateContract, ("revalidate", "identity_ws", "weight_shape")),
    )
}
_FUNCTIONS = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in (
        "check_actual_staged_pair",
        "_original_methods",
        "_current_implementation",
        "_transport",
        "_pointer_fields",
        "_normalized_pointer_copies",
        "_decode_arm",
        "_field_reader",
        "_logical",
        "_digest",
        "compare_decoded_pair",
        "verify_staged_dense",
        "_pin",
        "_revalidate_sources",
    )
)
_IMPORTED_SOURCE = _pin(__file__)
_IMPORTED_SOURCES = {
    name: _pin(module_source_path(name))
    for name in (
        "gemmini_v26_staged_pair_effects",
        "merlin.compile.scheduling.staged_dense",
        "merlin.compile.scheduling.staged_dense_verification",
    )
}
