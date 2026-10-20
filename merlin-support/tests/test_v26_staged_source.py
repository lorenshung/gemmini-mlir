"""Actual original-emitter host checks, never target/pass qualification."""

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import gemmini_v26_staged_source as adapter
import pytest
from merlin.common import mlir_query as mq
from merlin.compile.scheduling.export import export_shared_scheduling
from merlin.compile.scheduling.staged_dense import StagedDenseContract
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.package_runtime import CertFailure, integrity_scan
from merlin_experiments.phase0.connected_integer_bounds import derive

PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "phase1_generated/el4_fire_aligned_v26_original_29213759/package"
)


def inputs(tmp_path, m=32, k=16, n=32):
    source = tmp_path / "source.mlir"
    text = f"""module {{
func.func @forward(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
%z = arith.constant 0 : i32
%s = tensor.splat %z : tensor<{m}x{n}xi32>
%r = linalg.generic {{
indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
iterator_types = ["parallel","parallel","reduction"]
}} ins(%a,%b : tensor<{m}x{k}xi8>,tensor<{k}x{n}xi8>) outs(%s : tensor<{m}x{n}xi32>) {{
^bb0(%x:i8,%y:i8,%acc:i32):
%u = arith.extsi %x : i8 to i32
%v = arith.extsi %y : i8 to i32
%p = arith.muli %u,%v : i32
%sum = arith.addi %acc,%p : i32
linalg.yield %sum : i32
}} -> tensor<{m}x{n}xi32>
func.return %r : tensor<{m}x{n}xi32>
}}}}
"""
    source.write_text(text)
    proof = derive(mq.parse(text), [(-3, 3), (-3, 3)])
    proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    buffer = {
        "abi_version": "0.1",
        "target": "gemmini",
        "compiler_pointer_abi": {"version": 1, "arguments": ["a", "b", "out"]},
        "tensors": {
            "a": {"shape": [m, k], "dtype": "i8", "role": "input"},
            "b": {"shape": [k, n], "dtype": "i8", "role": "input"},
            "out": {"shape": [m, n], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {
                "opcode": "RES_PACK",
                "operands": {"src": "b", "dst": "handle"},
                "attributes": {"layout": "packed_rhs"},
            },
            {
                "opcode": "MATMUL_RESIDENT",
                "operands": {"lhs": "a", "rhs": "handle", "dst": "acc"},
            },
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "out"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": "handle"}},
        ],
    }
    contract = StagedDenseContract(
        "facts",
        "software",
        "numeric",
        "execution",
        16,
        32,
        16,
        8,
        20,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        False,
    )
    return {
        "package": PACKAGE,
        "source": source,
        "proof": proof,
        "command_buffer": buffer,
        "contract": contract,
        "enabled": True,
    }


def original_finite(options):
    modules, _ = adapter._original(PACKAGE)
    b = modules["builder.py"].Builder(["a", "b", "out"])
    recorder = adapter._FiniteBuilder(b, 4096)
    (m, k), (_, n) = [
        row["shape"]
        for name, row in options["command_buffer"]["tensors"].items()
        if name != "out"
    ]
    modules["contraction.py"].matmul(
        recorder,
        "%arg0",
        "%arg1",
        "%arg2",
        m,
        k,
        n,
        {"output_dtype": "i32", "epilogue": []},
        {},
        {},
    )
    recorder.fence()
    return b, recorder


@pytest.mark.parametrize("selected", [False, True])
def test_actual_adapter_runs_only_with_explicit_source_binding_sdk(tmp_path, selected):
    options = inputs(tmp_path)
    options["contract"] = asdict(options["contract"])
    options["source"] = str(options["source"])
    options["package"] = str(options["package"])
    sdk = tmp_path / "sdk"
    document = export_shared_scheduling(sdk, source_binding=selected)
    contract = freeze_compiler_library(
        sdk,
        review_id="explicit-v26-host-emission-test",
        public_modules=tuple(document["public_modules"]),
        sources=tuple((row["path"], row["module"]) for row in document["members"])
        + (("shared-library.json", None),),
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    leaf = candidate / "staged_source.py"
    shutil.copyfile(adapter.__file__, leaf)
    package = SimpleNamespace(directory=candidate, integrity_exempt=False)
    if selected:
        integrity_scan(package, compiler_library=contract, compiler_library_root=sdk)
    else:
        with pytest.raises(CertFailure):
            integrity_scan(package, compiler_library=contract, compiler_library_root=sdk)
    result = tmp_path / "result.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                sdk=str(sdk),
                selected=selected,
                dependencies=sorted(
                    {
                        str(Path(importlib.util.find_spec(name).origin).parent.parent)
                        for name in ("xdsl", "numpy", "yaml", "z3")
                    }
                ),
                adapter=str(leaf),
                options=options,
                result=str(result),
            )
        )
    )
    worker = Path(__file__).parent / "fixtures/v26_staged_source_worker.py"
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(worker), str(config)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    observed = json.loads(result.read_bytes())
    if selected:
        assert observed["host_emission"] == "PASS"
        assert observed["target_executions"] == 0
    else:
        assert observed == {"unselected_import": "refused"}
    contract.verify(sdk)


def test_disabled_default_never_reads_package_or_source():
    assert (
        adapter.emit_v26_staged_dense(
            package=None, source=None, proof=None, command_buffer=None, contract=None
        )
        is None
    )
    with pytest.raises(ValueError, match="Boolean"):
        adapter.emit_v26_staged_dense(
            package=None,
            source=None,
            proof=None,
            command_buffer=None,
            contract=None,
            enabled=1,
        )


@pytest.mark.parametrize(
    "m,k,n",
    [
        (32, 16, 32),
        (17, 16, 32),
        (31, 15, 17),
        (32, 3, 4),
        (1, 3, 4),
        (2, 17, 4),
        (2, 16, 33),
        (17, 17, 17),
    ],
)
def test_actual_original_lines_shared_rewrites_and_complete_work(tmp_path, m, k, n):
    options = inputs(tmp_path, m, k, n)
    before = copy.deepcopy(options)
    original, _recorder = original_finite(options)
    old = original.finish()
    emitted, report = adapter.emit_v26_staged_dense(**options)
    assert options == before
    assert report["proposal"]["rewrite_order"] == ["n_before_m", "rhs_reloads"]
    assert all(
        row["products"] == m * k * n and row["outputs"] == m * n
        for row in report["proposal"]["checks"]
    )
    old_lines, new_lines = Counter(old.splitlines()), Counter(emitted.splitlines())
    assert not new_lines - old_lines
    omitted = list((old_lines - new_lines).elements())
    assert len(omitted) == report["physical_rhs_calls_removed"]
    assert all("0x7b, 3, 1," in line for line in omitted)
    assert old.count('asm sideeffect "fence"') == emitted.count(
        'asm sideeffect "fence"'
    )
    assert old.count("0x7b, 3, 0,") == emitted.count("0x7b, 3, 0,")
    assert old.count("0x7b, 3, 14,") == emitted.count("0x7b, 3, 14,")
    assert old.count("0x7b, 3, 6,") == emitted.count("0x7b, 3, 6,")
    assert old.count("0x7b, 3, 4,") == emitted.count("0x7b, 3, 4,")
    assert old.count("0x7b, 3, 3,") == emitted.count("0x7b, 3, 3,")
    assert (
        report["target_capability_issued"] is False
        and report["timing_authority"] == "NONE"
    )
    assert report["original_runtime_loop_control_preserved"] is False
    if (m, k, n) == (32, 16, 32):
        assert report["physical_rhs_calls_removed"] == 2
        assert report["proposal"]["checks"][0]["computes"] == 4
        assert report["physical_fence_count"] == 15
    if m <= 16 or k > 16:
        assert report["physical_rhs_calls_removed"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "bias",
        "transpose",
        "extra",
        "source",
        "state",
        "layout",
        "alias",
        "budget",
        "tiles",
        "arithmetic",
    ],
)
def test_unknown_effects_ownership_and_budgets_refuse(tmp_path, fault):
    options = inputs(tmp_path)
    cb = options["command_buffer"]
    if fault == "bias":
        cb["commands"][2]["attributes"]["bias"] = "a"
    elif fault == "transpose":
        cb["commands"][0]["attributes"]["layout"] = "transpose"
    elif fault == "extra":
        cb["commands"].append(copy.deepcopy(cb["commands"][-1]))
    elif fault == "source":
        options["source"].write_text(options["source"].read_text() + "\n")
    elif fault == "state":
        options["contract"] = replace(options["contract"], retained_weights=True)
    elif fault == "layout":
        cb["tensors"]["b"]["layout"] = "column_major"
    elif fault == "alias":
        cb["commands"][0]["operands"]["dst"] = "a"
    elif fault == "budget":
        options["max_products"] = 10
    elif fault == "tiles":
        options["max_tiles"] = 1
    else:
        options["contract"] = replace(options["contract"], primitive_bits=21)
    with pytest.raises(ValueError):
        adapter.emit_v26_staged_dense(**options)


def test_original_source_mutation_and_symlink_refuse(tmp_path):
    options = inputs(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    for name in adapter._ORIGINAL:
        (package / name).write_bytes((PACKAGE / name).read_bytes())
    options["package"] = package
    (package / "contraction.py").write_text(
        (package / "contraction.py").read_text() + "\n"
    )
    with pytest.raises(ValueError, match="source changed"):
        adapter.emit_v26_staged_dense(**options)
    (package / "contraction.py").unlink()
    (package / "contraction.py").symlink_to(PACKAGE / "contraction.py")
    with pytest.raises(ValueError, match="symlinks"):
        adapter.emit_v26_staged_dense(**options)


def test_verify_only_control_is_original_finite_emission(tmp_path):
    options = inputs(tmp_path)
    original, _ = original_finite(options)
    expected = original.finish()
    emitted, report = adapter.emit_v26_staged_dense(
        **options, rewrite_policy="verify_only"
    )
    assert emitted == expected
    assert report["physical_rhs_calls_removed"] == 0
    assert report["proposal"]["rewrite_order"] == []
    with pytest.raises(ValueError, match="rewrite policy"):
        adapter.emit_v26_staged_dense(**options, rewrite_policy="unknown")


@pytest.mark.parametrize("fault", ["source", "proof", "commands", "emitter_source"])
def test_late_original_input_drift_refuses(tmp_path, monkeypatch, fault):
    options = inputs(tmp_path)
    if fault == "emitter_source":
        package = tmp_path / "package"
        package.mkdir()
        for name in adapter._ORIGINAL:
            (package / name).write_bytes((PACKAGE / name).read_bytes())
        options["package"] = package
    original_rewrite = adapter.rewrite_staged_dense

    def change_after_independent_check(*args, **kwargs):
        result = original_rewrite(*args, **kwargs)
        if fault == "source":
            options["source"].write_text(options["source"].read_text() + "\n")
        elif fault == "proof":
            options["proof"]["operations"][2]["reduction_extent"] = 1
        elif fault == "commands":
            options["command_buffer"]["commands"][2]["attributes"]["output_dtype"] = (
                "i8"
            )
        else:
            path = options["package"] / "builder.py"
            path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(adapter, "rewrite_staged_dense", change_after_independent_check)
    with pytest.raises(ValueError, match="changed"):
        adapter.emit_v26_staged_dense(**options)


def test_finite_projection_missing_load_state_and_changed_zero_refuse(tmp_path):
    options = inputs(tmp_path)
    _, recorder = original_finite(options)
    problem = adapter.StagedDenseProblem(32, 16, 32)
    events = list(recorder.events)
    recorder.events = [
        event
        for event in events
        if not (event[0] == 0 and event[1] & 3 == 1 and (event[1] >> 3) & 3 == 1)
    ]
    with pytest.raises(ValueError, match="complete configuration"):
        adapter._project(recorder, problem, 16)
    recorder.events = events
    recorder.builder.globals[0] = recorder.builder.globals[0].replace(
        "zeroinitializer", "undef"
    )
    with pytest.raises(ValueError, match="zero row"):
        adapter._project(recorder, problem, 16)


def test_requested_limits_cannot_raise_fixed_emission_ceiling(tmp_path):
    options = inputs(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        adapter.emit_v26_staged_dense(**options, max_products=(4 << 20) + 1)
    with pytest.raises(ValueError, match="bounded"):
        adapter.emit_v26_staged_dense(**options, max_tiles=4097)


def test_finite_compute_packed_width_must_match_descriptor(tmp_path):
    options = inputs(tmp_path)
    _, recorder = original_finite(options)
    index = next(i for i, event in enumerate(recorder.events) if event[0] == 4)
    event = recorder.events[index]
    recorder.events[index] = (*event[:2], event[2] ^ (1 << 32), *event[3:])
    with pytest.raises(ValueError, match="compute mode"):
        adapter._project(recorder, adapter.StagedDenseProblem(32, 16, 32), 16)


@pytest.mark.parametrize(
    "line", ["  %v2 = add i64 %foreign, 1", "  br label %next", "label:"]
)
def test_cross_tile_ssa_and_control_refuse(line):
    with pytest.raises(ValueError):
        adapter._closed_chunk([line], ())


def test_structural_lexer_keeps_asm_strings_out_of_ssa_and_control():
    adapter._closed_chunk(
        ['  call void asm sideeffect "br %foreign", "~{memory}"()'], ()
    )
    with pytest.raises(ValueError, match="duplicates"):
        adapter._closed_chunk(
            ["  %v1 = ptrtoint ptr %arg0 to i64", "  %v1 = ptrtoint ptr %arg1 to i64"],
            (),
        )
    with pytest.raises(ValueError, match="spelling"):
        adapter._closed_chunk(['  %"quoted" = ptrtoint ptr %arg0 to i64'], ())
    with pytest.raises(ValueError, match="bound"):
        adapter._closed_chunk([" " * 16385], ())
