"""Lower the extension-opcode buffers through ``gemmini-opt``'s iface route and grade them.

Per case: Merlin's ``emit_interface_mlir`` renders the buffer, the backend parses and lowers it to
one ``gemmini_kernel`` plus a command buffer, and the kernel's bit-level model must equal Merlin's
``reference_outputs`` EXACTLY.  With ``--spike`` every case is linked into one ELF that
compares the device bytes with the reference (tolerance 0).

    python run_ext.py [--rep] [--only name ...] [--spike] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "device"))

import ext_cases as C  # noqa: E402
import runner  # noqa: E402
from mlir_oot.core_aten import iface  # noqa: E402
from mlir_oot.core_aten.emit import build_module, compile_to_object  # noqa: E402
from mlir_oot.core_aten.stream import simulate  # noqa: E402
from mlir_oot.frontend.parse import parse_module  # noqa: E402


_MNEMONIC = {"AFFINE_MAP": "affine_map", "MOVEMENT": "movement", "CONCAT": "concat",
             "INSERT_SLICE": "insert_slice", "FILL": "fill", "WINDOW_SUM": "window_sum"}
_ORDER = {"AFFINE_MAP": ["x", "y"], "MOVEMENT": ["src"], "CONCAT": [f"src{i}" for i in range(8)],
          "INSERT_SLICE": ["dest", "src"], "FILL": [], "WINDOW_SUM": ["src"]}


def render(cb) -> str:
    """The buffer as merlin_iface text (grammar v0.1), attributes spelled by Merlin's formatter."""
    from merlin.targetgen.contract.interface_emit import _fmt_attrs, _tensor_type

    produced = {c["operands"].get("dst") for c in cb["commands"]}
    lines = ['module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
             'merlin_iface.abi_version = "0.1"} {']
    for name, spec in cb["tensors"].items():
        if name not in produced:
            lines.append(f"  %{name} = merlin_iface.tensor {_fmt_attrs({'name': name, 'role': spec['role']})} : "
                         f"{_tensor_type(spec)}")
    for c in cb["commands"]:
        if c["opcode"] == "FENCE":
            lines.append("  merlin_iface.fence : () -> ()")
            continue
        ops = c["operands"]
        srcs = [ops[k] for k in _ORDER[c["opcode"]] if k in ops]
        dst = ops["dst"]
        attrs = _fmt_attrs({"name": dst, **c["attributes"]})
        types = ", ".join(_tensor_type(cb["tensors"][x]) for x in srcs)
        operands = ", ".join(f"%{x}" for x in srcs)
        lines.append(f"  %{dst} = merlin_iface.{_MNEMONIC[c['opcode']]} {operands}{' ' if operands else ''}{attrs} : "
                     f"({types}) -> {_tensor_type(cb['tensors'][dst])}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def lower(cb):
    """(Merlin's reading of the text, the backend's lowering of the same text)."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    text = render(cb)
    merlin_cb = parse_interface_mlir(text)
    # Merlin's text parser (interface_emit._parse_value @ c750ae65) reads MLIR `true`/`false` as the
    # strings "true"/"false", which integer_ops.window_sum then refuses; restore the booleans the
    # text states so the reference reads what the backend reads.
    for c in merlin_cb.get("commands", []):
        attrs = c.get("attributes") or {}
        for k, v in list(attrs.items()):
            if v in ("true", "false"):
                attrs[k] = v == "true"
    return merlin_cb, iface.build(parse_module(text), text)


def reference(cb, inputs):
    sys.path.insert(0, str(HERE.parent))
    from exact_oracle import command_reference
    return command_reference(cb, inputs)


def merlin_checks(cb, inputs, ref) -> list[str]:
    """Merlin's own view of the emitted buffer: structure, fences, and its simulator == reference."""
    from merlin.runtime.commandbuffer import validate_command_buffer
    from merlin.runtime.fences import fence_violations
    from merlin.runtime.simulator import simulate as cb_simulate

    problems = list(validate_command_buffer(cb))
    problems += [f"fence: {v}" for v in fence_violations(cb)]
    try:
        sim = cb_simulate(cb, {k: v.tolist() for k, v in inputs.items()})
    except UnboundLocalError as exc:
        # This external simulator is informational. Exact grading above uses
        # the independent rational opcode reference, never its float formula.
        return problems + [f'informational Merlin simulator error: {exc}']
    sim = sim.get("outputs", sim) if isinstance(sim, dict) else sim
    for k, v in ref.items():
        got = sim.get(k)
        if got is None or not np.array_equal(np.asarray(got, np.int64).reshape(-1), v):
            problems.append(f"merlin simulate(our buffer)[{k}] != reference")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--spike", action="store_true")
    ap.add_argument("--batch", default="iface_ext")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cs = C.cases(a.rep)
    if a.only:
        cs = {k: v for k, v in cs.items() if k in a.only}
    rows, objects, buffers, calls, checks, outputs = {}, [], {}, [], [], set()
    wd = runner.work_root() / a.batch
    ok = True
    for i, (name, cb) in enumerate(cs.items()):
        t0 = time.time()
        inputs = C.stimulus(cb, i)
        try:
            merlin_cb, (ours, _, stream) = lower(cb)
            ref = reference(merlin_cb, inputs)
            model = simulate(stream, inputs)
            bad = {k: int(np.sum(model[k].astype(np.int64).reshape(-1) != ref[k])) for k in ref}
            diagnostics = merlin_checks(ours, inputs, ref)
            divergence = [p for p in diagnostics if p.startswith("merlin simulate")]
            problems = [p for p in diagnostics if p not in divergence]
            row = dict(status="PASS" if not any(bad.values()) and not problems else "FAIL", mismatches=bad,
                       buffer_problems=problems, eager_float_divergence=dict(informational=True, diagnostics=divergence),
                       commands=[c["opcode"] for c in ours["commands"]], instructions=len(stream.ins))
            if a.spike:
                sym = f"ifx{i}_{name}"
                module, _ = build_module(stream, sym)
                objects.append(Path(compile_to_object(module, wd, stem=sym)["object"]))
                argnames = []
                for arg in stream.args:
                    if arg.role == "const":
                        continue
                    bn = f"c{i}_{arg.name}"
                    argnames.append(bn)
                    dt = np.int32 if arg.dtype == "i32" else np.int8
                    if arg.name in inputs:
                        buffers[bn] = np.ascontiguousarray(inputs[arg.name].astype(dt).reshape(arg.shape))
                    else:
                        buffers[bn] = np.zeros(arg.shape, dt)
                        outputs.add(bn)
                        if arg.name in ref:
                            checks.append(runner.Check(bn, ref[arg.name].astype(dt), 0, f"{name}.{arg.name}_vs_ref"))
                calls.append((sym, argnames))
        except Exception as exc:  # noqa: BLE001
            from mlir_oot.core_aten.iface import IfaceDecline
            row = dict(status="REFUSED" if isinstance(exc, IfaceDecline) else "ERROR", error=f"{type(exc).__name__}: {exc}"[-800:])
        row["seconds"] = round(time.time() - t0, 2)
        rows[name] = row
        ok &= row["status"] in ("PASS", "REFUSED")
        print(f"{name:28s} {row['status']} {row.get('mismatches', row.get('error', ''))} {row.get('buffer_problems') or ''}",
              flush=True)
    result = {"model_vs_reference": rows}
    if a.spike and calls:
        elf = runner.build_elf(runner.DeviceTest(a.batch, objects, buffers, calls, checks, outputs), wd)
        from mlir_oot.golden_device_catalog import final_elf_audit
        final_elf_audit(elf)
        r = runner.run_spike(elf)
        result["spike"] = dict(status=r.status, passed=len(r.passed), failed=r.failed, info=r.info,
                               seconds=round(r.seconds, 1), tail=None if r.ok else r.log[-3000:])
        print("spike", r.status, len(r.passed), "checks passed", r.failed or "", flush=True)
        ok &= r.ok
    if a.json:
        a.json.write_text(json.dumps(result, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
