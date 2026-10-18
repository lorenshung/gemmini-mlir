"""Compile quantized op cases into one bare-metal ELF and run it on Spike and/or RTL.

Per case the ELF checks the device output against the kernel's bit-level model
and the exact-arithmetic reference, using equality of every output code.  Usage:

    python run_cases.py [--rtl] [--rep] [--only substr ...] [--batch NAME]
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
sys.path.insert(0, str(HERE.parent))

import runner  # noqa: E402
from mlir_oot.core_aten.emit import build_module, compile_to_object  # noqa: E402
from mlir_oot.core_aten.lint import lint  # noqa: E402
from mlir_oot.core_aten.stream import simulate  # noqa: E402


def device_test(cases, name: str) -> tuple[runner.DeviceTest, dict]:
    wd = runner.work_root() / name
    objects, buffers, calls, checks, outputs = [], {}, [], [], set()
    summary = {}
    for i, c in enumerate(cases):
        sym = f"ca_k{i}_{c.name}".replace(".", "_")
        t0 = time.time()
        lint(c.kernel.stream)
        module, tree = build_module(c.kernel.stream, sym)
        obj = compile_to_object(module, wd, stem=sym)["object"]
        objects.append(Path(obj))
        model = simulate(c.kernel.stream, c.inputs)
        argnames = []
        for a in c.kernel.args:
            bn = f"c{i}_{a.name}"
            argnames.append(bn)
            dt = np.int32 if a.dtype == "i32" else np.int8
            if a.role == "input":
                buffers[bn] = np.ascontiguousarray(c.inputs[a.name].astype(dt).reshape(a.shape))
            else:
                buffers[bn] = np.zeros(a.shape, dt)
                outputs.add(bn)
                checks.append(runner.Check(bn, model[a.name].reshape(-1), 0, f"{c.name}.{a.name}_vs_model"))
                if a.name in c.reference:
                    ref = c.reference[a.name].astype(dt).reshape(-1)
                    checks.append(runner.Check(bn, ref, c.tol, f"{c.name}.{a.name}_vs_ref_tol{c.tol}"))
                    e = np.abs(model[a.name].astype(np.int64).reshape(-1) - ref.astype(np.int64))
                    summary[f"{c.name}.{a.name}"] = dict(model_vs_ref_max=int(e.max()),
                                                          model_vs_ref_exact=float((e == 0).mean()),
                                                          instructions=len(c.kernel.stream.ins),
                                                          build_s=round(time.time() - t0, 2))
        calls.append((sym, argnames))
    return runner.DeviceTest(name, objects, buffers, calls, checks, outputs), summary


def run(cases, name: str, rtl: bool = False, spike: bool = True) -> dict:
    test, summary = device_test(cases, name)
    elf = runner.build_elf(test, runner.work_root() / name)
    result = {"elf": str(elf), "cases": [c.name for c in cases], "model_vs_ref": summary}
    for engine, fn in (("spike", runner.run_spike if spike else None), ("rtl", runner.run_rtl if rtl else None)):
        if fn is None:
            continue
        r = fn(elf)
        result[engine] = dict(status=r.status, passed=len(r.passed), failed=r.failed, info=r.info, perf=r.perf,
                              seconds=round(r.seconds, 1), exit=r.exit_code,
                              tail=None if r.ok else r.log[-3000:])
    return result


def main() -> int:
    import cases as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", action="store_true")
    ap.add_argument("--no-spike", action="store_true")
    ap.add_argument("--rep", action="store_true", help="representative shapes only")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--batch", default="paper")
    ap.add_argument("--json")
    args = ap.parse_args()
    cs = C.paper_cases(representative=args.rep)
    if args.only:
        cs = [c for c in cs if any(o in c.name for o in args.only)]
    res = run(cs, args.batch, rtl=args.rtl, spike=not args.no_spike)
    text = json.dumps(res, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text[-6000:])
    ok = all(res[e]["status"] == "PASS" for e in ("spike", "rtl") if e in res)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
