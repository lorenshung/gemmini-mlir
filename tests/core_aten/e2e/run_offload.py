"""Whole-graph path: capture -> catalog -> offload rewrite -> standard MLIR lowering -> Spike/RTL.

Unlike ``run_e2e.py`` (which calls the kernels directly), this lowers the rewritten model with an
upstream ``mlir-opt`` pipeline, so the bound regions are reached through ``func.call`` and the
generated C-interface shim (``mlir_oot.core_aten.offload``) exactly as a host compiler would
reach them.  The model's own ``_mlir_ciface_forward`` is called from a bare-metal driver and its
result is compared with the exact-arithmetic reference. Eager-float divergence is informational.

    python run_offload.py CAPTURE_DIR [--only name ...] [--rtl] [--batch NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "device"))

import runner  # noqa: E402
from mlir_oot.core_aten.catalog import compile_catalog  # noqa: E402
from mlir_oot.core_aten.emit import llvm_bin_dir  # noqa: E402
from mlir_oot.core_aten.offload import rewrite, shim_source  # noqa: E402

PIPELINE = ["--allow-unregistered-dialect", "--canonicalize",
            "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
            "--convert-linalg-to-loops", "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
            "--finalize-memref-to-llvm", "--convert-arith-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts"]

RUNTIME = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
static uint8_t ca_heap[64u << 20] __attribute__((aligned(64)));
static size_t ca_top = 0;
void *malloc(size_t n) { size_t a = (ca_top + 63) & ~(size_t)63; ca_top = a + n; return ca_top <= sizeof ca_heap ? ca_heap + a : 0; }
void *aligned_alloc(size_t al, size_t n) { (void)al; return malloc(n); }
void free(void *p) { (void)p; }
void exit(int);
int puts(const char *s) { return printf("%s\n", s); }
void ca_abort(const char *why) { printf("FAIL abort %s\n", why); exit(1); }
"""


def _ctype(dt):
    return {"int8": "int8_t", "int32": "int32_t"}[dt]


def build_case(case: Path, wd: Path) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    manifest = compile_catalog(case / "model.mlir", wd)
    (wd / "offloaded.mlir").write_text(rewrite((case / "model.mlir").read_text(), manifest) + "\n")
    (wd / "shim.c").write_text(shim_source(manifest))
    sys.path.insert(0, str(HERE.parent))
    from exact_oracle import evaluate, host_guard_values
    from mlir_oot.core_aten.catalog import build
    from run_e2e import _key, _return_sources, DeviceRefusal
    text = (case / "model.mlir").read_text()
    _, _, matches = build(text)
    values = {("arg", int(k[3:])): v for k,v in np.load(case / "inputs.npz").items()}
    values.update(host_guard_values(text, dict(np.load(case / "inputs.npz"))))
    for binding, match in zip(manifest["bindings"], matches):
        if not all(_key(src) in values for src in binding["operands"].values()):
            raise DeviceRefusal(str(manifest["declines"]))
        exact = evaluate(match, {name: values[_key(src)] for name,src in binding["operands"].items()})
        (name, src), = binding["result"].items()
        values[_key(src)] = exact[name]
    if not all(src in values for src in _return_sources(text)):
        raise DeviceRefusal(str(manifest["declines"]))
    exact_outputs = [values[src] for src in _return_sources(text)]
    bindir = llvm_bin_dir()
    subprocess.run([str(bindir / "mlir-opt"), str(wd / "offloaded.mlir"), *PIPELINE, "-o", str(wd / "host.mlir")],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(bindir / "mlir-translate"), "--mlir-to-llvmir", str(wd / "host.mlir"), "-o",
                    str(wd / "host.ll")], check=True, capture_output=True, text=True)
    subprocess.run([str(bindir / "llc"), "-mtriple=riscv64-unknown-elf", "-mattr=+m,+a,+f,+d,+c", "-target-abi=lp64d",
                    "-code-model=medium", "-O2", "-filetype=obj", str(wd / "host.ll"), "-o", str(wd / "host.o")],
                   check=True, capture_output=True, text=True)
    inputs = np.load(case / "inputs.npz")
    exps = exact_outputs
    names = sorted(inputs.files, key=lambda k: int(k[3:]))
    c = ['#include "ca_harness.h"', RUNTIME]
    asm = [".section .data"]
    blobs = {nm: inputs[nm] for nm in names} | {f"expected{k}": e for k, e in enumerate(exps)}
    for nm, arr in blobs.items():
        (wd / f"{nm}.bin").write_bytes(np.ascontiguousarray(arr).tobytes() or b"\0")
        asm += [f".global ca_{nm}", ".balign 64", f"ca_{nm}:", f'.incbin "{wd / (nm + ".bin")}"']
        c.append(f"extern {_ctype(str(arr.dtype))} ca_{nm}[];")
    ranks = sorted({inputs[n].ndim for n in names} | {e.ndim for e in exps})
    for r in ranks:
        c.append(f"typedef struct {{ void *allocated; void *aligned; int64_t offset; int64_t sizes[{r}]; "
                 f"int64_t strides[{r}]; }} md{r};")
    if len(exps) == 1:
        rtype = f"md{exps[0].ndim}"
    else:
        c.append("typedef struct { " + " ".join(f"md{e.ndim} r{k};" for k, e in enumerate(exps)) + " } results_t;")
        rtype = "results_t"
    c.append(f"extern void _mlir_ciface_forward({rtype} *res, "
             + ", ".join(f"md{inputs[n].ndim} *" for n in names) + ");")
    c.append("int main(void) {")
    for n in names:
        a = inputs[n]
        st = [int(np.prod(a.shape[i + 1:])) for i in range(a.ndim)]
        c.append(f"  md{a.ndim} d_{n} = {{ ca_{n}, ca_{n}, 0, {{{', '.join(map(str, a.shape))}}}, {{{', '.join(map(str, st))}}} }};")
    c.append(f"  {rtype} res;")
    c.append("  uint64_t t0 = ca_rdcycle();")
    c.append("  _mlir_ciface_forward(&res, " + ", ".join(f"&d_{n}" for n in names) + ");")
    c.append("  ca_fence();")
    c.append(f'  printf("PERF {case.name} cycles=%lu\\n", (unsigned long)(ca_rdcycle() - t0));')
    for k, e in enumerate(exps):
        field = "res" if len(exps) == 1 else f"res.r{k}"
        tag = case.name if len(exps) == 1 else f"{case.name}.out{k}"
        c.append(f'  ca_check_{"i8" if e.dtype == np.int8 else "i32"}("{tag}_offload_vs_exact_arithmetic", '
                 f"({_ctype(str(e.dtype))} *){field}.aligned + {field}.offset, ca_expected{k}, {e.size}, 0);")
    c.append("  return ca_finish();")
    c.append("}")
    (wd / "main.c").write_text("\n".join(c) + "\n")
    (wd / "blobs.S").write_text("\n".join(asm) + "\n")
    tools = runner.env("GEMMINI_RISCV_TOOLS")
    common = runner.env("GEMMINI_ROCC_TESTS") / "riscv-tests" / "benchmarks" / "common"
    elf = wd / f"{case.name}.elf"
    cmd = [str(tools / "bin" / "riscv64-unknown-elf-gcc"), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
           "-std=gnu99", "-O2", "-fno-common", "-fno-builtin-printf", "-fno-builtin-malloc",
           "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-mabi=lp64d", f"-I{HERE.parent / 'device'}",
           f"-I{common}", f"-I{runner.env('GEMMINI_ROCC_TESTS') / 'riscv-tests' / 'env'}", "-nostdlib",
           "-nostartfiles", "-static", "-T", str(common / "test.ld"), "-Wl,--no-warn-rwx-segments",
           str(wd / "main.c"), str(wd / "shim.c"), str(wd / "blobs.S"), str(wd / "host.o"), str(wd / "core_aten.o"),
           str(common / "syscalls.c"), str(common / "crt.S"), "-lm", "-lgcc", "-o", str(elf)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    from mlir_oot.golden_device_catalog import final_elf_audit
    final_elf_audit(elf)
    return elf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--rtl", action="store_true")
    ap.add_argument("--batch", default="offload")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    cases = sorted(p for p in a.capture.iterdir() if (p / "model.mlir").exists())
    if a.only:
        cases = [c for c in cases if c.name in a.only]
    rows = []
    for case in cases:
        try:
            elf = build_case(case, runner.work_root() / a.batch / case.name)
            r = (runner.run_rtl if a.rtl else runner.run_spike)(elf)
            rows.append(dict(case=case.name, status=r.status, info=r.info, failed=r.failed, perf=r.perf,
                             tail=None if r.ok else r.log[-1500:]))
        except Exception as exc:  # noqa: BLE001
            from run_e2e import DeviceRefusal
            if isinstance(exc, DeviceRefusal):
                rows.append(dict(case=case.name, status="REFUSED", reason=str(exc)))
                print(case.name, "device refusal:", str(exc), flush=True)
                continue
            rows.append(dict(case=case.name, status="ERROR", error=f"{type(exc).__name__}: {str(exc)[-1500:]}"))
        print(f"{case.name:32s} {rows[-1]['status']} {rows[-1].get('info') or rows[-1].get('error', '')}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["status"] in ("PASS", "REFUSED") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
