"""Build and run bare-metal device tests of compiler-emitted Core ATen kernels.

Environment (all explicit; nothing is guessed from the caller's checkout):
  GEMMINI_LLVM_BIN     LLVM/MLIR bin directory with mlir-translate and clang or llc
  GEMMINI_RISCV_TOOLS  riscv-tools prefix with riscv64-unknown-elf-gcc, spike, lib/libgemmini.so
  GEMMINI_ROCC_TESTS   gemmini-rocc-tests checkout (riscv-tests benchmarks common: crt.S, syscalls.c, test.ld)
  GEMMINI_RTL_SIM      Verilator GemminiRocketConfig simulator (RTL runs only)
  GEMMINI_DRAMSIM_INI  dramsim2_ini directory for the RTL simulator
  GEMMINI_TEST_WORK    scratch directory for build products (default: ./out/core_aten_device)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def env(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is not set")
    return Path(value)


def work_root() -> Path:
    root = Path(os.environ.get("GEMMINI_TEST_WORK", "out/core_aten_device")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Check:
    buffer: str
    expected: np.ndarray
    tol: int
    label: str
    count: int | None = None      # leading elements to compare (default: all of expected)


@dataclass
class DeviceTest:
    name: str
    objects: list[Path]                                  # compiled kernel objects
    buffers: dict[str, np.ndarray]                        # initial contents (outputs: sentinel-filled)
    calls: list[tuple[str, list[str]]]                    # (symbol, argument buffer names)
    checks: list[Check] = field(default_factory=list)
    outputs: set[str] = field(default_factory=set)        # refilled with a sentinel before the calls


def _ctype(arr: np.ndarray) -> str:
    return {np.dtype(np.int8): "int8_t", np.dtype(np.int32): "int32_t"}[arr.dtype]


def write_sources(test: DeviceTest, wd: Path) -> tuple[Path, Path]:
    """C driver plus an assembly blob file; tensors are .incbin'd, not printed as C text."""
    wd.mkdir(parents=True, exist_ok=True)
    asm = [".section .data", ""]
    lines = ['#include "ca_harness.h"', ""]
    blobs = dict(test.buffers)
    for i, chk in enumerate(test.checks):
        blobs[f"exp{i}"] = chk.expected
    for name, arr in blobs.items():
        arr = np.ascontiguousarray(arr)
        path = wd / f"{name}.bin"
        data = arr.tobytes()
        if not data:
            data = b"\0"
        path.write_bytes(data + b"\0" * (-len(data) % 64))
        asm += [f".global ca_{name}", ".balign 64", f"ca_{name}:", f'.incbin "{path}"', ""]
        lines.append(f"extern {_ctype(arr)} ca_{name}[];")
    lines.append("")
    symbols = sorted({s for s, _ in test.calls})
    for s in symbols:
        nargs = {len(a) for sym, a in test.calls if sym == s}
        assert len(nargs) == 1
        lines.append(f"extern void {s}({', '.join(['void *'] * nargs.pop())});")
    lines += ["", "int main(void) {"]
    for name in sorted(test.outputs):
        lines.append(f"  ca_fill(ca_{name}, {test.buffers[name].nbytes}, 0x5a);")
    lines.append("  ca_fence();")
    lines.append("  uint64_t t0 = ca_rdcycle();")
    for sym, args in test.calls:
        lines.append(f"  {sym}({', '.join('ca_' + a for a in args)});")
    lines.append("  ca_fence();")
    lines.append("  uint64_t t1 = ca_rdcycle();")
    lines.append(f'  printf("PERF {test.name} cycles=%lu\\n", (unsigned long)(t1 - t0));')
    for i, chk in enumerate(test.checks):
        fn = "ca_check_i8" if chk.expected.dtype == np.int8 else "ca_check_i32"
        n = chk.count if chk.count is not None else chk.expected.size
        lines.append(f'  {fn}("{chk.label}", ca_{chk.buffer}, ca_exp{i}, {n}, {chk.tol});')
    lines += ["  return ca_finish();", "}", ""]
    c_path, s_path = wd / "main.c", wd / "blobs.S"
    c_path.write_text("\n".join(lines))
    s_path.write_text("\n".join(asm))
    return c_path, s_path


def build_elf(test: DeviceTest, wd: Path | None = None) -> Path:
    wd = wd or work_root() / test.name
    c_path, s_path = write_sources(test, wd)
    tools = env("GEMMINI_RISCV_TOOLS")
    common = env("GEMMINI_ROCC_TESTS") / "riscv-tests" / "benchmarks" / "common"
    elf = wd / f"{test.name}.elf"
    cmd = [str(tools / "bin" / "riscv64-unknown-elf-gcc"), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
           "-std=gnu99", "-O2", "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-mabi=lp64d",
           f"-I{HERE}", f"-I{common}", f"-I{env('GEMMINI_ROCC_TESTS') / 'riscv-tests' / 'env'}", "-nostdlib", "-nostartfiles", "-static", "-T", str(common / "test.ld"),
           "-Wl,--no-warn-rwx-segments", str(c_path), str(s_path), *map(str, test.objects),
           str(common / "syscalls.c"), str(common / "crt.S"), "-lm", "-lgcc", "-o", str(elf)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"ELF build failed: {p.stderr[-2000:]}")
    return elf


@dataclass
class RunResult:
    status: str
    passed: list[str]
    failed: list[str]
    info: list[str]
    perf: dict[str, int]
    exit_code: int
    seconds: float
    log: str

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


def parse(log: str, code: int, seconds: float, rtl: bool = False) -> RunResult:
    lines = log.splitlines()
    passed = [l.split()[1] for l in lines if l.startswith("PASS ")]
    failed = [l for l in lines if l.startswith("FAIL ")]
    info = [l for l in lines if l.startswith("INFO ")]
    perf = {}
    for l in lines:
        m = re.match(r"PERF (\S+) cycles=(\d+)", l)
        if m:
            perf[m.group(1)] = int(m.group(2))
    done = [l for l in lines if l.startswith("HARNESS_DONE")]
    asserted = any("Assertion failed" in l or "assertion" in l.lower() and "fail" in l.lower() for l in lines)
    if done and "fail=0" in done[-1] and code == 0 and not failed and not asserted:
        status = "PASS"
    elif done:
        status = "FAIL"
    else:
        status = "ERROR"
    return RunResult(status, passed, failed, info, perf, code, seconds, log)


def run_spike(elf: Path, timeout: int = 1800) -> RunResult:
    tools = env("GEMMINI_RISCV_TOOLS")
    cmd = [str(tools / "bin" / "spike"), f"--extlib={tools / 'lib' / 'libgemmini.so'}", "--extension=gemmini", str(elf)]
    t0 = time.time()
    envv = dict(os.environ, LD_LIBRARY_PATH=f"{tools / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=envv)
        log, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        log, code = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), 124
    return parse(log, code, time.time() - t0)


def run_rtl(elf: Path, timeout: int = 7200, max_cycles: int = 400_000_000) -> RunResult:
    sim = env("GEMMINI_RTL_SIM")
    ini = env("GEMMINI_DRAMSIM_INI")
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=work_root()) as tmp:
        cmd = ["bash", "-c",
               f"ulimit -s unlimited; exec {sim} +permissive +dramsim +dramsim_ini_dir={ini} "
               f"+max-cycles={max_cycles} +loadmem={elf} +permissive-off {elf} </dev/null"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
            log, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            log, code = "TIMEOUT", 124
    log = "\n".join(l for l in log.splitlines() if not re.match(r"^C\d+: ", l))
    return parse(log, code, time.time() - t0, rtl=True)
