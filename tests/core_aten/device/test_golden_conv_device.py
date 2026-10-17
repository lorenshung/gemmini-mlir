"""Padded 3x3 GoldenConv on Spike and RTL: boundary lanes are zero-filled, never stale."""

import unittest
from pathlib import Path

import numpy as np

from common import need_rtl, need_spike, runner
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.golden_conv import ConvShape, GoldenConv
from mlir_oot.golden_device_lower import lower

CASES = [ConvShape(5, 19, 20, 19, 1), ConvShape(6, 20, 32, 16, 2)]


def conv_ref(cs, X, W):
    Xp = np.zeros((cs.h + 2, cs.w + 2, cs.cin), np.int64)
    Xp[1:-1, 1:-1] = X
    out = np.zeros((cs.oh, cs.ow, cs.cout), np.int64)
    for kh in range(3):
        for kw in range(3):
            patch = Xp[kh:kh + cs.stride * (cs.oh - 1) + 1:cs.stride, kw:kw + cs.stride * (cs.ow - 1) + 1:cs.stride]
            out += patch @ W[(kh * 3 + kw) * cs.cin:(kh * 3 + kw + 1) * cs.cin].astype(np.int64)
    return out.astype(np.int32)


def build(name: str) -> Path:
    wd = runner.work_root() / name
    objects, buffers, calls, checks = [], {}, [], []
    rng = np.random.default_rng(7)
    for i, cs in enumerate(CASES):
        module = lower(GoldenConv(cs).build())
        obj = compile_to_object(module, wd, stem=f"conv{i}")["object"]
        sym = f"conv{i}_k"
        # Rename the shared symbol per case; each object defines gemmini_golden_conv.
        renamed = wd / f"conv{i}_r.o"
        import subprocess, os
        objcopy = Path(os.environ["GEMMINI_RISCV_TOOLS"]) / "bin" / "riscv64-unknown-elf-objcopy"
        subprocess.run([str(objcopy), f"--redefine-sym=gemmini_golden_conv={sym}", obj, str(renamed)], check=True)
        objects.append(renamed)
        # Fill the input with a bias so stale scratchpad data would be visible in border outputs.
        X = rng.integers(-128, 128, (cs.h, cs.w, cs.cin)).astype(np.int8)
        W = rng.integers(-127, 128, (9 * cs.cin, cs.cout)).astype(np.int8)
        buffers.update({f"x{i}": X, f"w{i}": W, f"y{i}": np.zeros((cs.oh, cs.ow, cs.cout), np.int32)})
        calls.append((sym, [f"x{i}", f"w{i}", f"y{i}"]))
        checks.append(runner.Check(f"y{i}", conv_ref(cs, X, W).reshape(-1), 0, f"conv{i}_pad1_vs_ref"))
    test = runner.DeviceTest(name, objects, buffers, calls, checks, {f"y{i}" for i in range(len(CASES))})
    return runner.build_elf(test, wd)


class TestGoldenConvDevice(unittest.TestCase):
    @need_spike
    def test_spike(self):
        r = runner.run_spike(build("golden_conv_pad1"))
        self.assertTrue(r.ok, r.log[-2000:])

    @need_rtl
    def test_rtl(self):
        r = runner.run_rtl(build("golden_conv_pad1_rtl"))
        self.assertTrue(r.ok, r.log[-2000:])


if __name__ == "__main__":
    unittest.main()
