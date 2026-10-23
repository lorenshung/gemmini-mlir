"""Shared gating for device tests: Spike needs the tool variables."""

import os
import unittest

import numpy as np
import runner  # noqa: F401  (re-exported to the device tests)

SPIKE_VARS = ("GEMMINI_LLVM_BIN", "GEMMINI_RISCV_TOOLS", "GEMMINI_ROCC_TESTS")

need_spike = unittest.skipUnless(all(os.environ.get(v) for v in SPIKE_VARS), "device tool variables unset")


def conv_ref(cs, X, W):
    """Exact int64 reference for a padded 3x3 GoldenConv (HWC input, tap-major weights)."""
    Xp = np.zeros((cs.h + 2, cs.w + 2, cs.cin), np.int64)
    Xp[1:-1, 1:-1] = X
    out = np.zeros((cs.oh, cs.ow, cs.cout), np.int64)
    for kh in range(3):
        for kw in range(3):
            patch = Xp[kh:kh + cs.stride * (cs.oh - 1) + 1:cs.stride, kw:kw + cs.stride * (cs.ow - 1) + 1:cs.stride]
            out += patch @ W[(kh * 3 + kw) * cs.cin:(kh * 3 + kw + 1) * cs.cin].astype(np.int64)
    return out
