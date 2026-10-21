"""Shared gating for device tests: Spike needs the tool variables."""

import os
import unittest

import runner  # noqa: F401  (re-exported to the device tests)

SPIKE_VARS = ("GEMMINI_LLVM_BIN", "GEMMINI_RISCV_TOOLS", "GEMMINI_ROCC_TESTS")

need_spike = unittest.skipUnless(all(os.environ.get(v) for v in SPIKE_VARS), "device tool variables unset")
