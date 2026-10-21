"""Shared gating for device tests: Spike needs the tool variables."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402

SPIKE_VARS = ("GEMMINI_LLVM_BIN", "GEMMINI_RISCV_TOOLS", "GEMMINI_ROCC_TESTS")

need_spike = unittest.skipUnless(all(os.environ.get(v) for v in SPIKE_VARS), "device tool variables unset")
