"""Shared gating for device tests: Spike needs the tool variables, RTL also GEMMINI_RUN_RTL=1."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner  # noqa: E402

SPIKE_VARS = ("GEMMINI_LLVM_BIN", "GEMMINI_RISCV_TOOLS", "GEMMINI_ROCC_TESTS")
RTL_VARS = SPIKE_VARS + ("GEMMINI_RTL_SIM", "GEMMINI_DRAMSIM_INI")

need_spike = unittest.skipUnless(all(os.environ.get(v) for v in SPIKE_VARS), "device tool variables unset")
need_rtl = unittest.skipUnless(all(os.environ.get(v) for v in RTL_VARS) and os.environ.get("GEMMINI_RUN_RTL") == "1",
                               "RTL runs need GEMMINI_RUN_RTL=1 and the simulator variables")
