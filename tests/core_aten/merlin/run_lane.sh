#!/usr/bin/env bash
# Merlin's own build path for one Core ATen op, routed through this backend's region offload.
#   run_lane.sh OVERLOAD WORKDIR
# Needs: PY (Merlin interpreter, live Merlin src + merlin-experiments on PYTHONPATH), M2M_PY (model2MLIR
# interpreter), CASES (PyTorchFullCoverageKernelSet), CAPTURES (Merlin fp32 captures of the corpus),
# MERLIN_CORE_ATEN_EXECUTION_PROVIDER, GMLIR (this checkout) and Merlin's usual Spike/LLVM variables.
# 1. Merlin builds the bundle (inputs and semantic I/O contract) from its fp32 capture.
# 2. The case is re-captured PT2E-style: per-tensor int8 quantize/dequantize around the op
#    (contraction biases int32 at s_a s_w), with the bundle's output contract.
# 3. spike_model.build lowers it with DeviceRouting.region_offload = provider.core_aten_offload,
#    links the shim and kernels, runs Spike and counts the executed accelerator instructions.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ov="$1"; d="$2"
rm -rf "$d"
"$PY" "$here/make_bundle.py" "$CASES/cases/core_aten_test_cases.json" "$CAPTURES" "$d" "$ov" >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$M2M_PY" "$here/cap_case.py" "$CASES/cases/core_aten_test_cases.json" "$ov" "$d" >/dev/null
"$PY" "$here/run_region.py" "$d"
