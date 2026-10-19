# Candidate zero-initializer grouping

`merlin-support/backend/gemmini_zero_initializer_preparation.py` prepares a
separate copied compiler. It keeps the selected staged SDK, its source ledger,
and its numerical/effect declarations byte for byte. The enabled flag selects
ordinary candidate policy in `zero_initializer_policy.py`; the generic Merlin
source recognizer is copied exactly into candidate-owned `passes/`. This code
is not a new SDK export or a verifier grant.

For a sole-use private zero splat immediately before a signed rank-two
contraction, the candidate retains every original source operation in one task's
`source_op_indices`, omits only virtual initializer storage, and uses the
original target accumulator-zero, products and complete readout. Shared or
returned initializers remain materialized. Nonzero initializers retain the
original compiler refusal. Unknown semantic attributes do not authorize a
source group.

Focused tests check actual emitted C, command buffers, full source/task/ABI
accountability, preservation of returned outputs, unchanged selected sources and
refusals. A fresh native diagnostic on the sealed 17x16x32 signed source checked
all 544 output values with the original independent source/packet checker.
The disabled copied compiler reproduced the previous baseline command buffer and
LLVM byte for byte, so its original native/trace evidence was explicitly reused;
the enabled arm was freshly built, executed and traced. Its original complete
product/state/readout checks passed. Artifact:
`out/artifacts/target-adapters/gemmini/zero-initializer-native-v1/status.json`
in the managed Phase 2 pass-integration task.

This establishes a source-bound functional diagnostic, not a generally qualified
capability, hardware timing, or a latency improvement. The old staged pair
checker deliberately rejects its changed private ABI. A separate original
composite check is required before selecting this transformation through the
ordinary production capability route. The preserved Phase 1 package is unchanged.
