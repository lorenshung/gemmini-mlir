# Gemmini capsule compiler

This self-contained Python compiler parses the typed sources with xDSL and emits LLVM/RoCC through the four declared entrypoints. It implements tiled weight-stationary contractions, direct DMA convolution gathers, accumulator movement and bias, bounded-error residual coefficient products, scaled/independent-column readout, and hardware maxpool. Captured programs retain source-bound host/device tasks and operation ownership. Runtime loops bound code growth.

Mandatory discovery completed before submission mutation: CCA bijection/escalation, selected RTL facts, derived levers, and scaffold generation. The public conformance report credits that sequence and reports no authored regex calls.

Measured validation:

- All 147 public non-model inputs parse and emit LLVM.
- All six captured sources pass stock MLIR verification and public source-plan preflight.
- The first complete sweep certified 152 of 153 public capsules, including all six captured programs with clean placement/host-compute checks and complete eligible-source coverage.
- The initial convolution scale ladder accepted every requested emission case within the 1 MiB resource limit. The bounded object-check tool returned source-preparation `ValueError` without cases; that is unavailable evidence.
- All 39 direct convolution/residual members passed their mandatory tiers after direct DMA gathers and coefficient products replaced CPU patch copies and operand-scale rounding.
- The remaining carried-state capsule was numerically exact on both simulators but lacked resolvable resident-load trace identities. Reused RHS handles now load once into a disjoint SRAM region and retain those weights across commits. Its focused certification is pending at this report's freeze.

The immutable final public receipt, compiler digest and retained timing/trace records are written alongside the package in `artifacts/`. Read the receipt associated with the frozen digest for the final certification result. Source-plan preflight and emission are not numerical certification.

Target identity: FireSimGemminiRocketConfig, 16x16 mesh, 64 KiB accumulator; selected facts SHA256 `b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af`. Actual gSIM cycle observations and simulator identities are retained without claiming timing-model accuracy. Spike timing is untuned.

Limits: native pooling windows are at most 3x3 with the declared target padding value; nonunit i32 residual readout is declined. Quantized residuals use Q=1024 coefficients under the source's output-step error allowance. Complete RHS residency requires its padded tiles to fit the selected SRAM after reserved staging rows; larger RHS operands use tiled reloads. External full-model validation on the exact frozen compiler remains pending.

Continuation variant: native per-column matmul readout replaces temporary readout buffers and CPU payload scattering with source-scale-derived column runs. Focused certification is pending at this variant freeze; the immutable final receipt must establish the promoted candidate status.

The bounded hybrid admits at most64 native segments and at most twice the ordinary tile count. Dense alternating gains retain the certified existing path. This bounds additional static lowering and avoids excessive repeated contraction.

The bounded hybrid passed all four affected capsules on mandatory tiers. Grouped aligned/tail gSIM observations improved3396->616 and4269->990 cycles; dense independent-column observations remain4422/5064. Only the two grouped public emissions change, both shrinking. The prior bb3e compiler has a complete153/153 receipt; the promoted compiler requires its own complete receipt, pending at source freeze.
