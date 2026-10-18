# Exact host multiply-shift readout

Task E changes the existing full-int32 Gemmini readout ABI's host epilogue.
There is one quantize rounding, using integer arithmetic only. Parameter classes
come from the current readout's rational terms, constant and tensor shape.
No source IDs, model names or calibration samples select the algorithm.

The compile-time certificate is in `kernels[].meta.host_requantization_proofs`.
Each record authenticates the coefficient class, element count, accumulator
intervals and their origin, combined numerator interval, denominator, magic
multiplier, shift, inequality operands, and verification coverage. Bounds are
part of the kernel digest because changing them can change emitted code.

The isolated `mlir_oot/core_aten/host_magic.py` module is a generic prototype for
promotion to Merlin's host generator. This task was restricted to this worktree;
it introduces no external dependency. On integration, move this arithmetic and
its semantic checks into Merlin and delegate from the OOT epilogue, retaining
the Gemmini tile-bound metadata and instruction audits here. Do not maintain a
second shared requantizer in the OOT backend.

## Certificate

Put the exact rational expression over a common positive denominator:

```
N = constant + sum(coeff[j] * accumulator[j])
result = clamp(RNE(N / D) + zero_point)
```

For each recorded tile interval, interval arithmetic bounds the product and
every intermediate sum, including constants and negative coefficients. Missing
tile bounds conservatively mean the complete signed-int32 storage range.
Zero-point correction tiles have their own bounds. All signed intermediates
must fit strictly between `-2^63` and `2^63`; `0 < D < 2^63`. The rounded
quotient must also fit signed int64. Unsupported widths retain the prior exact
eight-comparison search, including arbitrarily large rational parameters.

Let `A=max(abs(N_min), abs(N_max))`. Search shifts `s=0..127` for

```
M = ceil(2^s / D), M < 2^64
E = M*D - 2^s >= 0
A*E < 2^s
```

For every integer `0 <= n <= A`, write `n=q*D+r`, `0 <= r < D`. Then

```
n*M/2^s = q + r/D + n*E/(D*2^s)
q <= n*M/2^s < q + (D-1)/D + 1/D = q+1
```

Thus `q = (n*M) >> s` is the **exact floor quotient over the entire interval**.
Both multiplicands fit uint64 and their product fits uint128. LLVM lowers this
to the RV64 `mul`/`mulhu` pair, followed by constant shifts. A multiplier of one
needs only a shift. No division or floating-point instruction is needed.

Compute the exact remainder `r=n-q*D`. Increment `q` precisely when
`r>D-r`, or `r==D-r` and `q` is odd. This proves every tie: when integral,
`n=(2*k+1)*D/2` has quotient `k`, remainder `D/2`, and increments iff `k` is
odd. Signed symmetry `RNE(-x)=-RNE(x)` proves both signs for every `k`, including
intervals with billions of ties. Restore the sign, clamp in the unshifted code
domain, then add the zero point. Moving the clamp before that addition is
algebraically identical and prevents signed addition overflow. ReLU, when
requested, acts on `N` before rounding.

## Compile-time checks and fallback

Intervals of at most 65,537 integer numerators are exhaustively compared with
`round(Fraction(N,D))`. Independently, when the Cartesian product of recorded
accumulator intervals has at most 65,536 tuples, every tuple is checked against
that rational reference. This includes sparse reachable numerators whose
conservative interval is much larger. Checks are cached by arithmetic inputs.

Larger ranges retain the formal inequality certificate and test both extremes,
both signs, zero, and neighborhoods of every rounding boundary and clamp edge
in the declared output range. Every tie outside that range remains covered by
the symbolic tie proof above. Verification mismatches or failure to construct
a safe multiplier select the comparison-search fallback, with a recorded reason.

`tests/core_aten/test_host_magic.py` checks the full-int32-domain certificate,
tie families, near-int64 limits, intermediate overflow refusal, metadata and
deduplication. Native LLVM execution compares multiterm expressions and shifts
above 64 with an independent rational reference. The existing host readout tests
also execute the fast and fallback paths, signed scales, offsets, per-element
parameters, ReLU, byte quantization schemes and extreme rational factors.

## Measurement method

`tools/measure_host_requant.py` compiles three frozen Phase 0 source kernels:
mean.dim, avg_pool2d and add.Tensor. It uses their concrete shapes, qparams and
recorded accumulator bounds. Deterministic generated int8 inputs are checked
against the independent source oracle. The complete kernels execute once;
paired old/new host-only epilogues then process the same resulting int32 tiles.
Each timing is five warmed trials of ten calls by default (configurable), with output equality checked.
The old emitter is loaded from a separate file extracted from the base revision.
There is no production flag that selects algorithms by workload.

Reported savings concern the host epilogues only. Spike cycle CSR deltas belong
to its instruction model; RTL `mcycle` deltas are Rocket hardware cycles. Neither
is a full-model or FireSim measurement. The final linked ELF must pass the
independent design-rule checker and emitted objects are checked for floating
point instructions. At most one RTL simulation is needed for the paired ELF.
