## Exact int32 readout + host requant

The Gemmini backend returns dense row-major signed int32 accumulator tiles. Every
multiply, add, bias seed, and zero-point correction on Gemmini is integer-exact;
placement must establish int32 bounds for every partial sum. DMA readout uses the
full-width accumulator bit. MVIN and accumulator readout scales are unity by
default. A nonunity scale requires a recorded proof of exact input conversion and
exact FP32 multiplication over the entire reachable input interval, before RNE.

A host readout descriptor carries `scale_numerator` and `scale_denominator` as
reduced arbitrary-precision decimal integer strings, `zero_point`, `quant_min`,
`quant_max`, and `rounding="half_to_even"`. For one accumulator `a`, the result is
`clamp(RNE(a * numerator / denominator) + zero_point, quant_min, quant_max)`.
Zero point is added after rounding: odd zero points must not change tie parity.
ReLU, when present, clamps the exact real value to zero before quantization.
All source scale factors are the fp32-canonical values recorded by capture;
products and ratios of those factors are formed exactly, never rounded to fp32.

An affine readout may combine several independent int32 tiles and an exact
rational constant before this same single rounding. This represents distinct
source scales without approximating them into int8 weights or overflowing int32.
Broadcasts and permutations describe dense source-to-output indexing. Tiles remain
private and unchanged until the host readout has consumed them. These are ordinary
CPU operations, separated from the device by a completion fence.

`mlir_oot/core_aten/requant.py` provides the pure-Python reference (`evaluate`), an
exact transition-table generator (`thresholds`), and a portable C reference
(`c_reference`). A transition table has `quant_max-quant_min` signed int64 entries,
including INT32_MAX+1 for unreachable transitions. It is valid for all int32 input
values and arbitrary positive rational factors, including extreme fp32 exponents.
The C transition-table function takes `(int32_t *acc, uint8_t *out, size_t count,
int64_t *thresholds, int quant_min, int quant_max)` and writes storage bits; signed
int8 callers interpret those bits as int8. Negative/multitile affine factors use
the exact arbitrary-width integer host implementation in `host_readout.py`.
The C reference also exports a `_ratio` variant taking a signed int64 numerator
and positive uint64 denominator; its i32-by-i64 product and RNE are exact in
`__int128`, including negative factors and INT32_MIN.

The model2MLIR integer producer path retains the original exact readout IR on the
host (including runtime float bias and per-channel factors) and replaces only the
closed integer contraction. `scale` is optional informational metadata. Consumers
must use the rational pair, without dividing by `output_scale` again.

This section is the repository-local copy for the parent ABI_NOTES.md. The current
sandbox does not grant write access to the parent directory.
