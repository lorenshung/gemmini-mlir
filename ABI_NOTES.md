# Core ATen Gemmini source/catalog ABI

This repository-local note supplements the shared Merlin ABI notes. The interface
is source MLIR plus `gemmini_core_aten_catalog_v1`: `source_sha256` authenticates
the bytes; each binding identifies its `source_operation_ordinal`, `root_ordinal`,
`replaced_ordinals`, operands, result and kernel symbol. Bindings must cover each
live proposed integer producer exactly once. Dead producers are not emitted.
A clean decline carries a reason; a zero-kernel result claims no object path.
Provenance identifies source operations, never selects a numeric strategy.

Merlin owns host representation changes, packing, recombination, final dtype
semantics and generic host code generation. This backend owns target schedules,
resource/bound proofs, instruction encoding, hazard lint and target ABI glue.
The local host requantizers are prototypes for promotion to Merlin, not a second
independent numeric contract.

The family contracts live in two documents. [The exact epilogue](docs/core_aten_exact_epilogue.md)
covers the multi-term readout, combining arithmetic on Gemmini, device ReLU,
centered byte planes and raw/readout-backed integer producers. [The dtype
families](docs/core_aten_bitwise_movement.md) cover F1 byte movement, F2/F4
signed-byte components, F3 quantized storage codes, layout/DMA schedules and the
compiler budgets. This note keeps what they share and the Phase 1 and VCS rules.

## Exact int32 readout

Device arithmetic is integer-exact. Every mesh dot product and every accumulator
prefix, including seeds and zero-point corrections, needs a nonoverflow bound.
Independent WS dot products split K into at most 16 elements; exact i32 SRAM
accumulation combines them. Arithmetic MVOUT uses full-width signed int32, dense
row-major storage. MVIN scales and CONFIG_ST accumulator scales are unity in
these schedules. A nonunit data-path scale would require a recorded exact-product
proof; none of the families depends on it. Fences establish DMA completion
before host reads, overlapping local writes and buffer reuse.

The readout equation is the exact epilogue's: one RNE after the complete rational
sum and optional ReLU, zero point after rounding, then clamp. Private tiles remain
unchanged until their last host consumer. A consumer of a readout-backed integer
producer must not divide the recorded ratio by output_scale again.

Readout parameters carry reduced decimal-string `scale_numerator` and positive
`scale_denominator`, `zero_point`, `quant_min`, `quant_max`, and
`rounding="half_to_even"`. A single term is `v=a*numerator/denominator`;
multi-term `Readout.terms` and `.constant` express that equation. Exact
scalar conversion accepts Python/NumPy scalars and 0-D arrays via `.item()`;
positive-rank arrays require the explicit broadcast/channel path.

### Phase 1 plain exact host readout

The emitter records `host_readout_contract="exact_rational_RNE_v1"` in kernel
metadata and `gemmini.host_readout_contract` on an emitted module with readouts.
It puts each exact rational expression over a common positive denominator.
Host integer widths cover the entire signed-int32 domain of every input tile,
the constant, twice the numerator and all half-integer output boundaries.
A bounded search compares that exact numerator to half-integer boundaries,
including parity at a tie; it implements the equation above without FP or an
intermediate rounding. No multiply-shift approximation, stored transition table,
per-channel algorithm certificate or recorded reachable-range specialization is
used. Channel, broadcast, permutation and placement semantics are unchanged.

Readouts execute serially at their fenced `after` boundary (or after the complete
stream). There are no overlap slots, slot offsets, partial-index parameters or
`readout_pipeline`/`host_requantization_proofs` records. Each complete private i32
tile remains available for its host consumers. Merlin-owned retained-source
readouts are unchanged; this emitter contract applies to provider-owned readouts.

## Phase 1 placement

This provider has no cost gate or profitability policy. A native family needs
its semantic constraints, source-bound exactness proof and legal target schedule.
No runtime samples or expected outputs establish those proofs. Unsupported or
unsafe constructions decline with a reason. Generic hack detection and native
dtype semantics are Merlin obligations; a provider proof never waives them.
Structural compact DMA loops remain because supported model-size layouts need
them to fit compiler memory/code-generation budgets. Literal folding and the
requant-A performance implementation are preserved only on `gemmini-perf`.

## VCS RTL engine

The backend accepts `vcs` explicitly in `available`, `run_elf` and
`simulator_identity`, with oracle kind `rtl_vcs`. Discovery uses
`MERLIN_GEMMINI_SIMV`, then Chipyard's
`sims/vcs/simv-chipyard.harness-<rtl_sim_config>`. The VCS config override is
`MERLIN_GEMMINI_VCS_CONFIG`; Verilator's override does not select a VCS build.
An explicit VCS request never falls back to another engine.

Execution follows Chipyard's permissive plusarg convention: DRAMSim/ini directory,
cycle cap, seed, `+loadmem=<absolute ELF>`, then `+permissive-off <ELF>`.
Loadmem is unconditional and is the sole image loader, including timing runs;
the positional ELF supplies symbols/HTIF metadata. Cycle counts may be compared
only under this same loading convention, never against TSI-loaded receipts.
`-no_save` avoids VCS save/restore's ASLR re-exec in restricted sandboxes.
The ELF directory is the writable cwd; stdin is `/dev/null`. The caller owns
license environment/access and the wall timeout. DRAM geometry is the selected
RTL model, with the ordinary target linker/arena contract.

Identity binds the executable hash and a sorted recursive `.daidir` file
manifest, its `daidir_sha256`, and a combined `digest`. Missing or incomplete
runtime trees have no complete digest. Native invocation records also include
daidir and DRAM ini files as dependencies. The grader must grant the binary,
its resolved `.daidir`, vendor/runtime shared libraries, ini directory and ELF,
plus writable work/tmp directories. Preserve the site's license environment
and authorized network or file access; the backend supplies no license server.
Verilator and GSIM interfaces remain unchanged.
