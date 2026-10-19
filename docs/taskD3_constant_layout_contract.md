# Literal layouts and the region-offload contract

This extends the [bounded layout schedule contract](core_aten_bitwise_movement.md#bounded-layout-schedules). It adds no numeric policy or rounding operation.

## Target implementation

`layout_match.describe` proves rectangular selectors from operation semantics, static shapes and indexing maps. Native storage and identical-qparam codes use scratchpad byte DMA, with affine CPU loop descriptors retained through emission. Different qparams retain the full-i32/exact-host-requant boundary. Interpreted arithmetic/selector fallbacks retain their explicit capacity refusal; large layouts never enter the dense linear constructor.

`constant_dma.specialize` partially evaluates these **same target DMA rectangles** on literal storage before kernel construction. It uses bounded NumPy byte views, not floating operations, index tensors, coefficient maps or boxed Python elements. Rectangles copy complete element byte groups. This proves preservation of signed zero, NaN payloads and integer codes. The output becomes `arith.constant`; the layout has no runtime binding. Constant-like splat/fill initializers retain their source form for other numeric proofs, while supplying literal storage to downstream layouts. Source operation ordinals refer to the original module even after replacement. Dead operand edges are removed before region closure checks.

The target's internal convolution layout conversions apply the same rule to literal code operands: the permutation is stored as a private constant, and the conversion emits **zero instructions**. Original and transformed byte hashes, shapes and permutations appear in `kernels[].meta.constant_layout_proofs`. `constant_code_packing` authenticates the original codes and identity packing, or the existing exact uint8 XOR-128 signed-unit representation. `bindings[].constant_operands` identifies the original literal SSA values; these operands are absent from the runtime pointer ABI. Integer convolution arithmetic and its full-i32 exact requantization are unchanged.

Kernel globals, folded literal printing and splat recognition use packed bytes. Large literal printing must not call xDSL's whole-tensor `get_values`/`is_splat`, which create Python numbers even for eventual hex output. The tests forbid that expansion through folding and host rewriting.

Identical schedules share a hazard audit **within one catalog build**, keyed by the complete stream digest, argument declarations, constants, readouts and numeric proof records. Each distinct schedule still receives the existing complete trace audit and instruction-form/encoding verification. This does not replace the final-ELF audits.

## Manifest additions

Schema remains `gemmini_core_aten_catalog_v1`. Optional records are:

- `constant_folds[]`: original `root_ordinal`, `result`, `result_type`, `bytes`, output `storage_sha256`, input ordinal/result/storage hashes, and `proof = literal_storage_through_proved_dma_rectangles`.
- `bindings[].constant_operands`: kernel constant name to original source `(op_ordinal, result)` binding. These are evidence, not runtime pointer arguments.
- `kernels[].meta.constant_layout_proofs` and `constant_code_packing`: target-private literal layout and representation proofs described above; included in the symbol digest.

`offload.rewrite` rechecks the original `source_sha256`, independently replays the DMA specialization, and requires the complete fold record sequence to agree before replacing any device region. It uses the original ordinal table and forwards folded values to subsequent calls. Fold records contain no large duplicate payloads: output bytes are reconstructed from the authenticated source. A constant-only catalog contains zero kernels/bindings and a valid empty object/shim; a genuinely empty catalog with no folds still refuses.

## Merlin requirements (not implemented in this worktree)

**External parameters.** `catalog.build(source)` receives tensor SSA only. A function argument, even one labelled CONSTANT in capture metadata, is not a literal and cannot be frozen from the source alone. The measured ResNet source has 109 entry arguments and **zero dense tensor constants**. Its stored weights therefore still use runtime DMA under this API. Silently reading a nearby weights file or treating provenance as immutability would violate source binding and compiler ownership.

Merlin owns immutable-parameter discovery, packing, generic constant propagation and entry ABI projection. Before the provider callback in `merlin/llvmlower/device_build.py::apply_region_offload`, Merlin must supply a prepared source with authenticated immutable parameters materialized as typed literal constants (or a source-bound immutable-operand contract with equivalent guarantees). Reuse `merlin/llvmlower/entry_weight_projection.py::derive_entry_argument_table` and its `ArgumentBinding.kind = IMMUTABLE_STORED_WEIGHT`/`IMMUTABLE_GENERATED_ZERO_WEIGHT`; require the relevant `GeneratedDispatchABI` readonly-storage, unobserved-address and generated-caller guarantees. Pin the stored weight bytes, capture/prepared source hashes, argument index/type/shape, and quantization/hoist plan identities. Mutable model inputs, state and merely captured values cannot be frozen. Caller/packing and buffer ownership changes belong in Merlin. Preserve the immutable original accuracy inputs/goldens and record a derived prepared-source receipt. The OOT then performs target DMA specialization on those literals without a new workload selector or an unmerged dependency.

**Fold-only region results.** In `merlin/llvmlower/device_build.py::apply_region_offload`, the current `if not bindings: raise ValueError("region offload bound no region")` must distinguish an empty refusal from successful authenticated `document["constant_folds"]`. Accept the latter with an empty object/shim and zero routed device calls, check the provider's original-source/fold receipt, and retain it in `region_offload.json` alongside the artifact hashes. A consumer that validates binding fields must recognize `constant_operands` as literal evidence excluded from runtime arguments. Mixed catalogs already carry ordinary bindings. No Merlin file was edited here.
