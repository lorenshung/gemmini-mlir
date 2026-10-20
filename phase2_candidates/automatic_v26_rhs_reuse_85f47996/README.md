# Retained V26 RHS-reuse hypothesis

This exact two-file patch preserves source `85f47996…`, produced by the
sandboxed author from the immutable V26 compiler `29213759…`. It selects the
existing resident implementation when multiple M tiles can reuse an admitted
RHS, with a bounded tile-count threshold and existing scratchpad fallback.
It also checks the accepted signed scalar contraction and zero initializer.

The fresh original queue consumed both exact sources and emitted different LLVM
for the same command buffer. Both arms then refused before native build because
the aggregate queue timeout exceeded the unchanged native-leaf limit. This
epoch produced no executed outputs, ELF or fresh counts. Its original feedback
and UNKNOWN decision were published and the process tree cleaned up.

The later host timeout fix has regression tests but was not target-reexecuted
in this campaign. Resident first-product overwrite, independent shapes/tails,
capacity, connected identity and readout checks remain required. No compiler
incumbent, latency gain or complete roofline attainment is established.

To reconstruct source in a fresh writable candidate directory, copy
`phase1_generated/el4_fire_aligned_v26_original_29213759/package`, make that
copy writable, then apply `candidate.patch` with `patch -d CANDIDATE -p1`.
Verify all 16 files against `experiment.json` before original evaluation.
Preserve the Phase 1 package and handwritten backend.
