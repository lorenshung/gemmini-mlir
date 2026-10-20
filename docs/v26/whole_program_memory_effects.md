# Explicit whole-program memory effects

`source_memory_effects_protocol: gemmini.whole_program_memory_effects.v1` is an
optional input to the original generated-source factory. It is accepted only
with `source_binding_protocol: whole_program`. Omitting it preserves the original
load-state-zero/pixel-zero packet checker and its record format. This option is
separate from `gemmini.source_observation.v2`, which still records every committed
effect under the original 64 MiB raw-file and decoder ceilings.

Before execution, the factory derives a live `WholeProgramMemoryEffects` from the
original exact `SelectedSchedulingTarget` and `IntegerArithmeticOwner`. It reuses
the independently checked `SelectedLoadStreams` mapping and
`SelectedAccumulatorZeroFill` RTL/runtime derivation. The factory freezes the new
helper and its load, zero-fill and private-zero-storage dependencies before its
privileged callback runs. The observer forwards the explicit protocol and checks
that the resulting source owner has the same independently derived effects identity.

The source owner passes these live effects to the existing complete packet
induction. Load streams retain independent configured byte strides. Identity
scaling, no shrink, selected SPAD row stride and only the independently proven
pixel encodings remain mandatory. Accumulator initialization still requires the
existing exact overwrite/zero-fill proof or independently observed immutable
zero storage. Commands, pending preloads, cell ownership, every source product,
readouts, completion and full output comparison retain their original checks.
Global-plan/source/ABI/control-flow and host transpose validation remain in the
original whole-program verifier. Direct-source owners reject this option.

Source and callback changes, foreign or serialized owners, missing state,
unsupported scaling/shrink/repetition/strides and incomplete work continue to
refuse. The option grants no compiler transformation, numerical relaxation,
physical residency, hardware qualification or timing authority.

The motivating retained public transpose control had complete native outputs and
a full trace, but the legacy whole-program checker refused its states 0/1/2 and
pixel encoding 1. Tests reconstruct the full packet induction with its actual
public raw load fields and independent K/N tails, using explicitly neutral
tensor bytes and framing. They are not accelerator qualification. That historical
refusal remains immutable. Actual admission under the new source epoch requires
the original current producer/runtime/source checks and a fresh complete trace;
this document and a saved success record cannot issue an owner.
