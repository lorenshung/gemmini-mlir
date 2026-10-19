# Plain MVIN bank-write accounting

`SourceBankWriteEffects(source_effects=original, bank_service=original_bank)` in
`backend/gemmini_source_bank_writes.py` is an additive, host-side live join.
Both arguments must be original exact-type owners selecting the same target.
The bank owner is independently rederived; source pins, mapping, checks, cone
geometry and original bound callbacks are reopened and compared before use.
Modified state or replacement callbacks refuse admission.

`derive(capsule_dir)` first re-runs `SourceResourceEffects.derive`, including its
original sealed-source, actual inputs, complete outputs and product checks.
It then reopens that same original commit trace. Every instruction and fence is
retained. Plain identity byte-MVIN writes map through selected LoadController
addresses, actual packet envelopes/reservation offsets, merger masks, scale
row/tag preservation, the single-row PixelRepeater path and bank mask drivers.
The configured pixel count zero is normalized to one, then captured as zero
remaining repetitions. Broadcast, accumulator loads and unsupported flags refuse.

A logical row may need multiple bank writes: an unaligned source row can split
at the maximum packet boundary. Each fragment has its own selected local offset
and mask; fragments must cover every required byte exactly. These are
source-derived necessary accepted writes from executed commands, conditional on
the checked instruction effects completing. They are not observed physical bank
events, cycle timestamps, stalls, a liveness proof or hardware qualification.

All other effects, including execute reads, readout/accumulator service, spatial
paths, pooling and nontrivial repetition, remain explicit UNKNOWN rows. The
existing generic service-floor consumer aggregates only the admitted write
events as noncompulsory emitted-schedule demand. No counter-window join, measured
cycles, attainment, convergence or qualification is issued.

Original sealed source/runtime custody is mandatory even when compilation and
execution artifacts are unchanged. A missing runtime CAS refuses the complete
join; prior successful JSON does not replace the live source gate. Standalone
synthetic register/cone tests qualify only source-path plumbing. The frozen
DMA V2 and native timing models retain their original interpretations.
