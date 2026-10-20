# Phase 2 integration checkpoint — 20 October 2026

This preserves source work on the EL4 branch. It does not promote a compiler or
declare the performance campaign qualified. The immutable generated V23 and V26
packages are unchanged. No handwritten backend replaced either package.

## Included work

- Explicit accumulator first-product overwrite semantics and its independent
  checks. Old/default behavior is preserved.
- Independent raw-i32 store-configuration-hoist experiment, including empty and
  stride-overflow fallbacks.
- Experimental bounded RHS-panel adapter: shared dense/block planning, multiple
  output accumulator tiles, full-K continuation, tails and explicit raw commands.
  The 25 static controls pass. The adapter is not registered or enabled; native
  and timing checks remain pending. See
  [the proposal](../../docs/proposals/v26-rhs-panel/README.md).
- Exact disabled packing/layout and effect-cleanup drafts in
  `../phase2_drafts/`, with source-byte manifests. They are untested snapshots,
  not enabled passes; their test drafts are excluded from test discovery.
- Two exact saved model files: `models/selected-dc413.json` and
  `models/unpromoted-e2a.json`. The latter passed broader statistical checks but
  still fails activity support and is not a qualified successor. Saved model
  JSON does not restore a live runtime, admission owner or calibration receipt.

Merlin's corresponding EL4 integration contains scoped exploratory counts and
estimates, partial roofline feedback, the scoped roofline/Spike bridge, source
verification and lifecycle fixes, the technical-status note and generic draft
snapshots. It also merges the previously unpublished local EL4 implementation
line through `fa6a18e95` without selecting its prospective numerical recipes for
this frozen V26 campaign.

## Portable Spike source work

The managed `spike-timing` repository's configured remote is a local directory
inside Phase 1. We deliberately did not push to or edit that task. Instead all
eight source commits are preserved as `spike-patches/*.patch`, including finite
reservation windows, staged issue/completion, dependency-ordered phases,
conservative local-store holds and the separately selected execute-service model.

`checkpoint.json` records every patch hash, exact base/head commits and model
hashes. `patch-verification.json` confirms that applying the complete series to
the base in a separate temporary Git index reproduces the entire preserved tree:
`0dbe447588a44fe569ee134eab0860f923926ed6`.

To reproduce in a managed editable Spike task, start from public upstream commit
`806ec0adda2242597b253ebf17adfaa5cdb395e3` on `feat/annotated-spike-model`
in the recorded GitHub source repository. Apply all patches with `git am` in
filename order: two `0000-base-*` profile prerequisites, then eight timing
patches. The prerequisites reconstruct the original local base
`dea21091bc2164989eaacd0c7acf62c79d004ee8`; the complete ten-patch series also
passes the entire-tree check in `upstream-patch-verification.json`. Use task
tooling for repositories; do not clone or create worktrees manually. Build a
separately versioned observer and rerun the existing calibration protocol before
making timing claims. These patches do not change the running compiler evaluator
or qualify a new timing domain merely by being present.

## Final completed diagnostic batch

The retained four-job V5 batch ended with complete original-process cleanup:

- RHS residency and configuration hoist each compiled and passed the complete
  544-value public case, for both baseline and candidate arms.
- Fresh boxed timing observations returned `CensusUnavailable`; these completed
  jobs have no new cycle estimate or timing qualification.
- Both larger M64×K256×N64 jobs refused because their evaluation output exceeded
  the assigned disk grant. These failures remain in the original roster.
- No candidate was promoted and no complete-model speedup was established.

This does not replace the separately obtained 1,175→988 exploratory estimate.
That pair keeps its original provenance and limitations. The four-hour campaign
deadline was 23:05:02 UTC; no new optimization campaign was started. Current work
is user-requested source integration and verification.

The large source captures, traces, original receipts and simulator builds remain
in their managed task locations; small tracked manifests retain hashes and
reproduction pointers. They are not copied verdicts presented as fresh evidence.

See [checkpoint.json](checkpoint.json) for identities and original report paths.

## Post-merge checks and operator snapshots

`verification.json` records **195 Merlin + 138 OOT focused tests passing**, using
one shared CPU grant. The initial run exposed a dispatch test depending on an
ambient capture-runtime variable; its fixture now explicitly selects its own
sealed source. Production admission checks were not weakened. These checks are
host regressions/static emitter proofs, not fresh accelerator timing evidence.

`operators/` preserves exact historical launcher/constructor/test-driver bytes as
`.txt` snapshots with hashes. They include the explicit selected Chipyard path
fix. Reuse requires the original source seals, runtime/corpus owners and input
identities, or fresh explicit selections for a new task. They are not automatic
launchers or portable qualification receipts.
