# Phase 2 additions and immutable Phase 1 artifacts

`feat/el4-experiment` is the integration branch. The independent experiment
`phase1/el4-fire-aligned-v26-20261019T191400Z` remains at
`8b8fd3629f4e8e1c3349176241620621048b96c7`; it is not an optimization branch.
Publishing the integration changes no original compiler or qualification result.

## Preserved artifacts

- The original V23 package remains unchanged at
  `phase1_generated/el4_fire_aligned_v23_20261018/package`: 21 source files,
  SHA-256 `b847c973da82f408a32c7622d0c49fb6902f0a58d0f716f131d8501e34175338`.
- The imported V26 publication under
  `phase1_generated/el4_fire_aligned_v26_20261019T191400Z` preserves all 54
  original Git blobs, SHA-256
  `3e7d314530b00e966af3bbc7c244ae0423721cea98cdfd1e002d061fe1f3fc3d`.
  Its published compiler is the original 14-file projection with its own digest.
- `phase1_generated/el4_fire_aligned_v26_original_29213759/package` is a separate
  exact copy of the original 16-file submission, including both author reports:
  SHA-256 `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
  The portable public-fixture directory contains another byte-identical copy.

Package hashes use sorted relative POSIX path, NUL, complete file bytes, NUL,
SHA-256, excluding generated `build`, `__pycache__` and `.git` subtrees.
The 54-file archive is also compared directly against the published Git tree.

## Additions outside those artifacts

The integration contains executable Phase 2 support as well as documentation:

| Location | Purpose |
| --- | --- |
| `merlin-support/backend` | Target-specific observation, emitted-instruction/source checks, capability intake, diagnostic compiler preparation and execution adapters. |
| `merlin-support/tests` | Regression tests for those adapters and proof lifetimes. |
| `experiments` | Explicit experimental scheduling policies and compiler-library integration examples. |
| `phase2_candidates` | Seven separate V23-derived candidate packages for residency, configuration and buffering experiments. These are unpromoted hypotheses, not Phase 1 outputs or a selected V26 compiler. |
| `examples/phase2/public_pass_inputs_v26` | Three portable public synthetic sources with complete inputs/weights, independent references and new relative-path fixture descriptors. Fresh capture/admission and V26 execution remain required. |
| `docs/v26` | Reproduction instructions, runtime hashes and exact public RTL facts, software specification, descriptor and device tree. Runtime binaries and live admission owners are not included. |

Compared with the starting checkpoint
`50f34dbcbf23d1b489b988c02fd659cba0f262ee`, existing tracked implementation files
are unchanged. The only pre-existing tracked file edited is the root README,
which gains a documentation link. All support, candidate and fixture files above
are additions in separate paths. The handwritten root backend is not substituted
for the generated compiler.

Generic scheduling and optimization algorithms belong in Merlin; this repository
owns target encodings, resource/effect bindings and execution checks. An available
adapter or passing host fixture does not enable an unqualified target mechanism.
No supported capability, numerical tolerance or qualification gate is added to
the immutable Phase 1 compiler by these files.

## Qualification remains separate

The V26 publication records 152 public numerical passes, one expected refusal and
eight hidden aggregate passes. Original resource certification is incomplete;
the executor repair still requires a separately attributed check. Complete V26
ResNet50, TinyLlama and SmolVLA validation is not established. The new public
fixtures have integrity/reference checks, not a new V26 accelerator qualification.
No candidate is promoted by this integration and no historical timing result is
relabeled as a new measurement.
