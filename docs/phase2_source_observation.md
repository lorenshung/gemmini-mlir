# Explicit source observation formats

`SourceExecutionIssuer.create_source_execution_owner` and `SourceExecutionOwner`
accept an explicit `observation_protocol`. The default
`gemmini.source_observation.v1` keeps `-l --log-commits`, its original digest
payload shape and result fields. `gemmini.source_observation.v2` selects only
`--log-commits`: Spike independently enables complete commit logging, while
`-l` adds disassembly lines that the source-effect consumers ignore.

V2 binds its protocol and exact flags into a distinct owner identity. Unknown
protocols refuse before execution. The same actual ELF, functional runtime,
input bindings, complete commit/register/memory stream, independent output
comparison and source/product/state/host-store checks remain required. File,
instruction, execution-time and resource ceilings are unchanged. This option
does not select a timing model or alter functional simulator behavior.

Select V2 explicitly when directly issuing or constructing the original live
source owner. Existing public-source issuers and default calls continue to
select V1; no environment variable or saved receipt upgrades an observation.
Historical full logs and size refusals retain their original interpretation.

A paired replay of one existing signed public ELF observed identical complete
110,000 commit rows, 34 decoded commands, fences, 8,697 CPU/device store
observations and all 217 independent output values in both modes. Stderr was
13,341,731 bytes with V1 and 7,030,416 bytes with V2. These are storage results,
not accelerator performance or proof that a larger previously refused source
fits. A new complete live source admission remains necessary for each program.

If commits alone exceed the unchanged ceiling, a future independently checked
streaming effect projection is needed. Aggregate Spike counts cannot replace
ordered packets or host-memory effects. Restricting logs to a kernel window
would require a separate complete outside-window input/zero-storage veto.

For prospective public-source diagnostics, select the separate
`gemmini_public_source_observation_factory.original_factory` with the original
prototype inputs plus
`source_observation_protocol="gemmini.source_observation.v2"`. Its canonical
`PublicSourceObservationIssuer` retains an actual original live public issuer
and delegates the unchanged bounded trace algorithm with explicit V2 flags.
Both new leaves and all original checker sources enter the selection identity.
The old public factory and its input interpretation are unchanged; absent or
unknown protocols refuse in the new factory.

If native timing is independently selected, the separate
`gemmini_source_observation_native.SourceObservationNativeCorrespondenceOwner.bind`
accepts only the original live V2 source reviewed capability and original count
reviewed capability. It constructs the original V1 structural correspondence
from the actual underlying live issuer, checks both complete bindings, and
keeps the V2 source identity. Wrap it as the original reviewed mechanism
`compiler_source_native_correspondence` using the native count binding. This
relates composite and raw software namespaces; it does not equate their hashes,
grant timing qualification, widen the model domain or deserialize a verdict.

Focused neutral tests exercise exact mode selection, canonical private-factory
restoration, complete binding checks, foreign and stale owners, callback/source
mutations and saved-object refusal. They qualify transport and fail-closed
behavior only. The earlier actual paired ELF comparison remains the evidence
for equivalence of the two complete commit streams; a larger source must still
pass its own actual native and full live source/output admission.
