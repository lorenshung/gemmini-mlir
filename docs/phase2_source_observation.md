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
