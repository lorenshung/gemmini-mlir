# Selected DMA transfer service, version 2

`backend/gemmini_dma_resources.py` derives a new, explicit client-interface
contract from the exact FIRRTL selected by `SelectedSchedulingTarget`. The
original `gemmini_dma_service.py` packet projection and frozen timing models
retain their interpretation. This contract does not issue scheduling or timing
qualification.

Construct `DmaTransferContract(selected)` with the original live selected owner.
Its `channel_contract()` returns the generic
`merlin.perf.channel_service_contract.v1` input to
`merlin.perf.channel_service.compose_transfer_service`. `transfer(...)` decodes
the actual instruction and configuration through the original selected operand
observer and row/address packet cones. It returns explicit packets; the generic
consumer never infers rounding or alignment.

The selected design has separate reader and writer A/D client interfaces, each
with one ready/valid handshake per cycle. The data-bearing interfaces are 128
bits. Reads require one A request and the packet's D data beats; writes require
the packet's A data beats and one D acknowledgement, under legal Get/Put
responses. Packet sizes and data-beat widths are derived, not assumed by the
generic consumer. Payload bytes remain distinct from full-beat client request
envelopes and from observed physical transactions. A bound on each separate
client resource combines with other resources using max. This establishes
neither independent downstream channels nor achievable simultaneous service.

The executable local checks derive transaction ownership and backpressure:

- Reader request issue and source-ID allocation happen together at the
  **untranslated** request/reserve interface. They are not downstream A issue.
- Reader source-ID release is `BeatMerger.in.fire && final_D_beat`. It can precede
  final merged-row consumption and local storage completion. A released source
  ID cannot serve as a full-DMA fence.
- Writer allocation happens on the first untranslated A beat. Continuation
  beats retain that transaction and do not allocate again. The source store can
  stall new and continuation traffic when all IDs are occupied.
- An accepted D response releases its source ID. The writer remains busy if its
  issuing state is non-idle, independently of outstanding IDs. Responses without
  an outstanding owner are refused by this checked contract.

These checks evaluate original combinational cones with explicit state cuts.
They check opcode selection, response beat counters, handshake atomicity,
source-ID selection, allocation/acknowledgement updates and wrapper release
signals. They do not prove state reachability, response fairness, external
protocol correctness or cycle-exact arbitration. Source files and the exact
selected inputs are reopened by the live owner. Saved JSON cannot construct it.

`backend/gemmini_source_dma_transfers.py` supplies the original source/ELF/trace
join. Construct `SourceDmaTransferEffects(source_effects=original,
transfer_contract=contract)` with exact live `SourceResourceEffects` and
`DmaTransferContract` selecting the same original target, then call
`derive(capsule_dir)`. This re-runs the original complete source/input/product/
output checks, maps its actual command packets, and invokes the generic channel
consumer. The result retains original source, ELF and trace bindings. Fixed
workload floors and emitted-schedule bounds remain separate. Cycles, counter
window and attainment remain absent until independently bound evidence exists.

Translation retries, downstream cache/DRAM transactions, shared interconnect
ports, per-bank service, local write retirement, complete-call host/packing
cost and completion chains remain explicit unresolved terms. No new gSIM run,
fit, held-label selection or change to the incumbent timing model is involved.
