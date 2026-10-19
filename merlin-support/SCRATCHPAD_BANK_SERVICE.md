# Selected scratchpad bank service

`backend/gemmini_bank_service.py` is an additive structural adapter. It changes
neither DMA contract V2 nor the frozen native timing model. Construct
`ScratchpadBankService(selected)` from the original live selected target owner.

The selected FIRRTL contains four separate scratchpad bank instances on the
parent clock, with 4,096 rows and 16-byte row data per bank. Despite separate
read/write interface bundles, their accepted operations share one slot: write
ready is constant true and read ready is `q_will_be_empty && !write.fire`.
`BankPortCones` exhaustively checks all eight Boolean input cuts per bank,
including contested read/write requests and response-queue backpressure.
Accepted reads plus accepted writes are therefore at most one per bank/cycle.
This upper bound does not require a claim that every queue cut is reachable.

The adapter checks original instance/clock wiring and the actual guarded
MVIN-return write-address expressions. Bank selection is local row address bits
13:12, and the row is bits 11:0. It checks the complete conjunction selecting
the bank, the non-accumulator tag, write readiness and execute-write priority.
The implementation derives these slices from the original bank geometry;
tests also exercise a different two-bank/eight-row fixture.

`record()` exposes shared resource IDs with units `accepted_row_operation`.
`local_row(address)` describes an explicit local row cut. It does not bind a
machine instruction to that cut. `local_service_floor(operations)` uses the
existing generic `compose_service_floor` consumer: read and write demands on
one bank add, while distinct bank bounds combine using max. Its inputs are
explicit caller-supplied local cuts and remain noncompulsory emitted-schedule
evidence. They cannot set a fixed-workload target or claim runtime attainment.

Actual command-to-bank mapping through load/execute/readout pipelines remains
unresolved. Scaling, repeats, masks, pending local writes, accumulator ports,
instruction/fence completion and simultaneous issue across banks are not
qualified here. No stalls, physical memory latency or achievable overlap are
inferred. Saved JSON cannot construct the original live adapter; selected FIRRTL
and helper bytes are reopened on use.
