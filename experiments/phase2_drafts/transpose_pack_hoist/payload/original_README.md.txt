This separate candidate materializes a complete immutable transposed i8 RHS once
into private dense K-by-N storage when at least two M tiles reuse it. A fixed
64 KiB compiler workspace policy bounds the extra host storage. Unsupported
shapes, epilogues, resident state and narrow output retain the original path.
The generic byte transpose emitter lives in Merlin's `dense_layout_loop`. The
candidate carries an exact ordinary source copy of that algorithm, with its
source hash in the inventory, rather than importing an ungranted new SDK API.

Prepare from the exact complete original package with:

```
python -m experiments.transpose_pack_hoist.prepare --baseline ORIGINAL16 --destination FRESH --report FRESH_REPORT
```

`apply_to_copy(package)` composes via one strict unique entry anchor. Apply layout
preparation before the panel-schedule hook, then configuration/effect cleanup.
It never overwrites a complete candidate module. The helper consumes the
original immutable-input/disjoint-output contraction ABI; source owners must
independently reopen and check the complete actual host-layout and packet trace.
Host expansion tests are neither native execution nor target qualification.

The selected LoadController can represent 64 i8 columns, splitting them into
DIM-column local blocks at block_stride DIM. The current original complete-source
packet owner explicitly limits every load extent to DIM. Wider transfers remain
disabled here; a separately selected owner/version with exact block-address,
overlap, tail, lifetime and full-output checks is required. No FSM, async issue,
numerical change or verification-cap increase is introduced. Original device
fences remain; the repeated pack-visibility fence moves with its CPU copy to
the one preparation boundary.
