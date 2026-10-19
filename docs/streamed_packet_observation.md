# Complete streamed packet observation

`merlin-support/backend/gemmini_streamed_packet_observation.py` is an additive
read-only observation entrypoint. It verifies an original completed invocation
before and after replay, then feeds a `merlin.common.replayable_text.ReplayableText`
to the unchanged original `_events` and `_cpu_writes` consumers. Integer-register
replay, command decoding, every fence and every observed memory write retain
their original semantics. Complete iterations authenticate all raw bytes and
reopen them again at the end; no trace lines are filtered or sampled.

The raw stream must still fit **64 MiB**. The original authority is
`gemmini_source_packet_artifacts.check_execution_packets`: it rejects stderr
larger than `64 << 20` as exceeding its independent byte budget. The original
`experiments/shared_load_configuration/qualify.py::_events` also retains its
**2,000,000 retired-instruction** ceiling. These are verification ceilings, not
compressed-storage allowances. The new entrypoint does not accept gzip or
reinterpret either ceiling. Historical controls 5, 25 and 26 whose complete
traces exceed the raw ceiling remain refused.

The observation returns decoded commands, complete event framing, conservative
host-write extents, raw hash/byte identity and decoder-source hashes. ISA and
symbol inputs remain explicit and caller-owned. It grants no target, source,
arithmetic, compiler, writer or timing qualification. Original SourceExecutionOwner
and packet-checker modules remain unchanged. A later source-owner integration
must explicitly pin/select the new reader and preserve every original full
source/ELF/input/output/product correspondence check. This transport improvement
alone cannot close an oversized-trace admission refusal.

Run focused controls with the selected Merlin source package and the OOT backend
on PYTHONPATH:

```sh
python -m pytest merlin/merlin/tests/infra/test_replayable_text.py \
  gemmini-mlir/merlin-support/tests/test_streamed_packet_observation.py
```

The tests compare complete decoder results, fences and memory effects, include
actual harmless subprocess provenance, and retain unknown-hart/store refusals.
They check UTF-8 and all Python line separators across chunk boundaries, source
rewrites/replacement, malformed input, raw-size limits and bounded allocations.
They execute no target program and supply no accelerator-performance evidence.
