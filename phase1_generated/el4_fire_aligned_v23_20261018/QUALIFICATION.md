# Qualification at the provisional checkpoint

This is a publication of existing compiler output, not a new passing Phase 1
grade. The formal completion flag of the source run remains false. Publication
does not change earlier grades or waive source, instruction-trace, protocol,
numerical or full-model execution requirements.

The selected hardware is `FireSimGemminiRocketConfig` at 30 MHz. Its exact facts,
contract, elaborated FIRRTL and simulator identities are recorded in
[checkpoint.json](checkpoint.json). Spike is untuned; simulator walltime and its
cycle counters are not calibrated target latency.

## Whole-model execution observations

| Model | Executed workload | What remains |
| --- | --- | --- |
| TinyLlama | Full 1.1B model, all 22 layers, batch 1, 32-token prefill plus all 32 decode steps with carried KV state | New 32/32 quantized-reference comparison and complete formal qualification |
| SmolVLA | Full model, batch 1, one 512x512 camera, prefix encoding plus all 10 denoising steps and the complete 50-action decode | Explain compiled/reference discrepancy and complete formal qualification |
| ResNet50, historical path | Full older captured model executed; all 1,000 outputs bit-identical to its own captured reference | This uses an older recipe and raw accumulator output plus host processing; the currently selected fused-readout path remains unsupported |

The SmolVLA compiled final `[1, 50, 32]` output differs from its selected INT8
reference by relative L2 error `0.13358568277481506` and maximum absolute error
`0.3015758991241455`. The user accepted this observed approximation for iteration;
it is not a proof of compiler equivalence or robotics task accuracy.

Targeted stage comparisons use the exact saved compiled inputs at each boundary.
The prefix KV output already differs by relative L2 error
`0.10731511553367792`. The first denoising action-state output differs locally by
`0.0015032943094307812`; its carried KV and time outputs are exact. Final action
decoding is bit-exact on the same inputs. The first divergent internal operation
has not been identified. These local observations do not prove how much each
stage contributes to the final recurrent-chain error.

## Readout and corpus limitations

All 54 selected ResNet contractions route to Gemmini during preparation, but 32
kernels require different gains across output-channel blocks. This compiler
accepts a scalar `acc_scale`, and refuses `acc_scale_by_column`. It does not yet
implement the selected grouped-16 readout scheduling.

The new Phase 0 corpus generation also refused nine mandatory cases: eight
column-readout cases at software admission and one public micro-model capture.
The successful remainder was retained. This incomplete generation does not
establish a new sealed corpus, release or official Phase 2 baseline.

Older reported capsule scores do not establish complete current trace/protocol
certification. The preserved author report provides historical context only.
