# Native Spike timing handoff, 2026-10-19

The separately versioned native reservation model passes the unchanged archived
public timing gates. This is reused diagnostic validation; the held set was
examined during infrastructure development and is not unseen final evaluation.
No new gSIM run or compiler performance promotion supports this result.

The final replay contains 473 fresh native reports from the original ELF/input
pairs. All 125 raw counts in every report equal the original count-v3 report;
the 19 dataset feature columns also remain unchanged. The original 483 declared
records, 313 cases and ten unavailable counter reviews remain present. Source
groups stay split 51 training, 16 calibration and 20 held, without variant/repeat
leakage. Available cases are 168/176 training, 75/76 calibration and 61/61 held.

Final native validation gives 41/41 correct eligible orderings across convolution,
matmul and resident-chain families; median absolute relative error is 10.3831%
and p95 is 26.9472%. All 61 held cases have predictions. Residual intervals use
75 calibration cases from 16 independent source groups. Coefficients were
selected using training labels only. The final bounded refinement froze its
settings before running: at most 17 models, with 15 unique models evaluated.

Prior failures remain separate immutable artifacts. The earlier complete-domain
model failed at 53.7058% p95 error; its training-only refit failed at
30.4843304843% p95. Neither result is rounded into a pass. Earlier restricted
models failed domain coverage and pair/family counts.

## Frozen inputs and outputs

Let `T=/work/merlin/tasks/phase2-target-adapters-20261019`,
`R=/work/merlin/tasks/phase2-el4-explore-20261018`, and
`B=$T/out/build/native-staged-v2`.

| Artifact | SHA256 |
|---|---|
| Final `$B/replay-holds-refined-v1/model.json` | `dc413b315b21b0bb2874098c0421db5900d541aef99cf20f76fa8eaf02f3fc0d` |
| Original observer `$B/observer-build/spike` | `0657c995d0ad2c045c1328d568b3c85405d781a02aaf3e6c6b18bc4eb7b16393` |
| Generic observer source identity | `e3c5f4f8499d68adcc209f929d349d91abb201254ef9d7cd2a433eb33755d166` |
| Provider `$B/provider-holds-v1.so` | `00a82c941275fe0caffe2255368a93b0f85e68172b14f7be8da02f9de532c8da` |
| Unchanged functional extension | `2dbd8445376cdda0df63abcf217c7a19f974d34a16ada21b549ed82fffd718eb` |
| Final training settings | `d7111504464892ca0009d8c19cd5b2fbb9a988c83b746a046610bb130ab087ea` |
| Original gSIM timing archive | `f815a0d2f4d80415b1e89e6ccdd62a0eb7af039287f539c91bd235451f8cfd5a` |

The complete file pins, closed-provider source roster, runtime freeze and
independent comparison audit are under
`$T/out/artifacts/target-adapters/gemmini/native-timing-handoff-v1/`.
`artifact-manifest.json` uses file-byte SHA256. The dataset file digest is
`61997336056bb0a6b87f4ad743b49b9e41d90e3290cc7867b82c9c8780122dcf`;
its canonical-document identity used by the validation API is
`81f54debb14c6fcd42edde2673a87bc548fcbfc6daef09f1a49050dd845f0fe5`.

The original host preflight independently rederived the count correspondence,
reconstructed the live native source owner, rebuilt the closed provider and
passed the original native archive review. Its record is
`$R/out/artifacts/perf-studies/gemmini/el4-20261018/native-archive-preflight-v1/report.json`.
The original setup script is `$R/.tmp/review_native_archive_selection_v1.py`.
This preflight issues no formal Phase 1 admission and launches no automatic
optimization campaign. A native queue selection still requires both original
live capabilities, fresh complete candidate numerics and ordinary policy gates.

## Reproduction and boundaries

Recheck the retained reports without fitting or rerunning a simulator:

```sh
export TMPDIR="$T/.tmp"
export PYTHONPATH="$R/merlin/src:$R/merlin/packages/merlin-timing/src:$R/merlin/packages/merlin-experiments/src"
/work/merlin/tasks/phase1-harness-v2/merlin/.venv/bin/python \
  -m merlin_timing.cli validate \
  --dataset "$B/replay-holds-refined-v1/dataset.json" \
  --prediction-kind event_timing \
  --output "$T/.tmp/native-v2-repeat-validation.json"
```

The live construction API and source checks are documented in [README.md](README.md).
Use the original closed `$B/native-owner-v1/provider-source`, selected
`phase1-harness-v2/.tmp/chipyard-fire-aligned` source root and a fresh task-owned
rebuild directory. A count sidecar includes its original implementation path;
another checkout correctly derives its own sidecar rather than importing a
copied success receipt. Both capabilities bind the same exact five fields.

The engine distinguishes RS allocation, issue, controller consumption, effects
and release. Pending PRELOAD survives a fence and its output tag retires with
the later compute. Whole-command bank/accumulator holds, aggregate cache LRU and
DMA resources are explicit conservative approximations. They do not reproduce
cycle-exact packet arbitration. Fitted parameters are operational estimates,
not measured individual physical prices. Unmapped/unpriced instructions and
unsupported pending mesh/tag overlap remain UNKNOWN.

The timing window matches the original kernel-entry/completion counter pair.
The gSIM `mm_magic_t` label domain is retained; FireSim's DRAM bridge is not
substituted for it. Packing and host instructions inside the window are observed;
external staging/publication and complete-model host work are not newly measured.
Resource breakdowns describe the whole trace and are nonadditive. They cannot
be summed or attributed to the kernel window as utilization. Large capacity,
attention and full-model extrapolation remain UNKNOWN without matching evidence.
No predicted cycle value establishes a roofline attainment or measured winner.

## Costs and retained source

`costs.json` separates each build/test, simulation and training stage. The observer
build took 213.816 seconds plus a retained 3.949-second failed first attempt. The
final provider build took 1.913 seconds and its staged tests 22.076 seconds.
The three training-only fits took 44.090, 435.518 and 131.945 wall seconds,
including shared-resource queue time. The final 473-record replay took 200.205
wall seconds: 171.366 seconds of queue time and 28.119 seconds of child execution.
Fresh source-owner reconstruction and provider rebuild took 6.326 seconds after
52.333 seconds queued. Original-host preflight/rebuild took another 6.861 seconds.
These stage walls must not be added to overlapping author/child durations.
Author/operator/helper token counters and total author wall time are unavailable.

Source was committed locally: Spike `e8205a1` (including the staged engine and
canonical-header export), Merlin `d23416b55` (closed provider runtime-header
custody, following canonical primitives `3c7671b`) and OOT `72d8392` (live native
owner with adversarial refusal tests). All were merged into the parent task.
The immutable generated Phase 1 compiler and original functional extension were
preserved. The timing infrastructure work stopped after the single final frozen
refinement passed the fixed diagnostic gates; further held-driven fitting stopped.
