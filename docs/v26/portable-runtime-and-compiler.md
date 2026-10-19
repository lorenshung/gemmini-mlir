# V26 setup on another server

This guide selects source and runtime inputs. It starts no author, compiler,
simulator or campaign and transfers no qualification. Choose fresh paths under
an existing managed task; use the receiving server's resource admission tools.

## Published source and compiler identity

| Repository | Exact source |
|---|---|
| https://github.com/ucb-bar/merlin.git | published `feat/el4-experiment`, `f8a43c657f11ee417f4c96d8920fa2f7e66f3f5b` |
| https://github.com/ucb-bar/gemmini-mlir.git | published `phase1/el4-fire-aligned-v26-20261019T191400Z`, `8b8fd3629f4e8e1c3349176241620621048b96c7` |
| https://github.com/riscv-software-src/riscv-isa-sim.git | functional source `9c190a07c6838f6392bafa4ad83acea462c7f759` |
| https://github.com/ucb-bar/libgemmini.git | functional receiver `5b1254a7dc72f757442280e7f28ef30542c78533` |
| https://github.com/copparihollmann/riscv-isa-sim-gemmini.git | recorded standalone packager/build helper at `dea21091bc2164989eaacd0c7acf62c79d004ee8` |
| https://github.com/llvm/llvm-project.git | LLVM/MLIR23 `a47bddccec30255619bb8c37fa59700e661d4e66` |

The helper's `examples/gemmini/functional_baseline/build.py` matches the actual
source package builder SHA `10c6529b7fedebc7410a0d9a9ea3cee7bb94689d802a06187a154f24228d2468`.
Its simulator sources are the pristine pinned objects above, not its instrumented
working checkout. Check availability of exact Git objects on the receiving server;
if a public server does not expose one, provision an explicit source bundle.

The published compiler is a **14-file projection**, 102,715 bytes, identity
`e7fd1d4e3f5d168050ba212d853aa7ff3283bf2f8b7527932bd3704c1f455039`.
The actual P1 output has **16 files**, 112,238 bytes, identity
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Two authoring reports account for the difference. Hash algorithm is sorted relative
POSIX path + NUL + complete file bytes + NUL, SHA-256. Never label the projection
with the full-package digest.

Full16 is retained by local OOT import `b23ff7e83532f742911bc3a86fcd767deb91d94a`,
integrated into local EL4 `6173b15f0934437cb8f4f604f01e3b1fec2bda8d`, at
`phase1_generated/el4_fire_aligned_v26_original_29213759/package`.
**Those local commits are not asserted published.** An operator must provide this
exact source bundle or publish reviewed commits before the remote session can use
it. The original native checkpoint is `38579f414f516e884d381cfc23683fdf43d54a6b`;
it is not assumed reachable from the cleaned publication. Adjacent provenance
binds all16 hashes and the original clean source snapshot.

For the published projection, with existing local checkouts:

```sh
: "${TASK_DIR:?your canonical managed task}" "${TMPDIR:?inside your task}"
: "${OOT_REPO:?existing gemmini-mlir checkout}"
git -C "$OOT_REPO" fetch origin phase1/el4-fire-aligned-v26-20261019T191400Z
test "$(git -C "$OOT_REPO" rev-parse FETCH_HEAD)" = 8b8fd3629f4e8e1c3349176241620621048b96c7
mkdir "$TASK_DIR/v26-public-archive"
git -C "$OOT_REPO" archive 8b8fd3629f4e8e1c3349176241620621048b96c7 | tar -x -C "$TASK_DIR/v26-public-archive"
python3 -B "$TASK_DIR/v26-public-archive/reproduce/verify_archive.py"
```

Integrity verification does not confer the original certificate or profile.
Manifest operations remain parse/interface lowering/command-buffer/LLVM/analysis;
there is no declared `target_ir_consumer`. Do not retrofit a provider ABI flag.

## Provision an existing functional runtime, or rebuild

The minimal externally provisioned runtime layout is:

```text
runtime/bin/spike
runtime/lib/libgemmini.so
runtime/lib/libboost_regex.so.1.74.0
runtime/deployment.json
```

`runtime-files.json` gives original sizes and hashes. This is about11MiB, not a
Python environment or complete toolchain. Transfer these bytes through an explicit
operator-owned artifact channel. Keep deployment.json byte-identical as historical
provenance; its embedded old paths are not a usable remote selection. A current
location/runner binding and independent current functional checks are still needed.
Do not expose runtime source archives/headers as compiler-answer material.

Alternatively, provision the standalone functional **source package** containing
`build.py`, `profile.json`, `sources.json`, `README.md`, `spike.tar`, and
`libgemmini.tar`, or regenerate it from the public Git objects with the recorded
packager. It requires Linux, Python>=3.10, GCC/G++, Make, dtc and Boost development
packages. The builder installs/downloads nothing and accepts at most2 build jobs.

```sh
# SOURCE_PACKAGE is the hash-verified external package; work directory must be new.
python3 "$SOURCE_PACKAGE/build.py" --verify-only
python3 "$SOURCE_PACKAGE/build.py" --work-dir "$TMPDIR/functional-build" --jobs 2
# With explicitly staged Boost, add BOTH --boost-root and --boost-libdir.
```

Recorded steps are configure with an explicit prefix (plus selected Boost paths),
`make -j2 spike install-hdrs install-config-hdrs`, then
`make -j2 RISCV=<prefix>` in libgemmini. Compiler flags were `-O2 -g0`.
**A rebuild is not promised to reproduce original binary hashes.** Host toolchain,
Boost, linker and OS differences can change bytes. Retain its newly issued build
receipt and re-establish current functional/source/extension compatibility before
any qualified use; never edit expected hashes merely to accept different binaries.
Building alone has `built_not_executed` status and no timing qualification.

## Matched selection and explicit launch

| Input | Original identity |
|---|---|
| selected RTL facts v6 | `b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af` |
| V26 descriptor target contract | `bfbfac9e98ff028aa184b5b13ea44e7421afeac2c35e1d3bb572fd0cbad5aa55` |
| software specification | `3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f` |
| separate observed capability contract | `8e4e6eb5aff7e0f92012b9c306f5cc22978342c02fb8cafafee8788d8c517be4` |
| selected models configuration | `cd028d6af0a6f5540164b3026c8a6dc2e7137e6b5a03d8783a38a46dc28fb536` |
| FireSim DTS | `c4a2009a405945ea98f731713e0f4e7dc483031ab8cd5156d871e4eb7594f1df` |

Provision these separately from the archive through their original owners. The
models configuration is not the example `reproduce/numerics.yaml`. Neither model,
capture nor protected corpus content is redistributed by this guide. The V26
contract and observed capability contract are different namespaces; the old
observed JSON must never replace the selected V26 descriptor contract.

```sh
: "${MERLIN_REPO:?explicit reviewed harness}" "${RUNTIME:?verified runtime root}"
: "${V26_CONTRACT:?hash-verified V26 descriptor contract}" "${V26_FACTS:?verified facts}"
export MERLIN_TARGET_CONTRACT="$V26_CONTRACT" MERLIN_RTL_FACTS="$V26_FACTS"
export MERLIN_SPIKE="$RUNTIME/bin/spike"
export MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME="$RUNTIME"
export LD_LIBRARY_PATH="$RUNTIME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Example argv ONLY; use the original bounded runner and a freshly admitted ELF:
"$MERLIN_SPIKE" --extlib="$RUNTIME/lib/libgemmini.so" --extension=gemmini \
  --isa=rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs \
  --priv=m -p1 -m0x80000000:0x400000000 "$ADMITTED_ELF"
```

No loadmem/signature options are implicit. The functional allocation is one hart,
machine mode, a16GiB address span; these are executable compatibility selections,
not proof of physical memory or bitstream ISA. Hardware association is 30MHz
FireSimGemminiRocketConfig, DIM16, int8/int32; Chipyard
`009e85b05ca817a15fcd6c3fe4eedd86cc2a3617`, Gemmini
`63f0b68a68f185a016a2993f2fb32e1661ea9e6c`. The historical dirty FireSim overlay,
matched FPGA execution and physical timing equivalence remain unestablished.
Spike is **untuned**. A current tuned native prediction requires separate exact
Count/semantics/runtime/model-domain owners; none follows from this archive.

## Compiler toolchain and remaining gates

Provision LLVM/MLIR23 at the exact source revision above. Recorded CMake options:
`LLVM_ENABLE_PROJECTS=mlir;clang`, `LLVM_TARGETS_TO_BUILD=X86;RISCV`, Release,
assertionsON, RTTIOFF, MLIR execution engineON, testsOFF, install-utilsON.
Use fresh caller-owned build/install/cache directories and independently bounded
build resources. Select `MERLIN_MLIR_INSTALL` explicitly. Original translation
binary SHA is `05e1860f1751defd32157dd41969ddd607a45b79a54ec54fca7f54305337f429`;
a rebuild has its own identity. The selected RISC-V GCC binary hash was
`4edd51fc7e8aecc1967d95cff423685dfe6565739fa208f876413fa8277ab8a4`;
the archive does not supply a hermetic GCC distribution/build recipe.

Observed harness environment was Python3.12.15, xdsl0.73.0, NumPy2.5.4,
PyYAML6.0.3 and jsonschema4.26.0, not a complete lockfile. Provision the separately
selected model2MLIR/reference environment only when that route is used; selecting
MERLIN_M2M_PYTHON is independent from sealed-capture configuration. Do not invoke
model loaders as a dependency probe. All runtime/qualification/source selections
must be rebuilt by original owners for the receiving server and actual compiler.

153public+8hidden capsule observations passed, but resource certification and
complete-model validation remain incomplete. The120s/1MiB resource sample was not
admitted. Local resource-executor repair does not retroactively certify V26.
No full Phase1 package, SDK release, formal gate, timing model or promotion is
silently substituted merely by preparing this remote setup.
