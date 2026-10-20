# Portable public pass inputs for the original V26 compiler

This bundle contains the exact original16-file generated Gemmini compiler and three public synthetic development fixtures. It is intended for another host; its manifest and descriptors use relative paths and SHA-256 identities. It contains no private validation model, hidden answer, operator receipt, live issuer, runtime, toolchain or cache. Fresh capture and admission on the destination host remain required.

Run the standard-library integrity check after transfer:

```sh
python3 -B verify.py
```

The check verifies the complete file roster, original compiler digest, every full input/reference array, output order/shape/dtype, copied source golden values and complete weight payload. It does not import Torch or xDSL, run the compiler, execute target work, compare against a new model run, or issue qualification. `SHA256SUMS` must itself be checked against the separately supplied transfer digest.

| Case | Complete source | Numeric contract and limits |
|---|---|---|
| `cases/direct_raw_i32` | M17 K16 N32 single contraction; one544-value output | Signedi8 × signedi8, exact rawi32. Small engineering control, not broad capacity/timing coverage. |
| `cases/integer_gqa_kv_state` | Two append steps,3queryheads/1KVhead, headwidth7; all16outputs/331values | Exact integer QK/PV and cache states. Source explicitly uses floor((wide+4)/8), i8saturation. No softmax/FlashAttention or physicalKV-residency claim. |
| `cases/parameter_chain_3mib` | Three width1024 projections,3distinct1MiB i8weights; both wide_i32/state_i8 outputs | Exact source integer arithmetic, floor((wide+4)/8) and i8saturation at each source boundary. Synthetic complete graph, not a complete headline or reduced validation model. Historical build success does not establish V26 execution. |

Each `case.json` is a new public fixture descriptor, **not an original admitted/sealed capsule**. Original capsule metadata containing host/operator bindings was omitted; its SHA is retained only as provenance. Python loaders, typed source, weights, copied capture arrays and independent references retain their original exact bytes. Reference member `output_i.npy` corresponds to the explicit output ordinal/name in `case.json`; do not match by element count. All outputs and parameters are required.

Compiler identity: `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`,16files112238bytes, using sorted relativePOSIX path +NUL+completebytes+NUL. Its two author reports contain historical public compiler notes, not a certificate. The separate V26 public result projection records152numeric/L0–L3 passes and one expected refusal, while formal resource certification and full-model validation remain incomplete. No old result or private-model numerical recipe qualifies these fixtures under V26.

The source compiler's original fallback toolchain path is preserved to keep its bytes exact. On a new host set `MERLIN_MLIR_INSTALL` explicitly to the selected LLVM/MLIR installation. Compilation requires a compatible Python/xDSL environment (original observed xDSL0.73.0); the PyTorch source loaders used Torch2.10.0+cpu. These dependencies are not bundled or automatically provisioned. No compilation or recapture is authorized by running `verify.py`.

All V26 fixture compilation, numerical execution and source realization are NOT_RUN/NOT_QUALIFIED; writer admission is NOT_ISSUED. These already inspected development fixtures are not held-out workloads or a frozen performance objective. Future experiments must retain exact source arithmetic, all live-outs, the independent references, required accelerator execution, explicit shared-library selection and original host-owned gates.
