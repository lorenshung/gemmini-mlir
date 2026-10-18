"""Source-bound Core ATen device catalog: match a linalg-on-tensors module, emit one object.

``build(source)`` matches every supported region (``match.match_all``), lints each kernel
stream against the stock-RTL hazard rules, deduplicates identical kernels by their exact
command stream, and returns the combined device module plus a manifest:

    schema    gemmini_core_aten_catalog_v1
    abi       every argument a pointer to a dense row-major buffer, in ``kernels[i].args``
              order; ``scratch`` arguments are caller-allocated, uninitialized and private to
              the call; no argument aliases another; the kernel fences before returning.
    kernels   symbol, Core ATen op, argument table (name, dtype, shape, role), stream counts
    bindings  per matched region: root op ordinal (whose result the kernel writes), the
              ordinals it replaces, and each kernel argument's source value as a function
              argument index or an (op ordinal, result index) pair
    declines  recognised regions this backend refuses, with the reason

It is opt-in beside the integer GEMM catalog (``golden_device_catalog``) and does not alter
that catalog's bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import BlockArgument

from ..frontend.parse import parse_module
from .emit import build_module, compile_to_object, encode
from .lint import lint
from .match import match_all

SCHEMA = "gemmini_core_aten_catalog_v1"


def _stream_digest(kernel) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([(a.name, a.dtype, list(a.shape), a.role) for a in kernel.stream.args]).encode())
    for name, data in sorted(kernel.stream.consts.items()):
        h.update(name.encode() + data.tobytes())
    from dataclasses import asdict
    h.update(json.dumps(_jsonable([asdict(r) for r in kernel.stream.readouts]),sort_keys=True).encode())
    h.update(repr(kernel.stream.packs).encode())
    for ins in kernel.stream.ins:
        r = encode(ins)
        h.update(repr((r.key, r.values)).encode())
    return h.hexdigest()


def _source(value, ordinals, func_args):
    if isinstance(value, BlockArgument) and value in func_args:
        return {"func_arg": func_args.index(value)}
    owner = value.owner
    return {"op_ordinal": ordinals[owner], "result": list(owner.results).index(value)}


def build(source: str) -> tuple[ModuleOp, dict, list]:
    module = parse_module(source)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    funcs = [op for op in module.walk() if op.name == "func.func"]
    func_args = list(funcs[0].body.blocks[0].args) if funcs else []
    matches, declines = match_all(module)
    kernels: dict[str, dict] = {}
    device = ModuleOp([])
    bindings = []
    for m in matches:
        report = lint(m.kernel.stream)
        digest = _stream_digest(m.kernel)
        symbol = "gemmini_ca_" + digest[:16]
        if symbol not in kernels:
            km, tree = build_module(m.kernel.stream, symbol)
            for op in list(km.body.blocks[0].ops):
                op.detach()
                device.body.blocks[0].add_op(op)
            kernels[symbol] = dict(symbol=symbol, op=m.op, stream_sha256=digest,
                                   args=[dict(name=a.name, dtype=a.dtype, shape=list(a.shape), role=a.role)
                                         for a in m.kernel.args],
                                   instructions=report["instructions"], fences=report["fences"],
                                   counts=m.kernel.stream.counts(), meta=_jsonable(m.kernel.meta),
                                   requantization=_readout_contracts(m.kernel.stream))
        argmap = {name: _source(v, ordinals, func_args) for name, v in m.operands}
        out_name, out_val = m.result
        provs = [getattr(o.attributes.get("prov.aten"), "data", "") for o in m.consumed if o is not None]
        provs = [p for p in provs if p and not p.startswith("quantized_decomposed.")]
        bindings.append(dict(
            op=m.op, aten=provs[0] if provs else "", symbol=symbol, root_ordinal=ordinals[m.root],
            source_operation_ordinal=ordinals[m.consumed[0]],
            replaced_ordinals=sorted({ordinals[o] for o in m.consumed if o is not None}),
            operands=argmap, result={out_name: _source(out_val, ordinals, func_args)},
            result_type=str(out_val.type), operand_types={n: str(v.type) for n, v in m.operands},
            region=getattr(m.root.attributes.get("prov.region_id"), "data", ""),
            layout=m.layout, numeric=_jsonable(m.numeric)))
    device.verify()
    manifest = dict(schema=SCHEMA, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    abi=dict(argument_order="kernels[].args", pointee_layout="dense_row_major",
                             scratch="caller-allocated, uninitialized, private to the call",
                             aliasing="none", completion="kernel fences before returning",
                             quantization="per-tensor symmetric int8, zero point 0; int32 bias in accumulator units"),
                    kernels=list(kernels.values()), bindings=bindings, declines=declines,
                    matched=len(bindings), unique_kernels=len(kernels))
    return device, manifest, matches


def _readout_contracts(stream):
    from dataclasses import asdict
    return _jsonable([dict(asdict(r),rounding='half_to_even',execution='host',device_readout='full_i32') for r in stream.readouts])


def _jsonable(obj):
    import numpy as np
    if isinstance(obj,np.ndarray): return _jsonable(obj.tolist())
    if isinstance(obj,np.integer): return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    from fractions import Fraction
    if isinstance(obj, Fraction):
        return dict(numerator=str(obj.numerator), denominator=str(obj.denominator))
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def compile_catalog(source_path: Path, workdir: Path, *, llvm_bin: Path | None = None) -> dict:
    source = Path(source_path).read_text()
    device, manifest, _ = build(source)
    workdir.mkdir(parents=True, exist_ok=True)
    if manifest["kernels"]:
        receipt = compile_to_object(device, workdir, llvm_bin=llvm_bin, stem="core_aten")
        manifest["compilation"] = receipt
    (workdir / "core_aten_catalog.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def merlin_builder(llvm_bin: Path):
    """Build function with Merlin's ``catalog_builder`` signature: (prepared, workdir) -> (manifest, object)."""

    def build_fn(prepared: Path, workdir: Path):
        compile_catalog(Path(prepared), Path(workdir), llvm_bin=Path(llvm_bin))
        return Path(workdir) / "core_aten_catalog.json", Path(workdir) / "core_aten.o"

    return build_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llvm-bin", type=Path)
    args = ap.parse_args()
    m = compile_catalog(args.input, args.workdir, llvm_bin=args.llvm_bin)
    print(json.dumps({k: m[k] for k in ("matched", "unique_kernels")} | {"declines": len(m["declines"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
