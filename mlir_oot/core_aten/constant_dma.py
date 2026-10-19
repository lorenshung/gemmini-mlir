"""Specialize proven target DMA maps on literal storage, before instruction emission.

This evaluates the very same byte rectangles as layout_match/native, never an
arithmetic interpreter or a per-element Python map. External entry arguments
are not constants: immutable-weight discovery/ABI projection belongs to Merlin.
"""
from hashlib import sha256
from math import prod

import numpy as np
from xdsl.dialects import arith
from xdsl.dialects.builtin import BytesAttr, DenseIntOrFPElementsAttr, TensorType

from .layout_match import describe, literal
from .match import Decline, _attr_of, etype, shape
from .storage import itemsize, literal_bytes


def specialize(module, ordinals=None):
    """Return (source-bound proof, original op, replacement constant) triples.

    The ordinal table retains original operations after replacement. Detach
    their operand uses so dead source edges cannot escape later region proofs.
    Storage is immutable.
    """
    from .liveness import live_source_ops
    original = list(module.walk())
    if ordinals is None:
        ordinals = {op: i for i, op in enumerate(original)}
    live = live_source_ops(module)
    values, identities, pool, folds = {}, {}, {}, []
    for op in original:
        if op not in live or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
            continue
        result = op.results[0]
        sh = shape(result)
        if any(d <= 0 for d in sh):
            continue
        try:
            width = itemsize(etype(result))
        except (Decline, ValueError):
            continue
        nbytes = prod(sh)*width
        if nbytes > 64 << 20:
            continue
        if op.name == 'arith.constant':
            attr = _attr_of(op, 'value')
            if not isinstance(attr, DenseIntOrFPElementsAttr):
                continue
            raw = attr.data.data
            if len(raw) == width and nbytes != width:
                raw *= prod(sh)
            if len(raw) == nbytes:
                values[result] = raw
                identities[result] = dict(op_ordinal=ordinals[op], result=0,
                                          storage_sha256=sha256(raw).hexdigest())
            continue
        if any(key.startswith('prov.quant.') for key in op.attributes):
            continue
        try:
            value = literal(op)
            plan = ([], []) if value is not None else describe(op)
            if plan is None:
                continue
            sources, maps = plan
            if any(v not in values or itemsize(etype(v)) != width for v in sources):
                continue
            if value is not None:
                raw = literal_bytes(value, etype(result))*prod(sh)
                # The literal's defining scalar operation is still authenticated
                # by the original module hash and this operation's ordinal.
            else:
                output = np.empty(nbytes, dtype=np.uint8)
                for src, mapping in zip(sources, maps, strict=True):
                    dims, ss, offset, ds, *source_offset = mapping
                    start = (source_offset[0] if source_offset else 0)*width
                    src_raw = values[src]
                    if (start < 0 or offset < 0 or any(s < 0 for s in ss)
                            or any(s <= 0 for s in ds)
                            or start+sum((d-1)*s*width for d,s in zip(dims,ss))+width > len(src_raw)
                            or offset*width+sum((d-1)*s*width for d,s in zip(dims,ds))+width > nbytes):
                        raise ValueError('constant DMA exceeds declared storage')
                    source_view = np.ndarray((*dims,width), np.uint8, buffer=src_raw,
                                             offset=start, strides=(*[s*width for s in ss],1))
                    target_view = np.ndarray((*dims,width), np.uint8, buffer=output,
                                             offset=offset*width, strides=(*[s*width for s in ds],1))
                    np.copyto(target_view, source_view)
                raw = output.tobytes()
        except (Decline, ValueError, OverflowError, KeyError, NotImplementedError):
            continue  # No proved constant specialization: retain the exact source.
        digest = sha256(raw).hexdigest()
        if value is not None:
            # Splat/fill is already constant-like, including contraction
            # initializers whose literal form is part of a separate proof.
            # Expose its storage to subsequent DMA maps without rewriting it.
            values[result] = raw
            identities[result] = dict(op_ordinal=ordinals[op], result=0, storage_sha256=digest)
            continue
        key = (str(result.type), digest)
        attr = pool.get(key)
        if attr is None:
            attr = DenseIntOrFPElementsAttr(result.type, BytesAttr(raw))
            pool[key] = attr
        replacement = arith.ConstantOp(attr)
        op.parent_block().insert_op_before(replacement, op)
        result.replace_all_uses_with(replacement.result)
        op.detach()
        op.drop_all_references()
        values[replacement.result] = attr.data.data
        identities[replacement.result] = dict(op_ordinal=ordinals[op], result=0, storage_sha256=digest)
        ordinals[replacement] = ordinals[op]
        proof = dict(root_ordinal=ordinals[op], result=0, result_type=str(result.type),
                     storage_sha256=digest, bytes=nbytes,
                     proof='literal_storage_through_proved_dma_rectangles',
                     sources=[identities[v] for v in sources])
        folds.append((proof, op, replacement))
    return folds
