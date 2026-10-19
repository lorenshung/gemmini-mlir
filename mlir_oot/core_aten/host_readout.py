"""requant-A host epilogues: exact clamp/multiply/add/shift or threshold table."""
from fractions import Fraction
from math import gcd, lcm
from functools import reduce

import numpy as np
from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, IntegerAttr, IntegerType, DenseArrayBase, i8, i32, i64
from xdsl.ir import Block, Region
from ..codegen.builder import iconst
from .requant_a import plan, validate
from .readout_bounds import interval
from .exactness import rational


from .requant import parameters

def emit(emitter, readouts=None):
    s, fb = emitter.stream, emitter.fb
    bases = {a.name: fb.add(llvm.IntToPtrOp(emitter.base[i])).results[0]
             for i, a in enumerate(s.args) if a.role != 'const'}

    def binary(cls, a, b):
        return fb.add(cls(a, b)).results[0]

    def compare(a, b, pred):
        return fb.add(llvm.ICmpOp(a, b, IntegerAttr(llvm.ICmpPredicateFlag(pred).to_int(), i64))).results[0]

    def select(cond, a, b):
        return fb.add(llvm.SelectOp(cond, a, b)).results[0]

    def table(values, typ, suffix):
        name = f"{emitter.symbol}__{suffix}_{len(emitter.globals)}"
        array = llvm.LLVMArrayType(len(values), typ)
        if typ.bitwidth <= 64:
            global_op = llvm.GlobalOp(array, name, 'internal', constant=True,
                value=DenseIntOrFPElementsAttr.from_list(TensorType(typ, [len(values)]), values), alignment=64)
        else:
            # xDSL's dense packing supports only <=64-bit elements. LLVM's
            # typed initializer region supports exact wider thresholds too.
            block = Block()
            undef = llvm.UndefOp(array)
            block.add_op(undef)
            value = undef.results[0]
            for i, number in enumerate(values):
                literal = iconst(number, typ)
                block.add_op(literal)
                insert = llvm.InsertValueOp(DenseArrayBase.from_list(i64, [i]), value, literal.results[0])
                block.add_op(insert)
                value = insert.results[0]
            block.add_op(llvm.ReturnOp(value))
            global_op = llvm.GlobalOp(array, name, 'internal', constant=True, alignment=64, body=Region(block))
        emitter.globals.append(global_op)
        return fb.add(llvm.AddressOfOp(name, emitter.ptr_type)).results[0]

    for readout in s.readouts if readouts is None else readouts:
        outarg = next(a for a in s.args if a.name == readout.out)
        placement = readout.placement
        size = (int(np.prod(placement['shape'])) if isinstance(placement, dict) else
                placement[0] if placement is not None else int(np.prod(outarg.shape)))
        values = [f for _, f in readout.terms] + [readout.constant, readout.zero_point,
                                                readout.quant_min, readout.quant_max]
        uniform = all(np.ndim(v) == 0 for v in values)
        if placement is not None and not uniform:
            raise ValueError('structural placement requires scalar readout parameters')
        indices = range(size) if readout.indices is None else readout.indices
        classes = {}
        if uniform:
            if any(rational(v).denominator != 1 for v in values[-3:]):
                raise ValueError('noninteger output quantization parameter')
            signature = tuple(rational(v) for v in values[:-3]) + tuple(int(v) for v in values[-3:])
            classes[signature] = indices
        else:
            params = [parameters(readout, v, outarg.shape) for v in values]
            for ordinal, p in enumerate(indices):
                if any(rational(f[ordinal]).denominator != 1 for f in params[-3:]):
                    raise ValueError('noninteger output quantization parameter')
                signature = tuple(rational(f[ordinal]) for f in params[:-3])+tuple(int(f[ordinal]) for f in params[-3:])
                classes.setdefault(signature, []).append(int(p))
        for signature, positions in classes.items():
            fractions, (zp, qmin, qmax) = signature[:-3], signature[-3:]
            denominator = lcm(*(f.denominator for f in fractions))
            coeffs = [int(f*denominator) for f in fractions]
            common = reduce(gcd, (abs(c) for c in coeffs[:-1]), 0) or 1
            weights = [c//common for c in coeffs[:-1]]
            low, high, sources, _ = interval(readout, [*weights, 0], emitter.meta)
            record = plan(Fraction(common, denominator), Fraction(coeffs[-1], denominator),
                          zp, qmin, qmax, readout.relu, low, high)
            # Revalidate before emitting imported or locally cached constants.
            validate(record)
            step = positions[1]-positions[0] if len(positions) > 1 else 1
            regular = isinstance(positions, range) or all(b-a == step for a, b in zip(positions, positions[1:]))
            binding = dict(start=positions[0], stop=positions[-1]+1, step=step) if regular else dict(indices=positions)
            record.update(output=readout.out, elements=len(positions), output_indices=binding,
                          normal_form=[dict(tile=n, weight=str(w)) for (n, _), w in zip(readout.terms, weights)],
                          accumulator_bounds=sources)
            if placement is not None:
                record['output_placement'] = placement
            emitter.meta['host_requantization_proofs'].append(record)
            bound = sum(abs(w)*(1 << 31) for w in weights)
            required = max(64, bound.bit_length()+1)
            typ = IntegerType(1 << (required-1).bit_length())
            record["normal_form_width"] = typ.bitwidth
            def c(v):
                return fb.const(v, typ)
            fast = record['algorithm'] == 'clamp_mul_add_shift'
            thresholds = list(map(int, record.get('thresholds', [])))
            thresholds_ptr = table(thresholds, typ, 'thresholds') if not fast else None

            def body(iv):
                target = iv
                if isinstance(placement, dict):
                    target = fb.const(placement['offset'])
                    for axis, (extent, stride) in enumerate(zip(placement['shape'], placement['strides'])):
                        coord = fb.urem_i(fb.udiv_i(iv, fb.const(int(np.prod(placement['shape'][axis+1:])))), fb.const(extent))
                        target = fb.add_i(target, fb.mul_i(coord, fb.const(stride)))
                elif placement is not None:
                    _, block, offset, stride = placement
                    target = fb.add_i(fb.add_i(fb.mul_i(fb.udiv_i(iv, fb.const(block)), fb.const(stride)),
                                              fb.urem_i(iv, fb.const(block))), fb.const(offset))
                value = c(0)
                for (name, _), weight in zip(readout.terms, weights):
                    if not weight:
                        continue
                    index = target
                    if readout.permutations and name in readout.permutations:
                        shp, perm = readout.permutations[name]
                        index = fb.const(0)
                        for axis in range(len(shp)):
                            pos = perm.index(axis)
                            coordinate = fb.urem_i(fb.udiv_i(target, fb.const(int(np.prod(outarg.shape[pos+1:])))), fb.const(shp[axis]))
                            index = fb.add_i(index, fb.mul_i(coordinate, fb.const(int(np.prod(shp[axis+1:])))))
                    if readout.broadcasts and name in readout.broadcasts:
                        shp = readout.broadcasts[name]
                        index = fb.const(0)
                        for axis, dim in enumerate(shp):
                            if dim == 1:
                                continue
                            pos = len(outarg.shape)-len(shp)+axis
                            coordinate = fb.urem_i(fb.udiv_i(target, fb.const(int(np.prod(outarg.shape[pos+1:])))), fb.const(dim))
                            index = fb.add_i(index, fb.mul_i(coordinate, fb.const(int(np.prod(shp[axis+1:])))))
                    if readout.source_offsets and name in readout.source_offsets:
                        index = fb.add_i(index, fb.const(readout.source_offsets[name]))
                    raw = fb.add(llvm.LoadOp(fb.gep(bases[name], index, i32), i32)).results[0]
                    wide = fb.add(llvm.SExtOp(raw, typ)).results[0]
                    term = wide if weight == 1 else binary(llvm.MulOp, wide, c(weight))
                    value = binary(llvm.AddOp, value, term)
                if fast:
                    k = record['constants']
                    lower, upper = c(int(k['L'])), c(int(k['H']))
                    v = select(compare(value, lower, 'slt'), lower, value)
                    v = select(compare(v, upper, 'sgt'), upper, v)
                    if typ != i64:
                        v = fb.add(llvm.TruncOp(v, i64)).results[0]
                    product = binary(llvm.MulOp, v, fb.const(int(k['M'])))
                    added = binary(llvm.AddOp, product, fb.const(int(k['B'])))
                    code = binary(llvm.AShrOp, added, fb.const(k['S']))
                    if k['out_clamp']:
                        lower, upper = fb.const(k['qlo']), fb.const(k['qhi'])
                        code = select(compare(code, lower, 'slt'), lower, code)
                        code = select(compare(code, upper, 'sgt'), upper, code)
                else:
                    lo, hi = fb.const(0), fb.const(len(thresholds))
                    for _ in range(len(thresholds).bit_length()):
                        active = compare(lo, hi, 'slt')
                        mid = fb.ashr_i(fb.add_i(lo, hi), 1)
                        index = select(active, mid, fb.const(0))
                        threshold = fb.add(llvm.LoadOp(fb.gep(thresholds_ptr, index, typ), typ)).results[0]
                        above = binary(llvm.AndOp, active, compare(value, threshold, 'sge'))
                        lo, hi = select(above, fb.add_i(mid, fb.const(1)), lo), select(above, hi, mid)
                    code = fb.add_i(lo, fb.const(qmin))
                byte = fb.add(llvm.TruncOp(code, i8)).results[0]
                fb.add(llvm.StoreOp(byte, fb.gep(bases[readout.out], target, i8)))

            # Regular channel strides need neither a full index table nor one
            # loop per element, including per-column frozen biases.
            if regular:
                fb.for_loop(positions[0], positions[-1]+1, step, body)
            else:
                indices_ptr = table(positions, i64, 'readout_indices')
                fb.for_loop(0, len(positions), 1, lambda ordinal: body(fb.load_i64(indices_ptr, ordinal, 'i64')))

def emit_packs(emitter):
    s, fb = emitter.stream, emitter.fb
    for source, dest in s.packs:
        si, di = s._index[source], s._index[dest]
        src = fb.add(llvm.IntToPtrOp(emitter.base[si])).results[0]
        dst = fb.add(llvm.IntToPtrOp(emitter.base[di])).results[0]
        size = int(np.prod(s.args[si].shape))

        def body(iv):
            value = fb.add(llvm.LoadOp(fb.gep(src, iv, i8), i8)).results[0]
            packed = fb.add(llvm.XOrOp(value, fb.const(-128, i8))).results[0]
            fb.add(llvm.StoreOp(packed, fb.gep(dst, iv, i8)))

        fb.for_loop(0, size, 1, body)
