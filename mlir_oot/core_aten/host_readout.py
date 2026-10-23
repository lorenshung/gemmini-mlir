"""Plain exact rational host RNE; widths cover every full-int32 input."""
from math import lcm

import numpy as np
from xdsl.dialects import llvm
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, IntegerAttr, IntegerType, i8, i32, i64
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
        global_op = llvm.GlobalOp(array, name, 'internal', constant=True,
            value=DenseIntOrFPElementsAttr.from_list(TensorType(typ, [len(values)]), values), alignment=64)
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
        indices = range(size)
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
            weights = coeffs[:-1]
            if not 0 <= qmax-qmin <= 255 or not qmin <= zp <= qmax:
                raise ValueError('invalid byte output quantization range or zero point')
            # No reachable-value specialization: include all signed-i32 tile
            # inputs, the constant, twice the numerator and every RNE boundary.
            bound = abs(coeffs[-1]) + sum(abs(w)*(1 << 31) for w in weights)
            edge = max(abs(qmin-zp), abs(qmax-zp))
            required = max(64, (2*bound).bit_length()+2,
                           ((2*edge+1)*denominator).bit_length()+2)
            typ = IntegerType(required)
            def c(v):
                return fb.const(v, typ)
            step = positions[1]-positions[0] if len(positions) > 1 else 1
            regular = isinstance(positions, range) or all(b-a == step for a,b in zip(positions,positions[1:]))

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
                value = c(coeffs[-1])
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
                    raw = fb.add(llvm.LoadOp(fb.gep(bases[name], index, i32), i32)).results[0]
                    wide = fb.add(llvm.SExtOp(raw, typ)).results[0]
                    term = wide if weight == 1 else binary(llvm.MulOp, wide, c(weight))
                    value = binary(llvm.AddOp, value, term)
                if readout.relu:
                    value = select(compare(value, c(0), 'slt'), c(0), value)
                twice = binary(llvm.MulOp, value, c(2))
                lo, hi = c(qmin-zp), c(qmax-zp)
                # Compare exact rationals to half-integer boundaries, without
                # constructing a transition table or dividing/rounding twice.
                for _ in range((qmax-qmin).bit_length()):
                    mid = binary(llvm.AShrOp, binary(llvm.AddOp, lo, hi), c(1))
                    threshold = binary(llvm.MulOp,
                        binary(llvm.AddOp, binary(llvm.MulOp, mid, c(2)), c(1)), c(denominator))
                    greater = compare(twice, threshold, 'sgt')
                    equal = compare(twice, threshold, 'eq')
                    odd = compare(binary(llvm.AndOp, mid, c(1)), c(0), 'ne')
                    above = binary(llvm.AndOp, compare(lo, hi, 'slt'),
                        binary(llvm.OrOp, greater, binary(llvm.AndOp, equal, odd)))
                    lo, hi = select(above, binary(llvm.AddOp, mid, c(1)), lo), select(above, hi, mid)
                code = binary(llvm.AddOp, lo, c(zp))
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
