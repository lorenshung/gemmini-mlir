"""LLVM implementation of the exact readout ABI (host CPU, never Gemmini FP32).

Integer widths are derived from the actual rational parameters and all i32
inputs, not calibration values. Eight threshold comparisons perform one RNE.
"""
from fractions import Fraction
from math import lcm

import numpy as np
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType, i8, i32, i64


def emit(emitter, readouts=None):
    s, fb = emitter.stream, emitter.fb
    bases = {a.name: fb.add(llvm.IntToPtrOp(emitter.base[i])).results[0]
             for i,a in enumerate(s.args) if a.role != 'const'}
    for readout in s.readouts if readouts is None else readouts:
        outarg = next(a for a in s.args if a.name == readout.out)
        size = int(np.prod(outarg.shape))
        factors = [(np.asarray(f,dtype=object).reshape(-1) if np.size(f)==size else np.broadcast_to(np.asarray(f,dtype=object),outarg.shape).reshape(-1)) for _,f in readout.terms]
        constants = (np.asarray(readout.constant,dtype=object).reshape(-1) if np.size(readout.constant)==size else np.broadcast_to(np.asarray(readout.constant,dtype=object),outarg.shape).reshape(-1))
        # Keep long uniform runs as CPU loops. Heterogeneous factors retain
        # exact per-element constants, never binary64 dictionary keys.
        start = 0
        while start < size:
            signature = tuple(Fraction(f[start]) for f in factors) + (Fraction(constants[start]),)
            end = start+1
            while end < size and all(f[end] == v for f,v in zip([*factors,constants],signature)):
                end += 1
            denominator = lcm(*(v.denominator for v in signature))
            coeffs = [int(v*denominator) for v in signature]
            bound = sum(abs(c)*(1 << 31) for c in coeffs[:-1])+abs(coeffs[-1])
            width = max(64, (2*bound).bit_length()+2, (1024*denominator).bit_length()+2)
            typ = IntegerType(width)
            def c(v): return fb.const(v,typ)
            def binary(cls,a,b): return fb.add(cls(a,b)).results[0]
            def compare(a,b,pred):
                return fb.add(llvm.ICmpOp(a,b,IntegerAttr(llvm.ICmpPredicateFlag(pred).to_int(),i64))).results[0]
            def select(cond,a,b): return fb.add(llvm.SelectOp(cond,a,b)).results[0]
            def body(iv):
                value = c(coeffs[-1])
                for (name,_),weight in zip(readout.terms,coeffs):
                    index = iv
                    if readout.permutations and name in readout.permutations:
                        shp,perm = readout.permutations[name]
                        output_shape = [shp[d] for d in perm]
                        index = fb.const(0)
                        for axis in range(len(shp)):
                            pos = perm.index(axis)
                            trailing_out = int(np.prod(output_shape[pos+1:]))
                            coordinate = fb.urem_i(fb.udiv_i(iv,fb.const(trailing_out)),fb.const(shp[axis]))
                            index = fb.add_i(index,fb.mul_i(coordinate,fb.const(int(np.prod(shp[axis+1:])))))
                    if readout.broadcasts and name in readout.broadcasts:
                        shp = readout.broadcasts[name]
                        index = fb.const(0)
                        output_shape = outarg.shape
                        for axis,dim in enumerate(shp):
                            if dim == 1: continue
                            pos = len(output_shape)-len(shp)+axis
                            coordinate = fb.urem_i(fb.udiv_i(iv,fb.const(int(np.prod(output_shape[pos+1:])))),fb.const(dim))
                            index = fb.add_i(index,fb.mul_i(coordinate,fb.const(int(np.prod(shp[axis+1:])))))
                    raw = fb.add(llvm.LoadOp(fb.gep(bases[name],index,i32),i32)).results[0]
                    wide = fb.add(llvm.SExtOp(raw,typ)).results[0]
                    value = binary(llvm.AddOp,value,binary(llvm.MulOp,wide,c(weight)))
                if readout.relu:
                    value = select(compare(value,c(0),'slt'),c(0),value)
                twice = binary(llvm.MulOp,value,c(2))
                lo,hi = c(readout.quant_min-readout.zero_point),c(readout.quant_max-readout.zero_point)
                for _ in range(8):
                    mid = binary(llvm.AShrOp,binary(llvm.AddOp,lo,hi),c(1))
                    threshold = binary(llvm.MulOp,binary(llvm.AddOp,binary(llvm.MulOp,mid,c(2)),c(1)),c(denominator))
                    greater = compare(twice,threshold,'sgt')
                    equal = compare(twice,threshold,'eq')
                    odd = compare(binary(llvm.AndOp,mid,c(1)),c(0),'ne')
                    above = binary(llvm.AndOp,binary(llvm.OrOp,greater,binary(llvm.AndOp,equal,odd)),compare(lo,hi,'slt'))
                    lo,hi = select(above,binary(llvm.AddOp,mid,c(1)),lo),select(above,hi,mid)
                code = binary(llvm.AddOp,lo,c(readout.zero_point))
                byte = fb.add(llvm.TruncOp(code,i8)).results[0]
                fb.add(llvm.StoreOp(byte,fb.gep(bases[readout.out],iv,i8)))
            fb.for_loop(start,end,1,body)
            start = end


def emit_packs(emitter):
    s,fb=emitter.stream,emitter.fb
    for source,dest in s.packs:
        si,di=s._index[source],s._index[dest]
        src=fb.add(llvm.IntToPtrOp(emitter.base[si])).results[0]
        dst=fb.add(llvm.IntToPtrOp(emitter.base[di])).results[0]
        size=int(np.prod(s.args[si].shape))
        def body(iv):
            value=fb.add(llvm.LoadOp(fb.gep(src,iv,i8),i8)).results[0]
            packed=fb.add(llvm.XOrOp(value,fb.const(-128,i8))).results[0]
            fb.add(llvm.StoreOp(packed,fb.gep(dst,iv,i8)))
        fb.for_loop(0,size,1,body)
