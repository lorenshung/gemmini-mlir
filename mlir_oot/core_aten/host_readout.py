"""LLVM implementation of the exact readout ABI (host CPU, never Gemmini FP32).

Integer widths are derived from the actual rational parameters and all i32
inputs, not calibration values. Certified reciprocal multiplication performs
one RNE; unsupported widths retain the exact threshold comparison path.
"""

from fractions import Fraction
from math import lcm

import numpy as np
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    TensorType,
    IntegerAttr,
    IntegerType,
    i8,
    i32,
    i64,
)


def emit(emitter, readouts=None):
    s, fb = emitter.stream, emitter.fb
    bases = {
        a.name: fb.add(llvm.IntToPtrOp(emitter.base[i])).results[0]
        for i, a in enumerate(s.args)
        if a.role != "const"
    }
    for readout in s.readouts if readouts is None else readouts:
        outarg = next(a for a in s.args if a.name == readout.out)
        size = (int(np.prod(readout.placement['shape'])) if isinstance(readout.placement,dict) else
                readout.placement[0] if readout.placement else int(np.prod(outarg.shape)))
        uniform = all(np.ndim(f) == 0 for _,f in readout.terms) and np.ndim(readout.constant) == 0
        factors = [] if uniform else [
            (
                np.asarray(f, dtype=object).reshape(-1)
                if np.size(f) == size
                else np.broadcast_to(np.asarray(f, dtype=object), outarg.shape).reshape(
                    -1
                )
            )
            for _, f in readout.terms
        ]
        constants = None if uniform else (
            np.asarray(readout.constant, dtype=object).reshape(-1)
            if np.size(readout.constant) == size
            else np.broadcast_to(
                np.asarray(readout.constant, dtype=object), outarg.shape
            ).reshape(-1)
        )
        # Tensor outputs are private and cannot alias readout inputs. Share
        # one exact body for repeated parameter classes, including spatial
        # borders, instead of unrolling a body for every short run.
        runs = [(tuple(Fraction(f) for _,f in readout.terms)+(Fraction(readout.constant),),0,size)] if uniform else []
        start = 0
        while not uniform and start < size:
            signature = tuple(Fraction(f[start]) for f in factors) + (
                Fraction(constants[start]),
            )
            end = start + 1
            while end < size and all(
                f[end] == v for f, v in zip([*factors, constants], signature)
            ):
                end += 1
            runs.append((signature, start, end))
            start = end
        classes = {}
        for signature, start, end in runs:
            classes.setdefault(signature, []).append((start, end))
        for signature, pieces in classes.items():
            denominator = lcm(*(v.denominator for v in signature))
            coeffs = [int(v * denominator) for v in signature]
            bound = sum(abs(c) * (1 << 31) for c in coeffs[:-1]) + abs(coeffs[-1])
            width = max(
                64, (2 * bound).bit_length() + 2, (1024 * denominator).bit_length() + 2
            )
            from .host_magic import interval, prove, verify_reachable
            low, high, sources, safe = interval(readout, coeffs, emitter.meta)
            proof = dict(prove(low, high, denominator, readout.quant_min,
                               readout.quant_max, readout.zero_point, readout.relu))
            if not safe:
                proof = dict(method="comparison_search", reason="i64 intermediate bound failed")
            if proof["method"] == "exact_reciprocal_multiply_shift":
                verify_reachable(proof, coeffs, sources, denominator)
            proof.update(output=readout.out, accumulator_bounds=sources,
                         numerator_min=str(low), numerator_max=str(high),
                         denominator=str(denominator), int64_intermediates_proven=bool(safe),
                         coefficients=[str(v) for v in coeffs],
                         elements=sum(end-start for start, end in pieces))
            emitter.meta["host_requantization_proofs"].append(proof)
            magic = proof["method"] == "exact_reciprocal_multiply_shift"
            typ = i64 if magic else IntegerType(width)

            def c(v):
                return fb.const(v, typ)

            def binary(cls, a, b):
                return fb.add(cls(a, b)).results[0]

            def compare(a, b, pred):
                return fb.add(
                    llvm.ICmpOp(
                        a, b, IntegerAttr(llvm.ICmpPredicateFlag(pred).to_int(), i64)
                    )
                ).results[0]

            def select(cond, a, b):
                return fb.add(llvm.SelectOp(cond, a, b)).results[0]

            def body(iv):
                target = iv
                if isinstance(readout.placement,dict):
                    p = readout.placement
                    target = fb.const(p['offset'])
                    for axis,(extent,stride) in enumerate(zip(p['shape'],p['strides'])):
                        coord = fb.urem_i(fb.udiv_i(iv,fb.const(int(np.prod(p['shape'][axis+1:])))),fb.const(extent))
                        target = fb.add_i(target,fb.mul_i(coord,fb.const(stride)))
                elif readout.placement:
                    _, block, offset, stride = readout.placement
                    target = fb.add_i(fb.add_i(fb.mul_i(fb.udiv_i(iv,fb.const(block)),fb.const(stride)),
                                               fb.urem_i(iv,fb.const(block))),fb.const(offset))
                value = c(coeffs[-1])
                for (name, _), weight in zip(readout.terms, coeffs):
                    index = target
                    if readout.permutations and name in readout.permutations:
                        shp, perm = readout.permutations[name]
                        output_shape = outarg.shape
                        index = fb.const(0)
                        for axis in range(len(shp)):
                            pos = perm.index(axis)
                            trailing_out = int(np.prod(output_shape[pos + 1 :]))
                            coordinate = fb.urem_i(
                                fb.udiv_i(iv, fb.const(trailing_out)),
                                fb.const(shp[axis]),
                            )
                            index = fb.add_i(
                                index,
                                fb.mul_i(
                                    coordinate, fb.const(int(np.prod(shp[axis + 1 :])))
                                ),
                            )
                    if readout.broadcasts and name in readout.broadcasts:
                        shp = readout.broadcasts[name]
                        index = fb.const(0)
                        output_shape = outarg.shape
                        for axis, dim in enumerate(shp):
                            if dim == 1:
                                continue
                            pos = len(output_shape) - len(shp) + axis
                            coordinate = fb.urem_i(
                                fb.udiv_i(
                                    iv, fb.const(int(np.prod(output_shape[pos + 1 :])))
                                ),
                                fb.const(dim),
                            )
                            index = fb.add_i(
                                index,
                                fb.mul_i(
                                    coordinate, fb.const(int(np.prod(shp[axis + 1 :])))
                                ),
                            )
                    raw = fb.add(
                        llvm.LoadOp(fb.gep(bases[name], index, i32), i32)
                    ).results[0]
                    wide = fb.add(llvm.SExtOp(raw, typ)).results[0]
                    value = binary(
                        llvm.AddOp, value, binary(llvm.MulOp, wide, c(weight))
                    )
                if readout.relu:
                    value = select(compare(value, c(0), "slt"), c(0), value)
                if magic:
                    negative = compare(value, c(0), "slt")
                    magnitude = select(negative, binary(llvm.SubOp, c(0), value), value)
                    shift = int(proof["shift"])
                    multiplier = int(proof["multiplier"])
                    if multiplier == 1:
                        quotient = binary(llvm.LShrOp, magnitude, c(shift))
                    else:
                        wide = IntegerType(128)
                        operand = fb.add(llvm.ZExtOp(magnitude, wide)).results[0]
                        product = binary(llvm.MulOp, operand, fb.const(multiplier, wide))
                        shifted = binary(llvm.LShrOp, product, fb.const(shift, wide))
                        quotient = fb.add(llvm.TruncOp(shifted, i64)).results[0]
                    remainder = binary(llvm.SubOp, magnitude,
                                       binary(llvm.MulOp, quotient, c(denominator)))
                    complement = binary(llvm.SubOp, c(denominator), remainder)
                    odd = compare(binary(llvm.AndOp, quotient, c(1)), c(0), "ne")
                    increment = binary(llvm.OrOp, compare(remainder, complement, "ugt"),
                                       binary(llvm.AndOp, compare(remainder, complement, "eq"), odd))
                    rounded = binary(llvm.AddOp, quotient, select(increment, c(1), c(0)))
                    signed = select(negative, binary(llvm.SubOp, c(0), rounded), rounded)
                    # Clamp before adding zero point to avoid signed overflow.
                    lower, upper = c(readout.quant_min-readout.zero_point), c(readout.quant_max-readout.zero_point)
                    clamped = select(compare(signed, lower, "slt"), lower, signed)
                    clamped = select(compare(clamped, upper, "sgt"), upper, clamped)
                    code = binary(llvm.AddOp, clamped, c(readout.zero_point))
                else:
                    twice = binary(llvm.MulOp, value, c(2))
                    lo, hi = (
                        c(readout.quant_min - readout.zero_point),
                        c(readout.quant_max - readout.zero_point),
                    )
                    for _ in range(8):
                        mid = binary(llvm.AShrOp, binary(llvm.AddOp, lo, hi), c(1))
                        threshold = binary(
                            llvm.MulOp,
                            binary(llvm.AddOp, binary(llvm.MulOp, mid, c(2)), c(1)),
                            c(denominator),
                        )
                        greater = compare(twice, threshold, "sgt")
                        equal = compare(twice, threshold, "eq")
                        odd = compare(binary(llvm.AndOp, mid, c(1)), c(0), "ne")
                        above = binary(
                            llvm.AndOp,
                            binary(llvm.OrOp, greater, binary(llvm.AndOp, equal, odd)),
                            compare(lo, hi, "slt"),
                        )
                        lo, hi = (
                            select(above, binary(llvm.AddOp, mid, c(1)), lo),
                            select(above, hi, mid),
                        )
                    code = binary(llvm.AddOp, lo, c(readout.zero_point))
                byte = fb.add(llvm.TruncOp(code, i8)).results[0]
                fb.add(llvm.StoreOp(byte, fb.gep(bases[readout.out], target, i8)))

            if len(pieces) == 1:
                start, end = pieces[0]
                fb.for_loop(start, end, 1, body)
            else:
                indices = [p for start, end in pieces for p in range(start, end)]
                name = f"{emitter.symbol}__readout_indices_{len(emitter.globals)}"
                value = DenseIntOrFPElementsAttr.from_list(
                    TensorType(i64, [len(indices)]), indices
                )
                emitter.globals.append(
                    llvm.GlobalOp(
                        llvm.LLVMArrayType(len(indices), i64),
                        name,
                        "internal",
                        constant=True,
                        value=value,
                        alignment=64,
                    )
                )
                table = fb.add(llvm.AddressOfOp(name, emitter.ptr_type)).results[0]

                def indexed_body(ordinal):
                    body(fb.load_i64(table, ordinal, "i64"))

                fb.for_loop(0, len(indices), 1, indexed_body)


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
