"""Prove gather-loop weight gradients as exact bilinear index maps.

The frontend may expose vector/tail loops instead of a canonical linalg MAC.
Match their actual products, coefficients and complete contribution set before
selecting the existing Gemmini convolution weight-gradient schedule.
"""
from fractions import Fraction as Q

import numpy as np

from . import linmap


def axis_geometry(position, tap, output, error=linmap.NotLinear, subject=('convolution coordinates', 'convolution geometry')):
    """Solve output = stride*position + dilation*tap - padding exactly; rank-deficient axes get unit defaults."""
    rows = []
    for a,b,c in sorted(set(zip(position.tolist(),tap.tolist(),output.tolist()))):
        row = [Q(a),Q(b),Q(1),Q(c)]
        for pivot,prior in rows:
            factor = row[pivot]
            row = [x-factor*y for x,y in zip(row,prior)]
        pivot = next((j for j in range(3) if row[j]),None)
        if pivot is None:
            if row[3]: raise error('non-affine ' + subject[0])
            continue
        row = [x/row[pivot] for x in row]
        rows.append((pivot,row));rows.sort()
    answer = [Q(1),Q(1),Q(0)]
    for pivot,row in reversed(rows):
        answer[pivot] = row[3]-sum(row[j]*answer[j] for j in range(pivot+1,3))
    if any(v.denominator != 1 for v in answer):
        raise error('noninteger ' + subject[1])
    stride,dilation,offset = map(int,answer)
    return stride,dilation,-offset


def match(op):
    from .match import (name_of,shape,producer,iterators,_dq_leaves,quant_sink,
                        Match,_pack_inputs)
    from .recipes.conv import ConvGeom
    from . import ops
    if name_of(op) != 'quant_ext.quantize_per_tensor': return None
    root = op.operands[0];body = producer(root)
    if (body is None or body.name != 'linalg.generic' or len(body.operands) != 1 or
            iterators(body) != ['parallel']*4): return None
    leaves = _dq_leaves(root)
    if len(leaves) != 2 or any(len(shape(q.value)) != 4 for q in leaves): return None
    qout = quant_sink(root)
    if qout is None: return None
    for qg,qx in (leaves,leaves[::-1]):
        N,F,OH,OW = shape(qg.value)
        NX,C,H,W = shape(qx.value)
        FO,CG,KH,KW = shape(root)
        if N != NX or F != FO or not CG or C % CG: continue
        groups = C//CG
        if F % groups: continue
        nx = int(np.prod(shape(qx.value)))
        ng = int(np.prod(shape(qg.value)))
        if nx*ng >= 1 << 63: continue
        class Bilinear(linmap.Region):
            def _scalar(self, operation, vals, env, ivs, mask):
                if operation.name == 'arith.mulf' and all(isinstance(v,linmap.Sym) for v in vals):
                    a,b = vals
                    if np.any(a.const) or np.any(b.const):
                        raise linmap.NotLinear('nonzero bilinear operand constant')
                    terms = []
                    for sa,ia,ca in a.terms:
                        for sb,ib,cb in b.terms:
                            if {sa,sb} != {0,1}:
                                raise linmap.NotLinear('weight gradient is not a two-input bilinear map')
                            gi,xi = (ia,ib) if sa == 0 else (ib,ia)
                            terms.append((2,gi*nx+xi,ca*cb))
                    return linmap.Sym(terms,0)
                return super()._scalar(operation,vals,env,ivs,mask)
        def leaf(value):
            if value is qg.dq.results[0]: return 0,Q(1)
            if value is qx.dq.results[0]: return 1,Q(1)
            return None
        try:
            lm,region = linmap.extract(root,leaf,region_type=Bilinear)
            if not lm.out.size or np.any(lm.coef != 1) or np.any(lm.src != 2) or np.any(lm.const): continue
            bn,fc,oh,ow = np.unravel_index(lm.idx//nx,shape(qg.value))
            xn,xc,iy,ix = np.unravel_index(lm.idx%nx,shape(qx.value))
            of,oc,kh,kw = np.unravel_index(lm.out,shape(root))
            if not (np.array_equal(bn,xn) and np.array_equal(fc,of) and
                    np.array_equal(xc,oc+(of//(F//groups))*CG)): continue
            sh,dh,ph = axis_geometry(oh,kh,iy)
            sw,dw,pw = axis_geometry(ow,kw,ix)
            g = ConvGeom(N,H,W,C,F,KH,KW,sh,sw,ph,pw,dh,dw,groups)
            g.validate()
            if (g.oh,g.ow) != (OH,OW): continue
            yy = np.arange(OH)[:,None]*sh-ph+np.arange(KH)[None]*dh
            xx = np.arange(OW)[:,None]*sw-pw+np.arange(KW)[None]*dw
            expected = int(((yy>=0)&(yy<H)).sum()*((xx>=0)&(xx<W)).sum())*N*F*CG
            if lm.out.size != expected or len(np.unique(np.stack((lm.out,lm.idx),axis=1),axis=0)) != expected: continue
        except (linmap.NotLinear,ValueError):
            continue
        kernel = ops.conv_backward_weight(g,s_g=qg.scale,s_x=qx.scale,s_out=qout.scale)
        _pack_inputs(kernel,[('g',qg),('x',qx)])
        ops.convolution_zero_points(kernel,('g',qg),('x',qx),g,'weight')
        kernel.meta['source_proof'] = 'complete_closed_bilinear_gather_map'
        return Match('aten.convolution_backward.default',qout.op,[*region,qout.op,qg.dq,qx.dq],
                     [('g',qg.value),('x',qx.value)],('y',qout.value),kernel,
                     layout=dict(source='NCHW,NCHW->OIHW',transposes='on-device'),
                     numeric=dict(output='grad_weight',s_g=qg.scale,s_x=qx.scale,s_out=qout.scale,
                                  geometry=g.__dict__,verification='complete_closed_bilinear_gather_map'))
    return None
