"""Exhaustive byte-domain affine checks, including exact ties, on Spike."""
import argparse
from fractions import Fraction as Q
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE/'device'))
from cases import Case
from test_device_affine import quant
from run_cases import run
from mlir_oot.core_aten import ops
from mlir_oot.core_aten.stream import Stream, Arg, ACC


def cases():
    result = []
    a = np.resize(np.arange(-128,128,dtype=np.int16).astype(np.int8), (19,23))
    for broadcast in (False, True):
        b = np.resize(np.array([-128,127,0,-1,1],np.int8), (23,) if broadcast else a.shape)
        for w in (1,2,-1,-3):
            k = ops.add_tensor('affine',a.shape,b.shape,alpha=w,s_a=Q(1,32),s_b=Q(1,16),s_out=Q(3,64))
            result.append(Case(f'tensor_{int(broadcast)}_{"neg" if w<0 else "pos"}{abs(w)}','affine',k,
                dict(a=a,b=b),dict(y=quant((a.astype(object)+2*w*b.astype(object))*Q(2,3)))))
    for i, c in enumerate((Q(5,32),Q(-7,32),Q(1,10),Q(1,2))):
        k = ops.add_scalar('affine',a.shape,c,s_x=Q(1,32),s_out=Q(3,64))
        result.append(Case(f'scalar_constant_{i}','affine',k,dict(x=a),dict(y=quant(a.astype(object)*Q(2,3)+c/Q(3,64)))))
    for i, f in enumerate((-1,-3,4,Q(3,2),Q(1,3))):
        k = ops.scale_map('affine',a.shape,f,s_x=Q(1,32),s_out=Q(1,16))
        result.append(Case(f'scalar_numerator_{i}','affine',k,dict(x=a),dict(y=quant(a.astype(object)*f/2))))
    k = ops.relu(a.shape,s_x=Q(1,32),s_out=Q(3,64))
    result.append(Case('device_relu_all_codes','relu',k,dict(x=a),dict(y=quant(np.maximum(a.astype(object),0)*Q(2,3)))))
    k = ops.add_scalar('affine',a.shape,3,s_x=1,s_out=2,relu=True)
    result.append(Case('wide_relu_host_clamp','relu',k,dict(x=a),dict(y=quant(np.maximum(a.astype(object)+3,0)/Q(2)))))
    # Cartesian byte inputs with unrelated canonical fp32 scales exercise the
    # retained multi-term rational sum and negative/positive near-half cases.
    x = np.broadcast_to(np.arange(-128,128,dtype=np.int16).astype(np.int8)[:,None],(256,256)).copy()
    y = x.T.copy()
    sa, sb = (Q(float(np.float32(v))) for v in (.03933464363217354,.049954403191804886))
    k = ops.add_tensor('affine',x.shape,y.shape,s_a=sa,s_b=sb,s_out=Q(1,16))
    result.append(Case('rational_ratio_cartesian','affine',k,dict(a=x,b=y),dict(y=quant((x.astype(object)*sa+y.astype(object)*sb)*16))))
    s = Stream([Arg('x','i32',(3,),'input'),Arg('out','i32',(3,),'output')])
    s.config_ld(2,12);s.mvin(s.at('x'),ACC(0),3,1,2)
    s.config_st(12,1,1);s.mvout(s.at('out'),ACC(0,full=True),3,1)
    k = ops._end('raw_readout',s)
    v = np.array([-1000,0,1000],np.int32)
    result.append(Case('full_store_relu_bypass','raw_readout',k,dict(x=v),dict(out=v)))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path)
    args = ap.parse_args()
    report = run(cases(),'affine_exact')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['spike']))
    raise SystemExit(report['spike']['status'] != 'PASS')
