"""Build independent tie and seeded-MAC captures for run_e2e.py/run_offload.py.

Test-only constants describe source IR, never production dispatch decisions.
"""
import argparse
from fractions import Fraction as F
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from exact_oracle import quantize
from test_integer_regions import source as seeded_source


def generate(root):
    fixtures = HERE.parents[1] / 'fixtures/core_aten'
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(613)
    for m,k,n in [(1,1,1),(3,17,5),(17,33,19)]:
        case = root / f'seeded_{m}_{k}_{n}'
        case.mkdir(exist_ok=True)
        a = rng.integers(-128,128,(m,k),dtype=np.int8)
        b = rng.integers(-128,128,(k,n),dtype=np.int8)
        seed = np.array([2*j-7 for j in range(n)], dtype=object)
        (case/'model.mlir').write_text(seeded_source(m,k,n))
        np.savez(case/'inputs.npz',arg0=a,arg1=b)
        np.save(case/'expected.npy',quantize((a.astype(object) @ b.astype(object)+seed)*F(1,2)))
    case=root/'seeded_runtime'
    case.mkdir(exist_ok=True)
    m,k,n=3,17,5
    a=rng.integers(-128,128,(m,k),dtype=np.int8)
    b=rng.integers(-128,128,(k,n),dtype=np.int8)
    bound=k*16384
    seed=np.array([-(1<<31)+bound,(1<<31)-1-bound,-7,0,17],dtype=np.int32)
    (case/'model.mlir').write_text(seeded_source(m,k,n,runtime=True))
    np.savez(case/'inputs.npz',arg0=a,arg1=b,arg2=seed)
    np.save(case/'expected.npy',quantize((a.astype(object)@b.astype(object)+seed.astype(object))*F(1,2)))
    # The review's eager-float divergence: exact f32 scales have ratio 3.
    case = root / 'mean_tie'
    case.mkdir(exist_ok=True)
    text = (fixtures/'mean_dim/model.mlir').read_text()
    text = text.replace('0.0252246745 : f32','0.03 : f32').replace('0.00494601484 : f32','0.01 : f32')
    (case/'model.mlir').write_text(text)
    x = np.zeros((37,50),dtype=np.int8)
    x[:,:2] = [-62,87]
    np.savez(case/'inputs.npz',arg0=x)
    # Informational eager result; run_e2e independently derives rational outputs.
    np.save(case/'expected.npy',np.ones(37,dtype=np.int8))
    case = root / 'affine_all_pairs'
    case.mkdir(exist_ok=True)
    text = (fixtures/'add_Tensor/model.mlir').read_text()
    # Only the three scale constants change; source operation semantics stay add.
    pattern = r'(arith.constant[^\n]*\}) ([0-9.eE+-]+) : f32'
    scales = iter([1.0,1.0,2.0])
    text = re.sub(pattern,lambda m:m[1]+' '+str(next(scales))+' : f32',text)
    text = text.replace('37x21','256x256')
    (case/'model.mlir').write_text(text)
    a,b = np.meshgrid(np.arange(-128,128,dtype=np.int8),np.arange(-128,128,dtype=np.int8),indexing='ij')
    np.savez(case/'inputs.npz',arg0=a,arg1=b)
    np.save(case/'expected.npy',quantize((a.astype(object)+b.astype(object))/F(2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    generate(parser.parse_args().output)
