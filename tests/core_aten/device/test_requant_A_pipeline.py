"""Spike qualification of overlapped per-channel, biased multi-term readout."""
from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from common import need_spike
import runner


@need_spike
def test_per_channel_multiterm_frozen_bias_overlaps_exactly():
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.stream import Arg,simulate
    from mlir_oot.core_aten.emit import build_module,compile_to_object
    from mlir_oot.core_aten.lint import lint
    rng=np.random.default_rng(12345)
    M,K,N=133,19,130
    kernel=ops.matmul('aten.mm.default',M,K,N,s_a=F(1,7),s_out=F(1))
    stream=kernel.stream
    stream._index['bias']=len(stream.args)
    stream.args.append(Arg('bias','i32',(N,),'input'))
    r=stream.readouts[0]
    factors=np.resize(np.array([F(1),F(1,2),F(-3,8),F(17,31),F(1,3)],object),N)
    r.terms=[(r.terms[0][0],factors),('bias',F(1,3))]
    r.constant=np.array([F(j-60,7) for j in range(N)],object)
    r.zero_point=np.arange(N)%256
    r.quant_min,r.quant_max=0,255
    r.broadcasts={'bias':(1,N)}
    feed=dict(a=rng.integers(-4,5,(M,K),dtype=np.int8),b=rng.integers(-4,5,(K,N),dtype=np.int8),
              bias=rng.integers(-3,4,N,dtype=np.int32))
    acc=feed['a'].astype(np.int64)@feed['b'].astype(np.int64)
    expected=np.array([[min(255,max(0,round(int(acc[i,j])*factors[j]+F(int(feed['bias'][j]),3)+r.constant[j])+int(r.zero_point[j])))
                        for j in range(N)] for i in range(M)],np.uint8)
    wd=runner.work_root()/'requant_A_pipeline'
    module,_=build_module(stream,'rq_pipeline',meta=kernel.meta)
    assert kernel.meta['readout_pipeline']['status']=='enabled'
    assert len(kernel.meta['readout_pipeline']['tiles'])>2
    lint(stream)
    assert np.array_equal(simulate(stream,feed)['out'].view(np.uint8),expected)
    receipt=compile_to_object(module,wd)
    buffers={a.name:feed[a.name] if a.role=='input' else np.zeros(a.shape,np.int32 if a.dtype=='i32' else np.int8)
             for a in kernel.args}
    test=runner.DeviceTest('requant_A_pipeline',[Path(receipt['object'])],buffers,
        [('rq_pipeline',[a.name for a in kernel.args])],[runner.Check('out',expected.view(np.int8),0,'pipeline_exact')],{'out'})
    elf=runner.build_elf(test,wd)
    result=runner.run_spike(elf)
    (wd/'spike.log').write_text(result.log)
    import json
    (wd/'proofs.json').write_text(json.dumps(kernel.meta,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),indent=2))
    assert result.ok,result.log[-3000:]
