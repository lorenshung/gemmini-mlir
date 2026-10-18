"""Source-bound scalar/rank-five, wide-plan and legacy-layout checks on Spike."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from xdsl.dialects.builtin import ModuleOp

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'device'))
import runner
from test_bitwise_movement import reshape_source
from test_limbs import annotate, ranked_linear_source
from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.core_aten.storage import itemsize
from mlir_oot.core_aten.stream import simulate


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    combined, symbols = ModuleOp([]),set()
    calls, checks, outputs, buffers, records = [],[],set(),{},[]
    def append(label,module,symbol,args,feed,expected,bitwise=False):
        argv=[]
        for arg in args:
            name=label+'_'+arg.name
            argv.append(name)
            data=feed.get(arg.name)
            if data is None:
                data=np.zeros(arg.nbytes if bitwise else arg.shape,np.int8 if bitwise else np.int32)
            buffers[name]=data
            if arg.role=='output':
                outputs.add(name)
                checks.append(runner.Check(name,expected,0,label))
        calls.append((symbol,argv))
        for fn in list(module.body.block.ops):
            if fn.sym_name.data not in symbols:
                symbols.add(fn.sym_name.data);fn.detach();combined.body.block.add_op(fn)
        records.append(dict(label=label,symbol=symbol))
    for dtype in ['i1','i8','f16','bf16','i32','i64','f32','f64','complex<f32>']:
        for rank in [0,1,5]:
            module,manifest,matches=build(reshape_source(dtype,rank))
            match,=matches
            data=np.arange(itemsize(dtype),dtype=np.int8)+1
            if dtype=='i1':data[:]=1
            actual=simulate(match.kernel.stream,{'x0':data})['y']
            assert np.array_equal(np.ascontiguousarray(actual).view(np.int8).reshape(-1),data)
            append(f'bytes_{dtype.replace("<", "_").replace(">", "_")}_{rank}',module,
                   manifest['bindings'][0]['symbol'],match.kernel.args,{'x0':data},data,True)
    for shape in [(),(1,),(1,2,1,3,5)]:
        for count in [1,9,65]:
            for family in ['F2','F4']:
                source=annotate(ranked_linear_source(shape),indices=(count-1,),exponents=(0,),
                    family=family,component='real',mutate=lambda p:p['inputs'][0].update(count=count))
                module,manifest,matches=build(source);match,=matches
                x=np.resize(np.array([-128,127,-1,0,1],np.int8),shape)
                expected=x.astype(np.int32)*-3-3
                assert np.array_equal(simulate(match.kernel.stream,{'x0':x})['out'],expected)
                append(f'linear_{family}_{len(shape)}_{count}',module,manifest['bindings'][0]['symbol'],
                       match.kernel.args,{'x0':x},expected)
    # The source has actual host Q and layout producers before its raw GEMM.
    # The legacy ABI deliberately retains those source host operations.
    from mlir_oot.golden_device_catalog import build_catalog
    from mlir_oot.core_aten.stream import Arg
    from exact_oracle import SourceOracle
    source=(HERE/'fixtures/scalar_mean/host_quantized_gemm.mlir').read_text()
    parsed=parse_module(source);ops=list(parsed.walk())
    function=next(o for o in ops if o.name=='func.func')
    oracle=SourceOracle({arg:np.linspace(-1,1,np.prod(arg.type.get_shape())).astype(np.float32)
                         .reshape(arg.type.get_shape()).astype(object) for arg in function.body.block.args})
    module,manifest=build_catalog(source)
    binding,=manifest['bindings'];producer=ops[binding['source_operation_ordinal']]
    a,b=[np.asarray(oracle.tensor(v),np.int8) for v in producer.operands[:2]]
    expected=a.astype(np.int32)@b.astype(np.int32)
    append('legacy_layout',module,binding['symbol'],
           [Arg('a','i8',a.shape,'input'),Arg('b','i8',b.shape,'input'),Arg('out','i32',expected.shape,'output')],
           {'a':a,'b':b},expected)
    from mlir_oot.golden_device_lower import lower
    receipt=compile_to_object(lower(combined),out/'build')
    test=runner.DeviceTest('provider_shapes',[Path(receipt['object'])],buffers,calls,checks,outputs)
    elf=runner.build_elf(test,out/'build')
    from mlir_oot.golden_device_catalog import final_elf_audit
    final_elf_audit(elf)
    result=runner.run_spike(elf)
    (out/'spike.log').write_text(result.log)
    (out/'results.json').write_text(json.dumps(dict(elf=str(elf.resolve()),cases=records,
        spike=dict(status=result.status,passed=len(result.passed),failed=result.failed)),indent=2))
    print(result.status,len(result.passed),result.failed)
    return result.ok


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    raise SystemExit(0 if run(parser.parse_args().out) else 1)
