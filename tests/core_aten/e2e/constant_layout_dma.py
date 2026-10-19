"""Spike checks consume compile-time folded literals through a runtime byte cat."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from xdsl.dialects.builtin import ModuleOp
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'device'))
import runner
from test_constant_dma import constant_source
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.frontend.parse import parse_module
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.core_aten.stream import simulate,ACC_BIT
from mlir_oot.golden_device_catalog import final_elf_audit


def run(out,checker):
    out.mkdir(parents=True,exist_ok=True)
    combined=ModuleOp([])
    buffers,calls,checks,outputs={},[],[],set()
    for dtype in ('i8','f32'):
        source,expected_constant=constant_source(dtype,consumer=True)
        device,manifest,matches=build(source)
        assert len(matches)==1 and len(manifest['constant_folds'])==1
        rewritten=rewrite(source,manifest)
        folded=parse_module(rewritten)
        constant=next(op for op in folded.walk() if op.name=='arith.constant')
        raw=constant.value.data.data
        assert raw==expected_constant
        (out/(dtype+'.source.mlir')).write_text(source)
        (out/(dtype+'.offloaded.mlir')).write_text(rewritten)
        (out/(dtype+'.catalog.json')).write_text(json.dumps(manifest,indent=2))
        match=matches[0]
        width=4 if dtype=='f32' else 1
        shape=(2,5,7,3)
        storage=np.dtype('<f4') if width==4 else np.dtype('i1')
        literal=np.frombuffer(raw,storage).reshape(shape)
        runtime_raw=np.random.default_rng(976).integers(0,256,(*shape,width),dtype=np.uint8)
        runtime=runtime_raw.reshape(-1).view(storage).reshape(shape)
        expected=np.concatenate([np.frombuffer(expected_constant,np.uint8),runtime_raw.reshape(-1)])
        argv=[];feed={}
        for arg in match.kernel.args:
            name=dtype+'_'+arg.name
            if arg.role=='input':
                value=literal if arg.name=='x0' else runtime
                feed[arg.name]=value
                buffers[name]=np.ascontiguousarray(value).view(np.int8).reshape(-1)
            else:
                buffers[name]=np.zeros(arg.nbytes,np.int8)
            if arg.role=='output':
                outputs.add(name);checks.append(runner.Check(name,expected.view(np.int8),0,dtype))
            argv.append(name)
        actual=np.ascontiguousarray(simulate(match.kernel.stream,feed)['y']).view(np.uint8).reshape(-1)
        assert np.array_equal(actual,expected)
        assert not any(i.kind=='mvout' and i.a('local')&ACC_BIT for i in match.kernel.stream.ins)
        calls.append((manifest['bindings'][0]['symbol'],argv))
        for op in list(device.body.block.ops):
            op.detach();combined.body.block.add_op(op)
    compile_to_object(combined,out)
    (out/'ca_harness.h').write_text((HERE/'device/ca_harness.h').read_text().replace('"rdcycle %0"','"csrr %0, mcycle"'))
    test=runner.DeviceTest('constant_layout',[out/'kernel.o'],buffers,calls,checks,outputs)
    elf=runner.build_elf(test,out)
    final_elf_audit(elf)
    result=runner.run_spike(elf)
    (out/'spike.log').write_text(result.log)
    assert result.ok,result.log[-3000:]
    subprocess.run(['python3','-I',str(checker),str(elf),'268435456',str(out/'design_rule.json')],check=True)
    audit=json.loads((out/'design_rule.json').read_text())
    assert audit['spike_rc']==0 and audit['design_rule']=='OK_full_i32'
    assert audit['mvout']['acc_full']==audit['mvout']['acc_narrow']==0
    (out/'results.json').write_text(json.dumps(dict(checks=2,bit_exact=True,spike_cycles=result.perf,audit=audit),indent=2)+'\n')
    print('constant i8/f32: 2 bit-exact checks; no accumulator MVOUT',flush=True)


def run_internal(out,checker):
    from mlir_oot.core_aten import ops
    from mlir_oot.core_aten.recipes.conv import ConvGeom
    from mlir_oot.core_aten.emit import build_module
    out.mkdir(parents=True,exist_ok=True)
    combined=ModuleOp([]);buffers,calls,checks,outputs={},[],[],set()
    rng=np.random.default_rng(591)
    g=ConvGeom(1,4,5,2,3,2,2)
    x=rng.integers(-3,4,(1,2,4,5),dtype=np.int8)
    w=rng.integers(-3,4,(3,2,2,2),dtype=np.int8)
    grad=rng.integers(-3,4,(1,3,g.oh,g.ow),dtype=np.int8)
    proofs=[]
    for kind in ('forward','input','weight'):
        if kind=='forward':
            factory=lambda **kw:ops.convolution_nchw(g,s_x=1,s_w=1,s_out=1,**kw)
            feed={'x':x,'w':w};constants={'w':w}
        elif kind=='input':
            factory=lambda **kw:ops.conv_backward_input(g,s_g=1,s_w=1,s_out=1,**kw)
            feed={'g':grad,'w':w};constants={'w':w}
        else:
            factory=lambda **kw:ops.conv_backward_weight(g,s_g=1,s_x=1,s_out=1,**kw)
            feed={'g':grad,'x':x};constants={'x':x}
        ordinary=factory();kernel=factory(constants=constants)
        expected=simulate(ordinary.stream,feed)['y']
        live={name:data for name,data in feed.items() if name not in constants}
        assert np.array_equal(simulate(kernel.stream,live)['y'],expected)
        proof,=kernel.meta['constant_layout_proofs'];proofs.append(proof)
        target=kernel.stream._index[proof['destination']]
        assert not any(i.kind=='mvout' and i.arg==target for i in kernel.stream.ins)
        argv=[]
        for arg in kernel.args:
            name=kind+'_'+arg.name
            dt=np.int32 if arg.dtype=='i32' else np.int8
            buffers[name]=live[arg.name] if arg.role=='input' else np.zeros(arg.shape,dt)
            if arg.role=='output':
                outputs.add(name);checks.append(runner.Check(name,expected,0,kind))
            argv.append(name)
        symbol='gemmini_ca_constant_conv_'+kind
        device,_=build_module(kernel.stream,symbol)
        for op in list(device.body.block.ops):
            op.detach();combined.body.block.add_op(op)
        calls.append((symbol,argv))
    compile_to_object(combined,out)
    (out/'ca_harness.h').write_text((HERE/'device/ca_harness.h').read_text().replace('"rdcycle %0"','"csrr %0, mcycle"'))
    elf=runner.build_elf(runner.DeviceTest('constant_conv',[out/'kernel.o'],buffers,calls,checks,outputs),out)
    final_elf_audit(elf)
    result=runner.run_spike(elf)
    (out/'spike.log').write_text(result.log)
    assert result.ok,result.log[-3000:]
    subprocess.run(['python3','-I',str(checker),str(elf),'268435456',str(out/'design_rule.json')],check=True)
    audit=json.loads((out/'design_rule.json').read_text())
    assert audit['spike_rc']==0 and audit['design_rule']=='OK_full_i32' and audit['mvout']['acc_narrow']==0
    (out/'results.json').write_text(json.dumps(dict(checks=3,bit_exact=True,spike_cycles=result.perf,
                                                constant_layout_proofs=proofs,audit=audit),indent=2)+'\n')
    print('constant convolution layouts: 3 bit-exact checks; zero layout instructions',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path);ap.add_argument('--design-rule',type=Path,required=True)
    args=ap.parse_args();run(args.out.resolve(),args.design_rule)
    run_internal(args.out.resolve()/"internal-conv",args.design_rule)
