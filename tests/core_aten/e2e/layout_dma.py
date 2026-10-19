"""Small layout capsules: opaque bytes, exact code requantization, Spike and ELF audit."""
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
from test_layout_scalability import source,asymmetric_cat_source
from mlir_oot.core_aten.catalog import build
from mlir_oot.core_aten.emit import compile_to_object
from mlir_oot.core_aten.stream import simulate,ACC_BIT,FULL_BIT
from mlir_oot.golden_device_catalog import final_elf_audit


def run(out,checker):
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    rng=np.random.default_rng(8137)
    for group in ('raw','requant'):
        wd=out/group;wd.mkdir(exist_ok=True)
        combined=ModuleOp([]);symbols=set()
        buffers,calls,checks,outputs={ },[],[],set()
        modes=[('i8','requant')] if group=='requant' else [('i8','native'),('f32','native'),('i8','same')]
        for kind in ('transpose','permute','copy','cat'):
            selected=modes+([('i8','asymmetric1'),('i8','asymmetric3')] if group=='requant' and kind=='cat' else [])
            for dtype,scheme in selected:
                label=f'{kind}_{dtype}_{scheme}'
                shape=(1,18,3,5)
                sh=(1,9,3,5) if kind=='cat' else shape
                text=asymmetric_cat_source(int(scheme[-1])) if scheme.startswith('asymmetric') else source(kind,dtype,shape,scheme)
                device,catalog,matches=build(text)
                assert len(matches)==1,catalog['declines']
                match=matches[0];binding=catalog['bindings'][0]
                (wd/(label+'.mlir')).write_text(text)
                (wd/(label+'.json')).write_text(json.dumps(catalog,indent=2))
                width=4 if dtype=='f32' else 1
                arrays=[rng.integers(0,256,sh,dtype=np.uint8).view(np.int8) if width==1 else
                        rng.integers(0,256,(*sh,4),dtype=np.uint8).reshape(-1).view(np.float32).reshape(sh)
                        for _ in range(2 if kind=='cat' else 1)]
                expected=(arrays[0].transpose(0,2,3,1) if kind in ('transpose','permute') else
                          np.concatenate(arrays,axis=1) if kind=='cat' else arrays[0].copy())
                if scheme.startswith('asymmetric'):
                    converted=[np.clip(np.rint(arrays[0].astype(np.int64)/2),-128,127).astype(np.int8),
                               np.clip(np.rint((arrays[1].view(np.uint8).astype(np.int64)-128)*1.5),-128,127).astype(np.int8)]
                    expected=np.concatenate(converted,axis=int(scheme[-1]))
                if scheme=='requant':
                    # Integers divided by two are exactly represented; this
                    # includes both parities and negative half-to-even ties.
                    expected=np.rint(expected.astype(np.int64)/2).astype(np.int8)
                argv,feed=[],{}
                for arg in match.kernel.args:
                    name=label+'_'+arg.name
                    dt=np.int32 if arg.dtype=='i32' else np.int8
                    if arg.role=='input':
                        where=binding['operands'][arg.name]
                        assert 'func_arg' in where,where
                        arr=arrays[where['func_arg']]
                        feed[arg.name]=arr
                        buffers[name]=np.ascontiguousarray(arr).view(np.int8).reshape(-1)
                    else:
                        buffers[name]=np.zeros(arg.nbytes//np.dtype(dt).itemsize,dt)
                    argv.append(name)
                    if arg.role=='output':
                        outputs.add(name)
                        checks.append(runner.Check(name,np.ascontiguousarray(expected).view(np.int8).reshape(-1),0,label))
                actual=np.ascontiguousarray(simulate(match.kernel.stream,feed)['y']).view(np.int8).reshape(-1)
                assert np.array_equal(actual,np.ascontiguousarray(expected).view(np.int8).reshape(-1)),label
                stores=[i for i in match.kernel.stream.ins if i.kind=='mvout' and i.a('local')&ACC_BIT]
                assert all(i.a('local')&FULL_BIT for i in stores)
                assert bool(stores)==(group=='requant')
                calls.append((binding['symbol'],argv))
                for op in list(device.body.block.ops):
                    if op.sym_name.data not in symbols:
                        symbols.add(op.sym_name.data);op.detach();combined.body.block.add_op(op)
        receipt=compile_to_object(combined,wd)
        # Match the mandated checker's explicit ISA, which omits Zicntr.
        (wd/'ca_harness.h').write_text((HERE/'device/ca_harness.h').read_text().replace('"rdcycle %0"','"csrr %0, mcycle"'))
        test=runner.DeviceTest('layout_'+group,[wd/'kernel.o'],buffers,calls,checks,outputs)
        elf=runner.build_elf(test,wd)
        final_elf_audit(elf)
        result=runner.run_spike(elf)
        (wd/'spike.log').write_text(result.log)
        assert result.ok,result.log[-3000:]
        subprocess.run(['python3','-I',str(checker),str(elf),'268435456',str(wd/'design_rule.json')],check=True)
        audit=json.loads((wd/'design_rule.json').read_text())
        assert audit['spike_rc']==0 and audit['design_rule']=='OK_full_i32',audit
        assert audit['mvout']['acc_narrow']==0
        if group=='raw':
            assert audit['mvout']['acc_full']==0
        records.append(dict(group=group,checks=len(checks),bit_exact=True,spike_cycles=result.perf,
                            audit=audit,compilation=receipt))
        print(group,len(checks),'bit-exact',audit['mvout'],flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path);ap.add_argument('--design-rule',type=Path,required=True)
    args=ap.parse_args();run(args.out.resolve(),args.design_rule)
