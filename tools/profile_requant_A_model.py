"""ResNet paper-model build with experiment-only host-epilogue minstret windows.

Uses the frozen paper-model script/capture; writes only --out. No production
instrumentation switch or target/model special case is introduced. Each kernel
counter accumulates CPU instructions observed between machine-mode CSR reads;
the final logged value for each symbol is summed (not its repeated prefixes).
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stages',default='build,run')
    ap.add_argument('--driver',type=Path,default=Path('/scratch2/loren/gemmini-coverage/workloads/paper_models/build_model.py'))
    a=ap.parse_args()
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import IntegerAttr,i64
    from merlin.targetgen.core_aten_device import load_execution_provider
    package=Path(os.environ['GMLIR'])
    provider=load_execution_provider('gemmini',Path(os.environ['MERLIN_CORE_ATEN_EXECUTION_PROVIDER']))
    host=provider._core_aten_module(package,'host_readout')
    original=host.emit
    def counted(emitter, readouts=None):
        fb=emitter.fb
        name=emitter.symbol+'__rq_host_instructions'
        if not hasattr(emitter,'_rq_counter'):
            emitter.globals.append(llvm.GlobalOp(i64,name,'external',constant=False,value=IntegerAttr(0,i64),alignment=8))
            emitter._rq_counter=True
        def stamp():
            return fb.add(llvm.InlineAsmOp('csrr $0, minstret','=r,~{memory}',[],[i64],has_side_effects=True)).results[0]
        start=stamp()
        original(emitter,readouts)
        stop=stamp()
        delta=fb.sub_i(fb.sub_i(stop,start),fb.const(1))
        ptr=fb.add(llvm.AddressOfOp(name,emitter.ptr_type)).results[0]
        previous=fb.add(llvm.LoadOp(ptr,i64)).results[0]
        fb.add(llvm.StoreOp(fb.add_i(previous,delta),ptr))
    host.emit=counted
    spec=importlib.util.spec_from_file_location('paper_model_build',a.driver)
    paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)
    patch=paper.patch_shim
    def instrument_shim(shim,manifest):
        kernels=patch(shim,manifest)
        text=shim.read_text()
        declarations=['extern void htif_puts(const char *);','extern void htif_putd(long);']
        for k in manifest['kernels']:
            if not k.get('meta',{}).get('host_requantization_proofs'):continue
            sym=k['symbol'];counter=sym+'__rq_host_instructions'
            declarations.append(f'extern uint64_t {counter};')
            start=text.index(f'  {sym}(');end=text.index(';\n',start)+2
            line=f'  htif_puts("RQI {sym} "); htif_putd((long){counter}); htif_puts("\\n");\n'
            text=text[:end]+line+text[end:]
        shim.write_text('#include <stdint.h>\n'+'\n'.join(declarations)+'\n'+text)
        return kernels
    paper.patch_shim=instrument_shim
    a.out.mkdir(parents=True,exist_ok=True)
    for stage in a.stages.split(','):
        if stage=='build':paper.stage_build('resnet50',a.out,2048,1024,65536,0)
        elif stage=='run':
            # paper stage_run also grades the frozen immutable golden bytes.
            paper.stage_run('resnet50',a.out,7200)
            log=(a.out/'spike-console.txt').read_text()
            counts={};calls={}
            for symbol,value in re.findall(r'^RQI (\S+) (\d+)$',log,re.M):
                counts[symbol]=int(value);calls[symbol]=calls.get(symbol,0)+1
            assert counts,'no host epilogue instruction windows executed'
            result=dict(domain='Spike minstret host-epilogue windows',instructions=sum(counts.values()),
                        per_kernel=counts,calls=calls,window_correction='subtract one start-CSR instruction per window',
                        qualification='instrumented build; excludes whole-model performance claims; compiler may move pure constant setup outside windows')
            (a.out/'host-epilogue-instructions.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
        else:raise ValueError(stage)

if __name__=='__main__':main()
