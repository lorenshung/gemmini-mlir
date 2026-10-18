"""Lower the logical command stream to raw Gemmini instructions and LLVM IR."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from math import prod
from collections import Counter
from device_runtime import RUNTIME
from elementwise import ELEMENTWISE_RUNTIME
from device_fast import FAST_RUNTIME
from direct_codegen import emit_direct_matmul, emit_resident_pack
from pool_codegen import validate_pool
from epilogue_validation import validate_acc_scale
from bias_validation import validate_bias_epilogue

PUBLISHED = {'COMMIT','BIAS_ADD','RESIDUAL_ADD','ATTENTION_QK','ATTENTION_PV','CONV2D','BATCHED_MATMUL','MOVEMENT'}

def width(dtype):
    return {'i8':1,'i16':2,'i32':4,'i64':8,'f32':4}[dtype]

def make_c(cb):
    tensors=cb.get('tensors',{})
    commands=cb['commands']
    produced={c['operands']['dst'] for c in commands if 'dst' in c.get('operands',{})}
    derived=set(cb.get('params',{}).get('derived_tensors',{}))
    if cb.get('kernel_abi',{}).get('kind')=='whole_program':
        args=[a['tensor'] for a in cb['kernel_abi']['args']]
    else:
        args=[name for name,t in tensors.items() if name not in produced|derived and t.get('role') not in {'output','intermediate'}]
        for c in commands:
            if c['opcode'] in PUBLISHED and c['operands']['dst'] not in args:
                args.append(c['operands']['dst'])
        args += [n for n,t in tensors.items() if t.get('role')=='output' and n not in args]
    ptrs={name:f'arg{i}' for i,name in enumerate(args)}
    declarations=[]
    statements=['gm_fence();','gm_cmd(7,0,0);']
    for i,(name,t) in enumerate(tensors.items()):
        if name not in ptrs:
            size=prod(t['shape'])*width(t['dtype'])
            ident=f'buffer_{i}'
            declarations.append(f'static unsigned char {ident}[{size}] __attribute__((aligned(64)));')
            ptrs[name]=ident
    residents={}
    cached_rhs={}
    next_resident_row=32
    resident_uses=Counter(c.get('operands',{}).get('rhs') for c in commands
                          if c['opcode'] in ('MATMUL','MATMUL_RESIDENT'))
    accumulators={}
    def resolve(name):
        while name in residents: name=residents[name]
        return name
    def matmul(a,b,dst,attrs,uid,trans=False,resident_base=None):
        validate_acc_scale(attrs)
        a,b=resolve(a),resolve(b)
        sa,sb=tensors[a]['shape'],tensors[b]['shape']
        M,K=prod(sa[:-1]),sa[-1]
        N=sb[-2] if trans else sb[-1]
        bp=ptrs[b]
        if trans:
            ident=f'transpose_{uid}'
            declarations.append(f'static int8_t {ident}[{K*N}] __attribute__((aligned(64)));')
            statements.append(f'for(int i=0;i<{K};i++) for(int j=0;j<{N};j++) {ident}[i*{N}+j]=((int8_t*){bp})[j*{K}+i];')
            bp=ident
        epi=attrs.get('epilogue',[])
        dtype=attrs.get('output_dtype',tensors[dst]['dtype'])
        scale=attrs.get('acc_scale',1.0)
        if 'requant' in epi:
            raise ValueError('integer requant is unsupported by the selected readout')
        if width(dtype) not in (1,4): raise ValueError('unsupported matmul output dtype')
        bias=attrs.get('bias')
        validate_bias_epilogue(attrs,{},tensors,N)
        biasp=f'(int32_t*){ptrs[bias]}' if bias else '0'
        outp=ptrs[dst]
        if 'maxpool' in epi:
            if tensors[dst]['dtype'] != dtype:
                raise ValueError('maxpool output_dtype disagrees with destination dtype')
            pool_geometry=validate_pool(attrs,M,N,tensors[dst]['shape'],dtype)
            ident=f'pool_source_{uid}'
            declarations.append(f'static int8_t {ident}[{M*N}] __attribute__((aligned(64)));')
            outp=ident
        if not trans:
            options=dict(out_bytes=width(dtype),scale=scale,relu='relu' in epi,
                         bias=None if not bias else biasp,tag=f'matmul_{uid}')
            if resident_base is not None:
                options['resident_base']=resident_base
            d,s=emit_direct_matmul(ptrs[a],bp,outp,M,K,N,**options)
            declarations.extend(d);statements.extend(s)
        else:
            statements.append(f'gm_matmul((int8_t*){ptrs[a]},(int8_t*){bp},(void*){outp},{M},{K},{N},{width(dtype)},{float(scale).hex()}f,{int("relu" in epi)},{biasp});')
        if 'maxpool' in epi:
            batch,h,w,ph,pw,sh,sw,pt,pl,pb,pr,pad=pool_geometry
            statements.append(f'gm_pool({outp},(int8_t*){ptrs[dst]},{batch},{h},{w},{N},{ph},{pw},{sh},{sw},{pt},{pl},{pb},{pr},{pad});')
    for uid,c in enumerate(commands):
        op=c['opcode']; o=c.get('operands',{}); at=c.get('attributes',{})
        if op=='RES_PACK':
            source=resolve(o['src']); handle=o['dst']
            residents[handle]=source
            shape=tensors[source]['shape']
            if resident_uses[handle]>1 and len(shape)==2 and tensors[source]['dtype']=='i8':
                K,N=shape
                rows=((K+15)//16)*((N+15)//16)*16
                # Rows 0..31 remain available to every ordinary tile schedule.
                # Static allocations are never recycled while deferred commits
                # may still refer to their packed weights.
                if min(K,N)>0 and next_resident_row+rows<=16384:
                    d,s=emit_resident_pack(ptrs[source],K,N,next_resident_row,
                                           tag=f'resident_{uid}')
                    declarations.extend(d);statements.extend(s)
                    cached_rhs[handle]=next_resident_row
                    next_resident_row+=rows
        elif op=='EVICT':
            cached_rhs.pop(o['handle'],None)
        elif op in ('MATMUL_RESIDENT','MATMUL'):
            accumulators[o['dst']]=(o['lhs'],resolve(o['rhs']),cached_rhs.get(o['rhs']))
        elif op=='COMMIT':
            if o['src'] not in accumulators: raise ValueError('commit of uncomputed accumulator')
            lhs,rhs,resident_base=accumulators[o['src']]
            validate_bias_epilogue(at,o,tensors,tensors[resolve(rhs)]['shape'][-1])
            attrs=dict(at)
            if 'bias' in o: attrs['bias']=o['bias']
            matmul(lhs,rhs,o['dst'],attrs,uid,resident_base=resident_base)
        elif op in ('ATTENTION_QK','ATTENTION_PV'):
            a,b=(o['q'],o['k']) if op=='ATTENTION_QK' else (o['p'],o['v'])
            matmul(a,b,o['dst'],at,uid,op=='ATTENTION_QK')
        elif op=='CONV2D':
            from conv_codegen import emit_conv
            cmd=copy.deepcopy(c); cmd['operands']['weight']=resolve(o['weight'])
            d,s=emit_conv(cb,cmd,ptrs,uid); declarations.extend(d); statements.extend(s)
        elif op=='BATCHED_MATMUL':
            a,b,d=o['a'],o['w'],o['dst']
            sa,sb=tensors[a]['shape'],tensors[b]['shape']; M,K=sa[-2:]; N=sb[-1]; batch=prod(sa[:-2])
            if sa[:-2]!=sb[:-2] or sb[-2]!=K: raise ValueError('invalid independent batched contraction')
            for bi in range(batch):
                dec,stm=emit_direct_matmul(f'(int8_t*){ptrs[a]}+{bi*M*K}',f'(int8_t*){ptrs[b]}+{bi*K*N}',f'(int32_t*){ptrs[d]}+{bi*M*N}',M,K,N,tag=f'batch_{uid}_{bi}')
                declarations.extend(dec);statements.extend(stm)
        elif op=='MOVEMENT':
            src,dst=o['src'],o['dst']; shape=tensors[src]['shape']
            statements.append(f'gm_move((int8_t*){ptrs[src]},{ptrs[dst]},{prod(shape[:-1])},{shape[-1]},{width(tensors[dst]["dtype"])});')
        elif op=='BIAS_ADD':
            src,dst=o['src'],o['dst']; shape=tensors[src]['shape']
            validate_bias_epilogue({'epilogue':['bias_add']},o,tensors,shape[-1])
            statements.append(f'gm_bias((int32_t*){ptrs[src]},(int32_t*){ptrs[o["bias"]]},(int32_t*){ptrs[dst]},{prod(shape[:-1])},{shape[-1]});')
        elif op=='RESIDUAL_ADD':
            src,dst=o['lhs'],o['dst']; shape=tensors[src]['shape']
            if tensors[dst]['dtype']!='i8': raise ValueError('residual output must be i8')
            ls=float(at['lhs_scale']); rs=float(at['rhs_scale'])
            statements.append(f'gm_residual((int8_t*){ptrs[src]},(int8_t*){ptrs[o["rhs"]]},(int8_t*){ptrs[dst]},{prod(shape[:-1])},{shape[-1]},{ls.hex()}f,{rs.hex()}f,{int("relu" in at.get("epilogue",[]))});')
        else: raise ValueError(f'unsupported device opcode {op}')
    statements.append('gm_fence();')
    try:
        from pool_runtime import POOL_RUNTIME
    except ImportError:
        POOL_RUNTIME=''
    return '\n'.join([RUNTIME,FAST_RUNTIME,ELEMENTWISE_RUNTIME,POOL_RUNTIME,*declarations,
        'void gemmini_kernel('+','.join(f'void *arg{i}' for i in range(len(args)))+') {',*statements,'}'])

def tool(name):
    roots=[os.environ.get('MERLIN_MLIR_INSTALL',''),'/work/merlin/toolchains/llvm-a47bddcc/install',str(Path(__file__).resolve().parent.parent/'install')]
    for root in roots:
        if root and (Path(root)/'bin'/name).is_file(): return str(Path(root)/'bin'/name)
    found=shutil.which(name)
    if found:return found
    raise ValueError('required LLVM tool is missing: '+name)

def emit(cb):
    return compile_source(make_c(cb))

def compile_source(source, task_functions=None, plan=None):
    from kernel_abi import adapt_source, copied_entry_commands, prevent_entry_copy
    source=adapt_source(source,whole_program=bool(task_functions))
    with tempfile.TemporaryDirectory(prefix='gemmini-emit-') as td:
        c=Path(td)/'kernel.c'; ll=Path(td)/'kernel.ll'
        for attempt in range(2):
            c.write_text(source)
            run=subprocess.run([tool('clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-S','-emit-llvm','-O2','-ffreestanding','-fno-builtin','-fno-vectorize','-fno-slp-vectorize','-fno-stack-protector','-ffp-contract=off',str(c),'-o',str(ll)],capture_output=True,text=True)
            if run.returncode: raise ValueError(run.stderr)
            if task_functions or not copied_entry_commands(ll.read_text()):break
            if attempt:raise ValueError('native wrapper still duplicates accelerator instructions')
            source=prevent_entry_copy(source)
        run=subprocess.run([tool('mlir-translate'),'--import-llvm',str(ll)],capture_output=True,text=True)
        if run.returncode: raise ValueError(run.stderr)
        if task_functions:
            from provenance import tag_llvm
            return tag_llvm(run.stdout,task_functions,tool,plan)
        from provenance import generic_llvm
        return generic_llvm(run.stdout,tool)
