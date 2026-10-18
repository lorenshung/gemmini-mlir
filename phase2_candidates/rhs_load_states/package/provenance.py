"""Attach source task identities to the actual generated LLVM operations."""
from io import StringIO
import subprocess

def tag_llvm(mlir, task_functions, tool, plan=None):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr
    from xdsl.parser import Parser
    from xdsl.printer import Printer
    converted=subprocess.run([tool('mlir-opt'),'--mlir-print-op-generic','--mlir-print-local-scope'],input=mlir,capture_output=True,text=True)
    if converted.returncode:raise ValueError(converted.stderr)
    ctx=Context(allow_unregistered=True);ctx.load_dialect(Builtin)
    module=Parser(ctx,converted.stdout).parse_module()
    reverse={v:int(k) for k,v in task_functions.items()}
    for op in list(module.body.block.ops):
        name=op.properties.get('sym_name',op.attributes.get('sym_name'))
        if not isinstance(name,StringAttr):continue
        if name.data=='llvm.compiler.used':
            module.body.block.erase_op(op)
            continue
        op.attributes.pop('no_inline',None)
        op.properties.pop('no_inline',None)
        task=reverse.get(name.data)
        if task is None:continue
        for nested in op.walk():
            nested.attributes['merlin.global_task']=IntegerAttr(task,64)
            nested.attributes['merlin.source_op_index']=IntegerAttr(task,64)
            if plan is not None:
                regions=plan['tasks'][task].get('source_region_ids',[])
                if regions:nested.attributes['merlin.source_region_id']=StringAttr(regions[0])
    out=StringIO();Printer(stream=out,print_generic_format=True).print_op(module)
    result=subprocess.run([tool('mlir-opt'),'--inline=inlining-threshold=100000000','--symbol-dce','--mlir-print-op-generic','--mlir-print-local-scope'],input=out.getvalue(),capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stderr)
    module=Parser(ctx,result.stdout).parse_module()
    counts={i:0 for i in reverse.values()}
    counts[-1]=counts[-2]=0
    for op in list(module.body.block.ops):
        name=op.properties.get('sym_name',op.attributes.get('sym_name'))
        if isinstance(name,StringAttr) and name.data in reverse:
            module.body.block.erase_op(op)
            continue
        if not isinstance(name,StringAttr) or name.data!='kernel_entry':continue
        blocks=list(op.regions[0].blocks)
        entry=blocks[0]
        def opname(node):return getattr(node.attributes.get('op_name__'),'data',node.name)
        # The stock inliner hoists ConstantLike values. Their address/constant
        # definitions belong to the wrapper, while their actual users retain
        # the task that performs the source computation.
        for node in entry.ops:
            if opname(node) in {'llvm.mlir.constant','llvm.mlir.addressof','llvm.alloca'}:
                node.attributes['merlin.global_task']=IntegerAttr(-1,64)
                node.attributes.pop('merlin.source_op_index',None)
        preds={b:[] for b in blocks}
        for b in blocks:
            for succ in b.last_op.successors:preds[succ].append(b)
        outgoing={b:-1 for b in blocks}
        for _ in range(len(blocks)+1):
            changed=False
            for b in blocks:
                current=max((outgoing[p] for p in preds[b]),default=-1)
                for node in b.ops:
                    attr=node.attributes.get('merlin.global_task')
                    if attr is not None and attr.value.data>=0:current=max(current,attr.value.data)
                if current!=outgoing[b]:outgoing[b]=current;changed=True
            if not changed:break
        for b in blocks:
            current=max((outgoing[p] for p in preds[b]),default=-1)
            for node in b.ops:
                attr=node.attributes.get('merlin.global_task')
                if attr is None:
                    task=-2 if opname(node)=='llvm.return' else current
                    node.attributes['merlin.global_task']=IntegerAttr(task,64)
                else:
                    task=attr.value.data
                    if task>=0:current=task
                counts[task]=counts.get(task,0)+1
    if plan is not None:
        cursor=counts[-1]
        plan['prologue_instruction_range']=[0,cursor]
        for row in plan['tasks']:
            n=counts[row['task_index']]
            if n<=0:raise ValueError('source task vanished during LLVM lowering')
            row['instruction_start']=cursor
            row['instruction_end']=cursor+n
            cursor+=n
        plan['epilogue_instruction_range']=[cursor,cursor+counts[-2]]
        plan['schedule_instruction_count']=cursor+counts[-2]
    for node in module.walk():
        if getattr(node.attributes.get('op_name__'),'data','') in {'llvm.fpext','llvm.fptrunc'}:
            flag=node.properties.get('fastmathFlags')
            if flag is not None and str(flag)=='#llvm.fastmath<none>':
                node.properties.pop('fastmathFlags')
    out=StringIO();Printer(stream=out,print_generic_format=True).print_op(module)
    return out.getvalue()

def generic_llvm(mlir, tool):
    """Print standard LLVM generically for the supplied xDSL reader.

    Per-call optimization annotations introduced after this reader's release
    are omitted. Removing these promises preserves computation and effects.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser
    from xdsl.printer import Printer
    converted=subprocess.run([tool('mlir-opt'),'--mlir-print-op-generic','--mlir-print-local-scope'],input=mlir,capture_output=True,text=True)
    if converted.returncode:raise ValueError(converted.stderr)
    context=Context(allow_unregistered=True);context.load_dialect(Builtin)
    module=Parser(context,converted.stdout).parse_module()
    for op in module.walk():
        if getattr(op.attributes.get('op_name__'),'data','')=='llvm.call':
            for key in ('arg_attrs','res_attrs','nobuiltin','nobuiltins','memory_effects'):
                op.properties.pop(key,None)
    output=StringIO();Printer(stream=output,print_generic_format=True).print_op(module)
    return output.getvalue()
