"""Source-bound mixed Gemmini and scalar-host program compiler.

Each direct source operation owns one noinline C function and one planned task.
Every materialized source value has its own caller-provided buffer. Integer
contractions are dispatched through the selected Gemmini runtime; scalar region
bodies are reserved for operations outside that contraction path.
"""
from __future__ import annotations
import hashlib
import math
from math import prod

from linalg_frontend import read_module, _name, _attrs, _shape, _dtype, _constant, _canonical_multiply_accumulate

_TYPES = {'i1': 'uint8_t', 'i8': 'int8_t', 'i16': 'int16_t', 'i32': 'int32_t', 'i64': 'int64_t',
          'ui8': 'uint8_t', 'ui16': 'uint16_t', 'ui32': 'uint32_t', 'ui64': 'uint64_t',
          'f32': 'float', 'f64': 'double', 'index': 'int64_t', 'f16': '_Float16'}

def ctype(t):
    name = _dtype(t)
    if name not in _TYPES:
        raise ValueError(f'unsupported mixed scalar type {name}')
    return _TYPES[name]


def _at(pointer, value_type, index='0'):
    return f'(({ctype(value_type)}*)({pointer}))[{index}]'


def _literal(value, result_type):
    dtype = _dtype(result_type)
    if dtype.startswith('f'):
        val = float(value)
        if math.isnan(val):
            return '__builtin_nanf("")'
        if math.isinf(val):
            return ('-' if val < 0 else '') + '__builtin_inff()'
        return val.hex() + ('f' if dtype != 'f64' else '')
    return str(int(value)) + ('LL' if dtype in ('i64', 'index') else '')


MIXED_RUNTIME = r'''
static inline int64_t mx_roundeven(float x) {
  /* Conversion truncates; correct the fractional part and tie parity once. */
  int64_t q = (int64_t)x;
  float fraction = x - (float)q;
  if (fraction > 0.5f || (fraction == 0.5f && (q & 1))) ++q;
  if (fraction < -0.5f || (fraction == -0.5f && (q & 1))) --q;
  return q;
}
'''


def _emit_constant(op, result):
    attrs = _attrs(op)
    attr = attrs.get('value')
    value = getattr(getattr(attr, 'value', None), 'data', None)
    if value is not None:
        return f'{_at(result, op.results[0].type)} = {_literal(value, op.results[0].type)};'
    if hasattr(attr, 'get_values'):
        values = list(attr.get_values())
        length = prod(_shape(op.results[0].type))
        if len(values) == 1:
            return f'for (long i=0;i<{length};++i) {_at(result, op.results[0].type,"i")} = {_literal(values[0],op.results[0].type)};'
        literals = ','.join(_literal(v, op.results[0].type) for v in values)
        return f'{{ static const {ctype(op.results[0].type)} values[{length}] = {{{literals}}}; for(long i=0;i<{length};++i) {_at(result,op.results[0].type,"i")} = values[i]; }}'
    raise ValueError('arith.constant: unsupported literal attribute')


def _emit_quant(op, operands, result):
    kind = _name(op)
    if len(operands) != 3:
        raise ValueError(f'{kind}: expected value, scale and zero point')
    inp, scale, zp = [_at(p, v.type) for p, v in zip(operands, op.operands)]
    inp_i = _at(operands[0], op.operands[0].type, 'i')
    length = prod(_shape(op.results[0].type))
    dst = _at(result, op.results[0].type, 'i')
    if kind == 'quant_ext.dequantize_per_tensor':
        return f'for(long i=0;i<{length};++i) {{ float v=(float)((int64_t){inp_i}-(int64_t){zp}); {dst}=v*(float){scale}; }}'
    attrs = _attrs(op)
    low = getattr(getattr(attrs.get('quant_min'), 'value', None), 'data', None)
    high = getattr(getattr(attrs.get('quant_max'), 'value', None), 'data', None)
    if low is None or high is None:
        raise ValueError('quantize_per_tensor: missing quantization limits')
    return f'''for(long i=0;i<{length};++i) {{
      float v=(float){inp_i}/(float){scale};
      int64_t z=(int64_t){zp};
      if (v > (float)({high}-z)) v=(float)({high}-z);
      if (v < (float)({low}-z)) v=(float)({low}-z);
      int64_t q=mx_roundeven(v)+z;
      if(q<{low}) q={low}; if(q>{high}) q={high};
      {dst}=({ctype(op.results[0].type)})q;
    }}'''


def _contraction(op):
    """Return the exact static contraction geometry, or None for other math."""
    kind = _name(op)
    if kind not in ('linalg.generic', 'linalg.matmul', 'linalg.batch_matmul'):
        return None
    ins = list(getattr(op, 'inputs', []))
    outs = list(getattr(op, 'outputs', []))
    if len(ins) != 2 or len(outs) != 1 or len(op.results) != 1:
        return None
    if _dtype(ins[0].type) != 'i8' or _dtype(ins[1].type) != 'i8' or _dtype(op.results[0].type) != 'i32':
        return None
    if not _canonical_multiply_accumulate(op):
        raise ValueError('i8 contraction scalar body does not implement signed multiply-accumulate')
    a,b,y = (_shape(v.type) for v in (ins[0],ins[1],op.results[0]))
    if len(a)<2 or len(a)!=len(b) or len(a)!=len(y) or a[:-2]!=b[:-2] or y!=a[:-1]+[b[-1]] or a[-1]!=b[-2]:
        raise ValueError('integer contraction has unsupported geometry')
    from xdsl.ir.affine import AffineDimExpr
    ndim=len(a)+1
    expected=[tuple(range(ndim-3))+(ndim-3,ndim-1), tuple(range(ndim-3))+(ndim-1,ndim-2), tuple(range(ndim-1))]
    if kind=='linalg.generic':
        maps=_attrs(op)['indexing_maps']
        actual=[]
        for attr in maps:
            m=attr.data
            if m.num_dims!=ndim or m.num_symbols or any(not isinstance(x,AffineDimExpr) for x in m.results):
                raise ValueError('integer contraction requires canonical projected indexing maps')
            actual.append(tuple(x.position for x in m.results))
        if actual!=expected:
            raise ValueError('integer contraction requires canonical matrix indexing')
        iterators=_attrs(op).get('iterator_types')
        if iterators is None or len(iterators)!=ndim or any(
                ('reduction' if j==ndim-1 else 'parallel') not in str(it)
                for j,it in enumerate(iterators)):
            raise ValueError('integer contraction requires parallel matrix axes and a reduction K axis')
    return (prod(a[:-2]),a[-2],a[-1],b[-1],ins,outs)


def emit_mixed(text):
    from device_runtime import RUNTIME
    from direct_codegen import emit_direct_matmul
    from scalar_codegen import emit_generic, emit_reduce, SCALAR_RUNTIME
    from tensor_codegen import emit_tensor
    from xdsl.dialects.builtin import TensorType
    _, fn = read_module(text)
    block=fn.body.blocks[0]
    operations=[op for op in block.ops if _name(op)!='func.return']
    returned=list(block.ops.last.operands)
    if len(set(returned))!=len(returned):
        raise ValueError('repeated return aliases require distinct source result ownership')
    output_name={v:f'Y{i}' for i,v in enumerate(returned)}
    names={v:f'arg{i}' for i,v in enumerate(block.args)}
    tensors={names[v]:{'shape':_shape(v.type),'dtype':_dtype(v.type),'role':'input'} for v in block.args}
    source_values=[]
    scalar_locals=[]
    for i,op in enumerate(operations):
        for j,value in enumerate(op.results):
            if isinstance(value.type,TensorType):
                name=output_name.get(value,f'v{i}_{j}')
                names[value]=name
                tensors[name]={'shape':_shape(value.type),'dtype':_dtype(value.type),'role':'output' if value in output_name else 'intermediate'}
                source_values.append({'op_index':i,'result_index':j,'tensor':name})
            else:
                name=f'scalar_{i}_{j}'
                names[value]=name
                scalar_locals.append(f'{ctype(value.type)} {name};')
    if any(value not in names or value in block.args for value in returned):
        raise ValueError('returned values must have a direct source result owner')
    abi_names=list(tensors)
    abi_index={name:i for i,name in enumerate(abi_names)}
    value_addresses={value:(f'a{abi_index[name]}' if name in abi_index else f'&{name}') for value,name in names.items()}
    task_functions={}
    functions=[]
    calls=[]
    tasks=[]
    commands=[]
    placements=[]
    for i,op in enumerate(operations):
        # Include captures from a scalar region in the task's explicit boundary.
        needed=list(dict.fromkeys(op.operands))
        nested_results=set()
        nested_args=set()
        for region in op.regions:
            for b in region.blocks:
                nested_args.update(b.args)
                for inner in b.walk():
                    nested_results.update(inner.results)
        for inner in op.walk():
            if inner is op:
                continue
            for operand in inner.operands:
                if operand not in nested_args and operand not in nested_results and operand not in needed:
                    needed.append(operand)
        needed.extend(v for v in op.results if v not in needed)
        if any(v not in names for v in needed):
            raise ValueError(f'{_name(op)}: capture has no source buffer')
        local_ptrs={v:f'p{k}' for k,v in enumerate(needed)}
        operands=[local_ptrs[v] for v in op.operands]
        results=[local_ptrs[v] for v in op.results]
        kind=_name(op)
        device=False
        geometry=_contraction(op)
        if geometry:
            batch,m,k,n,ins,outs=geometry
            # Gemmini runtime seeds zero itself; reject nonzero source init.
            zero=_constant(outs[0]) == 0
            if not zero:
                init_owner=outs[0].owner
                zero=(_name(init_owner)=='linalg.fill' and init_owner.operands and _constant(init_owner.operands[0])==0)
            if not zero:
                raise ValueError('integer contraction requires proved zero source initializer')
            pa,pb=local_ptrs[ins[0]],local_ptrs[ins[1]]
            # All slice and edge geometry is known at compile time.  Direct
            # logical DMA avoids conditional host packing and payload loads.
            direct_parts=[]
            for batch_index in range(batch):
                a_expr=f'((const int8_t*)({pa})+{batch_index*m*k})'
                b_expr=f'((const int8_t*)({pb})+{batch_index*k*n})'
                c_expr=f'((int32_t*)({results[0]})+{batch_index*m*n})'
                declarations,statements=emit_direct_matmul(
                    a_expr,b_expr,c_expr,m,k,n,out_bytes=4,scale=1.0,
                    relu=False,bias=None,tag=f'mixed_{i}_{batch_index}')
                direct_parts.extend(declarations)
                direct_parts.extend(statements)
            body='\n'.join(direct_parts)
            device=True
            if batch==1 and len(_shape(ins[0].type))==2:
                acc=f'acc{i}'
                commands.extend([{'opcode':'MATMUL','operands':{'lhs':names[ins[0]],'rhs':names[ins[1]],'dst':acc},'attributes':{}},
                                 {'opcode':'COMMIT','operands':{'src':acc,'dst':names[op.results[0]]},'attributes':{'output_dtype':'i32','epilogue':[]}}])
            else:
                commands.append({'opcode':'BATCHED_MATMUL','operands':{'a':names[ins[0]],'w':names[ins[1]],'dst':names[op.results[0]]},'attributes':{}})
        elif kind=='arith.constant':
            if len(results)!=1:
                raise ValueError('constant requires one result')
            body=_emit_constant(op,results[0])
        elif kind=='tensor.empty':
            body='\n'.join(f'for(long i=0;i<{prod(_shape(v.type))};++i) {_at(p,v.type,"i")}=0;' for p,v in zip(results,op.results))
        elif kind=='tensor.splat':
            if len(results)!=1 or len(operands)!=1:
                raise ValueError('tensor.splat requires scalar and result')
            body=f'for(long i=0;i<{prod(_shape(op.results[0].type))};++i) {_at(results[0],op.results[0].type,"i")}={_at(operands[0],op.operands[0].type)};'
        elif kind in ('quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor'):
            body=_emit_quant(op,operands,results[0])
        elif kind=='linalg.generic':
            body=emit_generic(op,operands,results,local_ptrs,str(i))
        elif kind=='linalg.reduce':
            body=emit_reduce(op,operands,results,local_ptrs,str(i))
        elif kind.startswith('tensor.') or kind in ('linalg.transpose','linalg.fill','linalg.broadcast'):
            body=emit_tensor(op,operands,results,str(i))
        else:
            raise ValueError(f'{kind}: no mixed source lowering')
        func=f'task_{i}'
        task_functions[i]=func
        signature=', '.join('void *'+local_ptrs[v] for v in needed) or 'void'
        functions.append(f'__attribute__((noinline,used)) static void {func}({signature}) {{\n{body}\n}}')
        calls.append(f'{func}('+', '.join(value_addresses[v] for v in needed)+');')
        attrs=_attrs(op)
        region=getattr(attrs.get('prov.region_id'),'data',None)
        task={'task_index':i,'kind':'device' if device else 'host','source_op_indices':[i],
              'instruction_start':i,'instruction_end':i+1,
              'reads':list(dict.fromkeys(names[v] for v in needed if v not in op.results and names[v] in tensors)),
              'writes':[names[v] for v in op.results if names[v] in tensors]}
        if region:
            task['source_region_ids']=[region]
        tasks.append(task)
        placements.append({'source_op_index':i,'operation':kind,'lane':'accelerator' if device else 'host',
                           'reason':'integer Gemmini contraction' if device else 'source layout, initializer, quantization, or f32 scalar work'})
    args=[{'tensor':name,'access':'read' if tensors[name]['role']=='input' else 'write'} for name in abi_names]
    outputs=[output_name[v] for v in returned]
    plan={'schema':'mixed_program_plan_v1','source_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'source_op_count':len(operations),'tasks':tasks,'schedule_instruction_count':max(1,len(tasks)),
          'prologue_instruction_range':[0,0],'epilogue_instruction_range':[len(tasks),len(tasks)],
          'entry_bindings':[names[v] for v in block.args],'source_values':source_values,'output_bindings':outputs,
          'compiler_temporaries':[]}
    cb={'abi_version':'0.1','target':'gemmini','tensors':tensors,'commands':commands,
        'kernel_abi':{'kind':'whole_program','args':args,'outputs':outputs},
        'compiler_pointer_abi':{'version':1,'arguments':abi_names},
        'params':{'global_program_plan':plan,'lane_placement':placements}}
    signature=', '.join(f'void *a{i}' for i in range(len(abi_names))) or 'void'
    mixed_device_runtime=RUNTIME.replace('static void gm_matmul(', 'static __attribute__((always_inline)) inline void gm_matmul(')
    mixed_device_runtime=mixed_device_runtime.replace('static inline ', 'static __attribute__((always_inline)) inline ')
    mixed_scalar_runtime=SCALAR_RUNTIME.replace('static ', 'static __attribute__((always_inline)) inline ')
    c_source=mixed_device_runtime+'\n'+mixed_scalar_runtime+'\n'+MIXED_RUNTIME+'\n'+'\n'.join(functions)+f'\nvoid gemmini_kernel({signature}) {{\n'+'\n'.join(scalar_locals)+'\ngm_fence(); gm_cmd(7,0,0);\n'+'\n'.join(calls)+'\ngm_fence();\n}\n'
    return cb,c_source,task_functions
