"""Structural scalar lowering for host floating-point linalg regions.

Indexing maps, loop bounds, scalar SSA dependencies and yielded values are read
from the xDSL operations. Integer contractions are deliberately diagnosed: the
mixed emitter must route those to the accelerator implementation.
"""
from __future__ import annotations

import math
from math import prod

SCALAR_RUNTIME = r'''
static float gm_sqrtf(float x) {
#if defined(__riscv)
  float y; __asm__("fsqrt.s %0, %1" : "=f"(y) : "f"(x)); return y;
#else
  if (x == 0.0f || x != x) return x;
  if (x < 0.0f) return __builtin_nanf("");
  if (x == __builtin_inff()) return x;
  union {float f; unsigned int u;} b={x};
  b.u=(b.u>>1)+0x1fc00000u; double y=b.f;
  for (int i=0;i<8;i++) y=(y+(double)x/y)*0.5;
  return (float)y;
#endif
}
static float gm_expf(float x) {
  if (x != x) return x;
  if (x > 88.72283905206835f) return __builtin_inff();
  if (x < -103.9720770839918f) return 0.0f;
  int k=(int)((double)x*1.4426950408889634074+((x>=0.0f)?0.5:-0.5));
  double r=(double)x-(double)k*0.69314718055994530942;
  double p=1.0+r*(1.0+r*(0.5+r*(0.16666666666666666667+r*(0.041666666666666666667+r*(0.0083333333333333333333+r*(0.0013888888888888888889+r*(0.00019841269841269841270+r*0.000024801587301587301587)))))));
  union { unsigned long long u; double d; } s;
  s.u=(unsigned long long)(k+1023)<<52;
  return (float)(p*s.d);
}
static float gm_logf(float x) {
  if (x != x || x == __builtin_inff()) return x;
  if (x == 0.0f) return -__builtin_inff();
  if (x < 0.0f) return __builtin_nanf("");
  union {float f; unsigned int u;} b={x}; int adjust=0;
  if (b.u < 0x00800000u) { b.f=x*16777216.0f; adjust=-24; }
  int e=(int)(b.u>>23)-127+adjust;
  b.u=(b.u&0x007fffffu)|0x3f800000u;
  double m=b.f; if (m>1.4142135623730950488) { m*=0.5; e++; }
  double s=(m-1.0)/(m+1.0), z=s*s;
  double p=s*(1.0+z*(1.0/3.0+z*(1.0/5.0+z*(1.0/7.0+z*(1.0/9.0+z*(1.0/11.0+z*(1.0/13.0+z/15.0)))))));
  return (float)(2.0*p+e*0.69314718055994530942);
}
static float gm_tanhf(float x) {
  if (x != x) return x;
  float a=x<0.0f?-x:x;
  if (a>10.0f) return x<0.0f?-1.0f:1.0f;
  float e=gm_expf(-2.0f*a), y=(1.0f-e)/(1.0f+e);
  return x<0.0f?-y:y;
}
static double gm_sin_core(double x) {
  if (x != x || x == __builtin_inf() || x == -__builtin_inf()) return __builtin_nan("");
  const double pi=3.1415926535897932384626433832795;
  double turns=x*(1.0/(2.0*pi));
  if (turns>9223372036854774784.0 || turns< -9223372036854774784.0) return __builtin_nan("");
  long long k=(long long)(turns+(turns>=0.0?0.5:-0.5));
  x-=k*(2.0*pi);
  if (x>pi*0.5) x=pi-x;
  if (x< -pi*0.5) x=-pi-x;
  double z=x*x;
  return x*(1.0+z*(-1.0/6.0+z*(1.0/120.0+z*(-1.0/5040.0+z*(1.0/362880.0+z*(-1.0/39916800.0+z*(1.0/6227020800.0+z*(-1.0/1307674368000.0+z/355687428096000.0))))))));
}
static float gm_sinf(float x) { return (float)gm_sin_core((double)x); }
static float gm_cosf(float x) { return (float)gm_sin_core((double)x+1.57079632679489661923); }
static float gm_floorf(float x) {
  if (x != x || x>=8388608.0f || x<=-8388608.0f) return x;
  long y=(long)x; return (float)(y-((float)y>x));
}
static float gm_ceilf(float x) { return -gm_floorf(-x); }
static float gm_truncf(float x) { return x<0.0f?gm_ceilf(x):gm_floorf(x); }
static float gm_roundevenf(float x) {
  if (x != x || x>=8388608.0f || x<=-8388608.0f) return x;
  float lo=gm_floorf(x), d=x-lo;
  return lo+(d>0.5f || (d==0.5f && ((long)lo&1)));
}
static float gm_roundf(float x) { return x<0.0f?-gm_floorf(-x+0.5f):gm_floorf(x+0.5f); }
static float gm_maximumf(float a,float b) {
  if (a!=a) return a; if (b!=b) return b;
  if (a==0.0f && b==0.0f) { union {float f; unsigned int u;} x={a},y={b}; x.u&=y.u; return x.f; }
  return a>b?a:b;
}
static float gm_minimumf(float a,float b) {
  if (a!=a) return a; if (b!=b) return b;
  if (a==0.0f && b==0.0f) { union {float f; unsigned int u;} x={a},y={b}; x.u|=y.u; return x.f; }
  return a<b?a:b;
}
static float gm_maxnumf(float a,float b) { if(a!=a)return b; if(b!=b)return a; return gm_maximumf(a,b); }
static float gm_minnumf(float a,float b) { if(a!=a)return b; if(b!=b)return a; return gm_minimumf(a,b); }
static float gm_bitcast_i32_f32(int x) { union {int i;float f;} v={x}; return v.f; }
static int gm_bitcast_f32_i32(float x) { union {float f;int i;} v={x}; return v.i; }
static long gm_floordiv(long a,long b) { long q=a/b,r=a%b; return q-((r!=0)&&((r<0)!=(b<0))); }
static long gm_ceildiv(long a,long b) { return -gm_floordiv(-a,b); }
static long gm_mod(long a,long b) { return a-gm_floordiv(a,b)*b; }
'''


def _name(op):
    if op.name == 'builtin.unregistered':
        return op.attributes['op_name__'].data
    return op.name


def _attrs(op):
    return {**op.attributes, **dict(getattr(op, 'properties', {}))}


def _shape(ty):
    return tuple(ty.get_shape()) if hasattr(ty, 'get_shape') else ()


def _element(ty):
    return ty.element_type if hasattr(ty, 'element_type') else ty


def _ctype(ty, unsigned=False):
    name = str(_element(ty))
    types = {'f16':'_Float16', 'bf16':'__bf16', 'f32':'float', 'f64':'double',
             'i1':'unsigned char', 'i8':'signed char', 'i16':'short',
             'i32':'int', 'i64':'long long', 'index':'long'}
    if name not in types:
        raise ValueError(f'unsupported scalar type {name}')
    result = types[name]
    if unsigned and name not in ('i1',) and (name.startswith('i') or name=='index'):
        result = 'unsigned ' + result.replace('signed ', '')
    return result


def _load(ptr, ty, index='0'):
    return f'(({_ctype(ty)}*)({ptr}))[{index}]'


def _flat(indices, shape):
    if len(indices) != len(shape):
        raise ValueError('index rank does not match tensor rank')
    result = '0'
    for i, size in zip(indices, shape):
        result = f'(({result})*{size}+({i}))'
    return result


def _constant(attr):
    value = getattr(getattr(attr, 'value', None), 'data', None)
    if value is None:
        raise ValueError(f'unsupported scalar constant {attr}')
    if isinstance(value, float):
        if math.isnan(value): return '__builtin_nanf("")'
        if math.isinf(value): return ('-' if value<0 else '')+'__builtin_inff()'
        suffix = 'f' if str(getattr(attr, 'type', '')) == 'f32' else ''
        return value.hex() + suffix
    return str(value)


def _attr_int(attr):
    return int(getattr(getattr(attr, 'value', None), 'data', getattr(attr, 'data', attr)))


def _affine(expr, loops):
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr
    if isinstance(expr, AffineDimExpr): return loops[expr.position]
    if isinstance(expr, AffineConstantExpr): return str(expr.value)
    if isinstance(expr, AffineBinaryOpExpr):
        a,b=_affine(expr.lhs,loops),_affine(expr.rhs,loops)
        kind=expr.kind.name.lower()
        if kind=='add': return f'(({a})+({b}))'
        if kind=='mul': return f'(({a})*({b}))'
        if kind in ('mod','floordiv','ceildiv'):
            return f'gm_{kind}(({a}),({b}))'
    raise ValueError(f'unsupported affine expression {expr}')


def _body(block, arg_exprs, value_ptrs, loops, uid):
    if len(block.args)!=len(arg_exprs):
        raise ValueError('linalg block argument count does not match operands')
    env = {}
    code=[]
    for j,(arg,expr) in enumerate(zip(block.args,arg_exprs)):
        var=f'sg{uid}_a{j}'
        code.append(f'{_ctype(arg.type)} {var} = {expr};')
        env[arg]=var

    def scalar(value):
        if value in env: return env[value]
        owner=getattr(value,'owner',None)
        if getattr(owner,'name',None)=='arith.constant':
            return _constant(_attrs(owner)['value'])
        if value in value_ptrs: return _load(value_ptrs[value], value.type)
        raise ValueError(f'unknown captured scalar in linalg region: {value}')

    yields=None
    for i,inner in enumerate(block.ops):
        name=_name(inner); attrs=_attrs(inner)
        if name=='linalg.yield':
            yields=[scalar(v) for v in inner.operands]
            continue
        if len(inner.results)!=1:
            raise ValueError(f'scalar region operation {name} requires one result')
        ty=inner.results[0].type
        if name=='tensor.extract':
            tensor=inner.operands[0]
            if tensor not in value_ptrs:
                raise ValueError('tensor.extract captures an unknown source tensor')
            idx=_flat([scalar(v) for v in inner.operands[1:]],_shape(tensor.type))
            expr=_load(value_ptrs[tensor],tensor.type,idx)
        elif name=='linalg.index':
            expr=loops[_attr_int(attrs['dim'])]
        elif name=='arith.constant':
            expr=_constant(attrs['value'])
        else:
            args=[scalar(v) for v in inner.operands]
            binary={ 'arith.addf':'+','arith.subf':'-','arith.mulf':'*','arith.divf':'/',
                     'arith.addi':'+','arith.subi':'-','arith.muli':'*','arith.divsi':'/',
                     'arith.remsi':'%','arith.andi':'&','arith.ori':'|','arith.xori':'^',
                     'arith.shli':'<<','arith.shrsi':'>>' }
            casts={'arith.sitofp','arith.uitofp','arith.fptosi','arith.fptoui',
                   'arith.extsi','arith.extui','arith.trunci','arith.extf',
                   'arith.truncf','arith.index_cast','arith.index_castui'}
            unary={'math.sqrt':'gm_sqrtf','math.exp':'gm_expf','math.exp2':None,
                   'math.log':'gm_logf','math.log2':None,'math.log10':None,
                   'math.tanh':'gm_tanhf','math.sin':'gm_sinf','math.cos':'gm_cosf',
                   'math.floor':'gm_floorf','math.ceil':'gm_ceilf','math.trunc':'gm_truncf',
                   'math.round':'gm_roundf','math.roundeven':'gm_roundevenf'}
            if name in binary:
                if len(args)!=2: raise ValueError(f'invalid binary scalar {name}')
                expr=f'(({args[0]}) {binary[name]} ({args[1]}))'
            elif name in ('arith.divui','arith.remui','arith.shrui'):
                sym={'arith.divui':'/','arith.remui':'%','arith.shrui':'>>'}[name]
                unsigned=_ctype(inner.operands[0].type,True)
                expr=f'((({unsigned})({args[0]})) {sym} (({unsigned})({args[1]})))'
            elif name in casts:
                source=args[0]
                if name in ('arith.uitofp','arith.extui','arith.index_castui'):
                    source=f'({_ctype(inner.operands[0].type,True)})({source})'
                expr=f'({_ctype(ty)})({source})'
            elif name=='arith.negf': expr=f'-({args[0]})'
            elif name in ('math.absf','math.absi'):
                expr=f'(({args[0]})<0?-({args[0]}):({args[0]}))'
            elif name=='math.rsqrt': expr=f'(1.0f/gm_sqrtf({args[0]}))'
            elif name in unary:
                if name=='math.exp2': expr=f'gm_expf(({args[0]})*0.6931471805599453f)'
                elif name=='math.log2': expr=f'(gm_logf({args[0]})*1.4426950408889634f)'
                elif name=='math.log10': expr=f'(gm_logf({args[0]})*0.4342944819032518f)'
                else: expr=f'{unary[name]}({args[0]})'
            elif name in ('arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf'):
                expr=f'gm_{name.split(".")[1]}({args[0]},{args[1]})'
            elif name in ('arith.maxsi','arith.minsi','arith.maxui','arith.minui'):
                cmp='>' if 'max' in name else '<'
                a,b=args
                if name.endswith('ui'):
                    ct=_ctype(inner.operands[0].type,True);a,b=f'({ct})({a})',f'({ct})({b})'
                expr=f'(({a}){cmp}({b})?({args[0]}):({args[1]}))'
            elif name=='arith.select': expr=f'(({args[0]})?({args[1]}):({args[2]}))'
            elif name=='arith.cmpi':
                pred=_attr_int(attrs['predicate'])
                a,b=args
                if pred>=6:
                    ct=_ctype(inner.operands[0].type,True);a,b=f'({ct})({a})',f'({ct})({b})'
                cmp={0:'==',1:'!=',2:'<',3:'<=',4:'>',5:'>=',6:'<',7:'<=',8:'>',9:'>='}.get(pred)
                if cmp is None: raise ValueError(f'unknown integer comparison {pred}')
                expr=f'(({a}) {cmp} ({b}))'
            elif name=='arith.cmpf':
                pred=_attr_int(attrs['predicate']); a,b=args
                nan=f'((({a})!=({a}))||(({b})!=({b})))'
                cmp={1:'==',2:'>',3:'>=',4:'<',5:'<=',6:'!=',8:'==',9:'>',10:'>=',11:'<',12:'<=',13:'!='}
                if pred==0: expr='0'
                elif pred==15: expr='1'
                elif pred==7: expr=f'(!{nan})'
                elif pred==14: expr=nan
                elif pred in cmp:
                    comparison=f'(({a}){cmp[pred]}({b}))'
                    expr=f'((!{nan})&&{comparison})' if pred<8 else f'({nan}||{comparison})'
                else: raise ValueError(f'unknown floating comparison {pred}')
            elif name=='arith.bitcast':
                pair=(str(inner.operands[0].type),str(ty))
                if pair==('i32','f32'):expr=f'gm_bitcast_i32_f32({args[0]})'
                elif pair==('f32','i32'):expr=f'gm_bitcast_f32_i32({args[0]})'
                else:raise ValueError(f'unsupported bitcast {pair}')
            elif name in ('math.powf','math.fpowi'):
                expr=f'gm_expf(({args[1]})*gm_logf({args[0]}))'
            elif name=='math.fma':expr=f'__builtin_fmaf({args[0]},{args[1]},{args[2]})'
            elif name=='arith.ceildivsi':expr=f'gm_ceildiv({args[0]},{args[1]})'
            elif name=='arith.floordivsi':expr=f'gm_floordiv({args[0]},{args[1]})'
            else:
                raise ValueError(f'unsupported scalar region operation {name}')
        var=f'sg{uid}_v{i}'
        code.append(f'{_ctype(ty)} {var} = {expr};')
        env[inner.results[0]]=var
    if yields is None: raise ValueError('linalg region has no yield')
    return code,yields


def _initialize(op, operands, results, nin, uid):
    if len(results)!=len(op.results):
        raise ValueError('linalg result pointer count mismatch')
    code=[]
    for j,(result,dst) in enumerate(zip(op.results,results)):
        initial=op.operands[nin+j]
        if _shape(initial.type)!=_shape(result.type):
            raise ValueError('linalg initial/result shape mismatch')
        n=prod(_shape(result.type))
        it=f'sg{uid}_copy{j}'
        code.append(f'for(long {it}=0;{it}<{n};++{it}) '
                    f'{_load(dst,result.type,it)} = {_load(operands[nin+j],initial.type,it)};')
    return code


def emit_generic(op, operands: list[str], results: list[str], value_ptrs: dict, uid: int) -> str:
    from xdsl.ir.affine import AffineDimExpr
    attrs=_attrs(op)
    maps=[m.data for m in attrs['indexing_maps']]
    if len(maps)!=len(op.operands) or len(operands)!=len(op.operands):
        raise ValueError('linalg.generic indexing map/operand count mismatch')
    nin=len(op.operands)-len(op.results)
    if nin<0: raise ValueError('linalg.generic has no destination operands')
    iterators=list(attrs['iterator_types'])
    nloops=maps[0].num_dims if maps else len(iterators)
    if len(iterators)!=nloops: raise ValueError('linalg.generic iterator rank mismatch')
    reduction=any('reduction' in str(x) for x in iterators)
    if reduction and any(_name(x)=='arith.muli' for x in op.regions[0].blocks[0].ops):
        raise ValueError('integer contraction requires accelerator lowering')
    extents=[None]*nloops
    for value, amap in zip(op.operands,maps):
        if amap.num_symbols: raise ValueError('symbolic indexing maps are unsupported')
        if len(amap.results)!=len(_shape(value.type)):
            raise ValueError('linalg.generic map/tensor rank mismatch')
        for expr,size in zip(amap.results,_shape(value.type)):
            if isinstance(expr,AffineDimExpr):
                d=expr.position
                if extents[d] is not None and extents[d]!=size:
                    raise ValueError('inconsistent generic iteration extents')
                extents[d]=size
    if any(x is None or x<0 for x in extents):
        raise ValueError('generic loop bounds are not static projected dimensions')
    loops=[f'sg{uid}_d{i}' for i in range(nloops)]
    indices=[_flat([_affine(e,loops) for e in amap.results],_shape(v.type))
             for v,amap in zip(op.operands,maps)]
    sources=list(operands[:nin])+list(results)
    arguments=[_load(ptr,value.type,index) for ptr,value,index in zip(sources,op.operands,indices)]
    body,yields=_body(op.regions[0].blocks[0],arguments,value_ptrs,loops,uid)
    if len(yields)!=len(results):raise ValueError('generic yield/result count mismatch')
    code=['{']+_initialize(op,operands,results,nin,uid)
    for loop,size in zip(loops,extents):code.append(f'for(long {loop}=0;{loop}<{size};++{loop}) {{')
    code+=body
    code += [f'{_load(dst,value.type,index)} = {expr};' for dst,value,index,expr in
             zip(results,op.results,indices[nin:],yields)]
    code+=['}']*nloops+['}']
    return '\n'.join(code)


def emit_reduce(op, operands: list[str], results: list[str], value_ptrs: dict, uid: int) -> str:
    if len(operands)!=len(op.operands):raise ValueError('linalg.reduce operand count mismatch')
    nin=len(op.operands)-len(op.results)
    if nin<1:raise ValueError('linalg.reduce has no input')
    shape=_shape(op.operands[0].type)
    if any(_shape(x.type)!=shape for x in op.operands[:nin]):
        raise ValueError('reduce inputs must have the same shape')
    dimsattr=_attrs(op)['dimensions']
    dims=list(dimsattr.iter_values()) if hasattr(dimsattr,'iter_values') else [_attr_int(x) for x in dimsattr]
    if len(set(dims))!=len(dims) or any(d<0 or d>=len(shape) for d in dims):
        raise ValueError('invalid reduce dimensions')
    if any(str(_element(x.type))!='f32' for x in op.operands[:nin]):
        raise ValueError('integer reduction requires accelerator lowering')
    loops=[f'sg{uid}_d{i}' for i in range(len(shape))]
    outshape=tuple(s for i,s in enumerate(shape) if i not in dims)
    if any(_shape(x.type)!=outshape for x in op.results):
        raise ValueError('reduce output shape mismatch')
    inidx=_flat(loops,shape)
    outidx=_flat([v for i,v in enumerate(loops) if i not in dims],outshape)
    args=[_load(ptr,value.type,inidx) for ptr,value in zip(operands[:nin],op.operands[:nin])]
    args += [_load(ptr,value.type,outidx) for ptr,value in zip(results,op.results)]
    body,yields=_body(op.regions[0].blocks[0],args,value_ptrs,loops,uid)
    if len(yields)!=len(results):raise ValueError('reduce yield/result count mismatch')
    code=['{']+_initialize(op,operands,results,nin,uid)
    for loop,size in zip(loops,shape):code.append(f'for(long {loop}=0;{loop}<{size};++{loop}) {{')
    code+=body
    code += [f'{_load(dst,value.type,outidx)} = {expr};' for dst,value,expr in zip(results,op.results,yields)]
    code+=['}']*len(shape)+['}']
    return '\n'.join(code)
