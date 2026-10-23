"""Real centered-contraction captures and an independent Fraction reference."""
from fractions import Fraction as Q
from functools import lru_cache
import itertools

import numpy as np

KINDS = ['linear', 'mm', 'bmm', 'addmm', 'conv']
SCHEMES = ['asym8', 'uint8', 'narrow', 'per_channel', 'mixed_channel']


@lru_cache(None)
def capture(kind, scheme, wide=False):
    import torch
    import torch.ao.quantization.fx._decomposed  # noqa: F401
    import m2m
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    unsigned = scheme in ('uint8', 'per_channel', 'mixed_channel')
    channel = scheme in ('per_channel','mixed_channel')
    dtype = torch.uint8 if unsigned else torch.int8
    wdtype = torch.int8 if scheme == 'mixed_channel' else dtype
    wlo, whi = (-128,127) if wdtype == torch.int8 else (0,255)
    lo, hi = (0, 255) if unsigned else (-128, 127)
    qlo, qhi = (-8, 7) if scheme == 'narrow' else (lo, hi)
    zx = 255 if unsigned else -17
    zw = ([-128,3,127] if scheme == 'mixed_channel' else [0,128,255]) if channel else (0 if unsigned else 23)
    sx, sw = {'linear': ((2,17),(3,17)), 'mm': ((2,17),(17,3)),
              'bmm': ((2,2,17),(2,17,3)), 'addmm': ((2,17),(17,3)),
              'conv': ((1,3,3,4),(3,1,2,2))}[kind]
    if wide:
        assert kind == 'mm' and not channel
        sx,sw = (17,33),(33,19)
    # Include storage extrema even in narrow schemes: a DQ declaration does
    # not establish a narrower runtime storage range.
    def codes(shp,dt,low,high):
        values = np.resize(np.array([low,high,low+1,high-1,(low+high)//2,0], np.int64),shp)
        return torch.tensor(values, dtype=dt)
    x, w = codes(sx,dtype,lo,hi), codes(sw,wdtype,wlo,whi)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('zeros', torch.tensor(zw if isinstance(zw,list) else [zw]*3, dtype=torch.int32))
            self.register_buffer('scales', torch.tensor([.03,.125,.5],dtype=torch.float64))
            self.register_buffer('bias',torch.tensor([.125,-.25,.3]))
        def forward(self,x,w):
            qd = torch.ops.quantized_decomposed
            a = qd.dequantize_per_tensor.default(x,.03,zx,qlo,qhi,dtype)
            axis = 0 if kind in ('linear','conv') else w.ndim-1
            b = (qd.dequantize_per_channel.default(w,self.scales,self.zeros,axis,wlo,whi,wdtype)
                 if channel else qd.dequantize_per_tensor.default(w,.125,zw,qlo if scheme=='narrow' else wlo,qhi if scheme=='narrow' else whi,wdtype))
            if kind == 'linear': y = torch.nn.functional.linear(a,b,self.bias)
            elif kind == 'addmm': y = torch.addmm(self.bias,a,b,alpha=-1.3,beta=-.5)
            elif kind == 'conv': y = torch.conv2d(a,b,self.bias,padding=1,groups=3)
            elif kind == 'bmm': y = torch.bmm(a,b)
            else: y = torch.mm(a,b)
            return qd.quantize_per_tensor.default(y,.0371,129,1,254,torch.uint8)
    model = Model()
    integer, receipt = integerize_pt2e(torch.export.export(model,(x,w)).module(),(x,w))
    assert not receipt['refusals'] and receipt['quantized_contractions_integerized'] == 1
    allow_exported_model_train_eval(integer)
    converted = m2m.convert(integer,(x,w),backend='fx_importer',fully_standard=True)
    # The capture ABI lifts state to leading arguments. In this standalone
    # test, bind that exported immutable state as source constants, just as a
    # compiler's normal parameter specialization does. Never infer it from
    # provenance or from the numerical input samples.
    from torch.export.graph_signature import InputKind
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    from mlir_oot.core_aten.catalog import parse_module
    module = parse_module(converted.mlir_text)
    fn = next(o for o in module.walk() if o.name == 'func.func')
    arrays, user = [], iter((x,w))
    specs = converted.exported_program.graph_signature.input_specs
    assert len(specs) == len(fn.body.block.args)
    for arg,spec in zip(fn.body.block.args,specs):
        if spec.kind == InputKind.USER_INPUT:
            arrays.append(next(user).numpy())
        else:
            state = converted.exported_program.state_dict.get(spec.target)
            if state is None: state = converted.exported_program.constants[spec.target]
            data = state.detach().numpy();arrays.append(data)
            const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(arg.type,data.reshape(-1).tolist()))
            fn.body.block.insert_op_before(const,fn.body.block.first_op)
            arg.replace_all_uses_with(const.result)
    module.verify()
    xx = x.numpy().astype(np.int64)-zx
    axis = 0 if kind in ('linear','conv') else w.ndim-1
    zeros_shape = [1]*w.ndim;zeros_shape[axis] = -1
    ww = w.numpy().astype(np.int64)-np.asarray(zw).reshape(zeros_shape)
    if kind == 'linear': acc = xx@ww.T
    elif kind in ('mm','bmm','addmm'): acc = xx@ww
    else:
        acc = np.zeros((1,3,4,5),np.int64)
        for n,c,h,t in np.ndindex(acc.shape):
            for kh,kw in itertools.product(range(2),repeat=2):
                ih,iw = h+kh-1,t+kw-1
                if 0<=ih<3 and 0<=iw<4:
                    acc[n,c,h,t] += xx[n,c,ih,iw]*ww[c,0,kh,kw]
    expected = np.empty(acc.shape,np.uint8)
    for ix in np.ndindex(acc.shape):
        c = ix[1] if kind == 'conv' else ix[-1]
        scale = [float(np.float32(s)) for s in (.03,.125,.5)][c] if channel else .125
        value = Q(int(acc[ix]))*Q(float(np.float32(.03)))*Q(scale)
        if kind == 'addmm': value *= Q(-1.3)
        if kind in ('linear','conv','addmm'):
            value += Q(float(model.bias[c]))*Q(-.5 if kind == 'addmm' else 1.)
        expected[ix] = min(254,max(1,round(value/Q(float(np.float32(.0371))))+129))
    assert np.array_equal(integer(x,w).numpy(),expected)
    return str(module), arrays, expected


def check(source, inputs, expected):
    """Check source-bound raw device results and the retained exact host readout."""
    from exact_oracle import SourceOracle, evaluate
    from mlir_oot.core_aten.catalog import build, parse_module
    from mlir_oot.core_aten.stream import simulate
    parsed = parse_module(source)
    fn = next(o for o in parsed.walk() if o.name == 'func.func')
    oracle = SourceOracle({v:np.asarray(a.view(np.int8) if a.dtype == np.uint8 else a,object) for v,a in zip(fn.body.block.args,inputs)})
    returned = next(o for o in fn.walk() if o.name == 'func.return').operands[0]
    assert np.array_equal(np.asarray(oracle.tensor(returned),np.int64).astype(np.uint8),expected)
    device,manifest,matches = build(source)
    regions = [m for m in matches if m.numeric.get('integer_readout')]
    assert regions,manifest['declines']
    replays=[]
    match_fn = regions[0].root.parent_op()
    while match_fn.name != 'func.func': match_fn = match_fn.parent_op()
    local = SourceOracle({v:np.asarray(a.view(np.int8) if a.dtype == np.uint8 else a,object)
                          for v,a in zip(match_fn.body.block.args,inputs)})
    replacements = {}
    for match in regions:
        # Match values belong to build's parsed module, independently of the
        # oracle's module; function arguments keep the same checked order.
        feed = {name:np.asarray(local.tensor(v),np.int64).astype(np.int8) for name,v in match.operands}
        raw = evaluate(match,feed)['out']
        actual = simulate(match.kernel.stream,feed)['out']
        assert np.array_equal(actual,raw)
        # Replace precisely the producer, leaving the original exact rational
        # readout (including signed alpha, bias, channel scales and clamp).
        replacements[match.result[1]] = actual.astype(object)
        proof = match.kernel.meta['centered_byte_contraction']
        assert proof['readout'] == 'full_i32' and proof['device_scales'] == 'unit'
        replays.append((match,feed,raw))
    local.memo.clear();local.memo.update(replacements)
    result = next(o for o in match_fn.walk() if o.name == 'func.return').operands[0]
    assert np.array_equal(np.asarray(local.tensor(result),np.int64).astype(np.uint8),expected)
    from mlir_oot.core_aten.offload import rewrite
    rewritten = rewrite(source,manifest)
    assert 'func.call @ca_' in rewritten and 'prov.quant.arithmetic = "exact"' in rewritten
    return device,manifest,replays
