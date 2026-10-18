"""Exact-quantized FMA/NaN selectors and unsigned static index guards."""
from fractions import Fraction as Q

import numpy as np
import pytest

from mlir_oot.core_aten import linmap
from mlir_oot.frontend.parse import parse_module


def test_finite_quantized_affine_fma_discards_only_unselected_payload_bits():
    text='''builtin.module { func.func @forward(%input:tensor<2xf32>) -> tensor<2xf32> {
      %e = tensor.empty() : tensor<2xf32>
      %out = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
        ins(%input:tensor<2xf32>) outs(%e:tensor<2xf32>) {
        ^bb(%x:f32,%y:f32):
        %alpha = arith.constant 2.0 : f32
        %bias = arith.constant 0.25 : f32
        %fma = math.fma %x,%alpha,%bias : f32
        %bits = arith.bitcast %x : f32 to i32
        %mask = arith.constant 4194304 : i32
        %quiet = arith.ori %bits,%mask : i32
        %payload = arith.bitcast %quiet : i32 to f32
        %nan = arith.cmpf uno,%fma,%fma : f32
        %result = arith.select %nan,%payload,%fma : f32
        linalg.yield %result : f32
      } -> tensor<2xf32>
      func.return %out : tensor<2xf32>
    }}'''
    def extract(source,scale):
        module=parse_module(source)
        fn=next(o for o in module.walk() if o.name=='func.func')
        arg=fn.body.block.args[0]
        ret=next(o for o in fn.walk() if o.name=='func.return').operands[0]
        return linmap.Region(lambda v:(0,scale) if v is arg else None).tensor(ret)
    result=extract(text,Q(1,8))
    assert isinstance(result,linmap.Sym)
    assert np.all(result.const==Q(1,4))
    assert np.all(result.terms[0][2]==Q(1,4))
    with pytest.raises(linmap.NotLinear,match='payload bits affect tensor output'):
        extract(text.replace('linalg.yield %result','linalg.yield %payload'),Q(1,8))
    with pytest.raises(linmap.NotLinear,match='no finite byte-domain proof'):
        extract(text,Q(1 << 200))
    # This abstraction is an exact-quantized map proof, never admission of
    # unquantized native IEEE arithmetic merely because the IR contains FMA.
    from mlir_oot.core_aten.catalog import build
    assert not build(text)[1]['kernels']


@pytest.mark.parametrize('dtype,value,limit',[('index',-1,4),('i8',-1,4),('i32',-2147483648,4),('i64',-1,4)])
def test_unsigned_bounds_guard_excludes_negative_padding_coordinates(dtype,value,limit):
    module=parse_module(f'''builtin.module {{ func.func @guard(%a:{dtype},%b:{dtype}) -> i1 {{
      %ok = arith.cmpi ult,%a,%b : {dtype}
      func.return %ok : i1
    }} }}''')
    op=next(o for o in module.walk() if o.name=='arith.cmpi')
    result=linmap.Region(lambda _:None)._scalar(op,[np.asarray([np.int64(value)],object),np.asarray([np.int64(limit)],object)],{},[],True)
    assert not result
