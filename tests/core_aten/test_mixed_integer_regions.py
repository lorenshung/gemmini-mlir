"""One source-bound catalog covers raw and retained-readout integer producers."""
import json
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from exact_oracle import evaluate
from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.integer_regions import contraction
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.core_aten.stream import simulate
from conftest import provider,PACKAGE,POLICY

FIXTURE=Path(__file__).parent/'fixtures/mixed_integer/tiny.mlir'


def source(batch=False):
    text=FIXTURE.read_text()
    if batch:
        text=text.replace('2x2x','3x2x2x')
        text=text.replace('(d0, d1, d2) -> (d0, d2)', '(d0, d1, d2, d3) -> (d0, d1, d3)')
        text=text.replace('(d0, d1, d2) -> (d2, d1)', '(d0, d1, d2, d3) -> (d0, d3, d2)')
        text=text.replace('(d0, d1, d2) -> (d0, d1)', '(d0, d1, d2, d3) -> (d0, d1, d2)')
        text=text.replace('["parallel", "parallel", "reduction"]','["parallel", "parallel", "parallel", "reduction"]')
        text=text.replace('(d0, d1) -> (d0, d1)', '(d0, d1, d2) -> (d0, d1, d2)')
        text=text.replace('["parallel", "parallel"]','["parallel", "parallel", "parallel"]')
    return text


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('raw_first',[False,True])
def test_mixed_producers_are_both_proven_bound_and_exact(batch,raw_first):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_spec_int8 import provider_coverage
    module=parse_mlir_text(source(batch))
    producers=[o for o in module.walk() if contraction(o)]
    if not raw_first:
        raw,readout=producers
        raw.detach();readout.parent_block().insert_op_after(raw,readout)
    text=str(module)
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY)
    assert proof['status']=='proven',proof
    assert len(proof['kernels'])==2 and len(proof['bindings'])==2
    for op in module.walk():
        if contraction(op): assert provider_coverage(module,proof,op) is None
    _,manifest,matches=build(text)
    assert manifest['bindings']==proof['bindings']
    assert not manifest['declines']
    assert len(matches)==2
    kinds={}
    for match in matches:
        kinds[match.kernel.meta['raw_integer']]=match.kernel.meta
        feed={name:np.resize(np.array([-128,127,-1,0],np.int8),value.type.get_shape())
              for name,value in match.operands}
        assert np.array_equal(simulate(match.kernel.stream,feed)['out'],evaluate(match,feed)['out'])
        assert not match.kernel.stream.readouts
    assert kinds[True]['requantization']=='none'
    assert kinds[True]['output_dtype']=='i32' and kinds[True]['seed']=='zero'
    assert 'scale_numerator' not in kinds[True] and 'scale_denominator' not in kinds[True]
    assert kinds[False]['requantization']=='retained_source_exact_integer'
    assert kinds[False]['scale_numerator']==['1'] and kinds[False]['scale_denominator']==['1']
    rewritten=parse_module(rewrite(text,manifest))
    assert len([o for o in rewritten.walk() if o.name=='func.call'])==2
    assert any(o.name=='arith.sitofp' for o in rewritten.walk())
    assert not any(contraction(o) for o in rewritten.walk())


def test_uncovered_live_raw_producer_cannot_receive_partial_proven_status():
    module=parse_module(source())
    raw=next(o for o in module.walk() if contraction(o))
    raw.attributes['prov.quant.host_only']=StringAttr('true')
    proof=provider().admit_numeric_schedule(str(module),package=PACKAGE,numeric_policy=POLICY)
    assert proof['status']=='declined' and not proof['kernels'] and not proof['bindings']
    assert proof['reason']=='provider did not cover every live integer producer'
    assert any('host-only' in row['reason'] for row in proof['declines'])


def test_dead_raw_producer_does_not_create_a_coverage_obligation():
    module=parse_module(source())
    raw=next(o for o in module.walk() if contraction(o))
    raw.attributes['prov.quant.host_only']=StringAttr('true')
    ret=next(o for o in module.walk() if o.name=='func.return')
    ret.operands=(ret.operands[1],ret.operands[1])
    from xdsl.dialects.builtin import FunctionType
    fn=ret.parent_op()
    fn.properties['function_type']=FunctionType.from_lists(list(fn.function_type.inputs),[v.type for v in ret.operands])
    proof=provider().admit_numeric_schedule(str(module),package=PACKAGE,numeric_policy=POLICY)
    assert proof['status']=='proven' and len(proof['kernels'])==1
    assert not any(row['numeric'].get('raw_integer') for row in proof['bindings'])


def test_compile_and_offload_use_the_same_complete_catalog(tmp_path):
    from mlir_oot.core_aten.emit import llvm_bin_dir
    try:
        bindir=llvm_bin_dir()
    except RuntimeError:
        pytest.skip('requires MLIR compiler')
    if not (bindir/'mlir-translate').is_file(): pytest.skip('requires MLIR compiler')
    path=tmp_path/'source.mlir';path.write_text(source())
    api=provider()
    manifest,obj=api.core_aten_catalog(path,tmp_path/'catalog',package=PACKAGE,llvm_bin=bindir)
    model,shim,offload_obj,offload_manifest=api.core_aten_offload(path,tmp_path/'offload',package=PACKAGE,llvm_bin=bindir)
    assert all(p is not None and p.is_file() for p in (obj,model,shim,offload_obj))
    catalog=json.loads(manifest.read_text());offload=json.loads(offload_manifest.read_text())
    proof=api.admit_numeric_schedule(path.read_text(),package=PACKAGE,numeric_policy=POLICY)
    assert catalog['bindings']==offload['bindings']==proof['bindings']
    assert len(catalog['bindings'])==2


def test_unrecognized_raw_integer_body_is_not_silently_omitted():
    text=source().replace('arith.extsi %a_raw','arith.extui %a_raw')
    proof=provider().admit_numeric_schedule(text,package=PACKAGE,numeric_policy=POLICY)
    assert proof['status']=='declined' and not proof['kernels']
    assert proof['reason']=='provider did not cover every live integer producer'
