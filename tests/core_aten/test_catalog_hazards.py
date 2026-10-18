"""Golden schedules share the region lint; unused source work has no bindings."""
import hashlib

from mlir_oot.core_aten.catalog import build, parse_module
from mlir_oot.core_aten.offload import rewrite
from mlir_oot.golden_device_catalog import build_catalog
from mlir_oot.golden_gemm import GoldenGemm
from mlir_oot.ir import gemmini_dialect as G
from test_limbs import FIXTURE, annotate


def test_dead_bmm_widenings_and_replaced_zero_initializer_have_no_kernel():
    text = annotate((FIXTURE.parent/'bmm.mlir').read_text())
    original = list(parse_module(text).walk())
    _, manifest, matches = build(text)
    assert len(matches) == manifest['unique_kernels'] == 1
    binding, = manifest['bindings']
    assert original[binding['root_ordinal']].attributes.get('gemmini.limb')
    assert manifest['source_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert rewrite(text, manifest).count('func.call @ca_') == 1


def test_golden_gemm_and_bmm_record_executed_hazard_checks():
    for name in ('int_mm.mlir', 'bmm.mlir'):
        _, manifest = build_catalog((FIXTURE.parent/name).read_text())
        assert manifest['covered_contractions'] == 1, manifest
        kernel, = manifest['kernels']
        assert kernel['hazard_lint'] == 'pass'
        assert kernel['hazard_lint_method'] == 'complete_static_CPU_CFG_trace'
        assert kernel['fences'] >= 2 and kernel['instructions'] > kernel['fences']


def test_golden_catalog_declines_a_real_overlapping_mvin(monkeypatch):
    original = GoldenGemm.build
    def unsafe(self):
        module = original(self)
        load = next(op for op in module.walk() if isinstance(op, G.MvinOp))
        load.parent_block().insert_op_after(load.clone(), load)
        return module
    monkeypatch.setattr(GoldenGemm, 'build', unsafe)
    _, manifest = build_catalog(FIXTURE.read_text())
    assert manifest['covered_contractions'] == 0 and not manifest['kernels']
    assert any('HW3' in row['reason'] for row in manifest['declines'])
