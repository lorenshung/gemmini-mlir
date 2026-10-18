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


def test_dead_named_contraction_has_no_binding():
    text = '''builtin.module {
      func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x2xi8>) -> tensor<2x2xi32> {
        %z = arith.constant 0 : i32
        %init = tensor.splat %z : tensor<2x2xi32>
        %unused = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>
        %live = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>
        func.return %live : tensor<2x2xi32>
      }
    }'''
    _, manifest = build_catalog(text)
    assert manifest['covered_contractions'] == 1


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


def test_unused_runtime_assertion_is_not_dead_pure_tensor_work():
    text = '''builtin.module {
      func.func @forward(%x: tensor<3xi32>) -> tensor<3xi32> {
        %e = tensor.empty() : tensor<3xi32>
        %unused = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
          ins(%x : tensor<3xi32>) outs(%e : tensor<3xi32>) {
          ^bb0(%a: i32, %b: i32):
            %c = arith.constant 100 : i32
            %ok = arith.cmpi slt, %a, %c : i32
            cf.assert %ok, "runtime guard"
            linalg.yield %a : i32
          } -> tensor<3xi32>
        func.return %x : tensor<3xi32>
      }
    }'''
    _, manifest, _ = build(text)
    assert 'cf.assert' in rewrite(text,manifest)
