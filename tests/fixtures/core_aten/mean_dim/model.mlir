builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [37, 50], \22storage_offset\22: 0, \22stride\22: [50, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [37], \22storage_offset\22: 0, \22stride\22: [1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<37x50xi8>) -> tensor<37xi8> {
    %1 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0252246745 : f32
    %2 = tensor.splat %1 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %3 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %4 = tensor.splat %3 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %5 = "quant_ext.dequantize_per_tensor"(%0, %2, %4) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.025224674449247472 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<37x50xi8>, tensor<f32>, tensor<i64>) -> tensor<37x50xf32>
    %6 = arith.constant {prov.role = "sum", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %7 = tensor.splat %6 {prov.role = "sum", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<37xf32>
    %8 = "linalg.reduce"(%5, %7) <{dimensions = array<i64: 1>}> ({
    ^bb0(%9: f32, %10: f32):
      %11 = arith.addf %9, %10 : f32
      linalg.yield %11 : f32
    }) {prov.role = "sum", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : (tensor<37x50xf32>, tensor<37xf32>) -> tensor<37xf32>
    %12 = arith.constant {prov.role = "scale", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 5.000000e+01 : f32
    %13 = tensor.splat %12 {prov.role = "scale", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<37xf32>
    %14 = "tensor.empty"() {prov.role = "scale", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : () -> tensor<37xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%8, %13 : tensor<37xf32>, tensor<37xf32>) outs(%14 : tensor<37xf32>) attrs =  {prov.role = "scale", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.family = "reduce", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb1(%16: f32, %17: f32, %18: f32):
      %19 = arith.divf %16, %17 : f32
      linalg.yield %19 : f32
    } -> tensor<37xf32>
    %20 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00494601484 : f32
    %21 = tensor.splat %20 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %22 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %23 = tensor.splat %22 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %24 = "quant_ext.quantize_per_tensor"(%15, %21, %23) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0049460149278827744 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<37xf32>, tensor<f32>, tensor<i64>) -> tensor<37xi8>
    func.return %24 : tensor<37xi8>
  }
}