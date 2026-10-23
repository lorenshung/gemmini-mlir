builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 20, 3, 4], \22storage_offset\22: 0, \22stride\22: [240, 12, 4, 1]}, {\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 20, 9, 11], \22storage_offset\22: 0, \22stride\22: [1980, 99, 11, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [1, 20, 9, 11], \22storage_offset\22: 0, \22stride\22: [1980, 99, 11, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x20x3x4xi8>, %1: tensor<1x20x9x11xi8>) -> tensor<1x20x9x11xi8> {
    %2 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 4.918150e-03 : f32
    %3 = tensor.splat %2 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %4 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %5 = tensor.splat %4 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %6 = "quant_ext.dequantize_per_tensor"(%0, %3, %5) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0049181503408095417 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x20x3x4xi8>, tensor<f32>, tensor<i64>) -> tensor<1x20x3x4xf32>
    %7 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0311086867 : f32
    %8 = tensor.splat %7 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %9 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %10 = tensor.splat %9 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %11 = "quant_ext.dequantize_per_tensor"(%1, %8, %10) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.031108686035754635 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x20x9x11xi8>, tensor<f32>, tensor<i64>) -> tensor<1x20x9x11xf32>
    %12 = "tensor.empty"() {prov._pattern_hint = "avg_pool_backward", prov.op = "avg_pool_backward", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d_backward.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x9x11xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%12 : tensor<1x20x9x11xf32>) attrs =  {prov._pattern_hint = "avg_pool_backward", prov.op = "avg_pool_backward", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d_backward.default", prov.orig_dtype = "float32"} {
    ^bb0(%14: f32):
      %15 = linalg.index 0 : index
      %16 = linalg.index 1 : index
      %17 = linalg.index 2 : index
      %18 = linalg.index 3 : index
      %19 = arith.constant 0 : index
      %20 = arith.constant 1 : index
      %21 = arith.constant 12 : index
      %22 = arith.constant 0.000000e+00 : f32
      %23 = scf.for %24 = %19 to %21 step %20 iter_args(%25 = %22) -> (f32) {
        %26 = arith.constant 4 : index
        %27 = arith.remui %24, %26 : index
        %28 = arith.divui %24, %26 : index
        %29 = arith.constant 3 : index
        %30 = arith.remui %28, %29 : index
        %31 = arith.constant 9 : index
        %32 = arith.constant 3 : index
        %33 = arith.muli %30, %31 : index
        %34 = arith.divui %33, %32 : index
        %35 = arith.addi %30, %20 : index
        %36 = arith.muli %35, %31 : index
        %37 = arith.subi %32, %20 : index
        %38 = arith.addi %36, %37 : index
        %39 = arith.divui %38, %32 : index
        %40 = arith.maxsi %34, %19 : index
        %41 = arith.minsi %39, %31 : index
        %42 = arith.cmpi sge, %17, %40 : index
        %43 = arith.cmpi slt, %17, %41 : index
        %44 = arith.andi %42, %43 : i1
        %45 = arith.subi %41, %40 : index
        %46 = arith.muli %20, %45 : index
        %47 = arith.constant 11 : index
        %48 = arith.constant 4 : index
        %49 = arith.muli %27, %47 : index
        %50 = arith.divui %49, %48 : index
        %51 = arith.addi %27, %20 : index
        %52 = arith.muli %51, %47 : index
        %53 = arith.subi %48, %20 : index
        %54 = arith.addi %52, %53 : index
        %55 = arith.divui %54, %48 : index
        %56 = arith.maxsi %50, %19 : index
        %57 = arith.minsi %55, %47 : index
        %58 = arith.cmpi sge, %18, %56 : index
        %59 = arith.cmpi slt, %18, %57 : index
        %60 = arith.andi %58, %59 : i1
        %61 = arith.andi %44, %60 : i1
        %62 = arith.subi %57, %56 : index
        %63 = arith.muli %46, %62 : index
        %64 = scf.if %61 -> (f32) {
          %65 = tensor.extract %6[%15, %16, %30, %27] : tensor<1x20x3x4xf32>
          %66 = arith.index_cast %63 : index to i64
          %67 = arith.sitofp %66 : i64 to f32
          %68 = arith.divf %65, %67 : f32
          %69 = arith.addf %25, %68 : f32
          scf.yield %69 : f32
        } else {
          scf.yield %25 : f32
        }
        scf.yield %64 : f32
      }
      linalg.yield %23 : f32
    } -> tensor<1x20x9x11xf32>
    %70 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.000566819508 : f32
    %71 = tensor.splat %70 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %72 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %73 = tensor.splat %72 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %74 = "quant_ext.quantize_per_tensor"(%13, %71, %73) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.00056681948549607225 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x20x9x11xf32>, tensor<f32>, tensor<i64>) -> tensor<1x20x9x11xi8>
    func.return %74 : tensor<1x20x9x11xi8>
  }
}