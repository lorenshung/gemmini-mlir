builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 24, 9, 11], \22storage_offset\22: 0, \22stride\22: [2376, 99, 11, 1]}, {\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 20, 9, 11], \22storage_offset\22: 0, \22stride\22: [1980, 99, 11, 1]}, {\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [24, 20, 3, 3], \22storage_offset\22: 0, \22stride\22: [180, 9, 3, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [1, 20, 9, 11], \22storage_offset\22: 0, \22stride\22: [1980, 99, 11, 1]}, {\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [24, 20, 3, 3], \22storage_offset\22: 0, \22stride\22: [180, 9, 3, 1]}, {\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [24], \22storage_offset\22: 0, \22stride\22: [1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x24x9x11xi8>, %1: tensor<1x20x9x11xi8>, %2: tensor<24x20x3x3xi8>) -> (tensor<1x20x9x11xi8>, tensor<24x20x3x3xi8>, tensor<24xi8>) {
    %3 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.00844220538 : f32
    %4 = tensor.splat %3 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %5 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %6 = tensor.splat %5 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %7 = "quant_ext.dequantize_per_tensor"(%0, %4, %6) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0084422055412741275 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x24x9x11xi8>, tensor<f32>, tensor<i64>) -> tensor<1x24x9x11xf32>
    %8 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0298503637 : f32
    %9 = tensor.splat %8 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %10 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %11 = tensor.splat %10 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %12 = "quant_ext.dequantize_per_tensor"(%1, %9, %11) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.029850363263896867 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x20x9x11xi8>, tensor<f32>, tensor<i64>) -> tensor<1x20x9x11xf32>
    %13 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0015029537 : f32
    %14 = tensor.splat %13 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %15 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %16 = tensor.splat %15 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %17 = "quant_ext.dequantize_per_tensor"(%2, %14, %16) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0015029536742790072 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_2", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<24x20x3x3xi8>, tensor<f32>, tensor<i64>) -> tensor<24x20x3x3xf32>
    %18 = "tensor.empty"() {prov.role = "grad_input", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x9x11xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%18 : tensor<1x20x9x11xf32>) attrs =  {prov.role = "grad_input", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} {
    ^bb0(%20: f32):
      %21 = arith.constant 0 : index
      %22 = arith.constant 1 : index
      %23 = arith.constant 9 : index
      %24 = arith.constant 24 : index
      %25 = arith.constant 0.000000e+00 : f32
      %26 = linalg.index 0 : index
      %27 = linalg.index 1 : index
      %28 = linalg.index 2 : index
      %29 = linalg.index 3 : index
      %30 = arith.constant 0.000000e+00 : f32
      %31 = scf.for %32 = %21 to %23 step %22 iter_args(%33 = %30) -> (f32) {
        %34 = arith.constant 3 : index
        %35 = arith.divui %32, %34 : index
        %36 = arith.remui %32, %34 : index
        %37 = arith.constant 1 : index
        %38 = arith.constant 1 : index
        %39 = arith.constant 1 : index
        %40 = arith.constant 9 : index
        %41 = arith.addi %28, %37 : index
        %42 = arith.muli %35, %38 : index
        %43 = arith.subi %41, %42 : index
        %44 = arith.divsi %43, %39 : index
        %45 = arith.remsi %43, %39 : index
        %46 = arith.cmpi sge, %43, %21 : index
        %47 = arith.cmpi slt, %44, %40 : index
        %48 = arith.cmpi eq, %45, %21 : index
        %49 = arith.constant 1 : index
        %50 = arith.constant 1 : index
        %51 = arith.constant 1 : index
        %52 = arith.constant 11 : index
        %53 = arith.addi %29, %49 : index
        %54 = arith.muli %36, %50 : index
        %55 = arith.subi %53, %54 : index
        %56 = arith.divsi %55, %51 : index
        %57 = arith.remsi %55, %51 : index
        %58 = arith.cmpi sge, %55, %21 : index
        %59 = arith.cmpi slt, %56, %52 : index
        %60 = arith.cmpi eq, %57, %21 : index
        %61 = arith.andi %46, %47 : i1
        %62 = arith.andi %61, %48 : i1
        %63 = arith.andi %62, %58 : i1
        %64 = arith.andi %63, %59 : i1
        %65 = arith.andi %64, %60 : i1
        %66 = scf.if %65 -> (f32) {
          %67 = arith.constant 20 : index
          %68 = arith.divui %27, %67 : index
          %69 = arith.remui %27, %67 : index
          %70 = arith.muli %68, %24 : index
          %71 = scf.for %72 = %21 to %24 step %22 iter_args(%73 = %25) -> (f32) {
            %74 = arith.addi %70, %72 : index
            %75 = tensor.extract %7[%26, %74, %44, %56] : tensor<1x24x9x11xf32>
            %76 = tensor.extract %17[%74, %69, %35, %36] : tensor<24x20x3x3xf32>
            %77 = arith.mulf %75, %76 : f32
            %78 = arith.addf %73, %77 : f32
            scf.yield %78 : f32
          }
          %79 = arith.addf %33, %71 : f32
          scf.yield %79 : f32
        } else {
          scf.yield %33 : f32
        }
        scf.yield %66 : f32
      }
      linalg.yield %31 : f32
    } -> tensor<1x20x9x11xf32>
    %80 = arith.constant {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %81 = tensor.splat %80 {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : tensor<1x20x11x13xf32>
    %82 = "tensor.insert_slice"(%12, %81) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 20, 9, 11>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : (tensor<1x20x9x11xf32>, tensor<1x20x11x13xf32>) -> tensor<1x20x11x13xf32>
    %83 = arith.constant {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %84 = tensor.splat %83 {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : tensor<24x20x3x3xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, (((d0 floordiv 24) * 20) + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d0, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%82, %7 : tensor<1x20x11x13xf32>, tensor<1x24x9x11xf32>) outs(%84 : tensor<24x20x3x3xf32>) attrs =  {prov.role = "grad_weight", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} {
    ^bb1(%86: f32, %87: f32, %88: f32):
      %89 = arith.mulf %86, %87 : f32
      %90 = arith.addf %88, %89 : f32
      linalg.yield %90 : f32
    } -> tensor<24x20x3x3xf32>
    %91 = arith.constant {prov.role = "grad_bias", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %92 = tensor.splat %91 {prov.role = "grad_bias", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : tensor<24xf32>
    %93 = "linalg.reduce"(%7, %92) <{dimensions = array<i64: 0, 2, 3>}> ({
    ^bb2(%94: f32, %95: f32):
      %96 = arith.addf %94, %95 : f32
      linalg.yield %96 : f32
    }) {prov.role = "grad_bias", prov.region_id = "conv_backward_0", prov.family = "contraction", prov.conv_path = "backward_contractions", prov._pattern_hint = "convolution_backward", prov.op = "convolution_backward", prov.aten = "aten.convolution_backward.default", prov.orig_dtype = "float32"} : (tensor<1x24x9x11xf32>, tensor<24xf32>) -> tensor<24xf32>
    %97 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00605174527 : f32
    %98 = tensor.splat %97 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %99 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %100 = tensor.splat %99 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %101 = "quant_ext.quantize_per_tensor"(%19, %98, %100) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0060517451342414407 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x20x9x11xf32>, tensor<f32>, tensor<i64>) -> tensor<1x20x9x11xi8>
    %102 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0653744936 : f32
    %103 = tensor.splat %102 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %104 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %105 = tensor.splat %104 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %106 = "quant_ext.quantize_per_tensor"(%85, %103, %105) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.065374494066425398 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_1", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<24x20x3x3xf32>, tensor<f32>, tensor<i64>) -> tensor<24x20x3x3xi8>
    %107 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.0657498837 : f32
    %108 = tensor.splat %107 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %109 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %110 = tensor.splat %109 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %111 = "quant_ext.quantize_per_tensor"(%93, %108, %110) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.065749882716758579 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_2", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<24xf32>, tensor<f32>, tensor<i64>) -> tensor<24xi8>
    func.return %101, %106, %111 : tensor<1x20x9x11xi8>, tensor<24x20x3x3xi8>, tensor<24xi8>
  }
}