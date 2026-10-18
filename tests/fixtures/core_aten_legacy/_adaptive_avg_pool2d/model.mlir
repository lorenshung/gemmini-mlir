builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 20, 9, 11], \22storage_offset\22: 0, \22stride\22: [1980, 99, 11, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [1, 20, 3, 4], \22storage_offset\22: 0, \22stride\22: [240, 12, 4, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x20x9x11xi8>) -> tensor<1x20x3x4xi8> {
    %1 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0254488811 : f32
    %2 = tensor.splat %1 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %3 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %4 = tensor.splat %3 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %5 = "quant_ext.dequantize_per_tensor"(%0, %2, %4) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.025448881411084943 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x20x9x11xi8>, tensor<f32>, tensor<i64>) -> tensor<1x20x9x11xf32>
    %6 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %7 = tensor.splat %6 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x4xf32>
    %8 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %9 = tensor.splat %8 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %10 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<3x3xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 3) + d4), (d3 + d5))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %10 : tensor<1x20x9x11xf32>, tensor<3x3xf32>) outs(%9 : tensor<1x20x3x1xf32>) attrs =  {prov.window = "kernel=3x3,stride=3x1", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb0(%12: f32, %13: f32, %14: f32):
      %15 = arith.addf %14, %12 : f32
      linalg.yield %15 : f32
    } -> tensor<1x20x3x1xf32>
    %16 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %17 = tensor.splat %16 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %18 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x1xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %17 : tensor<1x20x3x1xf32>, tensor<1x20x3x1xf32>) outs(%18 : tensor<1x20x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb1(%20: f32, %21: f32, %22: f32):
      %23 = arith.divf %20, %21 : f32
      linalg.yield %23 : f32
    } -> tensor<1x20x3x1xf32>
    %24 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %25 = tensor.splat %24 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %26 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x1xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19, %25 : tensor<1x20x3x1xf32>, tensor<1x20x3x1xf32>) outs(%26 : tensor<1x20x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb2(%28: f32, %29: f32, %30: f32):
      %31 = arith.divf %28, %29 : f32
      linalg.yield %31 : f32
    } -> tensor<1x20x3x1xf32>
    %32 = "tensor.insert_slice"(%27, %7) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 20, 3, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : (tensor<1x20x3x1xf32>, tensor<1x20x3x4xf32>) -> tensor<1x20x3x4xf32>
    %33 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %34 = tensor.splat %33 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x2xf32>
    %35 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<3x4xf32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 3) + d4), (((d3 * 3) + d5) + 2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %35 : tensor<1x20x9x11xf32>, tensor<3x4xf32>) outs(%34 : tensor<1x20x3x2xf32>) attrs =  {prov.window = "kernel=3x4,stride=3x3", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb3(%37: f32, %38: f32, %39: f32):
      %40 = arith.addf %39, %37 : f32
      linalg.yield %40 : f32
    } -> tensor<1x20x3x2xf32>
    %41 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %42 = tensor.splat %41 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x2xf32>
    %43 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x2xf32>
    %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %42 : tensor<1x20x3x2xf32>, tensor<1x20x3x2xf32>) outs(%43 : tensor<1x20x3x2xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb4(%45: f32, %46: f32, %47: f32):
      %48 = arith.divf %45, %46 : f32
      linalg.yield %48 : f32
    } -> tensor<1x20x3x2xf32>
    %49 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 4.000000e+00 : f32
    %50 = tensor.splat %49 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x2xf32>
    %51 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x2xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44, %50 : tensor<1x20x3x2xf32>, tensor<1x20x3x2xf32>) outs(%51 : tensor<1x20x3x2xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb5(%53: f32, %54: f32, %55: f32):
      %56 = arith.divf %53, %54 : f32
      linalg.yield %56 : f32
    } -> tensor<1x20x3x2xf32>
    %57 = "tensor.insert_slice"(%52, %32) <{static_offsets = array<i64: 0, 0, 0, 1>, static_sizes = array<i64: 1, 20, 3, 2>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : (tensor<1x20x3x2xf32>, tensor<1x20x3x4xf32>) -> tensor<1x20x3x4xf32>
    %58 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %59 = tensor.splat %58 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %60 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<3x3xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 3) + d4), ((d3 + d5) + 8))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %60 : tensor<1x20x9x11xf32>, tensor<3x3xf32>) outs(%59 : tensor<1x20x3x1xf32>) attrs =  {prov.window = "kernel=3x3,stride=3x1", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb6(%62: f32, %63: f32, %64: f32):
      %65 = arith.addf %64, %62 : f32
      linalg.yield %65 : f32
    } -> tensor<1x20x3x1xf32>
    %66 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %67 = tensor.splat %66 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %68 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x1xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %67 : tensor<1x20x3x1xf32>, tensor<1x20x3x1xf32>) outs(%68 : tensor<1x20x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb7(%70: f32, %71: f32, %72: f32):
      %73 = arith.divf %70, %71 : f32
      linalg.yield %73 : f32
    } -> tensor<1x20x3x1xf32>
    %74 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %75 = tensor.splat %74 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : tensor<1x20x3x1xf32>
    %76 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : () -> tensor<1x20x3x1xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %75 : tensor<1x20x3x1xf32>, tensor<1x20x3x1xf32>) outs(%76 : tensor<1x20x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} {
    ^bb8(%78: f32, %79: f32, %80: f32):
      %81 = arith.divf %78, %79 : f32
      linalg.yield %81 : f32
    } -> tensor<1x20x3x1xf32>
    %82 = "tensor.insert_slice"(%77, %57) <{static_offsets = array<i64: 0, 0, 0, 3>, static_sizes = array<i64: 1, 20, 3, 1>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool2d.default", prov.orig_dtype = "float32"} : (tensor<1x20x3x1xf32>, tensor<1x20x3x4xf32>) -> tensor<1x20x3x4xf32>
    %83 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00825009495 : f32
    %84 = tensor.splat %83 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %85 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %86 = tensor.splat %85 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %87 = "quant_ext.quantize_per_tensor"(%82, %84, %86) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0082500953300326477 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x20x3x4xf32>, tensor<f32>, tensor<i64>) -> tensor<1x20x3x4xi8>
    func.return %87 : tensor<1x20x3x4xi8>
  }
}