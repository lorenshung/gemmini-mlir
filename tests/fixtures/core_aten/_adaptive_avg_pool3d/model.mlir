builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [1, 16, 5, 6, 7], \22storage_offset\22: 0, \22stride\22: [3360, 210, 42, 7, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int8\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [1, 16, 2, 3, 4], \22storage_offset\22: 0, \22stride\22: [384, 24, 12, 4, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1x16x5x6x7xi8>) -> tensor<1x16x2x3x4xi8> {
    %1 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0261582322 : f32
    %2 = tensor.splat %1 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %3 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %4 = tensor.splat %3 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %5 = "quant_ext.dequantize_per_tensor"(%0, %2, %4) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.02615823184742647 : f64, prov.quant.zero_point = 0 : i64, prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x16x5x6x7xi8>, tensor<f32>, tensor<i64>) -> tensor<1x16x5x6x7xf32>
    %6 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %7 = tensor.splat %6 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x4xf32>
    %8 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %9 = tensor.splat %8 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %10 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<3x2x2xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, ((d2 * 2) + d5), ((d3 * 2) + d6), (d4 + d7))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%5, %10 : tensor<1x16x5x6x7xf32>, tensor<3x2x2xf32>) outs(%9 : tensor<1x16x2x3x1xf32>) attrs =  {prov.window = "kernel=3x2x2,stride=2x2x1", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb0(%12: f32, %13: f32, %14: f32):
      %15 = arith.addf %14, %12 : f32
      linalg.yield %15 : f32
    } -> tensor<1x16x2x3x1xf32>
    %16 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %17 = tensor.splat %16 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %18 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%11, %17 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%18 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb1(%20: f32, %21: f32, %22: f32):
      %23 = arith.divf %20, %21 : f32
      linalg.yield %23 : f32
    } -> tensor<1x16x2x3x1xf32>
    %24 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %25 = tensor.splat %24 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %26 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%19, %25 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%26 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb2(%28: f32, %29: f32, %30: f32):
      %31 = arith.divf %28, %29 : f32
      linalg.yield %31 : f32
    } -> tensor<1x16x2x3x1xf32>
    %32 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %33 = tensor.splat %32 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %34 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%27, %33 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%34 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb3(%36: f32, %37: f32, %38: f32):
      %39 = arith.divf %36, %37 : f32
      linalg.yield %39 : f32
    } -> tensor<1x16x2x3x1xf32>
    %40 = "tensor.insert_slice"(%35, %7) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 16, 2, 3, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : (tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x4xf32>) -> tensor<1x16x2x3x4xf32>
    %41 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %42 = tensor.splat %41 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x2xf32>
    %43 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<3x2x3xf32>
    %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, ((d2 * 2) + d5), ((d3 * 2) + d6), (((d4 * 2) + d7) + 1))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%5, %43 : tensor<1x16x5x6x7xf32>, tensor<3x2x3xf32>) outs(%42 : tensor<1x16x2x3x2xf32>) attrs =  {prov.window = "kernel=3x2x3,stride=2x2x2", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb4(%45: f32, %46: f32, %47: f32):
      %48 = arith.addf %47, %45 : f32
      linalg.yield %48 : f32
    } -> tensor<1x16x2x3x2xf32>
    %49 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %50 = tensor.splat %49 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x2xf32>
    %51 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x2xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%44, %50 : tensor<1x16x2x3x2xf32>, tensor<1x16x2x3x2xf32>) outs(%51 : tensor<1x16x2x3x2xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb5(%53: f32, %54: f32, %55: f32):
      %56 = arith.divf %53, %54 : f32
      linalg.yield %56 : f32
    } -> tensor<1x16x2x3x2xf32>
    %57 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %58 = tensor.splat %57 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x2xf32>
    %59 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x2xf32>
    %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%52, %58 : tensor<1x16x2x3x2xf32>, tensor<1x16x2x3x2xf32>) outs(%59 : tensor<1x16x2x3x2xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb6(%61: f32, %62: f32, %63: f32):
      %64 = arith.divf %61, %62 : f32
      linalg.yield %64 : f32
    } -> tensor<1x16x2x3x2xf32>
    %65 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %66 = tensor.splat %65 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x2xf32>
    %67 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x2xf32>
    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%60, %66 : tensor<1x16x2x3x2xf32>, tensor<1x16x2x3x2xf32>) outs(%67 : tensor<1x16x2x3x2xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb7(%69: f32, %70: f32, %71: f32):
      %72 = arith.divf %69, %70 : f32
      linalg.yield %72 : f32
    } -> tensor<1x16x2x3x2xf32>
    %73 = "tensor.insert_slice"(%68, %40) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 16, 2, 3, 2>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : (tensor<1x16x2x3x2xf32>, tensor<1x16x2x3x4xf32>) -> tensor<1x16x2x3x4xf32>
    %74 = arith.constant {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %75 = tensor.splat %74 {prov.role = "acc_init", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %76 = "tensor.empty"() {prov.role = "window_shape", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<3x2x2xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, ((d2 * 2) + d5), ((d3 * 2) + d6), ((d4 + d7) + 5))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%5, %76 : tensor<1x16x5x6x7xf32>, tensor<3x2x2xf32>) outs(%75 : tensor<1x16x2x3x1xf32>) attrs =  {prov.window = "kernel=3x2x2,stride=2x2x1", prov.role = "window_sum", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb8(%78: f32, %79: f32, %80: f32):
      %81 = arith.addf %80, %78 : f32
      linalg.yield %81 : f32
    } -> tensor<1x16x2x3x1xf32>
    %82 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %83 = tensor.splat %82 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %84 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%77, %83 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%84 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb9(%86: f32, %87: f32, %88: f32):
      %89 = arith.divf %86, %87 : f32
      linalg.yield %89 : f32
    } -> tensor<1x16x2x3x1xf32>
    %90 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %91 = tensor.splat %90 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %92 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%85, %91 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%92 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb10(%94: f32, %95: f32, %96: f32):
      %97 = arith.divf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<1x16x2x3x1xf32>
    %98 = arith.constant {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %99 = tensor.splat %98 {prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : tensor<1x16x2x3x1xf32>
    %100 = "tensor.empty"() {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : () -> tensor<1x16x2x3x1xf32>
    %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%93, %99 : tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x1xf32>) outs(%100 : tensor<1x16x2x3x1xf32>) attrs =  {prov.role = "divide", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} {
    ^bb11(%102: f32, %103: f32, %104: f32):
      %105 = arith.divf %102, %103 : f32
      linalg.yield %105 : f32
    } -> tensor<1x16x2x3x1xf32>
    %106 = "tensor.insert_slice"(%101, %73) <{static_offsets = array<i64: 0, 0, 0, 0, 3>, static_sizes = array<i64: 1, 16, 2, 3, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.role = "assemble", prov.movement_form = "copy", prov._pattern_hint = "avg_pool", prov.op = "avg_pool", prov.family = "reduce", prov.aten = "aten._adaptive_avg_pool3d.default", prov.orig_dtype = "float32"} : (tensor<1x16x2x3x1xf32>, tensor<1x16x2x3x4xf32>) -> tensor<1x16x2x3x4xf32>
    %107 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.00742004765 : f32
    %108 = tensor.splat %107 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %109 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %110 = tensor.splat %109 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %111 = "quant_ext.quantize_per_tensor"(%106, %108, %110) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.qmin = -128 : i64, prov.quant.qmax = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale = 0.0074200475917142981 : f64, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even", prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x16x2x3x4xf32>, tensor<f32>, tensor<i64>) -> tensor<1x16x2x3x4xi8>
    func.return %111 : tensor<1x16x2x3x4xi8>
  }
}