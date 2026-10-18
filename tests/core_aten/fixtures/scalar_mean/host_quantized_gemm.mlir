builtin.module {
  func.func @forward(%arg0: tensor<1x2x3x3xf32>, %arg2: tensor<2x1x3x3xf32>) -> tensor<9x9xi32> {
    %c0 = arith.constant 0.0078431377187371254 : f64
    %c0_1 = tensor.splat %c0 : tensor<f64>
    %c0_2 = arith.constant 0 : i64
    %c0_3 = tensor.splat %c0_2 : tensor<i64>
    %c0_4 = "quant_ext.quantize_per_tensor"(%arg0, %c0_1, %c0_3) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["8421505"], prov.quant.scale_denominator = ["1073741824"], prov.quant.scale = 0.00784313772 : f32, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even"} : (tensor<1x2x3x3xf32>, tensor<f64>, tensor<i64>) -> tensor<1x2x3x3xi8>
    %c0_5 = arith.constant 0.0097962627187371254 : f64
    %c0_6 = tensor.splat %c0_5 : tensor<f64>
    %c0_7 = arith.constant 0 : i64
    %c0_8 = tensor.splat %c0_7 : tensor<i64>
    %c0_9 = "quant_ext.quantize_per_tensor"(%arg2, %c0_6, %c0_8) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["10518657"], prov.quant.scale_denominator = ["1073741824"], prov.quant.scale = 0.00979626272 : f32, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even"} : (tensor<2x1x3x3xf32>, tensor<f64>, tensor<i64>) -> tensor<2x1x3x3xi8>
    %c0_10 = tensor.empty() : tensor<2x1x3x3xi8>
    %c0_11 = linalg.transpose ins(%c0_4:tensor<1x2x3x3xi8>) outs(%c0_10:tensor<2x1x3x3xi8>) permutation = [1, 0, 2, 3]
    %c0_12 = tensor.collapse_shape %c0_11 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<2x1x3x3xi8> into tensor<18xi8>
    %c0_13 = tensor.expand_shape %c0_12 [[0 : i64, 1 : i64]] output_shape [2, 9] : tensor<18xi8> into tensor<2x9xi8>
    %c0_14 = tensor.collapse_shape %c0_9 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<2x1x3x3xi8> into tensor<18xi8>
    %c0_15 = tensor.expand_shape %c0_14 [[0 : i64, 1 : i64]] output_shape [2, 9] : tensor<18xi8> into tensor<2x9xi8>
    %c0_16 = tensor.empty() : tensor<9x2xi8>
    %c0_17 = linalg.transpose ins(%c0_15:tensor<2x9xi8>) outs(%c0_16:tensor<9x2xi8>) permutation = [1, 0]
    %c0_18 = arith.constant {prov.quant.lhs_scale = 0.0097962627187371254 : f64, prov.quant.rhs_scale = 0.0078431377187371254 : f64, prov.quant.accumulator = "i32", prov.quant.numerics = "integer_readout", prov.quant.arithmetic = "exact", prov.quant.host_only = "false", prov.quant.gradient = "input", prov.quant.scale_numerator = ["88582922518785"], prov.quant.scale_denominator = ["1152921504606846976"]} 0 : i32
    %c0_19 = tensor.splat %c0_18 {prov.quant.lhs_scale = 0.0097962627187371254 : f64, prov.quant.rhs_scale = 0.0078431377187371254 : f64, prov.quant.accumulator = "i32", prov.quant.numerics = "integer_readout", prov.quant.arithmetic = "exact", prov.quant.host_only = "false", prov.quant.gradient = "input", prov.quant.scale_numerator = ["88582922518785"], prov.quant.scale_denominator = ["1152921504606846976"]} : tensor<9x9xi32>
    %c0_20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%c0_17, %c0_13 : tensor<9x2xi8>, tensor<2x9xi8>) outs(%c0_19 : tensor<9x9xi32>) attrs =  {prov.quant.lhs_scale = 0.0097962627187371254 : f64, prov.quant.rhs_scale = 0.0078431377187371254 : f64, prov.quant.accumulator = "i32", prov.quant.numerics = "integer_readout", prov.quant.arithmetic = "exact", prov.quant.host_only = "false", prov.quant.gradient = "input", prov.quant.scale_numerator = ["88582922518785"], prov.quant.scale_denominator = ["1152921504606846976"]} {
    ^c0_bb3(%c0_21: i8, %c0_22: i8, %c0_23: i32):
      %c0_24 = arith.extsi %c0_21 : i8 to i32
      %c0_25 = arith.extsi %c0_22 : i8 to i32
      %c0_26 = arith.muli %c0_24, %c0_25 : i32
      %c0_27 = arith.addi %c0_23, %c0_26 : i32
      linalg.yield %c0_27 : i32
    } -> tensor<9x9xi32>
    func.return %c0_20 : tensor<9x9xi32>
  }
}
