builtin.module {
  func.func @forward(%arg0: tensor<2x3xf32>) -> (tensor<1x1xf32>, tensor<2x3xf32>) {
    %c0 = arith.constant 0.0097962627187371254 : f64
    %c0_1 = tensor.splat %c0 : tensor<f64>
    %c0_2 = arith.constant 0 : i64
    %c0_3 = tensor.splat %c0_2 : tensor<i64>
    %c0_4 = "quant_ext.quantize_per_tensor"(%arg0, %c0_1, %c0_3) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<2x3xf32>, tensor<f64>, tensor<i64>) -> tensor<2x3xi8>
    %c0_5 = arith.constant 0.0097962627187371254 : f64
    %c0_6 = tensor.splat %c0_5 : tensor<f64>
    %c0_7 = arith.constant 0 : i64
    %c0_8 = tensor.splat %c0_7 : tensor<i64>
    %c0_9 = "quant_ext.dequantize_per_tensor"(%c0_4, %c0_6, %c0_8) <{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}> : (tensor<2x3xi8>, tensor<f64>, tensor<i64>) -> tensor<2x3xf32>
    %c0_10 = arith.constant 0.000000e+00 : f32
    %c0_11 = tensor.splat %c0_10 : tensor<f32>
    %c0_12 = linalg.reduce ins(%c0_9:tensor<2x3xf32>) outs(%c0_11:tensor<f32>) dimensions = [0, 1]
    (%c0_13: f32, %c0_14: f32) {
      %c0_15 = arith.addf %c0_13, %c0_14 : f32
      linalg.yield %c0_15 : f32
    }
    %c0_16 = arith.constant 6.000000e+00 : f32
    %c0_17 = tensor.splat %c0_16 : tensor<f32>
    %c0_18 = tensor.empty() : tensor<f32>
    %c0_19 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%c0_12, %c0_17 : tensor<f32>, tensor<f32>) outs(%c0_18 : tensor<f32>) {
    ^c0_bb2(%c0_20: f32, %c0_21: f32, %c0_22: f32):
      %c0_23 = arith.divf %c0_20, %c0_21 : f32
      linalg.yield %c0_23 : f32
    } -> tensor<f32>
    %c0_24 = arith.constant dense<1> : tensor<2xi64>
    %c0_25 = tensor.reshape %c0_19(%c0_24) : (tensor<f32>, tensor<2xi64>) -> tensor<1x1xf32>
    %c0_26 = arith.constant 0.0019607844296842813 : f64
    %c0_27 = tensor.splat %c0_26 : tensor<f64>
    %c0_28 = arith.constant 0 : i64
    %c0_29 = tensor.splat %c0_28 : tensor<i64>
    %c0_30 = "quant_ext.quantize_per_tensor"(%c0_25, %c0_27, %c0_29) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<1x1xf32>, tensor<f64>, tensor<i64>) -> tensor<1x1xi8>
    %c0_31 = arith.constant 0.0019607844296842813 : f64
    %c0_32 = tensor.splat %c0_31 : tensor<f64>
    %c0_33 = arith.constant 0 : i64
    %c0_34 = tensor.splat %c0_33 : tensor<i64>
    %c0_35 = "quant_ext.dequantize_per_tensor"(%c0_30, %c0_32, %c0_34) <{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}> : (tensor<1x1xi8>, tensor<f64>, tensor<i64>) -> tensor<1x1xf32>
    func.return %c0_35, %arg0 : tensor<1x1xf32>, tensor<2x3xf32>
  }
}
