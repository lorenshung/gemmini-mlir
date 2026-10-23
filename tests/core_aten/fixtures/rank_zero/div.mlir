"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<f32>) -> tensor<f32>}> ({
  ^bb0(%arg0: tensor<f32>):
    %c0 = "arith.constant"() <{value = 0.000244140625 : f64}> : () -> f64
    %c0_1 = "tensor.splat"(%c0) : (f64) -> tensor<f64>
    %c0_2 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %c0_3 = "tensor.splat"(%c0_2) : (i64) -> tensor<i64>
    %c0_4 = "quant_ext.quantize_per_tensor"(%arg0, %c0_1, %c0_3) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["1"], prov.quant.scale_denominator = ["4096"], prov.quant.scale = 0.000244140625 : f32, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even"} : (tensor<f32>, tensor<f64>, tensor<i64>) -> tensor<i8>
    %c0_5 = "arith.constant"() <{value = 0.000244140625 : f64}> : () -> f64
    %c0_6 = "tensor.splat"(%c0_5) : (f64) -> tensor<f64>
    %c0_7 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %c0_8 = "tensor.splat"(%c0_7) : (i64) -> tensor<i64>
    %c0_9 = "quant_ext.dequantize_per_tensor"(%c0_4, %c0_6, %c0_8) <{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["1"], prov.quant.scale_denominator = ["4096"], prov.quant.scale = 0.000244140625 : f32, prov.quant.zero_point = 0 : i64} : (tensor<i8>, tensor<f64>, tensor<i64>) -> tensor<f32>
    %c0_10 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %c0_11 = "tensor.splat"(%c0_10) : (f32) -> tensor<f32>
    %c0_12 = "tensor.empty"() : () -> tensor<f32>
    %c0_13 = "linalg.generic"(%c0_9, %c0_11, %c0_12) <{indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = [], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^c0_bb1(%c0_14: f32, %c0_15: f32, %c0_16: f32):
      %c0_17 = "arith.divf"(%c0_14, %c0_15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%c0_17) : (f32) -> ()
    }) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %c0_18 = "arith.constant"() <{value = 0.000244140625 : f64}> : () -> f64
    %c0_19 = "tensor.splat"(%c0_18) : (f64) -> tensor<f64>
    %c0_20 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %c0_21 = "tensor.splat"(%c0_20) : (i64) -> tensor<i64>
    %c0_22 = "quant_ext.quantize_per_tensor"(%c0_13, %c0_19, %c0_21) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["1"], prov.quant.scale_denominator = ["4096"], prov.quant.scale = 0.000244140625 : f32, prov.quant.zero_point = 0 : i64, prov.quant.rounding = "half_to_even"} : (tensor<f32>, tensor<f64>, tensor<i64>) -> tensor<i8>
    %c0_23 = "arith.constant"() <{value = 0.000244140625 : f64}> : () -> f64
    %c0_24 = "tensor.splat"(%c0_23) : (f64) -> tensor<f64>
    %c0_25 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %c0_26 = "tensor.splat"(%c0_25) : (i64) -> tensor<i64>
    %c0_27 = "quant_ext.dequantize_per_tensor"(%c0_22, %c0_24, %c0_26) <{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}> {prov.quant.storage_dtype = "int8", prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.granularity = "tensor", prov.quant.scale_numerator = ["1"], prov.quant.scale_denominator = ["4096"], prov.quant.scale = 0.000244140625 : f32, prov.quant.zero_point = 0 : i64} : (tensor<i8>, tensor<f64>, tensor<i64>) -> tensor<f32>
    "func.return"(%c0_27) : (tensor<f32>) -> ()
  }) : () -> ()
}) : () -> ()