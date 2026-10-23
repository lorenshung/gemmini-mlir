// model2mlir cb1a3da: bmm(a.to(int32), b.to(int32)); signed i8 inputs.
builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [2, 3, 17], \22storage_offset\22: 0, \22stride\22: [51, 17, 1]}, {\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [2, 17, 5], \22storage_offset\22: 0, \22stride\22: [85, 5, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [2, 3, 5], \22storage_offset\22: 0, \22stride\22: [15, 5, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x3x17xi8>, %1: tensor<2x17x5xi8>) -> tensor<2x3x5xi32> {
    %2 = "tensor.empty"() {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} : () -> tensor<2x3x17xi32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<2x3x17xi8>) outs(%2 : tensor<2x3x17xi32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb0(%4: i8, %5: i32):
      %6 = arith.extsi %4 : i8 to i32
      linalg.yield %6 : i32
    } -> tensor<2x3x17xi32>
    %7 = "tensor.empty"() {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} : () -> tensor<2x17x5xi32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<2x17x5xi8>) outs(%7 : tensor<2x17x5xi32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb1(%9: i8, %10: i32):
      %11 = arith.extsi %9 : i8 to i32
      linalg.yield %11 : i32
    } -> tensor<2x17x5xi32>
    %12 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "int32"} 0 : i32
    %13 = tensor.splat %12 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "int32"} : tensor<2x3x5xi32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<2x3x17xi8>, tensor<2x17x5xi8>) outs(%13 : tensor<2x3x5xi32>) attrs =  {prov.op = "int_matmul", prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "int32"} {
    ^bb2(%15: i8, %16: i8, %17: i32):
      %18 = arith.extsi %15 : i8 to i32
      %19 = arith.extsi %16 : i8 to i32
      %20 = arith.muli %18, %19 : i32
      %21 = arith.addi %17, %20 : i32
      linalg.yield %21 : i32
    } -> tensor<2x3x5xi32>
    func.return %14 : tensor<2x3x5xi32>
  }
}